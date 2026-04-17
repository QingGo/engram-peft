import gc
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Literal, cast

import torch
import wandb
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import examples.benchmarks.methods as methods
from engram_peft.utils import evaluate_model_loss
from examples.benchmarks.data import prepare_dataset
from examples.benchmarks.persistence import BenchmarkResult, ResultManager


class Logger:
    """Tee logger to write to both stdout and a file."""

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


class BenchmarkEngine:
    def __init__(self, model_name: str, args: Any):
        self.model_name = model_name
        self.args = args
        self.result_manager = ResultManager()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup Logging
        os.makedirs("outputs/benchmarks", exist_ok=True)
        log_path = os.path.join(
            "outputs/benchmarks", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        sys.stdout = Logger(log_path)
        sys.stderr = sys.stdout
        print(f"Logging started. Saving to {log_path}")

        # WandB Setup
        self.wandb_enabled = args.wandb and wandb is not None
        if args.wandb and wandb is None:
            print("Warning: wandb is not installed. Tracking disabled.")

        self.train_dataset: Any
        self.eval_dataset: Any
        self.train_dataset, self.eval_dataset = prepare_dataset(
            self.tokenizer,
            subset_size=args.subset,
            eval_size=200,
            max_length=args.max_length,
            num_proc=args.num_workers,
        )

        self.base_model: PreTrainedModel | None = None
        self.is_dirty = False
        self.results: dict[str, BenchmarkResult] = {}

    def load_model(self) -> PreTrainedModel:
        print(f"Loading Base Model: {self.model_name}...")
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map="auto"
        )
        return model

    def get_fresh_model(self) -> PreTrainedModel:
        if self.base_model is not None:
            del self.base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.base_model = self.load_model()
        self.is_dirty = False
        return self.base_model

    def run_method(self, method_spec: str) -> None:
        # Parse "method:param1=val1,param2=val2"
        # Support for complex types (lists, nested dicts) via OmegaConf
        overrides: dict[str, Any] = {}
        if ":" in method_spec:
            method_name, overrides_str = method_spec.split(":", 1)
            # Bracket-aware split to avoid breaking parameters like target_layers=[2,11]
            pairs = []
            current: list[str] = []
            bracket_level = 0
            for char in overrides_str:
                if char == "[":
                    bracket_level += 1
                elif char == "]":
                    bracket_level -= 1

                if char == "," and bracket_level == 0:
                    pairs.append("".join(current))
                    current = []
                else:
                    current.append(char)
            if current:
                pairs.append("".join(current))

            conf = OmegaConf.from_dotlist(pairs)
            overrides = cast(
                "dict[str, Any]", OmegaConf.to_container(conf, resolve=True)
            )
        else:
            method_name = method_spec

        print(f"\n{'=' * 20} Running: {method_spec} {'=' * 20}")

        # Start WandB Run for this method
        run = None
        if self.wandb_enabled:
            mode: Literal["online", "offline", "disabled", "shared"] = (
                "offline" if getattr(self.args, "wandb_offline", False) else "online"
            )
            run = wandb.init(
                project=getattr(self.args, "wandb_project", "engram-peft-benchmarks"),
                entity=getattr(self.args, "wandb_entity", None),
                name=f"{method_spec}_{datetime.now().strftime('%m%d_%H%M')}",
                mode=mode,
                config={**vars(self.args), **overrides},
                reinit=True,
            )

        if self.is_dirty or self.base_model is None:
            self.get_fresh_model()
        assert self.base_model is not None

        # Phase 0: Capture Baseline if not exists in results
        if "base" not in self.results:
            print("Capturing Zero-shot Baseline...")
            base_loss = evaluate_model_loss(
                self.base_model,
                self.tokenizer,
                self.eval_dataset,
                batch_size=self.args.batch_size,
                max_length=self.args.max_length,
            )
            self.results["base"] = BenchmarkResult(
                method="base",
                params={"model": self.model_name},
                metrics={"eval_loss": base_loss},
            )

        # Log baseline to WandB summary if run exists
        if run:
            run.summary["baseline_eval_loss"] = self.results["base"].metrics[
                "eval_loss"
            ]

        # Run training
        train_fn = getattr(methods, f"train_{method_name}", None)
        if train_fn is None:
            print(f"Warning: Unknown method {method_name}")
            if run:
                run.finish()
            return

        metrics = train_fn(
            self.base_model,
            self.tokenizer,
            self.train_dataset,
            self.eval_dataset,
            self.args,
            overrides=overrides,
        )

        # Save result (local JSON)
        full_params = {**vars(self.args), **overrides}
        result = BenchmarkResult(
            method=method_spec, params=full_params, metrics=metrics
        )
        path = self.result_manager.save(result)
        print(f"Result saved to {path}")
        self.results[method_spec] = result

        # Log final metrics to WandB
        if run:
            run.log(
                {
                    "eval_loss": metrics["eval_loss"],
                    "peak_memory_gb": metrics["peak_memory_gb"],
                }
            )
            # Also log log_history as a table or flat line
            for log in metrics.get("log_history", []):
                run.log(log)
            run.finish()

        # Update dirty flag for full-ft methods
        if "full_finetune" in method_name:
            self.is_dirty = True

        # Cleanup tmp directory (checkpoints)
        tmp_dir = "outputs/benchmarks/tmp"
        if os.path.exists(tmp_dir):
            print(f"Cleaning up temporary checkpoints in {tmp_dir}...")
            shutil.rmtree(tmp_dir)

    def run_all(self, method_names: list[str]) -> None:
        for name in method_names:
            self.run_method(name)
        self.print_summary()

    def print_summary(self) -> None:
        """Prints a ASCII summary table of the recent results."""
        print("\n" + "=" * 25 + " Summary Table " + "=" * 25)
        print(
            f"{'Method':<20} | {'Peak Mem (GB)':<15} | {'Avg Time/Step (s)':<18} | {'Eval Loss'}"
        )
        print("-" * 70)

        for method, result in self.results.items():
            metrics = result.metrics
            print(
                f"{method.capitalize():<20} | "
                f"{metrics.get('peak_memory_gb', 0):<15.2f} | "
                f"{metrics.get('avg_time_per_step', 0):<18.4f} | "
                f"{metrics.get('eval_loss', 0):.4f}"
            )
        print("=" * 65 + "\n")
