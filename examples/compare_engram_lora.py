"""
Engram-PEFT Baseline Comparison Script.

This script demonstrates how Engram compares against standard PEFT methods (like LoRA):
1. Benchmarking: Compares training loss, peak memory, and time-per-step.
2. Baselines: Evaluates Zero-shot Base Model and trained LoRA adapter.
3. Visualization: Generates a premium loss curve comparison using Seaborn.
4. Lifecycle: Demonstrates full training and dynamic management.

Usage:
    uv run python examples/compare_engram_lora.py --max_steps 3000 --batch_size 16 --grad_accum 2 --num_workers 4 --subset 100000 --methods lora engram lora+engram
"""

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, Tuple, cast

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset  # type: ignore
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramLayer,
    EngramModel,
    EngramTrainer,
    get_engram_model,
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR = "outputs/engram_test"
ENGRAM_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "engram_weights")
LORA_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "lora_weights")
LORA_ENGRAM_DIR = os.path.join(OUTPUT_DIR, "lora_engram_weights")
SEED = 42

set_seed(SEED)


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


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    subset_size: int,
    eval_size: int,
    max_length: int,
    num_proc: int = 4,
) -> Tuple[Any, Any]:
    print(f"Loading TinyStories dataset (train split)...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    print(f"Loading TinyStories dataset (validation split)...")
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)

    # Subsample immediately for speed
    train_ds = train_ds.select(range(subset_size))
    val_ds = val_ds.select(range(min(len(val_ds), eval_size)))

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Convert to dict to avoid BatchEncoding .copy() issues
        tokenized_dict = dict(tokenized)
        tokenized_dict["labels"] = copy.deepcopy(tokenized_dict["input_ids"])
        return tokenized_dict

    print(f"Tokenizing datasets with {num_proc} processes...")
    train_dataset = train_ds.map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=num_proc
    )
    eval_dataset = val_ds.map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=num_proc
    )

    print(
        f"Dataset prepared: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples (official split)."
    )
    return train_dataset, eval_dataset


def get_base_model_eval_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Any,
    batch_size: int,
) -> float:
    print("\n>>> Phase 1: Evaluating Base Model (Zero-shot Loss) ")
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "base_eval"),
        per_device_eval_batch_size=batch_size,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        and torch.cuda.is_available(),
        dataloader_num_workers=4,  # Constant for quick eval
        dataloader_pin_memory=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=dataset,
        data_collator=DefaultDataCollator(),
    )
    results = trainer.evaluate()
    return cast(float, results.get("eval_loss", 0.0))


def train_lora(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n>>> Phase 2: Training LoRA Baseline on {MODEL_NAME}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    warmup_steps = int(args.max_steps * 0.03)
    # Configure Warmup-Stable-Decay (WSD) parameters
    # 3% Warmup, 20% Stable (Hold), 77% Decay
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 3e-4,
    }

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "lora_outputs"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_steps=warmup_steps,
        logging_steps=20,
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
    )

    print("Starting LoRA training...")
    train_result = trainer.train()

    print(f"Evaluating LoRA and saving to {LORA_WEIGHTS_DIR}...")
    eval_results = trainer.evaluate()
    eval_loss = cast(float, eval_results.get("eval_loss", 0.0))
    trainer.save_model(LORA_WEIGHTS_DIR)

    avg_time_per_step = train_result.metrics["train_runtime"] / train_result.global_step
    print(f"LoRA Avg Time Per Step: {avg_time_per_step:.4f}s")

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"LoRA Peak Memory: {peak_memory:.2f} GB")

    log_history = []
    for log in trainer.state.log_history:
        if "step" in log:
            entry = {"step": log["step"]}
            if "loss" in log:
                entry["loss"] = log["loss"]
            if "eval_loss" in log:
                entry["eval_loss"] = log["eval_loss"]
            if len(entry) > 1:
                log_history.append(entry)

    # Restore base model to normal by unloading LoRA (clean slate for Engram)
    model = model.unload()

    return {
        "log_history": log_history,
        "peak_memory_gb": peak_memory,
        "avg_time_per_step": avg_time_per_step,
        "eval_loss": eval_loss,
    }


def train_engram_model(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n>>> Phase 3: Training Engram on {MODEL_NAME}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    config = EngramConfig(
        target_layers=[2, 11],
        engram_vocab_size_per_ngram=[256000, 256000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=1024,
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=MODEL_NAME,
        pad_id=tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0,
        learning_rate_multiplier=3.0,
    )

    print("Injecting Engram layers and freezing base model...")
    model = get_engram_model(base_model, config, tokenizer)
    model.print_trainable_parameters()

    warmup_steps = int(args.max_steps * 0.03)
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 3e-4,
    }

    collator = EngramDataCollator(tokenizer=tokenizer, config=config)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_steps=warmup_steps,
        logging_steps=20,
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("Starting Engram training...")
    train_result = trainer.train()

    print("Evaluating Engram...")
    eval_results = trainer.evaluate()
    eval_loss = cast(float, eval_results.get("eval_loss", 0.0))

    avg_time_per_step = train_result.metrics["train_runtime"] / train_result.global_step
    print(f"Engram Avg Time Per Step: {avg_time_per_step:.4f}s")

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"Engram Peak Memory: {peak_memory:.2f} GB")

    log_history = []
    for log in trainer.state.log_history:
        if "step" in log:
            entry = {"step": log["step"]}
            if "loss" in log:
                entry["loss"] = log["loss"]
            if "eval_loss" in log:
                entry["eval_loss"] = log["eval_loss"]
            if len(entry) > 1:
                log_history.append(entry)

    print(f"Saving Engram weights to {ENGRAM_WEIGHTS_DIR}")
    model.save_pretrained(ENGRAM_WEIGHTS_DIR)

    # CRITICAL: Unload Engram hooks from the base model so that the next phase
    # (inference) starts with a clean slate when it creates its own wrapper.
    model.unload_engram()

    return {
        "log_history": log_history,
        "peak_memory_gb": peak_memory,
        "avg_time_per_step": avg_time_per_step,
        "eval_loss": eval_loss,
    }


def train_lora_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n>>> Phase 2+3: Training LoRA + Engram on {MODEL_NAME}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1. Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, peft_config)

    # 2. Apply Engram
    engram_config = EngramConfig(
        target_layers=[2, 11],
        engram_vocab_size_per_ngram=[256000, 256000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=1024,
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=MODEL_NAME,
        pad_id=tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0,
        learning_rate_multiplier=3.0,
    )
    model = get_engram_model(cast(PreTrainedModel, model), engram_config, tokenizer, wrap_peft=True)
    model.print_trainable_parameters()

    warmup_steps = int(args.max_steps * 0.03)
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 3e-4,
    }

    collator = EngramDataCollator(tokenizer=tokenizer, config=engram_config)
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "lora_engram_outputs"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_steps=warmup_steps,
        logging_steps=20,
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("Starting LoRA + Engram training...")
    train_result = trainer.train()

    print("Evaluating Combined Model...")
    eval_results = trainer.evaluate()
    eval_loss = cast(float, eval_results.get("eval_loss", 0.0))

    avg_time_per_step = train_result.metrics["train_runtime"] / train_result.global_step
    print(f"Combined Avg Time Per Step: {avg_time_per_step:.4f}s")

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"Combined Peak Memory: {peak_memory:.2f} GB")

    log_history = []
    for log in trainer.state.log_history:
        if "step" in log:
            entry = {"step": log["step"]}
            if "loss" in log:
                entry["loss"] = log["loss"]
            if "eval_loss" in log:
                entry["eval_loss"] = log["eval_loss"]
            if len(entry) > 1:
                log_history.append(entry)

    print(f"Saving Combined weights to {LORA_ENGRAM_DIR}")
    model.save_pretrained(LORA_ENGRAM_DIR)
    # Also save LoRA part
    model.base_model.save_pretrained(LORA_ENGRAM_DIR)

    # Unload both
    model.unload_engram()
    model = model.base_model.unload()

    return {
        "log_history": log_history,
        "peak_memory_gb": peak_memory,
        "avg_time_per_step": avg_time_per_step,
        "eval_loss": eval_loss,
    }


def save_and_plot_results(results: Dict[str, Any]) -> None:
    print(f"\n>>> Saving outputs and generating plot in {OUTPUT_DIR}")

    json_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved metric data to {json_path}")

    # Print Summary comparison
    print("\n" + "=" * 20 + " Performance Summary " + "=" * 20)
    print(
        f"{'Method':<15} | {'Peak Mem (GB)':<15} | {'Avg Time/Step (s)':<18} | {'Eval Loss':<10}"
    )
    print("-" * 65)

    base = results.get("base_model", {})
    print(
        f"{'Base':<15} | {base.get('peak_memory_gb', 0):<15.2f} | {'N/A':<18} | {base.get('eval_loss', 0):.4f}"
    )

    for method in ["lora", "engram", "lora+engram"]:
        if method in results:
            data = results[method]
            print(
                f"{method.capitalize():<15} | {data.get('peak_memory_gb', 0):<15.2f} | {data.get('avg_time_per_step', 0):<18.4f} | {data.get('eval_loss', 0):.4f}"
            )
    print("=" * 65 + "\n")

    # Set premium Seaborn theme
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))

    import pandas as pd

    # Prepare data for Seaborn
    plot_data = []
    total_eval_df = []

    palette_train = {
        "engram": "seagreen",
        "lora": "royalblue",
        "lora+engram": "darkviolet",
    }
    palette_eval = {
        "engram": "springgreen",
        "lora": "skyblue",
        "lora+engram": "plum",
        "engram_final": "darkgreen",
        "lora_final": "midnightblue",
        "lora+engram_final": "indigo",
    }

    for method, data in results.items():
        if method == "base_model":
            continue

        method_logs = data.get("log_history", [])
        for item in method_logs:
            if "loss" in item:
                plot_data.append(
                    {
                        "Step": item["step"],
                        "Loss": item["loss"],
                        "Method": f"{method} (Train)",
                    }
                )
            if "eval_loss" in item:
                total_eval_df.append(
                    {
                        "Step": item["step"],
                        "Loss": item["eval_loss"],
                        "Method": f"{method} (Eval)",
                    }
                )

        # Add final eval point
        if "eval_loss" in data:
            max_step = max([item["step"] for item in method_logs]) if method_logs else 0
            total_eval_df.append(
                {
                    "Step": max_step,
                    "Loss": data["eval_loss"],
                    "Method": f"{method} (Final Eval)",
                }
            )

    ax = plt.gca()

    if plot_data:
        df = pd.DataFrame(plot_data)
        # Create mapping for lineplot palette
        line_palette = {
            f"{m} (Train)": palette_train.get(m, "gray")
            for m in results.keys()
            if m != "base_model"
        }
        sns.lineplot(
            data=df,
            x="Step",
            y="Loss",
            hue="Method",
            linewidth=2.5,
            palette=line_palette,
            ax=ax,
        )

    if total_eval_df:
        eval_df = pd.DataFrame(total_eval_df)
        # Create mapping for scatter palette
        scatter_palette = {}
        for m in results.keys():
            if m == "base_model":
                continue
            scatter_palette[f"{m} (Eval)"] = palette_eval.get(m, "gray")
            scatter_palette[f"{m} (Final Eval)"] = palette_eval.get(
                f"{m}_final", "black"
            )

        sns.scatterplot(
            data=eval_df,
            x="Step",
            y="Loss",
            hue="Method",
            s=120,
            marker="o",
            palette=scatter_palette,
            legend=True,
            ax=ax,
            edgecolor="white",
            zorder=5,
        )

    # Base Loss (Zero-shot)
    base_loss = results.get("base_model", {}).get("eval_loss", 0.0)
    if base_loss > 0:
        ax.axhline(
            y=base_loss,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label="Base Model (Zero-shot)",
            alpha=0.8,
        )

    plt.title("Training Loss Comparison", pad=20, fontweight="bold")
    plt.xlabel("Training Steps", labelpad=10)
    plt.ylabel("Loss", labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved premium loss curve to {plot_path}")


def inference_demo(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    results: Dict[str, Any],
) -> None:
    print(f"\n>>> Phase 4: Inference & Gating Visualization")

    prompt = "Once upon a time, there was a little robot named"
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

    print(f"\nPrompt: {prompt}")

    # 1. Base Model Inference (Zero-shot)
    print("\nGenerating with Base Model (No PEFT)...")
    output_base = base_model.generate(
        **inputs, max_new_tokens=40, max_length=None, do_sample=False
    )
    print(
        f"Output (Base):   {tokenizer.decode(output_base[0], skip_special_tokens=True)}"
    )

    # 2. LoRA Inference
    if "lora" in results:
        print(f"\nLoading trained LoRA from {LORA_WEIGHTS_DIR}")
        lora_model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_DIR)
        print("Generating with LoRA ENABLED...")
        output_lora = lora_model.generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
        print(
            f"Output (LoRA):   {tokenizer.decode(output_lora[0], skip_special_tokens=True)}"
        )
        lora_model = lora_model.unload()  # Back to base

    # 3. Engram Inference
    if "engram" in results:
        print(f"\nLoading trained Engram from {ENGRAM_WEIGHTS_DIR}")
        model = EngramModel.from_pretrained(base_model, ENGRAM_WEIGHTS_DIR)

        # Generate with Engram enabled
        print("Generating with Engram ENABLED...")
        output_engram = model.generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
        print(
            f"Output (Engram): {tokenizer.decode(output_engram[0], skip_special_tokens=True)}"
        )

        # Visualization: Print gates for the target layers
        print("\nCapture Gating Activation (Mean per branch):")
        for layer_id in model.config.target_layers:
            engram_layer = cast(EngramLayer, model.engram_layers[str(layer_id)])
            gate = engram_layer.gating.last_gate  # [B, L, M, 1]
            if gate is not None:
                mean_gates = gate.mean(dim=(0, 1, 3)).cpu().tolist()
                gate_str = " | ".join(
                    [f"B{i}: {g:.3f}" for i, g in enumerate(mean_gates)]
                )
                print(f"Layer {layer_id}: {gate_str}")
        model.unload_engram()

    # 4. LoRA + Engram Inference (Optional)
    if "lora+engram" in results:
        print(f"\nLoading trained Combined Model from {LORA_ENGRAM_DIR}")
        # Load LoRA first
        combined_model = PeftModel.from_pretrained(base_model, LORA_ENGRAM_DIR)
        # Load Engram wrapper
        combined_model = EngramModel.from_pretrained(
            cast(PreTrainedModel, combined_model), LORA_ENGRAM_DIR
        )

        print("Generating with LoRA + Engram ENABLED...")
        output_combined = combined_model.generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
        print(
            f"Output (Combined): {tokenizer.decode(output_combined[0], skip_special_tokens=True)}"
        )
        combined_model.unload_engram()
        combined_model = combined_model.base_model.unload()

    print("\nEnd-to-end example completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Engram End-to-End Test")
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per device (Recommended 8-32 for 4090)",
    )
    parser.add_argument(
        "--subset", type=int, default=1000, help="Dataset subset size to load"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=2, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length (Increase to fill GPU)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["lora", "engram"],
        choices=["lora", "engram", "lora+engram"],
        help="Methods to benchmark",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging to file
    log_path = os.path.join(OUTPUT_DIR, "training.log")
    sys.stdout = Logger(log_path)  # type: ignore
    sys.stderr = sys.stdout  # Redirect stderr as well

    print(f"Logging started. Saving to {log_path}")

    print(f"Loading tokenizer & base model: {MODEL_NAME}")
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(MODEL_NAME))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base Model once
    print(f"Loading Base Model to {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )

    if torch.cuda.is_available():
        base_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Base Model Peak Memory: {base_mem:.2f} GB")
    else:
        base_mem = 0.0

    # 1. Prepare Data (Training + Unseen Validation)
    # Increased eval_size to 200 for better representativeness
    # Added num_proc for parallel tokenization
    train_dataset, eval_dataset = prepare_dataset(
        tokenizer,
        subset_size=args.subset,
        eval_size=200,
        max_length=args.max_length,
        num_proc=args.num_workers,
    )

    results: Dict[str, Any] = {}

    # 2. Get baseline performance (Zero-shot loss on unseen eval data)
    base_loss = get_base_model_eval_loss(
        base_model, tokenizer, eval_dataset, batch_size=args.batch_size
    )
    results["base_model"] = {
        "eval_loss": base_loss,
        "peak_memory_gb": base_mem,
    }

    # 3. Iterate through methods
    for method in args.methods:
        print("\n" + "=" * 50)
        if method == "lora":
            results["lora"] = train_lora(
                base_model, tokenizer, train_dataset, eval_dataset, args
            )
        elif method == "engram":
            results["engram"] = train_engram_model(
                base_model, tokenizer, train_dataset, eval_dataset, args
            )
        elif method == "lora+engram":
            results["lora+engram"] = train_lora_engram(
                base_model, tokenizer, train_dataset, eval_dataset, args
            )

    # 4. Save & Plot
    print("\n" + "=" * 50)
    save_and_plot_results(results)

    # 5. Inference & dynamic usage Demo
    print("\n" + "=" * 50)
    inference_demo(base_model, tokenizer, results)
