"""
Gemma-4 (Effective 2B) LoRA + Engram Fine-tuning Example.

This script demonstrates combining LoRA with Engram-PEFT for Google's Gemma-4
family, focusing on extreme on-device efficiency and knowledge storage.

Usage:
    uv run python examples/gemma4_engram_lora.py --model_id google/gemma-4-E2B-it --max_steps 300
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from collections.abc import Iterable
from typing import Any, cast

from engram_peft.types import ModelProtocol, SafeTrainingArguments, SizedEncoding

# Add the project root to sys.path to allow absolute imports from the 'examples' package
# when running the script directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramModel,
    EngramTrainer,
    get_engram_model,
)
from engram_peft.utils import (
    apply_peft_patches,
    get_optimal_precision_config,
)
from engram_peft.utils.compat import (
    create_safe_training_args,
    wash_model,
    wash_tokenizer,
)
from examples.benchmarks.data_utils import get_dataset_template

# Try to import optional visualization components
try:
    from examples.benchmarks.persistence import BenchmarkResult
    from examples.benchmarks.plotting import plot_benchmark_comparison
except ImportError:
    BenchmarkResult = None
    plot_benchmark_comparison = None

# Defaults
DEFAULT_MODEL = "google/gemma-4-E2B-it"
OUTPUT_DIR = "outputs/gemma4_engram_lora"
SEED = 42

set_seed(SEED)


def prepare_alpaca_dataset(
    tokenizer: PreTrainedTokenizerBase, max_length: int = 512, eval_ratio: float = 0.05
) -> dict[str, Dataset]:
    """Load and format the Alpaca dataset using Gemma Instruct template."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    # Aggressively cap dataset for fast example execution
    dataset = dataset.select(range(min(600, len(dataset))))
    assert isinstance(dataset, Dataset)

    template = get_dataset_template("gemma")

    def format_alpaca(example: dict[str, Any]) -> dict[str, Any]:
        prompt = template.format(
            instruction=example["instruction"],
            input=f"\n\nInput:\n{example['input']}" if example.get("input") else "",
        )
        response = f"{example['output']}<end_of_turn>"
        full_text = prompt + response

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # Use isinstance for narrowing to avoid cast (Zero-Cast Principle)
        encoding_ids = tokenized["input_ids"]
        if not isinstance(encoding_ids, SizedEncoding):
            # Fallback for unexpected types, though tokenizers.Encoding should match structurally
            labels = list(encoding_ids) if isinstance(encoding_ids, Iterable) else []
        else:
            labels = list(encoding_ids)
        # Mask the prompt part in labels (Padding masking will be handled by SmartDataCollator)
        prompt_tokenized = tokenizer(prompt, max_length=max_length, truncation=True)
        if not isinstance(prompt_tokenized, BatchEncoding):
            prompt_tokenized = BatchEncoding(prompt_tokenized)

        # Use isinstance for narrowing to avoid cast (Zero-Cast Principle)
        prompt_ids = prompt_tokenized["input_ids"]
        if isinstance(prompt_ids, SizedEncoding):
            sized_prompt_ids: SizedEncoding = prompt_ids
            prompt_len = len(sized_prompt_ids)
        else:
            # Fallback for unexpected types
            prompt_len = 0
        for i in range(min(prompt_len, max_length)):
            labels[i] = -100

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    tokenized_ds = dataset.map(format_alpaca, remove_columns=dataset.column_names)

    # Split into train and eval
    if eval_ratio > 0:
        split_ds = tokenized_ds.train_test_split(test_size=eval_ratio, seed=SEED)
        return {"train": split_ds["train"], "eval": split_ds["test"]}
    return {"train": tokenized_ds}


def run_example(args: argparse.Namespace) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup file logging
    log_file = os.path.join(OUTPUT_DIR, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    logging.info(f"Starting Gemma-4 Engram+LoRA example with model: {args.model_id}")

    # 0. Apply PEFT deep patches to support Gemma-4 custom layers
    apply_peft_patches()

    print(f"\n>>> Initializing Gemma-4 Example with model: {args.model_id}")

    # 1. Load Processor & Model using official Transformers recommended way
    print(f"Loading tokenizer: {args.model_id}")

    # Gemma-4 might use AutoProcessor for multimodal support
    try:
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        tokenizer = processor.tokenizer
    except Exception:
        print("AutoProcessor failed, falling back to AutoTokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-load config to check for existing quantization and architecture details
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    # Defensive fix: ensure required attributes exist for Gemma4Model
    # Handle nested text_config which is common in new multimodal or complex architectures
    # CRITICAL: We DO NOT unpack text_config here to avoid losing top-level metadata.
    # Instead, we sync ALL attributes from text_config to the top-level config.
    if hasattr(config, "text_config"):
        print(
            "Detected nested text_config, synchronizing ALL attributes to top-level..."
        )
        text_config_dict = config.text_config.to_dict()
        for attr, value in text_config_dict.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, value)

        # Monkey patch config class for PEFT compatibility
        config_class = config.__class__
        if not hasattr(config_class, "vocab_size"):
            print(f"Monkey patching {config_class.__name__} for PEFT compatibility...")

            def get_vocab_size(self: Any) -> int | None:
                return (
                    self.text_config.vocab_size
                    if hasattr(self, "text_config")
                    else None
                )

            config_class.vocab_size = property(get_vocab_size)  # type: ignore

    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

    if not hasattr(config, "vocab_size") or config.vocab_size is None:
        config.vocab_size = len(tokenizer)

    has_existing_quant = getattr(config, "quantization_config", None) is not None

    # Initialize bnb config for quantization if requested AND not already quantized
    quantization_config = None
    if has_existing_quant:
        print(
            f"Notice: Model {args.model_id} is already quantized ({config.quantization_config.get('quant_method')})."
        )
        print(
            "Ignoring bitsandbytes flags (--load_in_4bit/--load_in_8bit) to avoid conflicts."
        )
    else:
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if not quantization_config:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if get_optimal_precision_config()["bf16"] else torch.float16
        )

    # Load model with final configuration
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_id, config=config, **model_kwargs
    )
    # Type it as Union to satisfy both transformers methods and avoid strange 'generate' callable errors
    base_model: PreTrainedModel | ModelProtocol = model_instance

    # 2. Apply LoRA
    print("Applying LoRA...")

    # PEFT patches for Gemma4 are now automatically applied by engram_peft upon import!
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )

    if not isinstance(base_model, PreTrainedModel):
        raise TypeError("base_model must be a PreTrainedModel for LoRA")

    model: PeftModel | PeftMixedModel | EngramModel = get_peft_model(
        base_model, lora_config
    )

    # 3. Apply Engram-PEFT
    print("Applying Engram-PEFT...")
    # Target layers for Gemma-2B/4B architecture
    num_layers = getattr(base_model.config, "num_hidden_layers", 18)
    target_layers = [num_layers // 2, num_layers - 2]

    engram_config = EngramConfig(
        engram_dim=args.engram_dim,
        target_layers=target_layers,
    )

    model = get_engram_model(
        model,
        engram_config,
        tokenizer=wash_tokenizer(tokenizer),
        train_mode="preserve_trainable",
    )
    model.print_trainable_parameters()

    # 4. Prepare Dataset
    print("Preparing Alpaca subsets (train + eval)...")
    datasets = prepare_alpaca_dataset(tokenizer)

    # 5. Training
    precision_config = get_optimal_precision_config()
    training_args_dict = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps" if "eval" in datasets else "no",
        "eval_steps": args.eval_steps,
        "per_device_eval_batch_size": 1,
        "remove_unused_columns": True,
        "report_to": "none",
        "bf16": precision_config.get("bf16", False),
        "fp16": precision_config.get("fp16", False),
    }

    training_args = create_safe_training_args(
        cast("SafeTrainingArguments", training_args_dict)
    )

    # 4. Prepare Trainer
    print("Preparing Trainer...")

    # Use the library's data collator which now handles padding masking automatically!
    # Use the explicit engram_config from the model wrapper
    engram_config_obj = model.config
    if not isinstance(engram_config_obj, EngramConfig):
        raise TypeError("Model config is not an EngramConfig")

    data_collator = EngramDataCollator(
        tokenizer=wash_tokenizer(tokenizer), config=engram_config_obj
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("eval"),
        data_collator=data_collator,
    )

    print("\n>>> Initial (Zero-shot) Evaluation")
    initial_metrics = trainer.evaluate()
    print(f"Initial Eval Loss: {initial_metrics.get('eval_loss', 0.0):.4f}")

    print("\n>>> Starting combined LoRA + Engram training...")
    trainer.train()

    # 5.1 Plot Results
    if BenchmarkResult is not None:
        print("\n>>> Generating Training Plots")
        # Prepare result objects for plotting
        main_res = BenchmarkResult(
            method="gemma4_lora_engram",
            metrics={
                "eval_loss": initial_metrics.get("eval_loss", 0.0),
                "log_history": trainer.state.log_history,
            },
            params=vars(args),
        )
        # Save structured logs
        main_res.save(OUTPUT_DIR)

        # Create a baseline result to trigger the horizontal line in plots
        base_res = BenchmarkResult(
            method="base",
            metrics={"eval_loss": initial_metrics.get("eval_loss", 0.0)},
            params=vars(args),
        )

        if plot_benchmark_comparison is not None:
            plot_benchmark_comparison(
                [main_res, base_res],
                output_path=os.path.join(OUTPUT_DIR, "training_curve.png"),
            )
    else:
        print("\n>>> Skipping plots (optional plotting tools not available)")

    # 6. Saving
    print(f"Saving combined adapters to {OUTPUT_DIR}")
    # Explicitly save LoRA adapters first to ensure adapter_config.json exists
    # Use the Protocol for saving/generating to avoid strange mypy attribute errors
    if isinstance(model.base_model, ModelProtocol):
        print("Saving LoRA adapters...")
        model.base_model.save_pretrained(OUTPUT_DIR)

    # Save Engram adapters separately for maximum robustness
    print("Saving Engram adapters...")
    model.save_pretrained_engram(OUTPUT_DIR)

    # 7. Inference Demo (Original Model)
    print("\n>>> Inference Demo (Original Model)")
    prompt = "<start_of_turn>user\nTell me a short fact about the moon.<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.base_model.device)

    print(f"Prompt: {prompt}")
    with torch.no_grad():
        gen_model = wash_model(model)
        output = gen_model.generate(
            **inputs,
            max_new_tokens=100,
            max_length=None,
            do_sample=True,
            temperature=0.7,
            stop_strings=["<end_of_turn>"],
            tokenizer=tokenizer,
        )
    print(f"Response: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # 8. Reload and Verify
    print("\n>>> Reloading Model for Verification")
    # To fully verify, we should be able to load both LoRA and Engram back
    try:
        # 1. Load LoRA part onto a fresh base model (or reuse base_model for efficiency)
        if not isinstance(base_model, torch.nn.Module):
            raise TypeError("base_model must be a torch.nn.Module for reloading")

        reloaded_peft = PeftModel.from_pretrained(
            base_model, OUTPUT_DIR, trust_remote_code=True
        )

        # 2. Re-wrap with Engram using the class method which is cleaner
        if not isinstance(reloaded_peft, torch.nn.Module):
            raise TypeError("reloaded_peft must be a torch.nn.Module")

        reloaded_model = EngramModel.from_pretrained(
            reloaded_peft, OUTPUT_DIR, tokenizer=wash_tokenizer(tokenizer)
        )

        print("Inference with Fully Reloaded Model (LoRA + Engram):")
        with torch.no_grad():
            reloaded_output = reloaded_model.generate(
                **inputs,
                max_new_tokens=100,
                max_length=None,
                do_sample=True,
                temperature=0.7,
                stop_strings=["<end_of_turn>"],
                tokenizer=tokenizer,
            )
        reloaded_resp = tokenizer.decode(reloaded_output[0], skip_special_tokens=True)
        print(f"Response: {reloaded_resp}")
    except Exception as e:
        print(f"Reloading failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma-4 LoRA + Engram Example")
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL, help="Model ID on HuggingFace"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum training steps"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--data_size", type=int, default=2000, help="Subset of Alpaca to use"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--engram_dim", type=int, default=1024, help="Engram embedding dimension"
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Load base model in 4-bit precision"
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load base model in 8-bit precision"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Evaluation frequency"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Logging frequency"
    )
    args = parser.parse_args()

    run_example(args)
