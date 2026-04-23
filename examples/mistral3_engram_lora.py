"""
Ministral-3 (3B Instruct) LoRA + Engram Fine-tuning Example.

This script demonstrates combining LoRA with Engram-PEFT for the Ministral-3
family of models, specifically optimized for edge devices and instruction following.

Usage:
    uv run python examples/mistral3_engram_lora.py --model_id mistralai/Ministral-3-3B-Instruct-2512 --max_steps 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from collections.abc import Iterable
from typing import Any

from engram_peft.types import GenerativeProtocol, SizedEncoding

# Add the project root to sys.path to allow absolute imports from the 'examples' package
# when running the script directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    set_seed,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramModel,
    EngramTrainer,
    get_engram_model,
)
from engram_peft.types import ModelProtocol
from engram_peft.utils import apply_peft_patches
from examples.benchmarks.data_utils import get_dataset_template

# Check for latest transformers features
try:
    from transformers import FineGrainedFP8Config
except ImportError:
    FineGrainedFP8Config = None

# Try to import optional visualization components
try:
    from examples.benchmarks.persistence import BenchmarkResult
    from examples.benchmarks.plotting import plot_benchmark_comparison
except ImportError:
    BenchmarkResult = None
    plot_benchmark_comparison = None

# Defaults
DEFAULT_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"
OUTPUT_DIR = "outputs/mistral3_engram_lora"
SEED = 42

set_seed(SEED)


def prepare_alpaca_dataset(
    tokenizer: PreTrainedTokenizerBase, max_length: int = 512, eval_ratio: float = 0.05
) -> dict[str, Dataset]:
    """Load and format the Alpaca dataset using Mistral Instruct template."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    # Aggressively cap dataset for fast example execution
    dataset = dataset.select(range(min(600, len(dataset))))
    assert isinstance(dataset, Dataset)

    template = get_dataset_template("mistral")

    def format_alpaca(example: dict[str, Any]) -> dict[str, Any]:
        prompt = template.format(
            instruction=example["instruction"],
            input=f"\n\nInput:\n{example['input']}" if example.get("input") else "",
        )
        response = f" {example['output']}</s>"
        full_text = prompt + response

        # Tokenize full text
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        if not isinstance(tokenized, BatchEncoding):
            tokenized = BatchEncoding(tokenized)

        # Use isinstance for narrowing to avoid cast (Zero-Cast Principle)
        encoding_ids = tokenized["input_ids"]
        if not isinstance(encoding_ids, SizedEncoding):
            labels = list(encoding_ids) if isinstance(encoding_ids, Iterable) else []
        else:
            labels = list(encoding_ids)

        # Mask the prompt part in labels (Padding masking handled by Collator)
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
    # 0. Apply PEFT deep patches
    apply_peft_patches()

    # Defensive patch for Ministral-3 naming inconsistency in some transformers versions
    try:
        if "mistral3" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["mistral3"] = "Ministral3ForCausalLM"
    except ImportError:
        pass

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

    logging.info(f"Starting Mistral-3 Engram+LoRA example with model: {args.model_id}")

    print(f"\n>>> Initializing Ministral-3 Example with model: {args.model_id}")

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load model using official Transformers Mistral3 class
    print(f"Loading base model: {args.model_id}")

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Handle quantization if requested
    if args.load_in_4bit:
        print("Enabling 4-bit quantization...")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        print("Enabling 8-bit quantization...")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # Handle BF16 dequantization if requested (for FP8 models)
    if args.load_in_bf16:
        print("Forcing model to BF16 precision to bypass FP8 kernel issues...")
        load_kwargs["torch_dtype"] = torch.bfloat16
        # If FineGrainedFP8Config is available, we use it, otherwise we rely on torch_dtype
        if FineGrainedFP8Config is not None:
            load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    elif not args.load_in_4bit and not args.load_in_8bit:
        # Default to FP8 as per official docs
        print("Loading in native FP8 precision...")

    # Type base_model as Union to satisfy both transformers methods and avoid strange 'generate' callable errors
    base_model: PreTrainedModel | ModelProtocol = (
        Mistral3ForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
    )

    # Tie weights after loading to ensure lm_head is correct
    base_model.tie_weights()

    # 1.1 Prepare for k-bit training if quantized
    if args.load_in_4bit or args.load_in_8bit:
        print("Preparing model for k-bit training...")
        base_model = prepare_model_for_kbit_training(base_model)

    # 2. Apply LoRA
    print("Applying LoRA...")
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
    # Mistral-3 layers
    num_layers = getattr(base_model.config, "num_hidden_layers", 32)
    target_layers = [num_layers // 2, num_layers - 2]

    engram_config = EngramConfig(
        engram_dim=args.engram_dim,
        target_layers=target_layers,
        k_dim=128,
        num_clusters=1024,
        backbone_freeze_steps=0,
    )
    # get_engram_model handles architecture-specific patching automatically!
    model = get_engram_model(model, engram_config, tokenizer=tokenizer)

    # 4. Prepare Dataset
    print("Preparing Alpaca subsets (train + eval)...")
    datasets = prepare_alpaca_dataset(tokenizer)

    # 5. Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if "eval" in datasets else "no",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=True,
    )

    # Use the explicit engram_config from the model wrapper
    engram_config_obj = model.config
    if not isinstance(engram_config_obj, EngramConfig):
        raise TypeError("Model config is not an EngramConfig")

    data_collator = EngramDataCollator(tokenizer=tokenizer, config=engram_config_obj)

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
            method="mistral3_lora_engram",
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
    prompt = "<s>[INST] List three benefits of using parameter-efficient fine-tuning for edge models. [/INST]"
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("tokenizer must be a PreTrainedTokenizerBase")

    # Use the Protocol to get the device cleanly
    target_device: torch.device | str
    if isinstance(model.base_model, ModelProtocol):
        target_device = model.base_model.device
    else:
        target_device = "cuda"
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

    print(f"Prompt: {prompt}")
    with torch.no_grad():
        gen_model: GenerativeProtocol = model
        output = gen_model.generate(**inputs, max_new_tokens=50, tokenizer=tokenizer)
    print(f"Response: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # 8. Reload and Verify
    print("\n>>> Reloading Model for Verification")
    # To fully verify, we should be able to load both LoRA and Engram back
    try:
        # 1. Load LoRA part onto a fresh base model (or reuse base_model for efficiency)
        if not isinstance(base_model, torch.nn.Module):
            raise TypeError("base_model must be a Module for reloading LoRA")
        reloaded_peft = PeftModel.from_pretrained(
            base_model, OUTPUT_DIR, trust_remote_code=True
        )

        # 2. Re-wrap with Engram using the class method which is cleaner
        reloaded_model = EngramModel.from_pretrained(
            reloaded_peft, OUTPUT_DIR, tokenizer=tokenizer
        )

        print("Inference with Fully Reloaded Model (LoRA + Engram):")
        with torch.no_grad():
            reloaded_output = reloaded_model.generate(
                **inputs,
                max_new_tokens=50,
                max_length=None,
                do_sample=True,
                temperature=0.7,
            )
        reloaded_resp = tokenizer.decode(reloaded_output[0], skip_special_tokens=True)
        print(f"Response: {reloaded_resp}")
    except Exception as e:
        print(f"Reloading failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ministral-3 LoRA + Engram Example")
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
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--load_in_bf16", action="store_true", help="Dequantize FP8 to BF16"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Evaluation frequency"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Logging frequency"
    )
    args = parser.parse_args()

    run_example(args)
