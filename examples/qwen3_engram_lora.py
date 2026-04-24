# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
"""
Qwen3.5 (Qwen 3.5 Compatible) LoRA + Engram Fine-tuning Example.

This script demonstrates combining LoRA (Low-Rank Adaptation) with Engram-PEFT
to inject persistent memory into Qwen-series models while maintaining standard
parameter-efficient fine-tuning throughput.

Usage:
    uv run python examples/qwen3_engram_lora.py --model_id Qwen/Qwen3.5-4B --max_steps 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from engram_peft.types import ModelProtocol, SizedEncoding

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
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    set_seed,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramModel,
    EngramTrainer,
    get_engram_model,
)
from engram_peft.utils import apply_peft_patches
from engram_peft.utils.compat import wash_model, wash_tokenizer

# Ensure benchmarks are importable
sys.path.append(os.getcwd())
try:
    from examples.benchmarks.persistence import BenchmarkResult
    from examples.benchmarks.plotting import plot_benchmark_comparison
except ImportError:
    BenchmarkResult = None
    plot_benchmark_comparison = None

# Defaults
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "outputs/qwen3_engram_lora"
SEED = 42

set_seed(SEED)


def prepare_alpaca_dataset(
    tokenizer: PreTrainedTokenizerBase, max_length: int = 512, eval_ratio: float = 0.05
) -> dict[str, Dataset]:
    """Load and format the Alpaca dataset using Qwen Instruct template."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    # Aggressively cap dataset for fast example execution
    dataset = dataset.select(range(min(600, len(dataset))))
    assert isinstance(dataset, Dataset)

    def format_alpaca(example: dict[str, Any]) -> dict[str, Any]:
        if example.get("input"):
            prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: "
        else:
            prompt = f"Instruction: {example['instruction']}\nResponse: "

        response = str(example["output"])
        full_text = str(prompt) + response + str(tokenizer.eos_token or "")
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # tokenized is already a BatchEncoding

        encoding_ids = tokenized["input_ids"]
        if isinstance(encoding_ids, list):
            labels = list(encoding_ids)
        elif isinstance(encoding_ids, Iterable):
            labels = list(encoding_ids)
        else:
            labels = []

        # Mask padding tokens in labels so they don't contribute to loss
        if tokenizer.pad_token_id is not None:
            for i in range(len(labels)):
                if labels[i] == tokenizer.pad_token_id:
                    labels[i] = -100

        # Mask the prompt part in labels
        prompt_tokenized = tokenizer(prompt, max_length=max_length, truncation=True)
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

    formatted = dataset.map(format_alpaca, remove_columns=dataset.column_names)
    split = formatted.train_test_split(test_size=eval_ratio, seed=SEED)
    return {"train": split["train"], "eval": split["test"]}


def run_example(args: argparse.Namespace) -> None:
    # Load environment variables from .env file if it exists (e.g. HF_TOKEN)
    dotenv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.isfile(dotenv_path):
        logging.info(f"Environment variables loaded from {dotenv_path}")

    # 0. Apply PEFT deep patches
    apply_peft_patches()

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

    logging.info(f"Starting Qwen3.5 Engram+LoRA example with model: {args.model_id}")

    print(f"\n>>> Initializing Qwen-3.5 Example with model: {args.model_id}")

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen models usually use <|endoftext|> as pad
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model (load_in_4bit={args.load_in_4bit})...")

    # Pre-load config to check for existing quantization
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    # Defensive fix: ensure required attributes exist for Qwen3Model
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
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    # Type base_model as Union to satisfy both transformers methods and avoid strange 'generate' callable errors
    base_model: PreTrainedModel | ModelProtocol = AutoModelForCausalLM.from_pretrained(
        args.model_id, config=config, **model_kwargs
    )

    # 1.1 Prepare for k-bit training if quantized
    if args.load_in_4bit or args.load_in_8bit:
        print("Preparing model for k-bit training...")

        # Note: we use gradient_checkpointing in TrainingArguments later,
        # but prepare_model_for_kbit_training also enables it by default.
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
    # Qwen-3.5 layers
    num_layers = getattr(base_model.config, "num_hidden_layers", 28)
    target_layers = [num_layers // 2, num_layers - 2]

    engram_config = EngramConfig(
        engram_dim=args.engram_dim,
        target_layers=target_layers,
        k_dim=128,
        num_clusters=1024,
        backbone_freeze_steps=0,
    )
    # get_engram_model handles text_config and vocab_size automatically!
    model = get_engram_model(model, engram_config, tokenizer=wash_tokenizer(tokenizer))

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
            method="qwen3_lora_engram",
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
    if hasattr(model.base_model, "save_pretrained"):
        print("Saving LoRA adapters...")
        model.base_model.save_pretrained(OUTPUT_DIR)
    else:
        print(
            "Warning: model.base_model does not have save_pretrained; LoRA adapter saving skipped."
        )

    # Save Engram adapters separately for maximum robustness
    print("Saving Engram adapters...")
    model.save_pretrained_engram(OUTPUT_DIR)

    # 7. Inference Demo (Original Model)
    print("\n>>> Inference Demo (Original Model)")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain the concept of quantum entanglement in one sentence.",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("tokenizer must be a PreTrainedTokenizerBase")

    # Use hasattr to get the device cleanly (works with PeftModel, PreTrainedModel, etc.)
    target_device: torch.device | str
    if hasattr(model.base_model, "device"):
        target_device = model.base_model.device
    else:
        target_device = "cuda"
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
    input_ids_val = inputs["input_ids"]
    if isinstance(input_ids_val, torch.Tensor):
        input_len = input_ids_val.shape[-1]
    elif isinstance(input_ids_val, SizedEncoding):
        sized_ids: SizedEncoding = input_ids_val
        input_len = len(sized_ids)
    else:
        # Fallback
        input_len = 0

    print(f"Prompt: {prompt}")
    with torch.no_grad():
        gen_model = wash_model(model)
        output = gen_model.generate(
            **inputs,
            max_new_tokens=200,
            max_length=None,
            do_sample=True,
            temperature=0.7,
            stop_strings=["<think>", "</think>", "<|im_end|>"],
            tokenizer=tokenizer,
        )
    original_resp = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    print(f"Response: {original_resp}")

    # 8. Reload and Verify
    print("\n>>> Reloading Model for Verification")
    # To fully verify, we should be able to load both LoRA and Engram back
    try:
        # 1. Load LoRA part onto a fresh base model (or reuse base_model for efficiency)
        # base_model is already a torch.nn.Module

        reloaded_peft = PeftModel.from_pretrained(
            base_model, OUTPUT_DIR, trust_remote_code=True
        )

        # 2. Re-wrap with Engram using the class method which is cleaner
        # reloaded_peft is already a torch.nn.Module

        reloaded_model = EngramModel.from_pretrained(
            reloaded_peft, OUTPUT_DIR, tokenizer=wash_tokenizer(tokenizer)
        )

        print("Inference with Fully Reloaded Model (LoRA + Engram):")
        with torch.no_grad():
            reloaded_output = reloaded_model.generate(
                **inputs,
                max_new_tokens=200,
                max_length=None,
                do_sample=True,
                temperature=0.7,
                stop_strings=["<think>", "</think>", "<|im_end|>"],
                tokenizer=tokenizer,
            )
        reloaded_resp = tokenizer.decode(
            reloaded_output[0][input_len:], skip_special_tokens=True
        )
        print(f"Response: {reloaded_resp}")
    except Exception as e:
        print(f"Reloading failed: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA + Engram Example")
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
        "--engram_dim", type=int, default=1280, help="Engram embedding dimension"
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
