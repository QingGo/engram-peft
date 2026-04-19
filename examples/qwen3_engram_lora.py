"""
Qwen3.5 (Qwen 3.5 Compatible) LoRA + Engram Fine-tuning Example.

This script demonstrates combining LoRA (Low-Rank Adaptation) with Engram-PEFT
to inject persistent memory into Qwen-series models while maintaining standard
parameter-efficient fine-tuning throughput.

Usage:
    uv run python examples/qwen3_engram_lora.py --model_id Qwen/Qwen3.5-4B --max_steps 50
"""

import argparse
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    set_seed,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramTrainer,
    get_engram_model,
)

# Defaults
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "outputs/qwen3_engram_lora"
SEED = 42

set_seed(SEED)


def prepare_alpaca_dataset(
    tokenizer: PreTrainedTokenizerBase, max_length: int = 512
) -> Dataset:
    """Load and format the Alpaca dataset."""
    dataset = load_dataset(
        "tatsu-lab/alpaca", split="train[:1%]"
    )  # Using 1% for demo speed
    assert isinstance(dataset, Dataset)

    def format_alpaca(example: dict[str, Any]) -> dict[str, Any]:
        if example.get("input"):
            prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: "
        else:
            prompt = f"Instruction: {example['instruction']}\nResponse: "

        response = str(example["output"])
        full_text = prompt + response + (tokenizer.eos_token or "")

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        labels = list(tokenized["input_ids"])
        # Mask the prompt part in labels so we only calculate loss on the response
        prompt_tokenized = tokenizer(prompt, max_length=max_length, truncation=True)
        prompt_len = len(prompt_tokenized["input_ids"])
        for i in range(min(prompt_len, max_length)):
            labels[i] = -100

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    return dataset.map(format_alpaca, remove_columns=dataset.column_names)


def run_example(args: argparse.Namespace) -> None:
    print(f"\n>>> Initializing Qwen3.5 Example with model: {args.model_id}")

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen models usually use <|endoftext|> as pad
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model (load_in_4bit={args.load_in_4bit})...")
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # 2. Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    # 3. Apply Engram-PEFT (Combined Mode)
    print("Applying Engram-PEFT...")
    # Target middle-to-deep layers for knowledge injection
    num_layers = getattr(base_model.config, "num_hidden_layers", 28)
    target_layers = [num_layers // 2, num_layers - 2]

    engram_config = EngramConfig(
        target_layers=target_layers,
        engram_vocab_size_per_ngram=[512000, 512000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=args.engram_dim,
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=args.model_id,
        pad_id=tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0,
    )

    # Use "preserve_trainable" to keep LoRA parameters trainable
    model = cast(
        "Any",
        get_engram_model(
            model, engram_config, tokenizer=tokenizer, train_mode="preserve_trainable"
        ),
    )
    model.print_trainable_parameters()

    # 4. Prepare Dataset
    print("Preparing Alpaca subset...")
    train_dataset = prepare_alpaca_dataset(tokenizer)

    # 5. Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
    )

    collator = EngramDataCollator(tokenizer=tokenizer, config=engram_config)
    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("Starting combined LoRA + Engram training...")
    trainer.train()

    # 6. Saving
    print(f"Saving combined adapters to {OUTPUT_DIR}")
    # This saves both LoRA (if it's a PeftModel) and Engram weights
    model.save_pretrained(OUTPUT_DIR)

    # 7. Inference Demo
    print("\n>>> Inference Demo")
    prompt = "Instruction: Explain the concept of quantum entanglement.\nResponse: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.base_model.device)

    print(f"Prompt: {prompt}")
    output = model.generate(
        **inputs, max_new_tokens=50, do_sample=True, temperature=0.7
    )
    print(f"Response: {tokenizer.decode(output[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA + Engram Example")
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL, help="Model ID on HuggingFace"
    )
    parser.add_argument(
        "--max_steps", type=int, default=300, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size per device"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
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
    args = parser.parse_args()

    run_example(args)
