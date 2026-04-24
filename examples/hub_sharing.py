# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
"""
Engram-PEFT Hugging Face Hub Sharing Example.

Demonstrates the full workflow:
1. Train Engram on a small dataset for a configurable number of steps.
2. Capture inference output before saving (ground truth).
3. Save/push the trained adapter to Hugging Face Hub (or local directory).
4. Reload the adapter from Hub/local.
5. Re-run inference and verify outputs are identical to ground truth.

Prerequisites:
    - Login: `huggingface-cli login` (only needed for Hub push)

Usage:
    # Push to Hub (requires login)
    uv run python examples/hub_sharing.py --repo_id "your-username/tinyllama-engram-test"

    # Save and reload locally instead (no Hub login needed)
    uv run python examples/hub_sharing.py --local ./my_adapter

    # Custom training steps
    uv run python examples/hub_sharing.py --max_steps 30 --local ./my_adapter
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import copy
import os
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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
    get_optimizer,
    get_scheduler,
)
from engram_peft.utils.compat import wash_tokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SEED = 42
PROMPT = "Once upon a time, there was a little robot named"


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase, subset_size: int, max_length: int
) -> Dataset:
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized_dict = dict(tokenized)
        tokenized_dict["labels"] = copy.deepcopy(tokenized_dict["input_ids"])
        return tokenized_dict

    train_data = []
    for i, item in enumerate(dataset):
        if i >= subset_size:
            break
        train_data.append(item)

    train_dataset = Dataset.from_list(train_data).map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return train_dataset


def run_inference(
    model: EngramModel, tokenizer: PreTrainedTokenizerBase
) -> str:
    device = model.base_model.device
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    output = model.generate(
        **inputs, max_new_tokens=20, max_length=None, do_sample=False
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Engram Hub Sharing Demo (Train -> Push -> Reload -> Verify)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Base model name",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hugging Face repository ID (e.g., 'username/my-adapter')",
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Local directory to save/reload adapter (avoids Hub upload)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=30, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Per-device batch size"
    )
    parser.add_argument(
        "--subset", type=int, default=500, help="Dataset subset size"
    )
    args = parser.parse_args()

    if not args.repo_id and not args.local:
        parser.error("Must provide either --repo_id (Hub) or --local (directory)")

    set_seed(SEED)

    # ---------- Setup ----------
    print(f"Loading tokenizer & base model: {args.model_name}")
    tokenizer = cast(
        "PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(args.model_name)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = cast(
        "PreTrainedModel",
        AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ),
    )

    # ---------- Prepare dataset ----------
    dataset = prepare_dataset(tokenizer, subset_size=args.subset, max_length=128)

    # ---------- Inject Engram ----------
    print("Initializing Engram layers...")
    config = EngramConfig(
        target_layers=[2],
        engram_vocab_size_per_ngram=[128000, 128000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=512,
        tokenizer_name_or_path=args.model_name,
    )

    model = get_engram_model(
        base_model,
        config,
        wash_tokenizer(tokenizer),
        train_mode="engram_only",
    )

    # ---------- Train ----------
    print(f"\n>>> Training Engram for {args.max_steps} steps...")
    collator = EngramDataCollator(tokenizer=wash_tokenizer(tokenizer), config=config)
    optimizer = get_optimizer(model, base_learning_rate=4e-4)
    scheduler = get_scheduler(
        optimizer, num_training_steps=args.max_steps, warmup_steps=5
    )

    training_args = TrainingArguments(
        output_dir="outputs/hub_sharing",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
    )
    trainer.train()

    # ---------- Capture ground truth inference ----------
    print("\n>>> Running inference BEFORE save (ground truth)...")
    output_before = run_inference(model, tokenizer)
    print(f"Output: {output_before}")

    # ---------- Save / Push ----------
    if args.repo_id:
        print(f"\n>>> Pushing adapter to Hub: {args.repo_id}")
        try:
            model.push_to_hub(args.repo_id)
            print(f"Successfully pushed to https://huggingface.co/{args.repo_id}")
            load_path: str = args.repo_id
        except Exception as e:
            print(f"Hub push failed: {e}")
            print("Falling back to local save...")
            load_path = "outputs/hub_sharing/fallback"
            model.save_pretrained(load_path)
    else:
        load_path = cast(str, args.local)
        os.makedirs(load_path, exist_ok=True)
        print(f"\n>>> Saving adapter locally: {load_path}")
        model.save_pretrained(load_path)

    # ---------- Reload ----------
    print(f"\n>>> Reloading adapter from: {load_path}")
    reloaded = EngramModel.from_pretrained(
        base_model, load_path, tokenizer=wash_tokenizer(tokenizer)
    )
    print(f"Active adapter: {reloaded.active_adapter}")
    print(f"Engram layers: {list(reloaded.engram_layers.keys())}")

    # ---------- Verify inference consistency ----------
    print("\n>>> Running inference AFTER reload (verification)...")
    output_after = run_inference(reloaded, tokenizer)
    print(f"Output: {output_after}")

    if output_before == output_after:
        print("\n✓ VERIFICATION PASSED: Outputs are IDENTICAL after save/reload cycle.")
    else:
        print("\n✗ VERIFICATION FAILED: Outputs differ after save/reload cycle!")

    # ---------- Cleanup ----------
    reloaded.unload_engram()
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
