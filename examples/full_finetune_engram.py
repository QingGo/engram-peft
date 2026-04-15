"""
Engram-PEFT Full Finetuning Example.

This script mirrors the training/evaluation flow used in `compare_engram_lora.py`
so the resulting metrics are directly comparable:
1. Dataset: Uses the same TinyStories train/validation splits and tokenization.
2. Training: Uses `train_mode="full_finetune"` with Engram injected on top.
3. Optimization: Uses layered optimizers for backbone, Engram dense, and Engram sparse params.
4. Reporting: Saves train/eval metrics, average time per step, and peak memory to JSON.

Usage:
    uv run python examples/full_finetune_engram.py --max_steps 3000 --batch_size 16 --grad_accum 2 --num_workers 4 --subset 100000
"""

import argparse
import copy
import json
import logging
import os
from typing import Any, Dict, Tuple, cast

import torch
from datasets import load_dataset  # type: ignore
from torch.optim import AdamW  # type: ignore
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
    EngramModel,
    EngramTrainer,
    get_engram_model,
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR = "outputs/full_finetune_engram"
ENGRAM_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "engram_weights")
BASE_MODEL_DIR = os.path.join(OUTPUT_DIR, "base_model")
METRICS_PATH = os.path.join(OUTPUT_DIR, "training_metrics.json")
SEED = 42

set_seed(SEED)


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    subset_size: int,
    eval_size: int,
    max_length: int,
    num_proc: int = 4,
) -> Tuple[Any, Any]:
    print("Loading TinyStories dataset (train split)...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    print("Loading TinyStories dataset (validation split)...")
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)

    train_ds = train_ds.select(range(subset_size))
    val_ds = val_ds.select(range(min(len(val_ds), eval_size)))

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
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
    model: Any,
    dataset: Any,
    batch_size: int,
) -> float:
    print("\n>>> Phase 1: Evaluating Base Model (Zero-shot Loss)")
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "base_eval"),
        per_device_eval_batch_size=batch_size,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        and torch.cuda.is_available(),
        dataloader_num_workers=4,
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


def train_full_finetune_engram(
    base_model: Any,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n>>> Phase 2: Training Full Finetune + Engram on {MODEL_NAME}")
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

    model = get_engram_model(
        base_model,
        config,
        tokenizer,
        train_mode="full_finetune",
    )
    model.print_trainable_parameters()

    warmup_steps = int(args.max_steps * 0.03)
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 3e-4,
    }

    collator = EngramDataCollator(tokenizer=tokenizer, config=config)
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "trainer_outputs"),
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
        optimizer_kwargs={
            "backbone_learning_rate": 5e-5,
            "engram_dense_learning_rate": 3e-4,
            "engram_sparse_learning_rate": 9e-4,
            "backbone_optimizer": AdamW,
            "engram_dense_optimizer": "adamw",
            "engram_sparse_optimizer": "sparse_adam",
        },
    )

    print("Starting Full Finetune + Engram training...")
    train_result = trainer.train()

    print("Evaluating Full Finetune + Engram...")
    eval_results = trainer.evaluate()
    eval_loss = cast(float, eval_results.get("eval_loss", 0.0))

    avg_time_per_step = train_result.metrics["train_runtime"] / train_result.global_step
    print(f"Full Finetune + Engram Avg Time Per Step: {avg_time_per_step:.4f}s")

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"Full Finetune + Engram Peak Memory: {peak_memory:.2f} GB")

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
    print(f"Saving finetuned base model to {BASE_MODEL_DIR}")
    model.base_model.save_pretrained(BASE_MODEL_DIR)
    tokenizer.save_pretrained(BASE_MODEL_DIR)

    model.unload_engram()

    return {
        "log_history": log_history,
        "peak_memory_gb": peak_memory,
        "avg_time_per_step": avg_time_per_step,
        "eval_loss": eval_loss,
    }


def save_metrics(metrics: Dict[str, Any]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


def inference_demo(
    base_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    print("\n>>> Phase 3: Inference Comparison")

    prompt = "Once upon a time, there was a little robot named"
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

    print(f"\nPrompt: {prompt}")

    print("\nGenerating with Base Model (No Engram)...")
    output_base = base_model.generate(
        **inputs, max_new_tokens=40, max_length=None, do_sample=False
    )  # type: ignore[attr-defined]
    print(
        f"Output (Base):                {tokenizer.decode(output_base[0], skip_special_tokens=True)}"
    )

    print("\nGenerating with Full Finetune + Engram ENABLED...")
    reloaded_base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else None
        ),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    full_ft_model = EngramModel.from_pretrained(reloaded_base_model, ENGRAM_WEIGHTS_DIR)
    output_full_ft = full_ft_model.generate(
        **inputs, max_new_tokens=40, max_length=None, do_sample=False
    )
    print(
        f"Output (Full FT + Engram):    {tokenizer.decode(output_full_ft[0], skip_special_tokens=True)}"
    )


def main() -> None:
    # Set logging level to INFO to see Engram-PEFT injection logs
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset", type=int, default=2000)
    parser.add_argument("--eval_size", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = prepare_dataset(
        tokenizer,
        subset_size=args.subset,
        eval_size=args.eval_size,
        max_length=args.max_length,
        num_proc=args.num_workers,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else None
        ),
        device_map="auto" if torch.cuda.is_available() else None,
    )

    base_eval_loss = get_base_model_eval_loss(
        base_model, eval_dataset, batch_size=args.batch_size
    )
    results = train_full_finetune_engram(
        base_model, tokenizer, train_dataset, eval_dataset, args
    )

    metrics = {
        "method": "full_finetune_engram",
        "model_name": MODEL_NAME,
        "base_eval_loss": base_eval_loss,
        **results,
    }
    save_metrics(metrics)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else None
        ),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    inference_demo(base_model, tokenizer)

    print("\nEnd-to-end full-finetune example completed successfully!")


if __name__ == "__main__":
    main()
