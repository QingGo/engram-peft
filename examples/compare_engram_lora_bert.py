"""
BERT Comparison Script: Engram vs LoRA.

This script benchmarks Engram-PEFT against LoRA on a real-world NLP task:
SST-2 (Stanford Sentiment Treebank) binary classification.

It compares:
1. Training/Eval Loss and Accuracy.
2. Peak VRAM usage.
3. Training throughput (Steps/sec).

Usage:
    uv run python examples/compare_engram_lora_bert.py --max_steps 500 --batch_size 32
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset  # type: ignore
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramTrainer,
    get_engram_model,
)

MODEL_NAME = "google-bert/bert-base-uncased"
OUTPUT_DIR = "outputs/bert_comparison"
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
    max_length: int,
) -> Tuple[Any, Any]:
    print("Loading GLUE SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    if subset_size > 0:
        train_ds = train_ds.select(range(min(len(train_ds), subset_size)))
        val_ds = val_ds.select(range(min(len(val_ds), subset_size // 5 + 100)))

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        )

    print("Tokenizing datasets...")
    train_dataset = train_ds.map(
        tokenize_function, batched=True, remove_columns=["sentence", "idx"]
    )
    eval_dataset = val_ds.map(
        tokenize_function, batched=True, remove_columns=["sentence", "idx"]
    )

    return train_dataset, eval_dataset


def train_lora(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print("\n>>> Phase 1: Training LoRA Baseline")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "lora_logs"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(1, args.max_steps // 5),
        save_strategy="no",
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    train_result = trainer.train()
    metrics = trainer.evaluate()

    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    avg_step_time = train_result.metrics["train_runtime"] / train_result.global_step

    # Unload LoRA to restore base model
    lora_model.unload()

    return {
        "log_history": trainer.state.log_history,
        "peak_memory_gb": peak_mem,
        "avg_time_per_step": avg_step_time,
        "eval_accuracy": metrics.get("eval_accuracy", 0.0),
        "eval_loss": metrics.get("eval_loss", 0.0),
    }


def train_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print("\n>>> Phase 2: Training Engram")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    engram_config = EngramConfig(
        target_layers=[2, 4, 8, 10],  # Spread across the model
        engram_vocab_size_per_ngram=[50000, 50000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=768,
        bidirectional_conv=True,  # BERT specific
        hashing_mode="centered",  # BERT specific
        stop_token_ids=[
            t for t in [tokenizer.sep_token_id, tokenizer.cls_token_id] if t is not None
        ],
        tokenizer_name_or_path=MODEL_NAME,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        learning_rate_multiplier=5.0,
    )

    model = get_engram_model(base_model, engram_config, tokenizer)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "engram_logs"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(1, args.max_steps // 5),
        save_strategy="no",
        report_to="none",
    )

    # Use specialized EngramDataCollator for pre-hashing
    collator = EngramDataCollator(tokenizer=tokenizer, config=engram_config, mlm=False)

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    train_result = trainer.train()
    metrics = trainer.evaluate()

    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    avg_step_time = train_result.metrics["train_runtime"] / train_result.global_step

    model.unload_engram()

    return {
        "log_history": trainer.state.log_history,
        "peak_memory_gb": peak_mem,
        "avg_time_per_step": avg_step_time,
        "eval_accuracy": metrics.get("eval_accuracy", 0.0),
        "eval_loss": metrics.get("eval_loss", 0.0),
    }


def save_and_plot(results: Dict[str, Any]) -> None:
    print(f"\n>>> Generating benchmarks and plots in {OUTPUT_DIR}")

    # Create Summary Table
    print("\n" + "=" * 80)
    print(
        f"{'Method':<15} | {'Accuracy':<10} | {'Loss':<10} | {'Peak VRAM (GB)':<15} | {'Step Time (s)':<15}"
    )
    print("-" * 80)
    for method, data in results.items():
        print(
            f"{method.upper():<15} | {data['eval_accuracy']:<10.4f} | {data['eval_loss']:<10.4f} | {data['peak_memory_gb']:<15.2f} | {data['avg_time_per_step']:<15.4f}"
        )
    print("=" * 80)

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for method, data in results.items():
        logs = [i for i in data["log_history"] if "loss" in i]
        steps = [i["step"] for i in logs]
        losses = [i["loss"] for i in logs]

        ax1.plot(steps, losses, label=f"{method.upper()} Loss", linewidth=2)

        eval_logs = [i for i in data["log_history"] if "eval_accuracy" in i]
        e_steps = [i["step"] for i in eval_logs]
        e_acc = [i["eval_accuracy"] for i in eval_logs]

        if e_steps:
            ax2.plot(e_steps, e_acc, label=f"{method.upper()} Accuracy", marker="o")

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_title("Evaluation Accuracy")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "bert_comparison_curves.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Save JSON
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        # Filter out large log_history for cleaner JSON summary
        summary = {
            k: {sk: sv for sk, sv in v.items() if sk != "log_history"}
            for k, v in results.items()
        }
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.stdout = Logger(os.path.join(OUTPUT_DIR, "benchmark.log"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    if torch.cuda.is_available():
        base_model = base_model.cuda()  # type: ignore

    train_ds, eval_ds = prepare_dataset(tokenizer, args.subset, args.max_length)

    results = {}

    # 1. LoRA
    results["lora"] = train_lora(base_model, tokenizer, train_ds, eval_ds, args)

    # 2. Engram
    results["engram"] = train_engram(base_model, tokenizer, train_ds, eval_ds, args)

    # Save and Plot
    save_and_plot(results)
