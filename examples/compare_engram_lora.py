"""
Engram-PEFT Baseline Comparison Script.

This script demonstrates how Engram compares against standard PEFT methods (like LoRA):
1. Benchmarking: Compares training loss, peak memory, and time-per-step.
2. Baselines: Evaluates Zero-shot Base Model and trained LoRA adapter.
3. Visualization: Generates a premium loss curve comparison using Seaborn.
4. Lifecycle: Demonstrates full training and dynamic management.

Usage:
    uv run python examples/compare_engram_lora.py --max_steps 50 --batch_size 4
"""

import argparse
import copy
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset  # type: ignore
from peft import LoraConfig, TaskType, get_peft_model

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
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
    get_optimizer,
    get_scheduler,
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR = "outputs/engram_test"
ENGRAM_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "engram_weights")
LORA_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "lora_weights")
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
    tokenizer: PreTrainedTokenizerBase, subset_size: int, max_length: int
) -> Any:
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

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

    train_data = []
    for i, item in enumerate(dataset):
        if i >= subset_size:
            break
        train_data.append(item)

    from datasets import Dataset

    train_dataset = Dataset.from_list(train_data).map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return train_dataset


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
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
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

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "lora_outputs"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        learning_rate=4e-4,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DefaultDataCollator(),
    )

    print("Starting LoRA training...")
    train_result = trainer.train()

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
        if "loss" in log and "step" in log:
            log_history.append({"step": log["step"], "loss": log["loss"]})

    # Restore base model to normal by unloading LoRA (clean slate for Engram)
    model = model.unload()

    return {
        "log_history": log_history,
        "peak_memory_gb": peak_memory,
        "avg_time_per_step": avg_time_per_step,
    }


def train_engram_model(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
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
    )

    print("Injecting Engram layers and freezing base model...")
    model = get_engram_model(base_model, config, tokenizer)

    collator = EngramDataCollator(tokenizer=tokenizer, config=config)
    optimizer = get_optimizer(model, base_learning_rate=4e-4)
    scheduler = get_scheduler(
        optimizer, num_training_steps=args.max_steps, warmup_steps=10
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
    )

    print("Starting Engram training...")
    train_result = trainer.train()

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
        if "loss" in log and "step" in log:
            log_history.append({"step": log["step"], "loss": log["loss"]})

    print(f"Saving Engram weights to {ENGRAM_WEIGHTS_DIR}")
    model.save_pretrained(ENGRAM_WEIGHTS_DIR)

    # CRITICAL: Unload Engram hooks from the base model so that the next phase
    # (inference) starts with a clean slate when it creates its own wrapper.
    model.unload_engram()

    return {
        "log_history": log_history,
        "peak_memory_gb": peak_memory,
        "avg_time_per_step": avg_time_per_step,
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
        f"{'Method':<12} | {'Peak Mem (GB)':<15} | {'Avg Time/Step (s)':<18} | {'Eval Loss':<10}"
    )
    print("-" * 65)

    base = results.get("base_model", {})
    print(
        f"{'Base':<12} | {base.get('peak_memory_gb', 0):<15.2f} | {'N/A':<18} | {base.get('eval_loss', 0):.4f}"
    )

    lora = results.get("lora", {})
    print(
        f"{'LoRA':<12} | {lora.get('peak_memory_gb', 0):<15.2f} | {lora.get('avg_time_per_step', 0):<18.4f} | {'N/A':<10}"
    )

    engram = results.get("engram", {})
    print(
        f"{'Engram':<12} | {engram.get('peak_memory_gb', 0):<15.2f} | {engram.get('avg_time_per_step', 0):<18.4f} | {'N/A':<10}"
    )
    print("=" * 61 + "\n")

    # Set premium Seaborn theme
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))

    # Prepare data for Seaborn
    plot_data = []

    # Engram Logs
    engram_logs = results.get("engram", {}).get("log_history", [])
    for item in engram_logs:
        plot_data.append(
            {"Step": item["step"], "Loss": item["loss"], "Method": "Engram"}
        )

    # LoRA Logs
    lora_logs = results.get("lora", {}).get("log_history", [])
    for item in lora_logs:
        plot_data.append({"Step": item["step"], "Loss": item["loss"], "Method": "LoRA"})

    # Create the line plot
    if plot_data:
        import pandas as pd

        df = pd.DataFrame(plot_data)
        ax = sns.lineplot(
            data=df,
            x="Step",
            y="Loss",
            hue="Method",
            marker="o",
            linewidth=2.5,
            palette="viridis",
        )
    else:
        ax = plt.gca()

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
    base_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    print(f"\n>>> Phase 4: Inference & Gating Visualization")

    # Load trained Engram onto the base model
    print(f"Loading trained Engram from {ENGRAM_WEIGHTS_DIR}")
    model = EngramModel.from_pretrained(base_model, ENGRAM_WEIGHTS_DIR)

    prompt = "Once upon a time, there was a little robot named"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.base_model.device)

    print(f"\nPrompt: {prompt}")

    # Generate with Engram enabled
    print("Generating with Engram ENABLED...")
    output_engram = model.generate(
        **inputs, max_new_tokens=20, max_length=None, do_sample=False
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
            gate_str = " | ".join([f"B{i}: {g:.3f}" for i, g in enumerate(mean_gates)])
            print(f"Layer {layer_id}: {gate_str}")

    # Dynamic Switching Demo
    print(f"\n>>> Phase 5: Dynamic Switching Demo")

    print("Unloading Engram (Switch back to Base Model)...")
    model.unload_engram()
    output_base = model.generate(
        **inputs, max_new_tokens=20, max_length=None, do_sample=False
    )
    print(
        f"Output (Base):   {tokenizer.decode(output_base[0], skip_special_tokens=True)}"
    )

    print("Reloading Engram weights...")
    model.load_engram(ENGRAM_WEIGHTS_DIR)
    output_reloaded = model.generate(
        **inputs, max_new_tokens=20, max_length=None, do_sample=False
    )
    print(
        f"Output (Reload): {tokenizer.decode(output_reloaded[0], skip_special_tokens=True)}"
    )

    print("\nEnd-to-end example completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Engram End-to-End Test")
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--subset", type=int, default=1000, help="Dataset subset size to load"
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

    # 1. Prepare Data
    train_dataset = prepare_dataset(tokenizer, subset_size=args.subset, max_length=128)

    results: Dict[str, Any] = {}

    # 2. Get baseline performance (Zero-shot loss on some eval data)
    eval_subset = train_dataset.select(range(min(len(train_dataset), 50)))
    base_loss = get_base_model_eval_loss(
        base_model, tokenizer, eval_subset, batch_size=args.batch_size
    )
    results["base_model"] = {
        "eval_loss": base_loss,
        "peak_memory_gb": base_mem,
    }

    # 3. Train LoRA setup
    print("\n" + "=" * 50)
    lora_results = train_lora(base_model, tokenizer, train_dataset, args)
    results["lora"] = lora_results

    # 4. Train Engram setup
    print("\n" + "=" * 50)
    engram_results = train_engram_model(base_model, tokenizer, train_dataset, args)
    results["engram"] = engram_results

    # 5. Save & Plot
    print("\n" + "=" * 50)
    save_and_plot_results(results)

    # 6. Inference & dynamic usage Demo
    print("\n" + "=" * 50)
    inference_demo(base_model, tokenizer)
