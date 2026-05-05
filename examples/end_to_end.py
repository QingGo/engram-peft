# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none, reportArgumentType=none
"""
Engram-PEFT End-to-End GPU Example.

This script demonstrates the core Engram-PEFT workflow:
1. Dataset Preparation: Loading and tokenizing a small subset of TinyStories.
2. Engram Injection: Injecting Context-Aware Hash-tables into a base model.
3. Focused Training: Updating only the Engram parameters using MixedOptimizer.
4. Saving & Inference: Persisting weights and demonstrating dynamic loading/generation.
5. Checkpoint Resume: Save mid-training checkpoint and resume seamlessly.

Usage:
    ```bash
uv run python examples/end_to_end.py --max_steps 50 --batch_size 4 --num_workers 4
```
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
from transformers.trainer_utils import get_last_checkpoint

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
from engram_peft.utils.compat import wash_tokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR = "outputs/engram_standard"
ENGRAM_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "engram_weights")
SEED = 42

set_seed(SEED)


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase, subset_size: int, max_length: int
) -> Any:
    print("Loading dataset from HF (may be slow without mirror)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

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


def train_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    args: argparse.Namespace,
) -> tuple[EngramTrainer, EngramModel, EngramConfig, EngramDataCollator]:
    print(f"\n>>> Phase 1: Training Engram on {MODEL_NAME}")
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
    # base_model is already a PreTrainedModel

    model = get_engram_model(
        base_model,
        config,
        wash_tokenizer(tokenizer),
        train_mode="engram_only",
    )

    collator = EngramDataCollator(tokenizer=wash_tokenizer(tokenizer), config=config)
    optimizer = get_optimizer(model, base_learning_rate=4e-4)
    scheduler = get_scheduler(
        optimizer, num_training_steps=args.max_steps, warmup_steps=10
    )

    save_steps = max(1, args.max_steps // 3)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
    )

    print("Starting Engram training...")
    result = trainer.train()
    print(f"Phase 1 complete. Global step: {result.global_step}")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak Memory: {peak_mem:.2f} GB")

    print(f"Saving Engram weights to {ENGRAM_WEIGHTS_DIR}")
    model.save_pretrained(ENGRAM_WEIGHTS_DIR)

    # Ready for inference
    model.unload_engram()

    return trainer, model, config, collator


def inference_demo(
    base_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    print("\n>>> Phase 2: Inference & Dynamic Usage")

    # Load trained Engram onto the base model
    print(f"Loading trained Engram from {ENGRAM_WEIGHTS_DIR}")
    # base_model is already a torch.nn.Module

    model = EngramModel.from_pretrained(base_model, ENGRAM_WEIGHTS_DIR)

    prompt = "Once upon a time, there was a little robot named"
    device = model.base_model.device
    if not isinstance(device, torch.device | str):
        device = str(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
        engram_layer = cast("EngramLayer", model.engram_layers[str(layer_id)])
        gate = engram_layer.gating.last_gate  # [B, L, M, 1]
        if gate is not None:
            mean_gates = gate.mean(dim=(0, 1, 3)).cpu().tolist()
            gate_str = " | ".join([f"B{i}: {g:.3f}" for i, g in enumerate(mean_gates)])
            print(f"Layer {layer_id}: {gate_str}")

    # Dynamic Switching Demo
    print("\nUnloading Engram (Back to Base Model)...")
    model.unload_engram()
    output_base = model.generate(
        **inputs, max_new_tokens=20, max_length=None, do_sample=False
    )
    print(
        f"Output (Base):   {tokenizer.decode(output_base[0], skip_special_tokens=True)}"
    )

    print("\nEnd-to-end demo completed!")


def resume_demo(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    collator: EngramDataCollator,
    args: argparse.Namespace,
) -> None:
    print("\n>>> Phase 3: Checkpoint Resume Test")

    last_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if last_ckpt is None:
        print(f"No checkpoint found in {OUTPUT_DIR}. Skipping resume test.")
        print("(Make sure save_strategy is set to 'steps' in Phase 1)")
        return

    print(f"Latest checkpoint: {last_ckpt}")

    # Simulate a script restart: load model from the saved checkpoint
    print("Loading model from checkpoint...")
    resume_model = EngramModel.from_pretrained(
        base_model, last_ckpt, tokenizer=wash_tokenizer(tokenizer)
    )
    print("Model loaded from checkpoint successfully.")

    resume_max_steps = args.max_steps + args.max_steps // 2

    optimizer = get_optimizer(resume_model, base_learning_rate=4e-4)
    scheduler = get_scheduler(
        optimizer, num_training_steps=resume_max_steps, warmup_steps=10
    )

    resume_output_dir = os.path.join(OUTPUT_DIR, "resume")
    resume_args = TrainingArguments(
        output_dir=resume_output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=resume_max_steps,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
    )

    resume_trainer = EngramTrainer(
        model=resume_model,
        args=resume_args,
        train_dataset=train_dataset,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
    )

    print(
        f"Resuming training from checkpoint (target max_steps={resume_max_steps})..."
    )
    result = resume_trainer.train(resume_from_checkpoint=last_ckpt)
    print(f"Resume complete. Global step: {result.global_step}")

    # Verify: after resume, the total steps should reach resume_max_steps
    if result.global_step >= resume_max_steps:
        print(
            "PASSED: Checkpoint resume works correctly. "
            + f"Reached step {result.global_step} >= {resume_max_steps}."
        )
    else:
        print(
            f"WARNING: Global step is {result.global_step}, "
            + f"expected >= {resume_max_steps}. Resume may not have worked as expected."
        )

    resume_model.unload_engram()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Engram End-to-End Demo")
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--subset", type=int, default=1000, help="Dataset subset size to load"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer & base model: {MODEL_NAME}")
    tokenizer = cast(
        "PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(MODEL_NAME)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)

    # 1. Prepare Data
    train_dataset = prepare_dataset(tokenizer, subset_size=args.subset, max_length=128)

    # 2. Train Engram (with checkpoint saving enabled)
    trainer, model, config, collator = train_engram(
        base_model, tokenizer, train_dataset, args
    )

    # 3. Inference Demo
    inference_demo(base_model, tokenizer)

    # 4. Checkpoint Resume Test
    resume_demo(base_model, tokenizer, train_dataset, collator, args)
