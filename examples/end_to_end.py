"""
Engram-PEFT End-to-End Example.

This script demonstrates a full workflow using Engram-PEFT:
1. Training: Loading a base model, injecting Engram layers, and training on TinyStories.
2. Inference: Generating text with trained Engram weights.
3. Visualization: Observing the Context-Aware Gating activation levels.
4. Dynamic Management: Loading and unloading Engram packs at runtime.

Usage:
    uv run python examples/end_to_end.py
"""

import os
from typing import Any, Callable, Dict, List, Optional, cast

import torch
import torch.nn as nn
from datasets import load_dataset  # type: ignore[import-untyped]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramLayer,
    EngramModel,
    get_engram_model,
    get_optimizer,
    get_scheduler,
)

# 1. Configuration & Constants
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR = "outputs/engram_test"
ENGRAM_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "engram_weights")
DATASET_SUBSET = 200  # Smaller subset for faster demonstration
MAX_LENGTH = 128
SEED = 42

set_seed(SEED)


def train_engram() -> PreTrainedTokenizer:
    print(f"\n>>> Phase 1: Training Engram on {MODEL_NAME}")

    # Load Tokenizer & Model
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )

    # Initialize Engram Configuration
    # We set hidden_size to 2048 to match TinyLlama's architecture
    config = EngramConfig(
        target_layers=[2, 11, 20],  # Inject into early, middle, and late layers
        hidden_size=2048,
        embedding_dim=1024,  # Engram retrieval dimension
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=MODEL_NAME,
        pad_id=tokenizer.pad_token_id,
    )

    # Inject Engram into base model
    print("Injecting Engram layers and freezing base model...")
    model = get_engram_model(base_model, config, tokenizer)  # type: ignore[arg-type]

    # Prepare TinyStories Dataset
    print(f"Loading TinyStories dataset (subset={DATASET_SUBSET})...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(  # type: ignore[no-any-return]
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    # Take a small sample and tokenize
    train_data = []
    for i, item in enumerate(dataset):
        if i >= DATASET_SUBSET:
            break
        train_data.append(item)

    from datasets import Dataset

    train_dataset = Dataset.from_list(train_data).map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Collator precomputes hash indices on CPU
    collator = EngramDataCollator(tokenizer=tokenizer, config=config)  # type: ignore[arg-type]

    # Setup Optimizer & Scheduler
    # MixedOptimizer manages SparseAdam for embeddings and Adam for conv/gating
    optimizer = get_optimizer(model, base_learning_rate=4e-4)

    # Calculate training steps for 1 epoch
    num_train_epochs = 1
    total_steps = len(train_dataset) * num_train_epochs // 8  # Assume batch_size=8
    scheduler = get_scheduler(
        optimizer, num_training_steps=total_steps, warmup_steps=10
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=num_train_epochs,
        learning_rate=4e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
    )

    # Trainer Execution
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
    )

    print("Starting training...")
    trainer.train()

    # Save ONLY Engram weights and config
    print(f"Saving Engram weights to {ENGRAM_WEIGHTS_DIR}")
    model.save_pretrained(ENGRAM_WEIGHTS_DIR)

    return cast(PreTrainedTokenizer, tokenizer)


def visualize_gating_hook(name: str) -> Callable[[nn.Module, Any, Any], None]:
    """Factory for forward hooks to capture gating values."""

    def hook(module: nn.Module, input: Any, output: Any) -> None:
        # gate has shape [B, L, M, D] in ContextAwareGating but we sum it to [B, L, M] for visualization
        # In our implementation, gate is computed and immediately applied.
        # We need to capture the 'gate' within ContextAwareGating.forward or modify it to store it.
        # For this demonstration, we'll just print a notification that hooks are active.
        pass

    return hook


def inference_demo(tokenizer: PreTrainedTokenizer) -> None:
    print(f"\n>>> Phase 2: Inference & Gating Visualization")

    # Load base model again (clean slate)
    print("Loading clean base model for inference...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )

    # Load trained Engram onto the base model
    print(f"Loading trained Engram from {ENGRAM_WEIGHTS_DIR}")
    model = EngramModel.from_pretrained(base_model, ENGRAM_WEIGHTS_DIR)

    # Setup Gating Visualization
    # We will hook into the ContextAwareGating modules to see activation levels
    gating_values: Dict[int, Any] = {}

    def get_gating_hook(layer_id: int) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # The gate is actually computed inside 'forward'.
            # In our current implementation, we'd need to modify 'layer.py' to expose it
            # or use a more complex hook.
            # For now, let's just demonstrate the generation and dynamic switching.
            pass

        return hook

    prompt = "Once upon a time, there was a little robot named"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.base_model.device)

    print(f"\nPrompt: {prompt}")

    # Generate with Engram enabled
    print("Generating with Engram ENABLED...")
    output_engram = model.generate(**inputs, max_new_tokens=20, do_sample=False)
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

    # 3. Dynamic Switching Demo
    print(f"\n>>> Phase 3: Dynamic Switching Demo")

    print("Unloading Engram (Switch back to Base Model)...")
    model.unload_engram()
    output_base = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(
        f"Output (Base):   {tokenizer.decode(output_base[0], skip_special_tokens=True)}"
    )

    print("Reloading Engram weights...")
    model.load_engram(ENGRAM_WEIGHTS_DIR)
    output_reloaded = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(
        f"Output (Reload): {tokenizer.decode(output_reloaded[0], skip_special_tokens=True)}"
    )

    print("\nEnd-to-end example completed successfully!")


if __name__ == "__main__":
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Train
    tokenizer = train_engram()

    # 2. Inference & Dynamic Switching
    inference_demo(tokenizer)
