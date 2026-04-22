"""
Quantization Support Example for Engram-PEFT.

This script demonstrates how to inject Engram layers into a 4-bit quantized
backbone (bitsandbytes) while maintaining correct computation precision.

Usage:
    uv run python examples/quantization_example.py --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse

import torch
from datasets import Dataset
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramTrainer,
    get_engram_model,
)


def run_quantization_example(model_id: str) -> None:
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in 4-bit: {model_id}")
    # 1. Setup bitsandbytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load backbone
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. Prepare for k-bit training (Crucial for PEFT + Quantization)
    print("Preparing model for k-bit training...")
    base_model = prepare_model_for_kbit_training(base_model)

    # 4. Inject Engram Layers
    print("Injecting Engram layers...")
    # Target some layers (e.g., middle and last)
    num_layers = getattr(base_model.config, "num_hidden_layers", 22)
    target_layers = [num_layers // 2, num_layers - 1]

    engram_config = EngramConfig(
        target_layers=target_layers,
        embedding_dim=128,  # Small for example
        n_head_per_ngram=4,
    )

    # get_engram_model will automatically detect bnb.compute_dtype
    model = get_engram_model(base_model, engram_config, tokenizer=tokenizer)

    # Verify precision
    layer_id = target_layers[0]
    engram_layer = model.engram_layers[str(layer_id)]
    print("\nVerification:")
    print(f"Backbone Compute Dtype: {bnb_config.bnb_4bit_compute_dtype}")
    print(f"Engram Layer Dtype: {next(engram_layer.parameters()).dtype}")

    # 5. Training Step (New)
    print("\nStarting minimal training (5 steps) to calibrate Engram...")
    # Prepare dummy data
    train_data = [
        {"text": "The capital of France is Paris. It is a beautiful city."},
        {"text": "Engram-PEFT allows for efficient memory injection in LLMs."},
        {"text": "Quantization reduces the memory footprint of large models."},
    ]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=64, padding="max_length"
        )

    train_dataset = Dataset.from_list(train_data).map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="outputs/quant_test",
        max_steps=5,
        per_device_train_batch_size=2,
        logging_steps=1,
        learning_rate=5e-4,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=True,
    )

    data_collator = EngramDataCollator(tokenizer=tokenizer, config=model.config)

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 6. Simple Inference Test
    print("\nRunning post-training inference test...")
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, max_length=None)

    print(f"Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    args = parser.parse_args()

    run_quantization_example(args.model_id)
