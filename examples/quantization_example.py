"""
Quantization Support Example for Engram-PEFT.

This script demonstrates how to inject Engram layers into a 4-bit quantized
backbone (bitsandbytes) while maintaining correct computation precision.

Usage:
    uv run python examples/quantization_example.py --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse

import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from engram_peft import EngramConfig, get_engram_model


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

    # Simple Inference Test
    print("\nRunning simple inference test...")
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)

    print(f"Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    args = parser.parse_args()

    run_quantization_example(args.model_id)
