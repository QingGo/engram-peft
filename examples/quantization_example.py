# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
"""
Quantization Support Example for Engram-PEFT.

This script demonstrates how to inject Engram layers into a 4-bit quantized
backbone (bitsandbytes) while maintaining correct computation precision.

Usage:
    uv run python examples/quantization_example.py --model_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
"""

import argparse
from typing import Any, cast

import torch
from datasets import load_dataset
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
    EngramModel,
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
        learning_rate_multiplier=1.0,
        enable_telemetry=True,
        gating_zero_init=False,  # Start with some contribution
    )

    # get_engram_model will automatically detect bnb.compute_dtype
    model = get_engram_model(base_model, engram_config, tokenizer=tokenizer)

    # Verify precision
    layer_id = target_layers[0]
    engram_layer = model.engram_layers[str(layer_id)]
    print("\nVerification:")
    print(f"Backbone Compute Dtype: {bnb_config.bnb_4bit_compute_dtype}")
    print(f"Engram Layer Dtype: {next(engram_layer.parameters()).dtype}")

    # 5. Training Step
    print("\nLoading real Alpaca dataset for calibration...")
    raw_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    # Take a subset for demonstration
    dataset_subset = raw_dataset.select(range(min(500, len(raw_dataset))))

    def tokenize_function(examples: dict[str, list[Any]]) -> dict[str, Any]:
        # Implementation of prompt masking for SFT
        prompts = []
        for i, instr in enumerate(examples["instruction"]):
            if examples["input"][i]:
                p = f"Instruction: {instr}\nInput: {examples['input'][i]}\nResponse: "
            else:
                p = f"Instruction: {instr}\nResponse: "
            prompts.append(p)

        full_texts = [
            p + out + tokenizer.eos_token
            for p, out in zip(prompts, examples["output"], strict=False)
        ]
        # Use fixed-length padding to avoid downstream collator issues
        tokenized = tokenizer(
            full_texts, truncation=True, max_length=128, padding="max_length"
        )

        # Create labels and mask the prompt part
        labels = []
        for i, input_ids in enumerate(tokenized["input_ids"]):
            # Get prompt length without padding
            prompt_ids = tokenizer(
                prompts[i],
                truncation=True,
                max_length=128,
                add_special_tokens=True,
                padding=False,
            )["input_ids"]
            prompt_len = len(prompt_ids)

            # Create labels: start with all -100 (masked)
            label = [-100] * 128

            # Only copy the non-prompt, non-pad tokens from input_ids
            # mask_limit is where the response starts
            mask_limit = min(prompt_len, 128)

            for j in range(mask_limit, 128):
                token_id = input_ids[j]
                if token_id != tokenizer.pad_token_id:
                    label[j] = token_id
                else:
                    # Keep as -100 for padding
                    label[j] = -100

            labels.append(label)

        tokenized["labels"] = labels
        return cast("dict[str, Any]", tokenized)

    train_dataset = dataset_subset.map(
        tokenize_function, batched=True, remove_columns=raw_dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir="outputs/quant_test",
        max_steps=50,
        per_device_train_batch_size=2,
        logging_steps=1,
        learning_rate=2e-5,  # Very conservative for 4-bit
        warmup_steps=2,
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
    OUTPUT_DIR = "outputs/quant_engram_final"

    # 6. Saving
    print(f"\nSaving fine-tuned Engram adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)

    # 7. Inference Demo
    print("\n>>> Inference Demo (Active Model)")
    model.eval()
    # Explicitly disable gradient checkpointing for inference stability
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Use the same format as training
    prompt = "Instruction: What is the capital of France?\nResponse: "
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    input_len = inputs["input_ids"].shape[-1]

    print(f"Prompt: {prompt}")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            max_length=None,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Decode only the NEW tokens
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(f"Response: {response}")

    # 8. Reload and Verify
    print("\n>>> Reloading for Verification")
    try:
        print("Re-loading base model in 4-bit...")
        # Re-use bnb_config from step 1
        reloaded_base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading Engram adapter from {OUTPUT_DIR}...")
        reloaded_model = EngramModel.from_pretrained(
            reloaded_base, OUTPUT_DIR, tokenizer=tokenizer
        )

        print("Inference with Reloaded Model:")
        with torch.no_grad():
            reloaded_outputs = reloaded_model.generate(
                **inputs,
                max_new_tokens=40,
                max_length=None,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        reloaded_resp = tokenizer.decode(
            reloaded_outputs[0][input_len:], skip_special_tokens=True
        )
        print(f"Response: {reloaded_resp}")
        print("\nReload verification successful!")
    except Exception as e:
        print(f"Reloading failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    )
    args = parser.parse_args()

    run_quantization_example(args.model_id)
