"""
Example script demonstrating how to use Engram-PEFT with trl's SFTTrainer.
"""

from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig

from engram_peft import (
    EngramConfig,
    EngramModel,
    create_engram_sft_trainer,
    get_engram_model,
)
from engram_peft.utils import get_optimal_precision_config


def main() -> None:
    # 1. Setup a dummy dataset for SFT
    data = {
        "instruction": [
            "Explain the concept of sparse memory.",
            "What is parameter-efficient fine-tuning?",
            "Write a short poem about AI.",
        ],
        "response": [
            "Sparse memory allows models to access specific information without full activation.",
            "PEFT methods reduce the number of trainable parameters during fine-tuning.",
            "In silicon minds, a spark does glow, a world of data, fast and slow.",
        ],
    }
    dataset = Dataset.from_dict(data)

    def formatting_prompts_func(example: dict[str, Any]) -> str:
        """Format the dataset into the expected prompt template."""
        return f"### Instruction: {example['instruction']}\n### Response: {example['response']}"

    # 2. Setup model and tokenizer
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    print(f"Loading base model and tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    # 3. Define Engram Configuration

    # EngramCompatibleSFTTrainer now supports sparse embeddings natively
    config = EngramConfig(
        target_layers=[0, 1],
        engram_vocab_size_per_ngram=[1000, 1000],
        ngram_sizes=[2, 3],
        n_head_per_ngram=2,
    )

    # 4. Initialize Engram Model
    print("Injecting Engram layers into the model...")
    model = get_engram_model(base_model, config, tokenizer=tokenizer)

    # 5. Define Training Arguments using SFTConfig
    # SFTConfig inherits from TrainingArguments and includes SFT-specific fields
    precision_cfg = get_optimal_precision_config()
    sft_config = SFTConfig(
        output_dir="outputs/engram_sft_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=5,
        logging_steps=1,
        save_steps=5,
        bf16=precision_cfg["bf16"],
        fp16=precision_cfg["fp16"],
        push_to_hub=False,
        report_to="none",
        max_length=128,
    )

    # 6. Create SFTTrainer using the Engram compatibility layer
    print("Initializing SFTTrainer via Engram compatibility layer...")
    trainer = create_engram_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        args=sft_config,
    )

    # 7. Start Training
    print("Starting training...")
    trainer.train()

    # 8. Save the final model
    print("Saving the fine-tuned Engram adapter...")
    model.save_pretrained("outputs/engram_sft_final")
    print("Success! Results saved to outputs/engram_sft_final")

    # 9. Inference Demo
    print("\n>>> Inference Demo")
    prompt = "### Instruction: What is the benefit of sparse memory?\n### Response: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.base_model.device)
    input_len = inputs["input_ids"].shape[-1]

    print(f"Prompt: {prompt}")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
        )
    resp = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    print(f"Response: {resp}")

    # 10. Reload and Verify
    print("\n>>> Reloading for Verification")
    try:
        # Load a fresh base model
        reloaded_base = AutoModelForCausalLM.from_pretrained(model_id)
        # Reload Engram adapter
        reloaded_model = EngramModel.from_pretrained(
            reloaded_base, "outputs/engram_sft_final", tokenizer=tokenizer
        )

        print("Inference with Reloaded Model:")
        with torch.no_grad():
            reloaded_output = reloaded_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
            )
        reloaded_resp = tokenizer.decode(
            reloaded_output[0][input_len:], skip_special_tokens=True
        )
        print(f"Response: {reloaded_resp}")
        print("\nReload verification successful!")
    except Exception as e:
        print(f"Reloading failed: {e}")


if __name__ == "__main__":
    main()
