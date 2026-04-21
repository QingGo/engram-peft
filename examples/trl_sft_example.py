"""
Example script demonstrating how to use Engram-PEFT with trl's SFTTrainer.
"""

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from engram_peft import EngramConfig, create_engram_sft_trainer, get_engram_model


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

    def formatting_prompts_func(example: dict[str, list[str]]) -> list[str]:
        """Format the dataset into the expected prompt template."""
        output_texts = []
        for i in range(len(example["instruction"])):
            text = f"### Instruction: {example['instruction'][i]}\n### Response: {example['response'][i]}"
            output_texts.append(text)
        return output_texts

    # 2. Setup model and tokenizer
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    print(f"Loading base model and tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    # 3. Define Engram Configuration
    config = EngramConfig(
        target_layers=[0, 1],
        engram_vocab_size_per_ngram=[1000, 1000],
        ngram_sizes=[2, 3],
        n_head_per_ngram=2,
    )

    # 4. Initialize Engram Model
    print("Injecting Engram layers into the model...")
    model = get_engram_model(base_model, config, tokenizer=tokenizer)

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./engram_sft_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=5,
        logging_steps=1,
        save_steps=5,
        fp16=False,
        push_to_hub=False,
        report_to="none",
    )

    # 6. Create SFTTrainer using the Engram compatibility layer
    print("Initializing SFTTrainer via Engram compatibility layer...")
    trainer = create_engram_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
        max_seq_length=128,
    )

    # 7. Start Training
    print("Starting training...")
    trainer.train()

    # 8. Save the results
    print("Saving the fine-tuned Engram adapter...")
    model.save_pretrained("./engram_sft_final")
    print("Success!")


if __name__ == "__main__":
    main()
