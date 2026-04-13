# Tutorials

Step-by-step guides to mastering Engram-PEFT for efficient LLM knowledge injection.

---

## Tutorial 1: 5-Minute Quickstart

Learn how to inject Engram conditional memory into a small model like TinyLlama and train it on a toy dataset.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from engram_peft import EngramConfig, get_engram_model, EngramDataCollator, get_optimizer

# 1. Setup
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. Configure Engram
config = EngramConfig(
    target_layers=[2, 11, 20],
    embedding_dim=1024,
    tokenizer_name_or_path=model_id
)

# 3. Inject & Freeze
base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
model = get_engram_model(base_model, config, tokenizer)

# 4. Train
collator = EngramDataCollator(tokenizer=tokenizer, config=config)
optimizer = get_optimizer(model)

# Standard HF Trainer usage...
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="engram_out", per_device_train_batch_size=4),
    data_collator=collator,
    optimizers=(optimizer, None), # Scheduler handled automatically by get_optimizer in full flow
    train_dataset=my_dataset
)
trainer.train()

# 5. Save ONLY the knowledge pack
model.save_pretrained("medical_knowledge_pack")
```

---

## Tutorial 2: Injecting Medical Knowledge into Llama-3

Engram is specifically designed to store vast amounts of static knowledge without interfering with the model's original reasoning capabilities.

**Scenario:** You want to fine-tune Llama-3-8B on a large corpus of medical textbooks (PubMed).

1.  **Initialize with full capacity**:
    Increase `engram_vocab_size_per_ngram` to handle millions of specialized medical terms.
    ```python
    config = EngramConfig(
        engram_vocab_size_per_ngram=[2262400, 2262400], # Large capacity
        target_layers=[2, 8, 16, 24], # More layers for deep knowledge
        tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B"
    )
    ```
2.  **Train on Domain Data**:
    Use the `EngramDataCollator` to ensure high-throughput training. Engram's sparse updates allow you to train on a much larger corpus than traditional LoRA without catastrophic forgetting of the base model's logic.

---

## Tutorial 3: Mixing Engram with LoRA

Engram and LoRA can be used together! LoRA is excellent for **task adaptation** (e.g., following instructions), while Engram is superior for **knowledge storage**.

### The "Double Adapter" Strategy
1.  **Apply LoRA** to the base model's Attention or MLP layers.
2.  **Apply Engram** to the Transformer Blocks.
3.  **Result**: A model that has the reasoning style of LoRA and the factual memory of Engram.

```python
from peft import LoraConfig, get_peft_model
from engram_peft import EngramConfig, get_engram_model

# 1. Apply LoRA first
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# 2. Inject Engram on top
engram_config = EngramConfig(target_layers=[2, 15])
model = get_engram_model(model, engram_config) # Engram handles the nesting

# Now both LoRA and Engram parameters are trainable!
# Base model parameters remain frozen.
```
