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

# Quick check on overhead
model.print_trainable_parameters()

# 4. Train
collator = EngramDataCollator(tokenizer=tokenizer, config=config)

# EngramTrainer handles the MixedOptimizer and Step Decay scheduler automatically
from engram_peft import EngramTrainer
trainer = EngramTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="engram_out",
        per_device_train_batch_size=4,
        learning_rate=4e-4  # Automatically passed to MixedOptimizer
    ),
    data_collator=collator,
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
2.  **Apply Engram** to the Transformer Blocks using the `wrap_peft=True` flag to keep LoRA weights trainable.
3.  **Result**: A model that has the reasoning style of LoRA and the factual memory of Engram.

```python
from peft import LoraConfig, get_peft_model
from engram_peft import EngramConfig, get_engram_model

# 1. Apply LoRA first
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# 2. Inject Engram on top
engram_config = EngramConfig(target_layers=[2, 15])
# IMPORTANT: Use wrap_peft=True so Engram doesn't freeze your LoRA parameters!
model = get_engram_model(model, engram_config, wrap_peft=True) 

# Now both LoRA and Engram parameters are trainable!
model.print_trainable_parameters()
```

> [!IMPORTANT]
> When stacking adapters, ensure you pass `wrap_peft=True` to `get_engram_model`. This instructs Engram to identify and preserve the `requires_grad=True` status of existing parameters (like LoRA weights) instead of performing a blanket freeze on the entire input model.

---

## Tutorial 4: Managing Multiple Knowledge Packs

Engram-PEFT supports a "Named Adapter" system similar to `peft`. You can load multiple specialized knowledge packs into the same base model and switch between them at runtime.

```python
# Assuming you have an engram model with 'default' knowledge
engram_model.print_trainable_parameters()

# 1. Add a second adapter for a different domain
legal_config = EngramConfig(target_layers=[2, 11, 20], embedding_dim=1024)
engram_model.add_adapter("legal", legal_config)

# 2. Switch to the new adapter for training
engram_model.set_adapter("legal")
# ... run training for legal knowledge ...

# 3. Switch back to medical knowledge
# 3. Switch back to medical knowledge
engram_model.set_adapter("default")
```

---

## Tutorial 5: Flexible Weight Migration

Engram-PEFT allows you to reuse pre-trained knowledge even if your target model has different layers, bucket capacities, or even a different tokenizer seed.

### Case A: Structural Alignment (Different Layers/Buckets)
If you have weights trained on layers `[0, 1]` but your new model uses layers `[5, 6]`:

```python
# Map layer 0 to 5, and layer 1 to 6
model.load_weights_flexible(
    "path/to/engram_weights.pt",
    layer_mapping={0: 5, 1: 6},
    reuse_structural=False # Recommended: Re-train Gating/Conv for the new layer position
)
```

### Case B: Logic Alignment (Different Seeds/Tokenizer)
If the hashing logic differs (e.g., a different `seed` was used in `EngramConfig`), use a reference corpus to "re-discover" the correct indices via best-effort remapping:

```python
# corpus_tokens should be a representative sample of your training data
model.remap_from_corpus(corpus_tokens, "path/to/engram_weights.pt")
```
