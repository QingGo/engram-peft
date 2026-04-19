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
model = get_engram_model(
    base_model,
    config,
    tokenizer,
    train_mode="engram_only",
)

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

## Tutorial 1.5: Simplified Training via CLI

For most standard use cases, you don't need to write a custom Python script. Engram-PEFT provides a high-level CLI powered by `typer`.

### 1. Generating a Configuration Template
We provide a minimal example in `examples/config.yaml`, but we highly recommend generating a full, documented template using the CLI to see all available options:

```bash
engram-peft config-template --output my_config.yaml
```

The generated file is divided into four main sections with detailed comments:
- `model_name_or_path`: The base model identifier.
- `engram_config`: Core hyperparameters for Engram layers.
- `lora_config`: (Optional) PEFT LoRA settings for hybrid adaptation.
- `training_args`: All standard `transformers.TrainingArguments`.
- `data_args`: Dataset name and tokenization settings.

### 2. Launching Training
Trigger the training pipeline using your configuration file:
```bash
engram-peft train --config my_config.yaml
```

### 3. Automated Inference Script
Once training is complete, the CLI automatically generates a ready-to-run `inference.py` script inside your `output_dir`. You can immediately test your trained model:

```bash
# Assuming output_dir is ./outputs/tinyllama-engram
uv run python outputs/tinyllama-engram/inference.py
```

### 4. Quick Overrides
You can override any nested configuration value using dot notation without editing the YAML file:
```bash
engram-peft train --config my_config.yaml \
    --overrides "training_args.learning_rate=1e-5" \
    --overrides "engram_config.target_layers=[2,15,31]"
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
2.  **Apply Engram** to the Transformer Blocks using `train_mode="preserve_trainable"` to keep LoRA weights trainable.
3.  **Result**: A model that has the reasoning style of LoRA and the factual memory of Engram.

```python
from peft import LoraConfig, get_peft_model
from engram_peft import EngramConfig, get_engram_model

# 1. Apply LoRA first
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# 2. Inject Engram on top
engram_config = EngramConfig(target_layers=[2, 15])
# IMPORTANT: preserve_trainable keeps existing LoRA parameters trainable.
model = get_engram_model(
    model,
    engram_config,
    train_mode="preserve_trainable",
)

# Now both LoRA and Engram parameters are trainable!
model.print_trainable_parameters()
```

> [!IMPORTANT]
> When stacking adapters, use `train_mode="preserve_trainable"` so Engram keeps the `requires_grad=True` status of existing parameters (like LoRA weights). `wrap_peft=True` is still supported as a backward-compatible alias, but `train_mode` is the recommended API.

---

## Tutorial 4: Transparent Injection & Custom Models

Engram-PEFT uses a multi-tiered strategy to find transformer layers. You can monitor this process via logs or override it for custom models.

### 1. Enabling Detailed Logs
By default, the library is quiet. To see exactly where and how Engram layers are being injected, enable `INFO` logging:

```python
import logging
# Only show INFO for engram_peft to avoid noise from other libraries
logging.basicConfig(level=logging.WARNING)
logging.getLogger("engram_peft").setLevel(logging.INFO)
```

**Expected Log Output:**
```text
[Engram-PEFT] Starting best-effort architecture discovery...
[Engram-PEFT] Determined layer_container_path='model.layers' (source: Architecture Registry (llama))
[Engram-PEFT] Attaching Engram layers to 32 blocks...
  - [Injected] Layer 2 -> LlamaDecoderLayer (device: cuda:0)
  - [Injected] Layer 15 -> LlamaDecoderLayer (device: cuda:0)
```

### 2. Targeting Custom Architectures
Engram-PEFT includes a built-in registry for common architectures including:
- **Llama-2/3, Mistral, Mixtral, Qwen2**
- **DeepSeek V2/V3**
- **Gemma/Gemma 2, Phi/Phi-3**
- **BERT, RoBERTa, Longformer**
- **GPT-2, GPT-NeoX**
- **GLM/ChatGLM**

If you are using a non-standard model that isn't in our built-in registry, Engram will fall back to a heuristic (finding the largest `nn.ModuleList`). If this fails, you can specify the path manually:

```python
config = EngramConfig(
    layer_container_path="my_model.transformer.h", # Explicit path
    target_layers=[0, 5, 10]
)
model = get_engram_model(base_model, config)
```

---

## Tutorial 5: Full Finetuning with Engram

If you want to train the backbone together with Engram, use `train_mode="full_finetune"` and configure separate optimizer groups for backbone, Engram dense layers, and Engram sparse embeddings.

```python
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from engram_peft import EngramConfig, EngramTrainer, get_engram_model

model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id)

config = EngramConfig(
    target_layers=[2, 11],
    tokenizer_name_or_path=model_id,
)

model = get_engram_model(
    base_model,
    config,
    tokenizer,
    train_mode="full_finetune",
)

optimizer = get_optimizer(
    model,
    backbone_learning_rate=5e-5,
    engram_dense_learning_rate=4e-4,
    engram_sparse_learning_rate=2e-3,
    backbone_optimizer=AdamW,
)

# Or let EngramTrainer build the layered optimizer for you
trainer = EngramTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="engram_full_ft_out",
        per_device_train_batch_size=2,
        learning_rate=4e-4,
    ),
    train_dataset=my_dataset,
    optimizer_kwargs={
        "backbone_learning_rate": 5e-5,
        "engram_dense_learning_rate": 4e-4,
        "engram_sparse_learning_rate": 2e-3,
        "backbone_optimizer": AdamW,
    },
)

# Save both parts after training
model.save_pretrained("engram_adapter_only")
model.base_model.save_pretrained("engram_full_model")
```

> [!IMPORTANT]
> In `train_mode="full_finetune"`, `model.save_pretrained(...)` still saves only Engram weights and config. Save `model.base_model` to a separate directory as well if you want a restorable full-finetuned checkpoint.

---

## Tutorial 6: Managing Multiple Knowledge Packs

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
engram_model.set_adapter("default")
```

---

## Tutorial 7: Flexible Weight Migration

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
# corpus should be a representative sample of your training data (tokens or strings)
model.remap_from_corpus(corpus, "path/to/engram_weights.pt")
```

---

## Tutorial 8: Performance Benchmarking & Comparison

To truly understand the benefits of Engram vs. traditional PEFT methods (like LoRA), you can use our built-in benchmarking suite.

### 1. Running All Methods
We provide a shortcut to run a standard suite of experiments (LoRA, Engram, LoRA+Engram, Full Finetune, etc.) in one go.

```bash
uv run python examples/compare_engram_lora.py --all --max_steps 500
```

### 2. Custom Sweeps
You can also run specific methods or perform hyperparameter sweeps (e.g., comparing different layer counts) using the `--methods` flag with overrides:

```bash
# Compare different layer configurations for Engram
uv run python examples/compare_engram_lora.py --methods engram:target_layers=[2,11] engram:target_layers=[11,21] 
```

---

## Tutorial 9: Production-Ready Mainstream Model Examples

For users looking to deploy Engram-PEFT with state-of-the-art models, we provide comprehensive templates for **Qwen 3.5**, **Ministral 3**, and **Gemma 4**. These scripts are located in the `examples/` directory and feature out-of-the-box support for:

1.  **Instruction Tuning**: Ready-made prompt templates for common chat formats.
2.  **Quantized Training**: Integration with `bitsandbytes` for 4-bit and 8-bit fine-tuning, enabling training on consumer GPUs.
3.  **Hybrid Adapters**: Demonstrates how to use LoRA for task-specific behavior while using Engram for massive knowledge storage.

### Running a Template
```bash
# Example: Training Qwen 3.5-4B using 4-bit quantization
uv run python examples/qwen3_engram_lora.py --load_in_4bit --max_steps 300
```

Refer to the [Examples README](file:///app/examples/README.md) for more details on each template.
