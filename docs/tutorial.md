# Tutorials

Step-by-step guides to mastering Engram-PEFT for efficient LLM knowledge injection.

---

## Tutorial 1: 5-Minute Quickstart

Learn how to inject Engram conditional memory into a small model like TinyLlama and train it on a toy dataset.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from engram_peft import EngramConfig, get_engram_model, EngramDataCollator, get_optimizer
from engram_peft.utils import get_optimal_precision_config

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
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
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
        learning_rate=4e-4,  # Automatically passed to MixedOptimizer
        **get_optimal_precision_config()  # Automatically handle bf16/fp16
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
engram_config = EngramConfig(target_layers=[1, 14])
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

### Technical Highlights of our Implementations

Our templates are designed to handle the unique complexities of these next-generation architectures:

*   **Recursive Layer Discovery**: Automatically detects transformer layers even in complex multimodal wrappers (e.g., Qwen3.5's hybrid linear/standard attention blocks).
*   **Multimodal Configuration Sync**: Seamlessly handles nested `text_config` structures used in Mistral-3 and Gemma-4, ensuring correct dimensionality (e.g., 3072 vs 4096) for memory injection.
*   **PLE-Aware Hooks**: Designed to work alongside Gemma 4's **Per-Layer Embeddings (PLE)** by injecting hooks at the layer boundary, capturing the optimal representation for sparse lookup.
*   **Thinking Mode Handling**: Templates include optimized generation parameters (e.g., `stop_strings`) to manage reasoning outputs in Qwen and Gemma, providing clean responses after the `<think>` or `<|think|>` phase.

### Running a Template
```bash
# Example: Training Qwen 3.5-4B using 4-bit quantization
uv run python examples/qwen3_engram_lora.py --load_in_4bit --max_steps 300
```

Refer to the [Examples README](file:///app/examples/README.md) for more details on each template.

## Quantization Support

Engram-PEFT natively supports fine-tuning on top of models quantized via `bitsandbytes` (4-bit/8-bit) or `GPTQ`.

### Core Mechanism

Since quantized models use low-precision weights (e.g., `uint8` or `nf4`) but perform computation in a `compute_dtype` (typically `float16` or `bfloat16`), Engram layers must adapt to this precision.

Engram-PEFT handles this via:
1.  **Smart Detection**: `get_engram_model` automatically detects the `compute_dtype` of targeted layers and aligns the injected Engram layers accordingly.
2.  **Explicit Control**: You can force a specific precision (e.g., `float32` for training stability) using the `engram_dtype` parameter in `EngramConfig`.

### Usage Example: 4-bit Training

This is the most common configuration for low-VRAM fine-tuning:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from engram_peft import get_engram_model, EngramConfig

model_id = "mistralai/Mistral-7B-v0.1"

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 2. Load backbone model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. [CRITICAL] Prepare quantized model for PEFT training
base_model = prepare_model_for_kbit_training(base_model)

# 4. Inject Engram
config = EngramConfig(target_layers=[10, 20])
model = get_engram_model(base_model, config)
```

### Combining with LoRA

You can use both LoRA and Engram to maximize parameter efficiency:

```python
from peft import LoraConfig, get_peft_model

# 1. Apply LoRA first
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(base_model, lora_config)

# 2. Apply Engram (use train_mode="preserve_trainable" to preserve LoRA's trainable state)
engram_model = get_engram_model(peft_model, config, train_mode="preserve_trainable")
```

### Important Notes

- **VRAM Usage**: While the backbone is quantized, Engram embedding tables can still be large. If you hit OOM, consider reducing `embedding_dim` or the number of `target_layers`.
- **Training Stability**: If loss diverges in extreme quantization scenarios, try setting `engram_dtype="float32"` in your `EngramConfig` to maintain higher precision for the memory module.
- **Saving/Loading**: Engram adapters are saved in full precision (or BF16) and will automatically re-align to the current backbone's precision when reloaded.

---

## NPU Support: Training on Huawei Ascend

Engram-PEFT fully supports training and inference on **Huawei Ascend NPU** hardware via the unified device backend in `engram_peft.utils.device`.

### Prerequisites

1. **Ascend CANN** installed and configured (see [Huawei Ascend documentation](https://www.hiascend.com/)).
2. **torch_npu** installed:
   ```bash
   pip install torch-npu --index-url https://repo.huaweicloud.com/repository/pypi/simple
   ```

### Usage

No code changes are needed — Engram-PEFT automatically detects NPU and switches all AMP settings accordingly:

```python
from engram_peft import EngramConfig, get_engram_model

config = EngramConfig(
    target_layers=[2, 11, 20],
    engram_dtype="float16",  # NPU typically does not support bfloat16
)
model = get_engram_model(base_model, config, tokenizer)
```

For training with the `EngramTrainer` or TRL integration:

```python
from engram_peft import EngramTrainer
from engram_peft.utils import get_optimal_precision_config

trainer = EngramTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="npu_engram_out",
        per_device_train_batch_size=4,
        **get_optimal_precision_config(),  # auto-detects NPU → fp16
    ),
    data_collator=collator,
    train_dataset=dataset,
)
trainer.train()
```

### Using with `accelerate` (Single Device)

Create an `accelerate` config that targets NPU:

```bash
accelerate config
# Choose: "This machine" → "None" (don't use DeepSpeed/FSDP) → "NO" (no distributed training)
# Then set `--mixed_precision fp16` explicitly.
```

Or use a config file:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: fp16
machine_rank: 0
main_process_ip: null
main_process_port: null
num_machines: 1
num_processes: 1
use_cpu: false
```

### Distributed Training on NPU (DDP)

NPU uses **HCCL** (Huawei Collective Communication Library) instead of NCCL. The unified backend provides automatic detection:

```python
import torch
import torch.distributed as dist
from engram_peft.utils.device import get_distributed_backend

backend = get_distributed_backend()  # "hccl" on NPU, "nccl" on CUDA, "" on CPU
if backend:
    dist.init_process_group(backend=backend)
```

When using `transformers.Trainer` or `EngramTrainer` with `accelerate`, you must configure the backend via environment variables or `accelerate config`:

```bash
# Option 1: Environment variables
export ACCELERATE_TORCH_DEVICE=npu
export NCCL_BACKEND=hccl   # If PyTorch-NPU build supports this

# Option 2: Use accelerate config for multi-NPU
accelerate config
# Choose: "This machine" → "Multi-NPU" → set backend to "hccl"
```

**`accelerate` config for multi-NPU:**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_NPU
machine_rank: 0
main_process_ip: null
main_process_port: null
num_machines: 1
num_processes: 8  # Number of NPU devices
mixed_precision: fp16
```

### Distributed Training on NPU (DeepSpeed)

DeepSpeed on Ascend is supported in `deepspeed>=0.12.0` (experimental). To use it:

1. Install DeepSpeed with NPU support:
   ```bash
   uv sync --group deepspeed  # installs deepspeed>=0.15.0
   ```

2. Set `torch.distributed` backend before trainer initialization:
   ```python
   import torch.distributed as dist
   from engram_peft.utils.device import get_distributed_backend
   
   backend = get_distributed_backend()
   if backend:
       dist.init_process_group(backend=backend, init_method="env://")
   ```

3. Pass a standard DeepSpeed config to `TrainingArguments`:
   ```python
   training_args = TrainingArguments(
       output_dir="npu_ds_out",
       deepspeed="ds_config.json",
       **get_optimal_precision_config(),
   )
   ```

> **Note:** DeepSpeed's NPU integration is less mature than CUDA. If you encounter issues, fall back to DDP with `accelerate`.

### Known Limitations

- **bfloat16** is generally not supported on NPU — use `engram_dtype="float16"` explicitly.
- **bitsandbytes** quantization does not support NPU; 4-bit/8-bit training is not available on NPU.
- `pin_memory=True` in DataLoader may behave differently on NPU; set `pin_memory=False` if you encounter issues.
- DeepSpeed's Ascend support is experimental — validate with small-scale tests before production.

---

## Tutorial 9: Seamless SFT with TRL

Engram-PEFT provides a deep integration with Hugging Face `trl`, including full support for sparse embeddings in instruction tuning.

### Using EngramCompatibleSFTTrainer

While the standard `SFTTrainer` doesn't natively support sparse gradient clipping, our `EngramCompatibleSFTTrainer` solves this by providing custom clipping and optimization logic.

```python
import torch
from datasets import Dataset
from trl import SFTConfig
from engram_peft import EngramConfig, get_engram_model, create_engram_sft_trainer
from engram_peft.utils import get_optimal_precision_config

# 1. Setup Model and Engram
model = ...
tokenizer = ...
config = EngramConfig(target_layers=[2, 11])
model = get_engram_model(model, config, tokenizer=tokenizer)

# 2. Configure SFT
sft_config = SFTConfig(
    output_dir="outputs/sft_results",
    learning_rate=2e-4,
    max_steps=500,
    **get_optimal_precision_config() # Optimal hardware support
)

# 3. Create & Train (factory handles everything)
trainer = create_engram_sft_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=my_dataset,
    args=sft_config,
)

trainer.train()
```

**Key Advantages:**
*   **Sparse Support**: `use_sparse_embeddings=True` (default) now works out-of-the-box in TRL.
*   **Mixed Optimizer**: Automatically uses `SparseAdam` for embeddings and `AdamW` for dense weights.
*   **Robust Clipping**: Bypasses PyTorch `NotImplementedError` for sparse gradients on all hardware.

---

## Tutorial 10: Knowledge Memorization Benchmark (PopQA)

This tutorial walks through the `examples/engram_knowledge_memory.py` script — a complete end-to-end benchmark that evaluates Engram's ability to memorize long-tail factual knowledge using the **PopQA** dataset.

The script compares four configurations on Exact Match (EM) accuracy:
- **Base** — no adapter (raw backbone)
- **+Engram** — Engram adapter only
- **+LoRA** — LoRA adapter only
- **+Engram+LoRA** — combined

### Running the Benchmark

```bash
# Train Engram only, then evaluate
python examples/engram_knowledge_memory.py --mode train

# Train Engram + LoRA, then evaluate
python examples/engram_knowledge_memory.py --mode train --train_lora

# Evaluate only (load previously saved adapters)
python examples/engram_knowledge_memory.py --mode eval \
    --engram_path outputs/popqa_benchmark/engram \
    --lora_path outputs/popqa_benchmark/lora

# Distributed training (DDP, 8 GPUs)
torchrun --nproc_per_node=8 examples/engram_knowledge_memory.py

# Distributed training (DeepSpeed ZeRO-2)
torchrun --nproc_per_node=8 examples/engram_knowledge_memory.py --use_deepspeed
```

### What the Script Does

1. **Loads PopQA** — 14,267 factual QA pairs (80/20 train/test split).
2. **Loads backbone in 4-bit** — uses `bitsandbytes` NF4 quantization to fit large models (default: `Qwen/Qwen3.6-27B`) on a single GPU.
3. **Builds Engram adapter** — sparse embeddings, configurable `embedding_dim` and `target_layers`.
4. **Trains with `EngramTrainer`** — handles sparse gradients, mixed optimizer, and Engram data collator automatically.
5. **(Optional) Trains LoRA adapter** — uses standard `peft.LoraConfig` + `Trainer`.
6. **Evaluates EM accuracy** — greedy-decodes answers on held-out questions, compares against ground-truth via normalized exact match.
7. **Prints comparison table** — shows accuracy and delta vs. base model.

### Configuration Highlights

```bash
# Key CLI arguments
--model "Qwen/Qwen3.6-27B"      # Backbone model
--embedding_dim 1280              # Engram embedding dimension
--target_layers 2 15              # Which layers to inject
--batch_size 1                    # Per-device batch size (4-bit is memory-heavy)
--grad_accum 8                    # Gradient accumulation steps
--learning_rate 2e-4              # Learning rate for Engram
--num_epochs 3                    # Training epochs
--entropy_loss_weight 0.01        # Gating entropy regularization
--lora_r 16                       # LoRA rank (if --train_lora)
```

### Distributed Training Details

The script automatically detects distributed environment variables (`LOCAL_RANK`, `WORLD_SIZE`, `RANK`) set by `torchrun`. Key behaviors:

- **DDP mode** (default): DDP forces dense gradients (its gradient bucket mechanism flattens all gradients before all-reduce). `nn.Embedding(sparse=True)` is automatically overridden to `sparse=False`, and `MixedOptimizer` is replaced with standard `AdamW`. For sparse `SparseAdam` benefits, train on a single GPU.
- **DeepSpeed mode** (`--use_deepspeed`): Same as DDP — unsupported sparse gradients. Falls back to dense embeddings with standard `AdamW`.
- **Device-agnostic**: When running on Ascend NPU, use `get_distributed_backend()` from `engram_peft.utils.device` to detect the correct HCCL backend (see [NPU Distributed Training](#distributed-training-on-npu-ddp) section).


