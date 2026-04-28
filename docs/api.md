# API Reference

Detailed documentation for all public classes and functions in `engram-peft`, 100% aligned with the DeepSeek Engram paper and official implementation.

---

## Configuration

### `EngramConfig`
`engram_peft.config.EngramConfig`

Configuration class for Engram PEFT module. Inherits from `transformers.PretrainedConfig`. All default values exactly match the specifications in the Engram paper Appendix A Table 5.

**Parameters:**
- `engram_vocab_size_per_ngram` (`List[int]`, default: `[1131200, 1131200]`): Total engram vocabulary size split per N-gram order.
- `ngram_sizes` (`List[int]`, default: `[2, 3]`): List of N-gram orders to use (e.g., `[2, 3]` means 2-grams and 3-grams).
- `n_head_per_ngram` (`int`, default: `8`): Number of hash heads per N-gram order.
- `embedding_dim` (`int`, default: `1280`): Dimension of the Engram retrieval embedding.
- `enable_tokenizer_compression` (`bool`, default: `True`): Whether to use NFKC/Lowercase normalization for token grouping.
- `target_layers` (`List[int]`, default: `[1, 14]`): Transformer layers where Engram modules are injected, 0-indexed.
- `target_modules` (`Optional[Union[List[str], str]]`, default: `None`): Specific module names or regex patterns to target for injection.
- `hc_mult` (`int`, default: `4`): Multi-head hyper-connection expansion factor.
- `combine_mhc` (`bool`, default: `True`): Whether to combine multi-head hyper-connections.
- `conv_kernel_size` (`int`, default: `4`): Convolution kernel size for short-term context.
- `conv_dilation` (`Optional[int]`, default: `None`): Convolution dilation (defaults to `max(ngram_sizes)`).
- `conv_zero_init` (`bool`, default: `True`): Initialize convolution weights to zero to ensure identity mapping at start.
- `learning_rate_multiplier` (`float`, default: `5.0`): LR multiplier for sparse embedding parameters.
- `tokenizer_name_or_path` (`Optional[str]`, default: `None`): Tokenizer used for precomputing hashes. Recommended to set explicitly (e.g., `"deepseek-ai/DeepSeek-V3"`).
- `seed` (`int`, default: `0`): Random seed for deterministic hashing primes.
- `weight_decay` (`float`, default: `0.0`): Weight decay for Engram parameters.
- `gating_zero_init` (`bool`, default: `True`): Whether to initialize gating parameters with zeros.
- `hidden_size` (`Optional[int]`, default: `None`): The hidden dimension of the base model. Auto-detected if not provided.
- `pad_id` (`Optional[int]`, default: `None`): The padding token ID. Auto-detected if not provided.
- `compressed_vocab_size` (`Optional[int]`, default: `None`): Resolved size of the hashing vocabulary. Automatically set and saved after first initialization.
- `layer_container_path` (`Optional[str]`, default: `None`): Explicit dot-separated path to the transformer layers.
- `max_ngram_size` (`int`, default: `3`): Maximum N-gram size, derived from `ngram_sizes`.
- `clip_grad_per_group` (`bool`, default: `False`): Whether to use group-wise gradient clipping.
- `enable_telemetry` (`bool`, default: `False`): Enables detailed metric logging (norms, drift, max values).
- `entropy_loss_weight` (`float`, default: `0.0`): Weight for the gating entropy penalty loss.
- `backbone_freeze_steps` (`int`, default: `0`): Initial steps where backbone is frozen (Adapter-First warm-up).
- `engram_dtype` (`Optional[str]`, default: `None`): Explicit precision for Engram layers (e.g., `"float32"`, `"float16"`, `"bfloat16"`). If `None`, it's auto-detected from the backbone's `compute_dtype`.

**Example Usage:**
```python
from engram_peft import EngramConfig

config = EngramConfig(
    target_layers=[2, 11, 20],
    embedding_dim=1024,
    learning_rate_multiplier=5.0,
    engram_dtype="bfloat16"
)
```

---

## Model Wrapping

### `get_engram_model`
`engram_peft.model.get_engram_model(model, config, tokenizer=None, wrap_peft=False, train_mode=None, backbone_freeze_steps=0, engram_dtype=None)`

Injects Engram layers into a base Transformer model and configures which backbone
parameters remain trainable.

**Args:**
- `model` (`Union[PreTrainedModel, nn.Module]`): The base model to wrap. Supports standard Hugging Face models and custom `torch.nn.Module` architectures.
- `config` (`EngramConfig`): Engram configuration.
- `tokenizer` (`Optional[PreTrainedTokenizer]`): Tokenizer for vocabulary/compression.
- `wrap_peft` (`bool`, default: `False`): Backward-compatible alias for `train_mode="preserve_trainable"`.
- `train_mode` (`Literal["engram_only", "preserve_trainable", "full_finetune"]`, optional): Controls backbone trainability.
- `backbone_freeze_steps` (`int`, default: `0`): Initial steps to freeze the backbone.
- `engram_dtype` (`Optional[str]`, default: `None`): Explicit precision for Engram layers.
  - `engram_only`: Freeze the backbone and train only Engram.
  - `preserve_trainable`: Preserve parameters that were already trainable before wrapping (e.g., LoRA), then add trainable Engram layers.
  - `full_finetune`: Train the full backbone together with Engram.

**Returns:**
- `EngramModel`: The wrapped model with injected forward hooks.

**Examples:**
```python
# Pure Engram PEFT
model = get_engram_model(base_model, config, tokenizer, train_mode="engram_only")

# LoRA + Engram
model = get_engram_model(model, config, tokenizer, train_mode="preserve_trainable")

# Full finetuning + Engram
model = get_engram_model(base_model, config, tokenizer, train_mode="full_finetune")
```

### `EngramModel`
`engram_peft.model.EngramModel`

The wrapper class for the base model. Handles dynamic hook management and weight serialization.

**Methods:**
- `print_trainable_parameters()`: Prints trainable counts for backbone, Engram, and total parameters.
- `add_adapter(adapter_name: str, config: EngramConfig)`: Adds a new set of Engram weights with its own configuration.
- `set_adapter(adapter_name: str)`: Switches the active knowledge pack to the specified adapter.
- `create_optimizer(base_learning_rate: float, **optimizer_kwargs)`: Returns a `MixedOptimizer` with configurable backbone/Engram optimizer groups.
- `create_scheduler(optimizer, num_training_steps, warmup_steps)`: Returns the paper-aligned Step Decay scheduler.
- `get_telemetry_stats()`: Collects activation statistics and diagnostics from active layers.
- `get_total_gating_entropy()`: Aggregates gating entropy tensors for regularization.
- `save_pretrained(save_directory: str, safe_serialization: bool = True, **kwargs)`: Saves Engram configurations and weights. Automatically saves base model adapters (e.g. LoRA) if present.
- `push_to_hub(repo_id: str, use_temp_dir: bool | None = None, commit_message: str | None = None, private: bool | None = None, token: str | bool | None = None, **kwargs)`: Pushes the Engram adapter (and any base model adapters) to the Hugging Face Hub.
- `from_pretrained(base_model, engram_path, tokenizer=None, **kwargs)`: Loads Engram weights onto a base model. `engram_path` can be a local directory or a Hugging Face Hub ID. Supports `token` and `revision` via `kwargs`.
- `unload_engram()`: Dynamically removes all PEFT hooks (reverts to base model).
- `load_engram(engram_path=None)`: Re-installs hooks and optionally loads weights.
- `load_weights_flexible(checkpoint_path, source_config_path=None, layer_mapping=None, reuse_structural=False)`: Loads weights from a checkpoint even if configurations (layers, buckets, n-grams) differ.
- `remap_from_corpus(corpus, checkpoint_path, source_config_path=None, layer_mapping=None, tokenizer=None, batch_size=1024)`: "Best-effort" remapping for cases where seeds or tokenizers differ, using a reference corpus to align indices.

---

## Data Utilities

### `EngramDataCollator`
`engram_peft.collator.EngramDataCollator`

A high-performance data collator that precomputes multi-head hash indices on the CPU during data loading, ensuring the GPU is dedicated to training.

**Args:**
- `tokenizer`: Hugging Face tokenizer.
- `config`: `EngramConfig` instance.
- `compressor`: `CompressedTokenizer` instance (optional).

**Example Usage:**
```python
from engram_peft import EngramDataCollator
from transformers import Trainer

collator = EngramDataCollator(tokenizer=tokenizer, config=config)
trainer = Trainer(..., data_collator=collator)
```

### `EngramTrainer`
`engram_peft.trainer.EngramTrainer`

Trainer subclass that handles sparse gradient clipping and can build Engram's
mixed optimizer automatically.

**Notable Args:**
- `optimizer_kwargs` (`Optional[Dict[str, Any]]`): Extra keyword arguments forwarded to `model.create_optimizer(...)` / `get_optimizer(...)`. Use this to configure layered optimizer behavior when relying on the trainer's default optimizer creation path.

**Example Usage:**
```python
trainer = EngramTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizer_kwargs={
        "backbone_learning_rate": 5e-5,
        "engram_dense_learning_rate": 4e-4,
        "engram_sparse_learning_rate": 2e-3,
    },
)
```

### `evaluate_model_loss`
`engram_peft.utils.evaluate_model_loss(model, tokenizer, dataset, batch_size=8, max_length=128)`

Standardizes the calculation of Zero-shot loss for language models. Uses `DataCollatorForLanguageModeling` to ensure correct label shifting.

---

## TRL Integration

A high-level factory function that creates an `EngramCompatibleSFTTrainer` (a subclass of `trl.SFTTrainer`) pre-configured for Engram models. It automatically handles model preparation, sets up the `EngramDataCollator`, and enables sparse gradient support.

### `EngramCompatibleSFTTrainer`
`engram_peft.trl.EngramCompatibleSFTTrainer`

A customized `SFTTrainer` that provides native support for Engram's sparse gradients within the TRL ecosystem.

**Key Features:**
- **`create_optimizer`**: Automatically uses `MixedOptimizer` to handle sparse embedding updates correctly.
- **`_clip_grad_norm`**: Implements a custom gradient clipping logic that supports sparse tensors on both CPU and GPU, bypassing PyTorch's `NotImplementedError`.
- **Automatic Config**: Inherits all features from `trl.SFTTrainer` while ensuring compatibility with `EngramModel` specific requirements.

### `prepare_engram_for_sft`
`engram_peft.trl.prepare_engram_for_sft(model, use_gradient_checkpointing=True)`

Prepares an `EngramModel` for SFT by disabling `use_cache`, enabling gradient checkpointing, and ensuring the model is in training mode.

### `create_engram_sft_trainer`
`engram_peft.trl.create_engram_sft_trainer(model, tokenizer, train_dataset, eval_dataset=None, args=None, **kwargs)`

High-level factory function that creates an `EngramCompatibleSFTTrainer` pre-configured for Engram models. It automatically handles model preparation, sets up the `EngramDataCollator`, and supports the SFTConfig migration for `trl>=1.2.0`.

**Args:**
- `model` (`EngramModel`): The Engram model instance.
- `tokenizer`: The tokenizer instance.
- `train_dataset`: The training dataset.
- `eval_dataset` (optional): Evaluation dataset.
- `args` (`TrainingArguments` or `SFTConfig`, optional): Training configuration.
- `**kwargs`: Additional arguments passed to `SFTTrainer`.

**Example Usage:**
```python
from engram_peft import create_engram_sft_trainer

trainer = create_engram_sft_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=sft_config,
)
trainer.train()
```

---

## Optimization

### `get_optimizer`
`engram_peft.utils.get_optimizer(model, base_learning_rate=4e-4, backbone_learning_rate=None, engram_dense_learning_rate=None, engram_sparse_learning_rate=None, backbone_weight_decay=None, engram_dense_weight_decay=None, engram_sparse_weight_decay=0.0, backbone_optimizer=None, engram_dense_optimizer="adam", engram_sparse_optimizer="sparse_adam")`

Creates a `MixedOptimizer` with separate optimizer groups for:
- Engram sparse embeddings (default: `SparseAdam`)
- Engram dense parameters (default: `Adam`)
- Backbone parameters (default: `AdamW`)

**Args:**
- `model` (`EngramModel`): The Engram model.
- `base_learning_rate` (`float`, default: `4e-4`): Base learning rate.
- `backbone_learning_rate` (`float`, optional): LR for backbone params (defaults to `base_learning_rate`).
- `engram_dense_learning_rate` (`float`, optional): LR for Engram dense params (defaults to `base_learning_rate * learning_rate_multiplier`).
- `engram_sparse_learning_rate` (`float`, optional): LR for Engram sparse embeddings (defaults to `base_learning_rate * learning_rate_multiplier`).
- `backbone_weight_decay` (`float`, optional): Weight decay for backbone params.
- `engram_dense_weight_decay` (`float`, optional): Weight decay for Engram dense params.
- `engram_sparse_weight_decay` (`float`, default: `0.0`): Weight decay for Engram sparse embeddings.
- `backbone_optimizer` (`OptimizerSpec`, optional): Optimizer for backbone (default: `"adamw"`).
- `engram_dense_optimizer` (`OptimizerSpec`, default: `"adam"`): Optimizer for Engram dense params.
- `engram_sparse_optimizer` (`OptimizerSpec`, default: `"sparse_adam"`): Optimizer for Engram sparse embeddings.

Supported optimizer specs:
- Built-in strings: `"adam"`, `"adamw"`, `"sgd"`, `"sparse_adam"`
- Optimizer classes (e.g., `torch.optim.AdamW`)
- Custom builder callables

**Example Usage:**
```python
from engram_peft import get_optimizer

optimizer = get_optimizer(
    model,
    backbone_learning_rate=5e-5,
    engram_dense_learning_rate=4e-4,
    engram_sparse_learning_rate=2e-3,
    backbone_optimizer="adamw",
    engram_dense_optimizer="adam",
    engram_sparse_optimizer="sparse_adam",
)
```

### `get_trainable_param_groups`
`engram_peft.utils.get_trainable_param_groups(model)`

Returns a dictionary with three trainable parameter lists:
- `backbone`
- `engram_dense`
- `engram_sparse`

### `get_scheduler`
`engram_peft.utils.get_scheduler(optimizer, num_training_steps, warmup_steps=0)`

Returns a `LambdaLR` scheduler implementing the Step Decay schedule from the DeepSeek paper (decay at 80% and 90% progress).

**Example Usage:**
```python
from engram_peft import get_scheduler

scheduler = get_scheduler(optimizer, num_training_steps=1000, warmup_steps=100)
```

---

## General Utilities

### `get_optimal_precision_config`
`engram_peft.utils.get_optimal_precision_config()`

Automatically detects the best available training precision based on hardware.

**Returns:**
- `dict[str, bool]`: A dictionary with keys `"bf16"` and `"fp16"`.
  - On Ampere+ GPUs: `{"bf16": True, "fp16": False}`
  - On older GPUs or NPU: `{"bf16": False, "fp16": True}`
  - On CPU: `{"bf16": False, "fp16": False}`

**Example:**
```python
training_args = TrainingArguments(
    output_dir="./results",
    **get_optimal_precision_config()
)
```

### Device Backend (`device.py`)
`engram_peft.utils.device`

Unified device backend abstraction supporting CUDA, NPU (Ascend), and CPU.

- `get_available_device() -> str`: Returns `"npu"`, `"cuda"`, or `"cpu"` based on priority (NPU > CUDA > CPU).
- `is_cuda_available() -> bool`: Wraps `torch.cuda.is_available()`.
- `is_npu_available() -> bool`: Lazily imports `torch_npu` and checks NPU availability. Returns `False` gracefully if `torch_npu` is not installed.
- `is_bf16_supported(device_type: str | None = None) -> bool`: Checks BF16 support for the given or detected device.
- `get_amp_device_type() -> str`: Returns the autocast device type (`"npu"`, `"cuda"`, or `"cpu"`).
- `create_grad_scaler(device_type: str | None = None) -> GradScalerProtocol | None`: Creates the appropriate GradScaler for the device. Returns `None` for CPU.
- `get_distributed_backend() -> str`: Returns `"hccl"` on NPU, `"nccl"` on CUDA, or `""` on CPU. Use as the `backend` argument to `torch.distributed.init_process_group()`.
- `is_hccl_available() -> bool`: Returns `True` when NPU + HCCL (Huawei Collective Communication Library) is available.

**Example:**
```python
from engram_peft.utils.device import get_available_device, create_grad_scaler, is_bf16_supported, get_distributed_backend

device = get_available_device()  # "npu", "cuda", or "cpu"
scaler = create_grad_scaler(device)
if scaler is not None:
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()

# Distributed backend detection
backend = get_distributed_backend()  # "hccl" on NPU, "nccl" on CUDA
if backend:
    torch.distributed.init_process_group(backend=backend)
```

### `apply_peft_patches`
`engram_peft.utils.apply_peft_patches()`

Applies necessary monkey-patches to third-party libraries (like PEFT) to improve compatibility with Engram modules.

---

## Core Components

### `EngramLayer`
`engram_peft.layer.EngramLayer`

The core torch module containing retrieval embeddings, context-aware gating, and short-term convolutions. Typically managed automatically via `get_engram_model`.
