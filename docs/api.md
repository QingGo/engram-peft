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
- `target_layers` (`List[int]`, default: `[2, 15]`): Transformer layers where Engram modules are injected, 0-indexed.
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
- `layer_container_path` (`Optional[str]`, default: `None`): Explicit dot-separated path to the `nn.ModuleList` containing transformer layers (e.g., `"model.layers"`). If provided, it bypasses the automatic architecture discovery.

**Example Usage:**
```python
from engram_peft import EngramConfig

config = EngramConfig(
    target_layers=[2, 11, 20],
    embedding_dim=1024,
    learning_rate_multiplier=5.0
)
```

---

## Model Wrapping

### `get_engram_model`
`engram_peft.model.get_engram_model(model, config, tokenizer=None, wrap_peft=False, train_mode=None)`

Injects Engram layers into a base Transformer model and configures which backbone
parameters remain trainable.

**Args:**
- `model` (`Union[PreTrainedModel, nn.Module]`): The base model to wrap. Supports standard Hugging Face models and custom `torch.nn.Module` architectures.
- `config` (`EngramConfig`): Engram configuration.
- `tokenizer` (`Optional[PreTrainedTokenizer]`): Tokenizer for vocabulary/compression.
- `wrap_peft` (`bool`, default: `False`): Backward-compatible alias for `train_mode="preserve_trainable"`.
- `train_mode` (`Literal["engram_only", "preserve_trainable", "full_finetune"]`, optional): Controls backbone trainability.
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
- `create_scheduler(optimizer, num_steps, warmup_steps)`: Returns the paper-aligned Step Decay scheduler.
- `save_pretrained(save_directory: str)`: Saves ONLY the active Engram weights and configuration.
- `from_pretrained(base_model, engram_path)`: Loads Engram weights onto a base model.
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

---

## Optimization

### `get_optimizer`
`engram_peft.utils.get_optimizer(model, base_learning_rate=4e-4, ...)`

Creates a `MixedOptimizer` with separate optimizer groups for:
- Engram sparse embeddings
- Engram dense parameters
- Backbone parameters

Supported optimizer specs:
- Built-in strings: `"adam"`, `"adamw"`, `"sgd"`, `"sparse_adam"`
- Optimizer classes
- Custom builder callables

**Example Usage:**
```python
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

---

## Core Components

### `EngramLayer`
`engram_peft.layer.EngramLayer`

The core torch module containing retrieval embeddings, context-aware gating, and short-term convolutions. Typically managed automatically via `get_engram_model`.
