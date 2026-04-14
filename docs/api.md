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
- `target_layers` (`List[int]`, default: `[2, 15]`): Transformer layers where Engram modules are injected.
- `hc_mult` (`int`, default: `4`): Multi-head hyper-connection expansion factor.
- `combine_mhc` (`bool`, default: `True`): Whether to combine multi-head hyper-connections.
- `conv_kernel_size` (`int`, default: `4`): Convolution kernel size for short-term context.
- `conv_dilation` (`Optional[int]`, default: `None`): Convolution dilation (defaults to `max(ngram_sizes)`).
- `conv_zero_init` (`bool`, default: `True`): Initialize convolution weights to zero to ensure identity mapping at start.
- `learning_rate_multiplier` (`float`, default: `5.0`): LR multiplier for sparse embedding parameters.
- `tokenizer_name_or_path` (`str`, default: `"deepseek-ai/DeepSeek-V3"`): Tokenizer used for precomputing hashes.

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
`engram_peft.model.get_engram_model(model, config, tokenizer=None)`

Injects Engram layers into a base Transformer model and freezes all base parameters.

**Args:**
- `model` (`PreTrainedModel`): The Hugging Face model to wrap (e.g., Llama, Qwen).
- `config` (`EngramConfig`): Engram configuration.
- `tokenizer` (`Optional[PreTrainedTokenizer]`): Tokenizer for vocabulary/compression.

**Returns:**
- `EngramModel`: The wrapped model with injected forward hooks.

### `EngramModel`
`engram_peft.model.EngramModel`

The wrapper class for the base model. Handles dynamic hook management and weight serialization.

**Methods:**
- `save_pretrained(save_directory: str)`: Saves ONLY the Engram weights and configuration.
- `from_pretrained(base_model, engram_path)`: Loads Engram weights onto a base model.
- `unload_engram()`: Dynamically removes all PEFT hooks (reverts to base model).
- `load_engram(engram_path=None)`: Re-installs hooks and optionally loads weights.

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

---

## Optimization

### `get_optimizer`
`engram_peft.utils.get_optimizer(model, base_learning_rate=4e-4)`

Creates a `MixedOptimizer` that combines `SparseAdam` (for embeddings) and `Adam` (for dense layers).

### `get_scheduler`
`engram_peft.utils.get_scheduler(optimizer, num_training_steps, warmup_steps=0)`

Returns a `LambdaLR` scheduler implementing the Step Decay schedule from the DeepSeek paper (decay at 80% and 90% progress).

---

## Core Components

### `EngramLayer`
`engram_peft.layer.EngramLayer`

The core torch module containing retrieval embeddings, context-aware gating, and short-term convolutions. Typically managed automatically via `get_engram_model`.
