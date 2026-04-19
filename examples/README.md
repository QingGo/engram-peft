# Engram-PEFT Examples

This directory contains examples demonstrating how to use the Engram-PEFT library for efficient Parameter-Efficient Fine-Tuning.

## 1. End-to-End Example (GPU): `end_to_end.py`

The `end_to_end.py` script provides a focused walkthrough of the Engram lifecycle on GPU hardware. This is the recommended script for a standard "Getting Started" experience.

### Features
- **Dataset Preparation**: Streaming download and tokenization of "TinyStories".
- **Engram Injection**: Automatic layer injection and base model freezing.
- **Sparse Training**: Efficient training using `EngramTrainer` and `MixedOptimizer`.
- **Inference Demo**: Interactive generation with dynamic `load`/`unload` switching.

### Running the Example
```bash
uv run python examples/end_to_end.py --max_steps 50 --batch_size 4 --num_workers 4
```

---

## 2. Baseline Comparison: `compare_engram_lora.py`

This script is used to benchmark Engram against other training methods like LoRA and Full Finetuning to showcase its advantages in memory, speed, and learning stability.

### Features
- **Benchmarking**: Tracks Peak VRAM usage and average wall-clock time per step.
- **Multiple Methods**: Evaluates `lora`, `engram`, `lora_engram`, `full_finetune`, and `full_finetune_engram` under identical conditions.
- **Hyperparameter Sweeps**: Support for per-method parameter overrides using the `method:param=val` syntax (handles lists like `target_layers=[2,15]`).
- **Layer-wise Clipping**: Toggleable per-layer gradient clipping via `clip_grad_per_layer=True` to improve sparse training stability.
- **Visualization**: Generates a high-quality comparison plot (`outputs/benchmarks/loss_curve.png`).
- **Reporting**: Persists all metrics into timestamped JSON files for historical comparison.

### Running the Comparison
```bash
# Compare standard Engram vs LoRA
uv run python examples/compare_engram_lora.py --methods lora engram

# Compare all available methods
uv run python examples/compare_engram_lora.py --all

# Hyperparameter Sweep: Compare different clipping strategies for Engram
uv run python examples/compare_engram_lora.py --methods engram:clip_grad_per_layer=False engram:clip_grad_per_layer=True

# Mix multiple methods with overrides (including list parameters)
uv run python examples/compare_engram_lora.py --methods lora:learning_rate=1e-4 engram:target_layers=[2,15]
```

---

## 3. CPU-Optimized Example: `end_to_end_cpu.py`

Designed for environments **without a GPU** or for users who want a **fast demonstration** (< 3 minutes) using a nano-sized model.

### Running the Example
```bash
uv run python examples/end_to_end_cpu.py
```

---

## 4. Flexible Weight Migration: `flexible_loading.py`

Demonstrates how to migrate Engram weights between different model configurations (e.g., different layer counts or bucket sizes).

### Features
- **Automatic Resizing**: Slices or zero-pads embedding tables to match target bucket capacities.
- **Layer Re-mapping**: Maps specific source layers to new target layers using a dictionary.

### Running the Example
```bash
uv run python examples/flexible_loading.py
```

---

## 5. Cross-Tokenizer Knowledge Transfer: `cross_tokenizer_migration.py`

Showcase the unique ability of Engram to transfer specialized knowledge between models using completely **different tokenizers** (e.g., Llama vs. Qwen).

### Features
- **Semantic Alignment**: Uses a raw text corpus as a "bridge" to calculate hash correspondences.
- **Best-Effort Remapping**: Transfers specialized "knowledge tokens" using character-level offset mapping.

### Running the Example
```bash
uv run python examples/cross_tokenizer_migration.py
```

---

## 6. YAML Configuration (CLI): `config.yaml`

A declarative way to define training experiments. This allows you to run the standardized `engram-peft train` pipeline without writing custom logic.

### Features
- **Separation of Concerns**: Decouples model config, training params, and dataset metadata.
- **Reproducibility**: Easily share specific experiment configurations.
- **Dynamic Overrides**: Combine the YAML file with CLI-level parameter injection.

### Running the Example
```bash
# source .venv/bin/activate
# Start training with the provided configuration template
engram-peft train --config examples/config.yaml

# Perform a quick learning rate sweep via overrides
engram-peft train --config examples/config.yaml --overrides "training_args.learning_rate=5e-5"
```

---

## 🛠 Prerequisites

The examples require additional libraries (`matplotlib`, `seaborn`, `pandas`, `peft`, `datasets`) which are included in the project's dev dependencies.

Ensure your environment is synchronized:
```bash
uv sync --all-groups
```
