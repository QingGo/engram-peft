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
uv run python examples/end_to_end.py --max_steps 50 --batch_size 4
```

---

## 2. Baseline Comparison: `compare_engram_lora.py`

This script is used to benchmark Engram against other PEFT methods like LoRA to showcase its advantages in memory and stability.

### Features
- **Benchmarking**: Tracks Peak VRAM usage and average wall-clock time per step.
- **LoRA vs Engram**: Trains both methods under identical conditions for fair comparison.
- **Visualization**: Generates a high-quality comparison plot (`outputs/engram_test/loss_curve.png`).
- **Reporting**: Persists all metrics into a `training_metrics.json` file.

### Running the Comparison
```bash
uv run python examples/compare_engram_lora.py --max_steps 100 --batch_size 4
```

---

## 3. CPU-Optimized Example: `end_to_end_cpu.py`

Designed for environments **without a GPU** or for users who want a **fast demonstration** (< 3 minutes) using a nano-sized model.

### Running the Example
```bash
uv run python examples/end_to_end_cpu.py
```

---

## 🛠 Prerequisites

The examples require additional libraries (`matplotlib`, `seaborn`, `pandas`, `peft`, `datasets`) which are included in the project's dev dependencies.

Ensure your environment is synchronized:
```bash
uv sync --all-groups
```
