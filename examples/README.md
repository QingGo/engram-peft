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
uv run python examples/compare_engram_lora.py --all --max_steps 50 --batch_size 16 --subset 10000

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
# Recommended: Generate a fresh template to see all options
engram-peft config-template --output my_experiment.yaml

# Start training with the provided configuration template
engram-peft train --config examples/config.yaml

# Perform a quick learning rate sweep via overrides
engram-peft train --config examples/config.yaml --overrides "training_args.learning_rate=5e-5"
```

---

## 7. Hugging Face Hub Sharing: `hub_sharing.py`

Demonstrates how to push a trained Engram adapter to the Hugging Face Hub and reload it directly from a Hub repository ID.

### Features
- **Upload**: Push a complete Engram adapter (config + weights) to the Hub.
- **Reload**: Load an adapter from a remote repository by its ID.
- **No Repository Creation Needed**: `push_to_hub` auto-creates the remote repository if it doesn't exist.

### Running the Example
```bash
# Login first
hf auth login

# Push and reload
uv run python examples/hub_sharing.py --repo_id "your-username/tinyllama-engram-test"
```

---

## 8. Quantization Support: `quantization_example.py`

Shows how to inject Engram layers into a 4-bit quantized backbone (bitsandbytes) while maintaining correct computation precision.

### Features
- **Quantized Backbone**: Loads a model with `BitsAndBytesConfig` (4-bit NF4).
- **Full-Precision Engram**: Keeps Engram embedding tables in FP32/BF16 for gradient stability.
- **End-to-End Training**: Runs sparse training steps with gradient checkpointing and CPU-offloaded mixed optimizer.

### Running the Example
```bash
uv run python examples/quantization_example.py --model_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

---

## 9. TRL SFT Integration: `trl_sft_example.py`

Demonstrates how to use Engram-PEFT with `trl`'s `SFTTrainer` for supervised fine-tuning.

### Features
- **Seamless Integration**: Uses `create_engram_sft_trainer(...)` to wrap the standard SFT pipeline.
- **Dataset Flexibility**: Works with any format trl accepts (instruction-style datasets).
- **Minimal Boilerplate**: Handles Engram injection, optimizer setup, and training loop automatically.

### Running the Example
```bash
uv run python examples/trl_sft_example.py
```

---

## 10. Mainstream Model Templates (Training + Inference)

These scripts provide ready-to-use templates for the latest mainstream models in the ecosystem. They demonstrate combined **LoRA + Engram** fine-tuning on the Alpaca instruction dataset.

### Models Covered
- **Qwen 3.5-4B**: `examples/qwen3_engram_lora.py`
- **Ministral-3-3B**: `examples/mistral3_engram_lora.py`
- **Gemma-4-E2B**: `examples/gemma4_engram_lora.py`

### Key Features
- **Bitsandbytes Support**: Built-in `--load_in_4bit` and `--load_in_8bit` flags for consumer GPU compatibility.
- **Instruct Formatting**: Uses model-specific chat templates (e.g., `[INST]` for Mistral, `<start_of_turn>` for Gemma).
- **Combined Training**: Jointly optimizes LoRA adapters (for task style) and Engram layers (for factual memory).
- **Inference Demo**: Includes a post-training generation hook to verify adapter functionality.

### Running Examples
```bash
# Run Qwen 3.5 with 4-bit quantization and 300 steps
uv run python examples/qwen3_engram_lora.py --load_in_4bit --max_steps 300

# Run Ministral 3 (multimodal) with BF16 to avoid FP8 errors
uv run python examples/mistral3_engram_lora.py --load_in_bf16

# Run Gemma 4 with custom learning rate
uv run python examples/gemma4_engram_lora.py --lr 1e-4
```

---

## 11. 🛠 Prerequisites

The examples require additional libraries (`matplotlib`, `seaborn`, `pandas`, `peft`, `datasets`, `bitsandbytes`, `accelerate`) which are included in the project's dev dependencies.

Ensure your environment is synchronized:
```bash
uv sync --all-groups
```
