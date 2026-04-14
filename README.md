# Engram-PEFT

[English] | [中文](README_zh.md)

> [!IMPORTANT]
> This is an **unofficial implementation** of the DeepSeek Engram paper ([arXiv:2601.07372](https://arxiv.org/abs/2601.07372)). [DeepSeek-AI official demo is here](https://github.com/deepseek-ai/Engram).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-MkDocs-blue.svg)](https://qinggo.github.io/engram-peft/)

**Engram-PEFT** is a high-performance, 100% paper-aligned implementation of the DeepSeek Engram architecture. It provides a Parameter-Efficient Fine-Tuning (PEFT) interface to inject conditional memory into any Transformer-based LLM.

Engram decouples **static knowledge storage** from **dynamic reasoning** using a sparse retrieval mechanism, allowing models to scale their factual memory without increasing inference FLOPs or interfering with core logic.

---

## 🚀 Quick Start

### Installation

```bash
pip install engram-peft
```

To run examples or contribute to development, install the project with development dependencies:

```bash
# Using uv (recommended)
uv sync --all-groups

# Using pip
pip install -e ".[dev]"
```

### 5-Minute Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram_peft import EngramConfig, get_engram_model

# 1. Load base model
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# 2. Inject Engram layers (aligned with arXiv:2601.07372)
config = EngramConfig(target_layers=[2, 11, 20])
model = get_engram_model(base_model, config, tokenizer)

# 3. Quick check on trainable parameters
model.print_trainable_parameters()
# trainable params: 86,938,368 || all params: 1,186,986,752 || trainable%: 7.3243
```

---

## 📊 Performance Comparison

| Method | Params Added | Speed (s/step) | Training Loss | Eval Loss | Peak Memory (JSON) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA** (r=16) | ~2.25 M | **0.2738 s** | 1.231 | 0.9890 | 8.07 GB |
| **Engram-PEFT** | **545.4 M** | 0.2961 s | 1.263 | 1.0165 | 9.38 GB |
| **LoRA+Engram** | ~547.7 M | 0.3360 s | **1.214** | **0.9656** | 10.33 GB |

> [!TIP]
> **Performance Insight**: In our latest benchmark (Test 8 & 9, TinyLlama-1.1B, 3000 steps), **LoRA+Engram** achieved the best convergence (lowest eval loss), outperforming standalone LoRA by ~2.3%. Engram-PEFT provides **240x more parameter capacity** (545M) for knowledge storage with minimal latency penalty. Use LoRA+Engram to leverage both structural adaptation and high-capacity sparse memory.

### Loss Curve Comparison
![Loss Curve Comparison](figures/loss_curve.png)

*\* Engram employs sparse lookup; only a tiny fraction of parameters (approx. 1%) are active and receive gradient updates per step. For a detailed breakdown of performance, computation, and memory, see our [Performance Analysis](docs/compare_engram_lora_analysis.md).*

---

## 🛠 Features

- **100% Paper Alignment**: Implements Appendix A Table 5 parameters and the official DeepSeek gating/hashing logic.
- **CPU Prefetching & Precomputation**: `EngramDataCollator` pre-calculates multi-head hash indices on the CPU. By using `num_workers > 0`, these indices are prefetched in parallel with training, ensuring zero hashing overhead on the GPU.
- **Tokenizer Compression**: Built-in NFKC and lowercase normalization for 23% vocabulary reduction.
- **Cross-Model Weight Migration**: A unique feature (see `weight_transfer.py`) that allows migrating Engram weights between different models (e.g., Llama to Qwen) using character-level alignment on a corpus—effectively "recycling" learned knowledge.
- **Zero-Invasive**: Injects via forward hooks; no modification to your base model architecture required.
- **Peft-like API**: Familiar methods like `print_trainable_parameters()` and `save_pretrained()`.
- **Combined Training (LoRA+Engram)**: Support for stacking adapters. Injects LoRA for structural fine-tuning and Engram for sparse knowledge retrieval in a single model.
- **Named Adapters**: Industry-standard named adapter management (add/set/unload) for modular knowledge packs.
- **Automated Training**: Native `EngramTrainer` with built-in sparse Adam support and automatic sync of optimizer hyperparameters.
- **Flexible Layer Discovery**: Recursive logic to find transformer layers regardless of PEFT wrapper nesting.

---

## 📖 Documentation

For full details, see our documentation:
- [Tutorials](docs/tutorial.md): Quickstart and domain knowledge injection.
- [API Reference](docs/api.md): Detailed class and function documentation.
- [Paper Alignment](docs/paper_alignment.md): How we match the DeepSeek research.

---

## 🎯 Citation

If you use this implementation in your research, please cite the original DeepSeek paper:

```bibtex
@article{deepseek2026engram,
  title={Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2601.07372},
  year={2026}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
