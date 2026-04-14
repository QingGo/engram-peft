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
# trainable params: 11,214,400 || all params: 1,111,214,400 || trainable%: 1.0092
```

---

## 📊 Performance Comparison

| Method | Params Added | Speed (s/step) | Training Loss | Eval Loss | VRAM (Total) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA** (r=16) | 2.88 M | 0.1777 s | **1.254** | 1.153 | 6.07 GiB |
| **Engram-PEFT** | **545.4 M** | **0.1643 s** | 1.311 | **1.141** | **6.97 GiB** |

> [!TIP]
> **Performance Insight**: Although LoRA achieves a slightly lower training loss, **Engram-PEFT yields a lower evaluation loss**, indicating superior generalization and knowledge capture from the stories dataset.

### Loss Curve Comparison
![Loss Curve Comparison](figures/loss_curve.png)

*\* Engram employs sparse lookup; only a tiny fraction of parameters (approx. 1%) are active and receive gradient updates per step. For a detailed breakdown of performance, computation, and memory, see our [Performance Analysis](docs/compare_engram_lora_analysis.md).*

---

## 🛠 Features

- **100% Paper Alignment**: Implements Appendix A Table 5 parameters and the official DeepSeek gating/hashing logic.
- **Vectorized CPU-Side Precomputation**: `EngramDataCollator` uses highly optimized NumPy broadcasting to precompute hashes on CPU. Supports multi-process loading (`num_workers > 0`) to ensure 100% GPU utilization.
- **Tokenizer Compression**: Built-in NFKC and lowercase normalization for 23% vocabulary reduction.
- **Zero-Invasive**: Injects via forward hooks; no modification to your base model architecture required.
- **Peft-like API**: Familiar methods like `print_trainable_parameters()` and `save_pretrained()`.
- **Named Adapters**: Industry-standard named adapter management (add/set/unload) for modular knowledge packs.
- **Weight Migration**: One-of-a-kind implementation supporting **Best-effort Cross-Tokenizer Migration** (best effort). Reuse weights between Llama, Qwen, and others via semantic offset alignment.
- **Structural Robustness**: Supports cross-layer mapping, dynamic bucket resizing, and N-gram subset loading.
- **Automated Training**: Native `EngramTrainer` with built-in sparse Adam support and automatic sync of optimizer hyperparameters.

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
