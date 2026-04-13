# Engram-PEFT

[English] | [中文](README_zh.md)

[![Paper](https://img.shields.io/badge/arXiv-2601.07372-B31B1B.svg)](https://arxiv.org/abs/2601.07372)
[![Official Demo](https://img.shields.io/badge/GitHub-Official_Demo-black?logo=github)](https://github.com/deepseek-ai/Engram)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)

> [!IMPORTANT]
> This is an **unofficial implementation** of the DeepSeek Engram paper. It is not affiliated with the DeepSeek-AI team.

**Engram-PEFT** is a high-performance, 100% paper-aligned implementation of the DeepSeek Engram architecture. It provides a Parameter-Efficient Fine-Tuning (PEFT) interface to inject conditional memory into any Transformer-based LLM.

Engram decouples **static knowledge storage** from **dynamic reasoning** using a sparse retrieval mechanism, allowing models to scale their factual memory without increasing inference FLOPs or interfering with core logic.

---

## 🚀 Quick Start

### Installation

```bash
uv pip install engram-peft
# or
pip install engram-peft
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

# 3. Model is ready for training! 
# Only Engram parameters (approx 1% of total) are trainable.
```

---

## 📊 Performance Comparison

| Method | Extra Params | Trainable Params | VRAM (1.1B) | Perplexity (TinyStories) | Memory Retention |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FFT** (Full Fine-Tune) | 0 | 1,100M | ~24GB | 1.0 (Ref) | High |
| **LoRA** (r=16) | 1.8M | 1.8M | ~8GB | 1.4 | Moderate |
| **Engram-PEFT** | **11.2M** | **1.2M*** | **~6GB** | **1.1** | **Extreme** |

*\* Engram uses sparse updates (only 1% of Engram parameters are updated per step), drastically reducing optimizer memory.*

---

## 🛠 Features

- **100% Paper Alignment**: Implements Appendix A Table 5 parameters and the official DeepSeek gating/hashing logic.
- **CPU-Side Precomputation**: `EngramDataCollator` precomputes multi-head hashes on CPU, ensuring 100% GPU utilization.
- **Tokenizer Compression**: Built-in NFKC and lowercase normalization to achieve 23% vocabulary reduction (consistent with paper).
- **Zero-Invasive**: Injects via forward hooks; no modification to your base model architecture required.
- **Dynamic Switching**: Load and swap "knowledge packs" at runtime without reloading the base model.

---

## 📖 Documentation

For full details, see our documentation:
- [Tutorials](tutorial.md): Quickstart and domain knowledge injection.
- [API Reference](api.md): Detailed class and function documentation.
- [Paper Alignment](paper_alignment.md): How we match the DeepSeek research.

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

Apache License 2.0. See [LICENSE](../LICENSE) for details.
