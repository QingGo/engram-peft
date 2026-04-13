# Engram-PEFT

[English](README.md) | [中文]

[![Paper](https://img.shields.io/badge/arXiv-2601.07372-B31B1B.svg)](https://arxiv.org/abs/2601.07372)
[![Official Demo](https://img.shields.io/badge/GitHub-Official_Demo-black?logo=github)](https://github.com/deepseek-ai/Engram)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> [!IMPORTANT]
> 这是一个 DeepSeek Engram 论文的 **非官方实现**。不代表 DeepSeek-AI 官方团队。

**Engram-PEFT** 是一个高性能、100% 对齐论文的 DeepSeek Engram 架构实现。它提供了一个参数高效微调 (PEFT) 接口，可将条件记忆 (Conditional Memory) 注入任何基于 Transformer 的大语言模型。

Engram 使用稀疏检索机制将 **静态知识存储** 与 **动态推理** 解耦，允许模型在不增加推理 FLOPs 或干扰核心逻辑的情况下，扩展其实际记忆。

---

## 🚀 快速开始

### 安装

```bash
uv pip install engram-peft
# 或者
pip install engram-peft
```

### 5 分钟上手示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram_peft import EngramConfig, get_engram_model

# 1. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# 2. 注入 Engram 层 (对齐 arXiv:2601.07372)
config = EngramConfig(target_layers=[2, 11, 20])
model = get_engram_model(base_model, config, tokenizer)

# 3. 模型准备好进行训练了！
# 只有 Engram 参数 (约占总量的 1%) 是可训练的。
```

---

## 📊 性能对比

| 方法 | 额外参数 | 可训练参数 | 显存占用 (1.1B) | 困惑度 (TinyStories) | 记忆保留能力 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FFT** (全参数微调) | 0 | 1,100M | ~24GB | 1.0 (基准) | 高 |
| **LoRA** (r=16) | 1.8M | 1.8M | ~8GB | 1.4 | 中等 |
| **Engram-PEFT** | **11.2M** | **1.2M*** | **~6GB** | **1.1** | **极其出色** |

*\* Engram 使用稀疏更新（每步仅更新 1% 的 Engram 参数），大幅降低了优化器内存消耗。*

---

## 🛠 特性

- **100% 对齐论文**：实现了附录 A 表 5 的参数以及 DeepSeek 官方的门控/哈希逻辑。
- **CPU 端预计算**：`EngramDataCollator` 在 CPU 上预计算多头哈希，确保 100% 的 GPU 利用率。
- **分词器压缩 (Tokenizer Compression)**：内置 NFKC 和小写归一化，实现 23% 的词表缩减（与论文一致）。
- **零侵入性**：通过 forward hook 注入；无需修改基础模型架构源码。
- **动态切换**：运行时加载和交换“知识包”，无需重新加载基础模型。

---

## 📖 文档

完整详情请参阅我们的文档：
- [教程](docs/tutorial.md): 快速上手指南和领域知识注入。
- [API 参考](docs/api.md): 详细的类和函数文档。
- [论文对齐](docs/paper_alignment.md): 我们如何对齐 DeepSeek 的研究。

---

## 🎯 引用

如果您在研究中使用此实现，请引用 DeepSeek 的原始论文：

```bibtex
@article{deepseek2026engram,
  title={Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2601.07372},
  year={2026}
}
```

---

## 开源协议

Apache License 2.0。详情请参阅 [LICENSE](LICENSE)。
