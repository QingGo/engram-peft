# Engram-PEFT

[English](README.md) | [中文]

> [!IMPORTANT]
> 这是一个 DeepSeek Engram 论文 ([arXiv:2601.07372](https://arxiv.org/abs/2601.07372)) 的 **非官方实现**。[DeepSeek-AI 官方示例在这里](https://github.com/deepseek-ai/Engram)。

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-MkDocs-blue.svg)](https://qinggo.github.io/engram-peft/)

**Engram-PEFT** 是一个高性能、100% 对齐论文的 DeepSeek Engram 架构实现。它提供了一个参数高效微调 (PEFT) 接口，可将条件记忆 (Conditional Memory) 注入任何基于 Transformer 的大语言模型。

Engram 使用稀疏检索机制将 **静态知识存储** 与 **动态推理** 解耦，允许模型在不增加推理 FLOPs 或干扰核心逻辑的情况下，扩展其实际记忆。

---

## 🚀 快速开始

### 安装

```bash
pip install engram-peft
```

如果您需要运行示例脚本或进行开发，请安装开发依赖：

```bash
# 使用 uv (推荐)
uv sync --all-groups

# 使用 pip
pip install -e ".[dev]"
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

# 3. 快速检查可训练参数
model.print_trainable_parameters()
# trainable params: 11,214,400 || all params: 1,111,214,400 || trainable%: 1.0092
```

---

## 📊 性能对比

| 方法 | 额外参数总量 | 梯度更新规模 | 显存占用 (1.1B) |
| :--- | :--- | :--- | :--- |
| **FFT** (全参数微调) | 0 | 1,100M | **~24GB (预估值)** |
| **LoRA** (r=16) | 1.8M | 1.8M | **~5.1GB** |
| **Engram-PEFT** | **11.2M** | **~1.2M*** | **~6.8GB** |

*\* Engram 采用稀疏检索机制；每步仅有极小比例（约 1%）的参数被激活并接收梯度更新。关于显存占用和扩展性的详细分析，请参阅 [显存分析报告 (英文)](docs/memory_analysis.md)。*

---

## 🛠 特性

- **100% 对齐论文**：实现了附录 A 表 5 的参数以及 DeepSeek 官方的门控/哈希逻辑。
- **CPU 端预计算**：`EngramDataCollator` 在 CPU 上预计算多头哈希，确保 100% 的 GPU 利用率。
- **分词器压缩 (Tokenizer Compression)**：内置 NFKC 和小写归一化，实现 23% 的词表缩减（与论文一致）。
- **零侵入性**：通过 forward hook 注入；无需修改基础模型架构源码。
- **类 PEFT API**：提供 `print_trainable_parameters()` 和 `save_pretrained()` 等熟悉的方法。
- **命名适配器 (Named Adapters)**：支持通过 `add_adapter()` 和 `set_adapter()` 同时管理多个知识包。
- **自动化训练**：`EngramTrainer` 在后台自动处理稀疏优化相关的复杂逻辑。

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
