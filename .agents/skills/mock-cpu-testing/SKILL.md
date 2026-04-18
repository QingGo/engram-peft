---
description: "在编写、修改或运行单元测试时触发。提供无 GPU 环境下安全测试大模型的最佳实践与 Mock 规范。"
globs: "tests/**/*.py"
---

# 🛡️ engram-peft 无 GPU 测试规范 (Mock CPU Testing)

当前开发环境**没有 GPU**，且系统内存有限。在编写和运行测试时，严禁初始化真实的、全参数的大规模预训练模型。请严格遵循以下规约：

## 1. 拦截重型计算 (Mocking)
如果测试核心关注点是“状态管理”、“缓存机制”或“路由逻辑”，而非真实的注意力矩阵乘法，**必须**使用 `unittest.mock.patch` 拦截底层的 `forward` 方法。

## 2. 构造极小模型配置 (Dummy Config)
如果必须进行真实的前向传播以验证 PyTorch 计算图连通性，**必须**使用自定义的极简配置来初始化模型。
```python
from transformers import PretrainedConfig, AutoModelForCausalLM

# 示例：极简模型配置，确保在 CPU 上能 < 1s 运行完毕
dummy_config = PretrainedConfig(
    hidden_size=16,
    num_attention_heads=2,
    num_hidden_layers=2,
    intermediate_size=64,
    vocab_size=1000
)
# 注意：务必随机初始化权重，不要去拉取真实权重的 safetensors
model = AutoModelForCausalLM.from_config(dummy_config)
```

## 3. 使用极小张量 (Dummy Tensors)
验证形变逻辑时，严禁使用大型张量。
```python
import torch

# 推荐使用的测试极小张量尺寸
batch_size, seq_len, hidden_size = 1, 4, 16
dummy_input = torch.randn(batch_size, seq_len, hidden_size)
```

## 4. 覆盖率与回归验证
- 运行测试时必须附带覆盖率检查：
  ```bash
  uv run pytest tests/unit --cov=src/engram_peft --cov-report=term-missing
  ```
- 所有新增代码必须被测试覆盖
- 修复Bug前必须先写一个失败的测试用例复现问题
- 修复后必须运行完整测试套件确保无回归