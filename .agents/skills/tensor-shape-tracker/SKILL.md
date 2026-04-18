---
description: "在修改模型 Forward 逻辑、Attention 层或 PEFT 内存注入逻辑时触发。规范张量维度推导与验证步骤。"
globs: "src/engram_peft/**/*.py"
---

# 📐 engram-peft 张量形状追踪规范 (Tensor Shape Tracker)

在 DeepSeek Engram 架构与 PEFT 机制的结合中，矩阵维度的 `Mismatch` 是最致命的错误。修改核心算子前，必须执行以下步骤：

## 1. 强制注释推导 (Mandatory Shape Docstrings)
在修改或新增涉及张量计算的函数时，必须在代码内部写明张量的变换过程。
```python
# [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
```

## 2. 防御性断言 (Defensive Assertions)
在进行 `view()`, `reshape()`, `transpose()` 或与 Engram 记忆库进行 `concat` 等高风险操作后，**强烈建议**插入临时或永久的 `assert` 语句进行维度自检。
```python
expected_shape = (batch_size, seq_length, self.hidden_size)
assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"
```

## 3. 行动准则：先推导后写代码
在向人类输出修改后的代码前，你必须在思考过程 (`Think Before You Code`) 中，明确写出维度的变化推演。确认逻辑闭环后，才能生成 Python 代码。

## 4. 常见形状错误案例
- 忘记处理 `batch_size=1` 的边界情况
- 混淆 `num_heads` 和 `head_dim` 的顺序
- 在转置后没有正确地连续化张量（`.contiguous()`）
- 没有考虑到 `past_key_values` 带来的序列长度变化