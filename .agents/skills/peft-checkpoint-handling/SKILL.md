---
description: "在编写或修复模型权重的保存 (save_pretrained) 与加载 (from_pretrained) 逻辑时触发。"
globs: "src/engram_peft/saving.py, src/engram_peft/model.py"
---

# 💾 PEFT & Engram 联合 Checkpoint 处理规范

原生的 `peft` 库在调用 `save_pretrained` 时，默认只会保存 Adapter 的权重（如 LoRA 的 A 和 B 矩阵）。但在 Engram 架构中，外挂的检索记忆库（Memory Bank/FAISS Index）也是状态的一部分，绝不能丢失。

## 1. 保存逻辑 (Save)
当你重写或调用保存逻辑时，必须确保：
1. 调用底层的 PEFT `save_pretrained` 保存 adapter 权重。
2. 必须将 Engram 的内部状态（如已存入的 Key-Value 记忆、聚类中心等）单独序列化，保存为 `engram_state.pt` 或 `engram_index.faiss`，并与 adapter 放在同一目录下。
3. 必须在 `adapter_config.json` 中添加 `engram_version` 字段，标识状态文件的格式版本。

## 2. 加载逻辑 (Load)
加载时必须是双轨制：
```python
# 1. 伪代码示例：先加载 PEFT adapter
model = PeftModel.from_pretrained(base_model, model_id)

# 2. 伪代码示例：再注入 Engram 记忆状态
engram_state_path = os.path.join(model_id, "engram_state.pt")
if os.path.exists(engram_state_path):
    model.load_engram_memory(torch.load(engram_state_path))
```

## 3. 验证原则
修复此类问题后，必须编写一个对应测试：初始化模型 -> 写入一点记忆 -> 保存到临时文件夹 (`tempfile.TemporaryDirectory`) -> 重新加载 -> 验证前向传播结果是否与保存前一致。

## 4. 版本兼容性与破坏性变更
- 所有状态文件格式变更必须递增 `engram_version`
- 必须提供向后兼容的加载逻辑，支持至少前两个版本的状态文件
- 如果必须引入破坏性变更，必须在加载时给出明确的错误提示和迁移指南
- 任何改变Checkpoint格式的修改都必须经过人类确认