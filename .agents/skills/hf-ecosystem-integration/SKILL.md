---
description: "当处理模型与 Hugging Face Trainer 结合训练，或使用 .generate() 进行文本生成时触发。"
globs: "src/engram_peft/**/*.py, examples/**/*.py"
---

# 🚀 Hugging Face 生态接入规范

为了让非官方的 `engram-peft` 模型能无缝对接 HF 的主流工具，修改接口层时必须注意以下红线：

## 1. 适配 `generate()` 文本生成
Engram 是一种序列依赖的架构，在自回归生成时，缓存机制至关重要。
- 必须确保模型正确实现了 `prepare_inputs_for_generation` 方法。
- 除了标准的 `input_ids` 和 `past_key_values`，如果 Engram 需要传递特定的 memory index 或 mask，必须在 `kwargs` 中正确透传。
- 注意 `past_key_values` 格式（通常是 `DynamicCache` 或 Tuple），不要破坏其原生结构。
- 必须支持 `use_cache=True` 和 `use_cache=False` 两种模式。

## 2. 适配 HF `Trainer`
`Trainer` 对模型的前向传播返回值有严格要求：
- 训练模式下，`forward()` 的返回值必须包含 `loss` 字段，且通常建议返回 `CausalLMOutputWithPast`。
- 必须处理好 `labels` 参数的偏移 (shift) 逻辑（即 `inputs` 和 `labels` 错位计算 CrossEntropy）。如果你的代码直接覆盖了原生的 `forward`，切记补齐损失计算逻辑。
- 必须支持 `gradient_checkpointing` 以节省显存。

## 3. 常见坑与注意事项
- `Trainer` 会自动将模型移动到正确的设备，不要在 `forward` 中手动调用 `.to(device)`
- 确保所有自定义张量都正确跟随模型的设备和 dtype
- 对于稀疏张量，必须处理好 `Trainer` 的梯度累积逻辑