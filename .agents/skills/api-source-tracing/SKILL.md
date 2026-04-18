---
description: "当遇到不确定的 transformers 或 peft 库 API，或者遇到了底层库抛出的未知异常时触发。提供防幻觉调查规范。"
globs: "*"
---

# 🕵️ 第三方 API 源码溯源规范 (Anti-Hallucination)

`transformers` 和 `peft` 的内部代码迭代极快（尤其是涉及到 `Cache` 机制和 `Attention` 变体）。在猜测某个冷门 API 的用法前，**必须执行底层抓取**。

## 1. 获取包绝对路径
使用以下命令定位虚拟环境中的包位置：
```bash
uv run python -c "import peft; print(peft.__path__[0])"
uv run python -c "import transformers; print(transformers.__path__[0])"
```

## 2. 优先使用 inspect 获取动态签名
在处理多行参数时，直接编写临时单行脚本提取签名，比使用 grep 更不易出错：
```bash
uv run python -c "import inspect, transformers; print(inspect.signature(transformers.GenerationMixin.generate))"
```

## 3. 版本差异处理
- 记录当前使用的 `transformers` 和 `peft` 版本号
- 如果发现API行为与文档不符，优先以本地源码为准
- 对于已废弃的API，必须查找替代方案并记录在 `.agent_memory.md` 中

## 4. 分析与决策
在你深入阅读了真实的 `.venv` 内部源码，并搞清楚了 `**kwargs` 到底包含了哪些隐藏参数后，才能在终端输出你的代码修改计划。严禁“盲人摸象”式的反复修改测试。