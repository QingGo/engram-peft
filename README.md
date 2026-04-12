# Engram-PEFT

Engram-PEFT: Efficient Parameter-Efficient Fine-Tuning with Engram.

## Introduction

`engram-peft` is a Python library designed to provide an efficient Parameter-Efficient Fine-Tuning (PEFT) method using Engram. It integrates seamlessly with Hugging Face's `transformers` and `peft` ecosystems.

## Installation

You can install `engram-peft` using `uv`:

```bash
uv pip install engram-peft
```

Or with `pip`:

```bash
pip install engram-peft
```

## Quick Start

```python
from engram_peft import EngramModel, EngramConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
config = EngramConfig()
engram_model = EngramModel(model, config)
```

## Project Structure

- `src/engram_peft/`: Core implementation.
- `tests/`: Unit tests.
- `examples/`: Usage examples.
- `docs/`: Documentation.

## License

Apache-2.0
