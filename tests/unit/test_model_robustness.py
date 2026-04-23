from typing import Any, cast

import pytest
import torch.nn as nn

from engram_peft.config import EngramConfig
from engram_peft.model import get_engram_model


def test_get_engram_model_invalid_mode(tiny_tokenizer: Any):
    class DummyModel(nn.Module):
        layers: nn.ModuleList
        config: Any

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])
            self.config = type(
                "Config",
                (),
                {
                    "model_type": "llama",
                    "hidden_size": 10,
                    "vocab_size": 10,
                    "pad_token_id": 0,
                },
            )()

    model = DummyModel()
    config = EngramConfig(
        original_vocab_size=10,
        hidden_size=10,
        layer_container_path="layers",
        tokenizer_name_or_path="mock/tiny",
        engram_vocab_size_per_ngram=[100, 100],
    )

    # Test valid modes
    get_engram_model(model, config, train_mode="engram_only", tokenizer=tiny_tokenizer)

    # Test invalid mode using cast to silence static error
    with pytest.raises(ValueError, match="Unsupported train_mode"):
        get_engram_model(
            model,
            config,
            train_mode=cast("Any", "invalid_mode"),
            tokenizer=tiny_tokenizer,
        )


def test_get_engram_model_wrap_peft_incompatibility(tiny_tokenizer: Any):
    class DummyModel(nn.Module):
        layers: nn.ModuleList
        config: Any

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])
            self.config = type(
                "Config",
                (),
                {
                    "model_type": "llama",
                    "hidden_size": 10,
                    "vocab_size": 10,
                    "pad_token_id": 0,
                },
            )()

    model = DummyModel()
    config = EngramConfig(
        original_vocab_size=10,
        hidden_size=10,
        layer_container_path="layers",
        tokenizer_name_or_path="mock/tiny",
        engram_vocab_size_per_ngram=[100, 100],
    )

    # wrap_peft=True is only compatible with preserve_trainable
    with pytest.raises(ValueError, match="compatible with"):
        get_engram_model(
            model,
            config,
            train_mode="engram_only",
            wrap_peft=True,
            tokenizer=tiny_tokenizer,
        )
