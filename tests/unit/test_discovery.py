import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from engram_peft.config import EngramConfig
from engram_peft.model import get_engram_model


def test_discovery_with_manual_path() -> None:
    """Tests discovery when a manual path is provided."""
    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    with torch.device("meta"):
        base_model = AutoModel.from_config(config)

    engram_config = EngramConfig(
        layer_container_path="layers",
        target_layers=[0],
        original_vocab_size=100,
        enable_tokenizer_compression=False,
    )

    model = get_engram_model(base_model, engram_config)
    found_layers = model._find_transformer_layers()
    assert isinstance(found_layers, nn.ModuleList)
    assert len(found_layers) == config.num_hidden_layers


def test_discovery_error_on_invalid_path() -> None:
    """Tests that an error is raised when an invalid path is provided."""
    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    with torch.device("meta"):
        base_model = AutoModel.from_config(config)

    engram_config = EngramConfig(
        layer_container_path="embed_tokens",
        target_layers=[0],
        original_vocab_size=100,
        enable_tokenizer_compression=False,
    )

    with pytest.raises(ValueError, match="is not a nn.ModuleList"):
        get_engram_model(base_model, engram_config)


def test_discovery_error_on_missing_path() -> None:
    """Tests that an error is raised when a non-existent path is provided."""
    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    with torch.device("meta"):
        base_model = AutoModel.from_config(config)

    engram_config = EngramConfig(
        layer_container_path="non_existent.path",
        target_layers=[0],
        original_vocab_size=100,
        enable_tokenizer_compression=False,
    )

    with pytest.raises(ValueError, match="not found in model"):
        get_engram_model(base_model, engram_config)


def test_discovery_heuristics_fallback() -> None:
    """Tests that heuristics are used when no path is provided and model is unknown."""

    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])
            self.config = type(
                "Config",
                (),
                {"model_type": "unknown", "vocab_size": 100, "pad_token_id": 0},
            )()

    base_model = MockModel()
    engram_config = EngramConfig(
        target_layers=[0],
        enable_tokenizer_compression=False,
    )

    model = get_engram_model(base_model, engram_config)
    found_layers = model._find_transformer_layers()
    assert len(found_layers) == 2
