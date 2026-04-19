from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from engram_peft.config import EngramConfig
from engram_peft.model import EngramModel, get_engram_model


def test_discovery_with_manual_path() -> None:
    """Tests discovery when a manual path is provided."""

    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])
            self.config = type(
                "Config", (), {"model_type": "custom", "vocab_size": 100}
            )()

    base_model = MockModel()
    engram_config = EngramConfig(
        layer_container_path="layers",
        target_layers=[0],
        enable_tokenizer_compression=False,
    )

    with (
        patch("engram_peft.model.NgramHashMapping", return_value=MagicMock()),
        patch("engram_peft.model.CompressedTokenizer", return_value=MagicMock()),
    ):
        model = get_engram_model(base_model, engram_config)
    found_layers = model._find_transformer_layers()
    assert isinstance(found_layers, nn.ModuleList)
    assert len(found_layers) == 2


def test_discovery_error_on_invalid_path() -> None:
    """Tests that an error is raised when an invalid path is provided."""

    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.some_attr = nn.Linear(10, 10)
            self.config = type(
                "Config", (), {"model_type": "custom", "vocab_size": 100}
            )()

    base_model = MockModel()
    engram_config = EngramConfig(
        layer_container_path="some_attr",
        target_layers=[0],
        enable_tokenizer_compression=False,
    )

    with (
        patch("engram_peft.model.EngramModel.__init__", return_value=None),
        patch("engram_peft.model.EngramModel.load_engram", return_value=None),
        pytest.raises(ValueError, match="is not a nn.ModuleList"),
    ):
        get_engram_model(base_model, engram_config)


def test_discovery_error_on_missing_path() -> None:
    """Tests that an error is raised when a non-existent path is provided."""

    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type(
                "Config", (), {"model_type": "custom", "vocab_size": 100}
            )()

    base_model = MockModel()
    engram_config = EngramConfig(
        layer_container_path="non_existent.path",
        target_layers=[0],
        enable_tokenizer_compression=False,
    )

    with (
        patch("engram_peft.model.EngramModel.__init__", return_value=None),
        patch("engram_peft.model.EngramModel.load_engram", return_value=None),
        pytest.raises(ValueError, match="not found in model"),
    ):
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

    with (
        patch("engram_peft.model.NgramHashMapping", return_value=MagicMock()),
        patch("engram_peft.model.CompressedTokenizer", return_value=MagicMock()),
    ):
        model = get_engram_model(base_model, engram_config)
    found_layers = model._find_transformer_layers()
    assert len(found_layers) == 2
