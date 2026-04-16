import torch
import torch.nn as nn

from engram_peft.config import EngramConfig
from engram_peft.discovery import ArchitectureResolver


class DummyModel(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=32000, pad_token_id=0):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "pad_token_id": pad_token_id,
                "model_type": "llama",
            },
        )
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(2)]
        )


def test_discovery_basic():
    """Test standard discovery from model config."""
    model = DummyModel(hidden_size=1024, vocab_size=50000, pad_token_id=1)
    config = EngramConfig()

    metadata = ArchitectureResolver.resolve(model, config=config)

    assert metadata.hidden_size == 1024
    assert metadata.original_vocab_size == 50000
    assert metadata.pad_token_id == 1
    assert metadata.layer_container_path in ["model.layers", "layers"]


def test_discovery_override():
    """Test that explicit config overrides discovery."""
    model = DummyModel(hidden_size=1024, vocab_size=50000, pad_token_id=1)
    config = EngramConfig(hidden_size=2048, pad_id=5)

    metadata = ArchitectureResolver.resolve(model, config=config)

    # Discovery should respect the None-check logic
    assert metadata.hidden_size == 2048
    assert metadata.pad_token_id == 5
    # vocab_size was not overridden in EngramConfig, so it should be detected
    assert metadata.original_vocab_size == 50000


def test_discovery_fallback():
    """Test fallback to defaults when detection fails."""

    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()

    model = EmptyModel()
    config = EngramConfig()

    # Should fallback for hidden_size and pad_id, but vocab_size will raise ValueError
    try:
        ArchitectureResolver.resolve(model, config=config)
    except ValueError as e:
        assert "Could not detect original vocab_size" in str(e)


def test_discovery_with_tokenizer():
    """Test discovery from tokenizer."""
    model = DummyModel()
    tokenizer = type("Tokenizer", (), {"pad_token_id": 9, "__len__": lambda s: 42000})()
    config = EngramConfig()

    metadata = ArchitectureResolver.resolve(model, tokenizer=tokenizer, config=config)

    assert metadata.pad_token_id == 9
    assert metadata.original_vocab_size == 42000
