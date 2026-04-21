from unittest.mock import MagicMock

import pytest

from engram_peft.utils.config_utils import patch_config


def test_patch_config_text_sync():
    """Test that attributes are synced from text_config to top-level."""
    config = MagicMock()
    config.text_config = MagicMock()
    config.text_config.to_dict.return_value = {"hidden_size": 1024, "num_heads": 16}
    # Initially missing
    del config.hidden_size
    del config.num_heads

    patched = patch_config(config)
    assert patched.hidden_size == 1024
    assert patched.num_heads == 16


def test_patch_config_vocab_size_inference():
    """Test that vocab_size is inferred from tokenizer."""

    class DummyConfig:
        pass

    config = DummyConfig()

    tokenizer = MagicMock()
    tokenizer.__len__.return_value = 50000

    patched = patch_config(config, tokenizer=tokenizer)
    assert patched.vocab_size == 50000


def test_patch_config_class_property_injection():
    """Test that the vocab_size property is injected into the class."""

    class DummyConfig:
        vocab_size: int

        def __init__(self):
            self.text_config = MagicMock()
            self.text_config.vocab_size = 32000

    config = DummyConfig()
    assert not hasattr(DummyConfig, "vocab_size")

    patch_config(config)
    assert hasattr(DummyConfig, "vocab_size")
    assert config.vocab_size == 32000


if __name__ == "__main__":
    pytest.main([__file__])
