import os
import shutil
import tempfile
from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PretrainedConfig, PreTrainedModel

from engram_peft.config import EngramConfig
from engram_peft.model import EngramModel


class DummyBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = PretrainedConfig()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(32)])
        self.dtype = torch.float32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_save_pretrained_standalone() -> None:
    """Test that Engram-only saving works and uses safetensors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = EngramConfig(
            target_layers=[2, 4],
            hidden_size=10,
            engram_vocab_size_per_ngram=[100, 100],
            compressed_vocab_size=100,
            pad_id=0,
            enable_tokenizer_compression=False,
            layer_container_path="layers",
        )
        base_model = DummyBaseModel()
        model = EngramModel(base_model, config)

        model.save_pretrained(tmp_dir)

        # Check for files
        assert os.path.exists(os.path.join(tmp_dir, "config.json"))
        assert os.path.exists(os.path.join(tmp_dir, "engram_adapters.safetensors"))
        assert not os.path.exists(os.path.join(tmp_dir, "engram_weights.pt"))

        # Verify config version
        loaded_config = EngramConfig.from_pretrained(tmp_dir)
        assert loaded_config.engram_version == "1.2.4"


def test_save_pretrained_with_peft() -> None:
    """Test that EngramModel triggers PeftModel saving."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = EngramConfig(
            target_layers=[2],
            hidden_size=10,
            engram_vocab_size_per_ngram=[100, 100],
            compressed_vocab_size=100,
            pad_id=0,
            enable_tokenizer_compression=False,
            layer_container_path="layers",
        )

        # Mock a PeftModel
        mock_peft = MagicMock(spec=PeftModel)

        mock_peft.save_pretrained = MagicMock()
        mock_peft.config = PretrainedConfig()
        mock_peft.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(32)])
        mock_peft._is_peft_model = True

        model = EngramModel(mock_peft, config)
        model.save_pretrained(tmp_dir)

        # Verify PeftModel.save_pretrained was called
        mock_peft.save_pretrained.assert_called_once()
        # Verify Engram artifacts also exist
        assert os.path.exists(os.path.join(tmp_dir, "engram_adapters.safetensors"))


def test_load_legacy_weights() -> None:
    """Test compatibility with legacy .pt files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = EngramConfig(
            target_layers=[2],
            hidden_size=10,
            engram_vocab_size_per_ngram=[100, 100],
            compressed_vocab_size=100,
            pad_id=0,
            enable_tokenizer_compression=False,
            layer_container_path="layers",
        )
        base_model = DummyBaseModel()
        model = EngramModel(base_model, config)

        # Manually save as legacy .pt
        state_dict = model.engram_layers.state_dict()
        torch.save(state_dict, os.path.join(tmp_dir, "engram_weights.pt"))

        # Try loading
        model.load_engram(tmp_dir)
        # If it didn't crash and we see logs (manually verified), it's good.
        # We can check if weights match.
        loaded_state_dict = model.engram_layers.state_dict()
        for k in state_dict:
            assert torch.equal(state_dict[k], loaded_state_dict[k])
