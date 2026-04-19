import os
import shutil
import tempfile
from typing import Any, cast
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from engram_peft.config import EngramConfig
from engram_peft.model import EngramModel, get_engram_model
from engram_peft.saving import ADAPTER_SAFE_NAME


class MockTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> tuple:
        return (self.linear(hidden_states),)


class MockModel(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = MagicMock()
        self.config.model_type = "mock"
        self.config.hidden_size = hidden_size
        self.config.vocab_size = 100
        self.config.pad_token_id = 0

        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockTransformerBlock(hidden_size) for _ in range(5)]
        )
        self.base_model = self.model

    def forward(
        self, input_ids: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids required")
        batch_size, seq_len = input_ids.shape
        hs = torch.randn(batch_size, seq_len, 32, device=input_ids.device)
        for layer in self.model.layers:
            hs = layer(hs)[0]
        return hs

    def generate(self, *args: Any, **kwargs: Any) -> str:
        return "mock_generation"


def create_mock_setup() -> tuple:
    config = EngramConfig(
        ngram_sizes=[2],
        n_head_per_ngram=2,
        embedding_dim=32,
        hidden_dim=32,
        hc_mult=1,
        target_layers=[1, 3],
        engram_vocab_size_per_ngram=[100],
        enable_tokenizer_compression=False,
        hidden_size=32,
    )
    base_model = MockModel(hidden_size=32)
    for param in base_model.parameters():
        param.requires_grad_(True)
    return config, base_model


def test_get_engram_model_freeze() -> None:
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    for name, param in base_model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen"

    engram_param_count = 0
    for name, param in engram_model.engram_layers.named_parameters():
        assert param.requires_grad, f"Engram parameter {name} should be trainable"
        engram_param_count += 1

    assert engram_param_count > 0


def test_layer_injection_instances() -> None:
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    assert len(engram_model.engram_layers) == 2
    assert "1" in engram_model.engram_layers
    assert "3" in engram_model.engram_layers
    assert len(engram_model._hook_handles) == 3

    input_ids = torch.randint(0, 100, (2, 5))
    output = engram_model(input_ids=input_ids)
    assert output.shape == (2, 5, 32)


def test_save_load_consistency() -> None:
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    with torch.no_grad():
        layer1 = cast("Any", engram_model.engram_layers["1"])
        layer1.multi_head_embedding.embedding.weight.add_(1.0)

    org_weight_sum = layer1.multi_head_embedding.embedding.weight.sum().item()

    temp_dir = tempfile.mkdtemp()
    try:
        engram_model.save_pretrained(temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "config.json"))
        assert os.path.exists(os.path.join(temp_dir, ADAPTER_SAFE_NAME))

        base_model_new = MockModel(hidden_size=32)
        loaded_model = EngramModel.from_pretrained(base_model_new, temp_dir)

        loaded_layer1 = cast("Any", loaded_model.engram_layers["1"])
        loaded_weight_sum = (
            loaded_layer1.multi_head_embedding.embedding.weight.sum().item()
        )
        assert abs(org_weight_sum - loaded_weight_sum) < 1e-5
    finally:
        shutil.rmtree(temp_dir)


def test_dynamic_load_unload() -> None:
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    engram_model.unload_engram()
    assert len(engram_model._hook_handles) == 0
    assert engram_model._engram_enabled is False

    engram_model.load_engram()
    assert len(engram_model._hook_handles) == 3
    assert engram_model._engram_enabled is True


def test_incremental_generation_hook() -> None:
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)
    engram_model.eval()

    # 1. First step: full prompt (length 5)
    input_ids = torch.randint(0, 100, (1, 5))
    engram_model(input_ids=input_ids)

    # 2. Second step: incremental token (length 1)
    # The hooks should NOT crash and should successfully slice the hash indices
    next_token_id = torch.randint(0, 100, (1, 1))
    output = engram_model(input_ids=next_token_id)

    assert output.shape == (1, 1, 32)
    # Verify buffer updated (5 tokens in step1 + 1 token in step2 = 6)
    assert engram_model._inference_token_buffer is not None
    assert engram_model._inference_token_buffer.size(1) == 6
