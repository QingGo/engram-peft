import os
import shutil
import tempfile
from typing import Any, Optional, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from engram_peft.config import EngramConfig
from engram_peft.layer import EngramLayer
from engram_peft.model import EngramModel, get_engram_model


class MockTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> tuple:
        # Simulate HuggingFace transformer block output (tuple)
        return (self.linear(hidden_states),)


class MockModelConfig(PretrainedConfig):
    model_type = "mock"
    hidden_size: int = 32


class MockPreTrainedModel(PreTrainedModel):
    config_class = MockModelConfig

    def __init__(self, config: MockModelConfig, hidden_size: int = 32) -> None:
        super().__init__(config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockTransformerBlock(hidden_size) for _ in range(5)]
        )

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, **kwargs: Any
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
        max_ngram_size=2,
        n_head_per_ngram=2,
        embedding_dim=32,
        hidden_dim=32,
        hc_mult=1,
        target_layers=[1, 3],
        engram_vocab_size_per_ngram=[100],
        enable_tokenizer_compression=False,
    )
    base_model = MockPreTrainedModel(MockModelConfig(), hidden_size=32)
    # Give some parameters true by default
    for param in base_model.parameters():
        param.requires_grad_(True)
    return config, base_model


def test_get_engram_model_freeze() -> None:
    """
    测试用例 1：验证 get_engram_model 能正确冻结骨干参数
    """
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    # 验证 base_model 参数全部被冻结
    for name, param in base_model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen"

    # 验证 engram_layers 参数全部可训练
    engram_param_count = 0
    for name, param in engram_model.engram_layers.named_parameters():
        assert param.requires_grad, f"Engram parameter {name} should be trainable"
        engram_param_count += 1

    assert engram_param_count > 0, "Should have engram parameters"


def test_layer_injection_instances() -> None:
    """
    测试用例 2：验证每个目标层都有独立的 EngramLayer 实例
    """
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    assert len(engram_model.engram_layers) == 2
    assert "1" in engram_model.engram_layers
    assert "3" in engram_model.engram_layers

    # Ensure hook handles are active (target layers + global model pre-hook)
    assert len(engram_model._hook_handles) == 3

    # Run a forward pass to ensure no crash from hooks
    input_ids = torch.randint(0, 100, (2, 5))
    output = engram_model(input_ids=input_ids)
    assert output.shape == (2, 5, 32)


def test_save_load_consistency() -> None:
    """
    测试用例 3：验证 save/load 功能正常
    """
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    # Modify weight slightly to test loading
    with torch.no_grad():
        layer1 = cast(EngramLayer, engram_model.engram_layers["1"])
        layer1.multi_head_embedding.embedding.weight.add_(1.0)

    org_weight_sum = layer1.multi_head_embedding.embedding.weight.sum().item()

    temp_dir = tempfile.mkdtemp()
    try:
        engram_model.save_pretrained(temp_dir)

        # Check raw files exist
        assert os.path.exists(os.path.join(temp_dir, "config.json"))
        assert os.path.exists(os.path.join(temp_dir, "engram_weights.pt"))

        # Load into new model
        base_model_new = MockPreTrainedModel(MockModelConfig(), hidden_size=32)
        loaded_model = EngramModel.from_pretrained(base_model_new, temp_dir)

        loaded_layer1 = cast(EngramLayer, loaded_model.engram_layers["1"])
        loaded_weight_sum = (
            loaded_layer1.multi_head_embedding.embedding.weight.sum().item()
        )
        assert abs(org_weight_sum - loaded_weight_sum) < 1e-5
    finally:
        shutil.rmtree(temp_dir)


def test_dynamic_load_unload() -> None:
    """
    测试用例 4：验证动态 load/unload_engram 功能
    """
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    assert len(engram_model._hook_handles) == 3
    assert engram_model._engram_enabled is True

    # Unload: should remove hooks and disable
    engram_model.unload_engram()
    assert len(engram_model._hook_handles) == 0
    assert engram_model._engram_enabled is False

    # Check forward pass without engram
    input_ids = torch.randint(0, 100, (1, 4))
    _ = engram_model(input_ids=input_ids)
    assert engram_model._current_hash_indices is None

    # Load again
    engram_model.load_engram()
    assert len(engram_model._hook_handles) == 3
    assert engram_model._engram_enabled is True


def test_transformers_compatibility() -> None:
    """
    测试用例 5：验证与 transformers Trainer 兼容 (代理 forward 和 generate)
    """
    config, base_model = create_mock_setup()
    engram_model = get_engram_model(base_model, config, tokenizer=None)

    # test generate delegation
    gen_out = engram_model.generate()
    assert gen_out == "mock_generation"

    # In HF Trainer, the model must be a subclass of nn.Module
    assert isinstance(engram_model, nn.Module)

    # ensure config is present at top level often needed by Trainer
    assert hasattr(engram_model, "config")

    # Input pass just works
    input_ids = torch.tensor([[1, 2, 3]])
    out = engram_model(input_ids=input_ids)
    assert out is not None
