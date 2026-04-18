from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import torch
import torch.nn as nn
from pytest import approx
from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments

from engram_peft import EngramConfig, EngramTrainer, get_engram_model

if TYPE_CHECKING:
    from engram_peft.utils import MixedOptimizer


class MockConfig(PretrainedConfig):
    model_type = "mock"
    hidden_size: int = 16
    vocab_size: int = 100
    pad_token_id: int = 0


class MockBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(
        self, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ) -> tuple:
        return (self.linear(hidden_states),)


class MockHFModel(PreTrainedModel):
    config_class = MockConfig

    def __init__(self) -> None:
        super().__init__(MockConfig())
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockBlock()])

    def forward(
        self, input_ids: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids required")
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(batch_size, seq_len, 16)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def test_engram_trainer_uses_transformers_optimizer_for_non_engram_model(
    tmp_path: str,
) -> None:
    model = MockHFModel()
    trainer = EngramTrainer(
        model=model,
        args=TrainingArguments(output_dir=tmp_path, learning_rate=1e-3),
    )

    sentinel = object()
    with patch(
        "transformers.Trainer.create_optimizer", return_value=sentinel
    ) as mocked:
        optimizer = trainer.create_optimizer()

    assert optimizer is sentinel
    mocked.assert_called_once()


def test_engram_trainer_uses_engram_optimizer_for_engram_model(
    tmp_path: str,
) -> None:
    base_model = MockHFModel()
    config = EngramConfig(
        target_layers=[0],
        enable_tokenizer_compression=False,
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=16,
        compressed_vocab_size=129280,
    )
    model = get_engram_model(base_model, config, train_mode="engram_only")

    trainer = EngramTrainer(
        model=model,
        args=TrainingArguments(output_dir=tmp_path, learning_rate=1e-3),
        optimizer_kwargs={"backbone_learning_rate": 1e-4},
    )

    optimizer = trainer.create_optimizer()

    assert optimizer is trainer.optimizer
    assert optimizer is not None
    # Use cast to satisfy Mypy as trainer.optimizer is typed as Optimizer base class
    assert len(cast("MixedOptimizer", trainer.optimizer).optimizers) == 2


def test_engram_trainer_compute_loss_fair_comparison(tmp_path: str) -> None:
    """验证 compute_loss 在评估时不包含熵惩罚，在训练时包含。"""
    base_model = MockHFModel()
    config = EngramConfig(
        target_layers=[0],
        enable_tokenizer_compression=False,
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=16,
        compressed_vocab_size=100,
        entropy_loss_weight=0.1,  # 开启熵惩罚
    )
    model = get_engram_model(base_model, config, train_mode="full_finetune")

    trainer = EngramTrainer(
        model=model,
        args=TrainingArguments(output_dir=tmp_path),
    )

    # 模拟输入
    inputs = {"input_ids": torch.zeros((1, 5), dtype=torch.long)}
    mock_ce_loss = torch.tensor(2.0, requires_grad=True)

    # 手动为 Engram 层设置一个非零的熵，模拟前向传播后的状态
    for layer in model.engram_layers.values():
        layer.gating.gating_entropy = torch.tensor(0.5, requires_grad=True)

    with patch("transformers.Trainer.compute_loss", return_value=(mock_ce_loss, {})):
        # 1. 模拟评估模式 (training=False)
        model.eval()
        eval_loss = trainer.compute_loss(model, inputs)
        # 评估损耗应等于纯 CE 损耗 (2.0)
        assert torch.allclose(eval_loss, mock_ce_loss)
        assert trainer._last_ce_loss == 2.0
        assert trainer._last_entropy_loss > 0  # 即使不返回，内部也应计算了惩罚

        # 2. 模拟训练模式 (training=True)
        model.train()
        train_loss = trainer.compute_loss(model, inputs)
        # 训练损耗应大于纯 CE 损耗 (2.0 + penalty)
        assert train_loss.item() > mock_ce_loss.item()

        assert train_loss.item() == approx(
            trainer._last_ce_loss + trainer._last_entropy_loss
        )
