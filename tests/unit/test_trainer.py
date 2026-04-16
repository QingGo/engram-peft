from unittest.mock import patch

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments

from engram_peft import EngramConfig, EngramTrainer, get_engram_model
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
        self, input_ids: torch.Tensor | None = None, **kwargs: object
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
    assert isinstance(optimizer, MixedOptimizer)
    assert len(optimizer.optimizers) == 2
