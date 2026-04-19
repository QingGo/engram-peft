from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock

import torch
import torch.nn as nn


@dataclass
class DummyConfig:
    model_type: str = "custom"
    hidden_size: int = 16
    vocab_size: int = 100
    pad_token_id: int = 0
    num_hidden_layers: int = 2
    enable_telemetry: bool = True
    clip_grad_per_group: bool = False
    backbone_freeze_steps: int = 0
    entropy_loss_weight: float = 0.0
    target_layers: list[int] = field(default_factory=lambda: [0])


class DummyModel(nn.Module):
    def __init__(self, config: Any | None = None) -> None:
        super().__init__()
        self.config = config or DummyConfig()
        # Mocking a transformer layer structure
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.Linear(
                            self.config.hidden_size, self.config.hidden_size
                        )
                    }
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        self.base_model = self.model  # For compatibility with EngramModel checks

    def forward(
        self, input_ids: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            return torch.randn(
                batch_size, seq_len, self.config.hidden_size, requires_grad=True
            )
        return torch.tensor(0.0, requires_grad=True)

    def get_telemetry_stats(self) -> dict[str, float]:
        return {}


@dataclass
class DummyTrainingArgs:
    output_dir: str = "tmp"
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    device: torch.device = torch.device("cpu")
    n_gpu: int = 0
    local_rank: int = -1
    world_size: int = 1

    def get_warmup_steps(self, num_training_steps: int) -> int:
        return self.warmup_steps
