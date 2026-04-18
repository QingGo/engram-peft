from typing import Any

import torch
import torch.nn as nn
from transformers import TrainingArguments

from engram_peft import EngramConfig, EngramTrainer


def test_telemetry_no_unfreeze_shock() -> None:
    """Smoke test to ensure telemetry doesn't crash even if some groups are missing."""

    class SimpleModel(nn.Module):
        def __init__(self, cfg: EngramConfig) -> None:
            super().__init__()
            self.config = cfg
            self.param = nn.Parameter(torch.randn(10))

        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.tensor(0.0)

    config = EngramConfig(target_layers=[0], enable_telemetry=True, hidden_size=10)
    model = SimpleModel(config)

    # Intentionally incomplete groups (missing engram_sparse etc)
    _groups: dict[str, list[nn.Parameter]] = {"backbone": [model.param]}

    args = TrainingArguments(output_dir="tmp_telemetry")
    trainer = EngramTrainer(model=model, args=args)
    # Mocking unwrap to avoid isinstance check issues in smoke test
    trainer._get_unwrapped_model = lambda: model  # type: ignore

    # This should not raise KeyError or AttributeError
    stats = trainer._collect_telemetry()
    assert isinstance(stats, dict)
