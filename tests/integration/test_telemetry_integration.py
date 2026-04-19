from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from engram_peft import EngramConfig, EngramModel, EngramTrainer


def test_telemetry_no_unfreeze_shock() -> None:
    """Smoke test to ensure telemetry doesn't crash even if some groups are missing."""

    class SimpleModel(EngramModel):
        def __init__(self, cfg: EngramConfig) -> None:
            # Skip real EngramModel.__init__ to avoid heavy initialization
            nn.Module.__init__(self)
            self.config = cfg
            self.param = nn.Parameter(torch.randn(10))

        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.tensor(0.0)

        def get_telemetry_stats(self) -> dict[str, float]:
            return {}

    config = EngramConfig(target_layers=[0], enable_telemetry=True, hidden_size=10)
    model = SimpleModel(config)

    # Bypass __init__ entirely to avoid heavy Trainer overhead.
    # We only need the instance to test the _collect_telemetry method.
    trainer = EngramTrainer.__new__(EngramTrainer)

    # Manually set up the minimal state required by _collect_telemetry
    trainer.model = model
    trainer.args = MagicMock()
    trainer.args.enable_telemetry = True
    trainer.args.logging_steps = 1
    trainer.args.max_grad_norm = 1.0
    trainer.state = MagicMock()
    trainer.state.global_step = 1
    trainer._initial_weights = {}

    # Initialize attributes that are normally set in EngramTrainer.__init__
    trainer._last_ce_loss = 0.0
    trainer._last_entropy_loss = 0.0

    # Mock parameter groups to avoid model traversal
    mock_groups = {"group1": [nn.Parameter(torch.randn(2, 2))]}

    with (
        patch("engram_peft.trainer.unwrap_model", return_value=model),
        patch(
            "engram_peft.trainer.get_trainable_param_groups", return_value=mock_groups
        ),
    ):
        # This should not raise KeyError or AttributeError
        stats = trainer._collect_telemetry()
        assert isinstance(stats, dict)
        assert "telemetry/group1/param_norm" in stats
