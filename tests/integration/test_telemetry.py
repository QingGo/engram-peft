from typing import Any, cast, override
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import TrainingArguments

from engram_peft import EngramConfig, EngramLayer, EngramModel, EngramTrainer


def test_model_telemetry_stats() -> None:
    # 1. Setup mock config and model
    config = EngramConfig(
        target_layers=[0],
        enable_telemetry=True,
        hidden_size=16,
        enable_tokenizer_compression=False,
        compressed_vocab_size=32000,
        pad_id=0,
    )

    # Mock EngramModel instance without calling __init__
    model = MagicMock(spec=EngramModel)
    model.config = config

    # 2. Setup mock layers that satisfy isinstance(layer, EngramLayer)
    layer0 = MagicMock(spec=EngramLayer)
    layer0.gating = MagicMock()
    layer0.gating.last_gate = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 4, 1, 1)

    layer1 = MagicMock(spec=EngramLayer)
    layer1.gating = MagicMock()
    layer1.gating.last_gate = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 4, 1, 1)

    model.engram_layers = nn.ModuleDict({"0": layer0, "1": layer1})

    # Use the real get_telemetry_stats implementation
    stats = EngramModel.get_telemetry_stats(model)

    assert "gating/mean" in stats
    assert abs(stats["gating/mean"] - 0.5) < 1e-5
    assert abs(stats["gating/inactive_rate"] - 0.5) < 1e-5
    assert stats["gating/max"] == 1.0
    assert stats["gating/min"] == 0.0


def test_trainer_telemetry_collection(tmp_path: Any) -> None:
    config = EngramConfig(target_layers=[0], enable_telemetry=True, hidden_size=16)

    # Inherit from EngramModel to pass isinstance checks reliably
    class MockModel(EngramModel):
        config: EngramConfig
        base_model: nn.Module
        backbone_param: nn.Parameter
        dense_param: nn.Parameter
        sparse_param: nn.Parameter

        def __init__(self, config_obj: EngramConfig) -> None:
            # Skip real EngramModel.__init__
            nn.Module.__init__(self)
            self.config = config_obj
            self.base_model = nn.Module()  # just to exist
            self.backbone_param = nn.Parameter(torch.ones(10))
            self.dense_param = nn.Parameter(torch.zeros(10))  # zero_rate should be 1.0
            self.sparse_param = nn.Parameter(torch.ones(10) * 0.5)

        @override
        def get_telemetry_stats(self) -> dict[str, float]:
            return {"mock_gate": 123.0}

    model = MockModel(config)

    # Mock gradients
    model.backbone_param.grad = torch.ones(10) * 0.1
    model.dense_param.grad = torch.zeros(10)  # grad_zero_rate should be 1.0

    args = TrainingArguments(output_dir=str(tmp_path))

    # Mock unwrap and groups
    groups = {
        "backbone": [model.backbone_param],
        "engram_dense": [model.dense_param],
        "engram_sparse": [model.sparse_param],
    }

    def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self.model = kwargs.get("model") or (args[0] if len(args) > 0 else None)
        self.args = kwargs.get("args") or (args[1] if len(args) > 1 else None)
        self._initial_weights = {}

    with (
        patch("engram_peft.trainer.unwrap_model", return_value=model),
        patch("engram_peft.trainer.get_trainable_param_groups", return_value=groups),
        patch("transformers.Trainer.__init__", autospec=True, side_effect=mock_init),
    ):
        trainer = EngramTrainer(model=model, args=args)
        # Initialize remaining required attributes
        trainer.state = MagicMock()
        trainer.state.global_step = 1  # Set step
        trainer.callback_handler = MagicMock()

        # Manually trigger capture since we skipped init
        trainer._initial_weights = {}
        trainer._capture_initial_weights()

        # Bypassing the skip-at-step-0 guard
        trainer.state.global_step = 1
        trainer.args.logging_steps = 1

        # Capture initial weights happens in __init__

        # Change a param to create drift
        with torch.no_grad():
            model.dense_param.add_(1.0)
            # Original was 0.0, now 1.0. Drift = ||1.0|| / (||0.0|| + eps) = sqrt(10) / eps (very large)
            # Actually norm(1.0-0.0) = sqrt(10), norm(0.0) = 0. Drift = sqrt(10) / 1e-6

        stats = trainer._collect_telemetry()

        # Verify groups are present
        assert "telemetry/backbone/param_norm" in stats
        assert "telemetry/engram_dense/param_zero_rate" in stats
        assert (
            stats["telemetry/engram_dense/param_zero_rate"] == 0.0
        )  # because we added 1.0
        assert "telemetry/engram_dense/grad_norm" in stats
        assert "telemetry/engram_sparse/param_max" in stats
        assert abs(stats["telemetry/engram_sparse/param_max"] - 0.5) < 1e-5

        # Drift
        assert "telemetry/engram_dense/weight_drift" in stats
        assert (
            stats["telemetry/engram_dense/weight_drift"] > 3.0
        )  # drifted from 0 to ones(10) (norm is sqrt(10) approx 3.16)

        # Model activation stats
        assert stats["telemetry/mock_gate"] == 123.0


if __name__ == "__main__":
    test_model_telemetry_stats()
    # Need to provide a mock path if running directly
    from pathlib import Path

    test_trainer_telemetry_collection(Path("tmp"))
    print("All telemetry tests passed!")
