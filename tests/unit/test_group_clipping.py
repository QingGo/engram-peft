from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import TrainingArguments

from engram_peft import EngramConfig, EngramModel, EngramTrainer


def test_group_wise_clipping_logic(tmp_path: Any) -> None:
    # 1. Setup mock model and config
    config = EngramConfig(target_layers=[0], clip_grad_per_group=True, hidden_size=16)

    # We need a proper nn.Module that mimics EngramModel structure for unwrap_model
    class MockModel(nn.Module):
        def __init__(self, config: EngramConfig) -> None:
            super().__init__()
            self.config = config
            self.backbone_param = nn.Parameter(torch.ones(10))
            self.engram_param = nn.Parameter(torch.ones(10))

    model = MockModel(config)

    # Mock gradients
    # Backbone norm = 2.0 * sqrt(10) = 6.32
    model.backbone_param.grad = torch.ones(10) * 2.0
    # Engram norm = 0.01 * sqrt(10) = 0.0316
    model.engram_param.grad = torch.ones(10) * 0.01

    # 2. Setup trainer
    args = TrainingArguments(output_dir=str(tmp_path), max_grad_norm=1.0)
    # We must treat the model as an EngramModel for the trainer to use group clipping
    with patch("engram_peft.trainer.unwrap_model", return_value=model):
        trainer = EngramTrainer(model=model, args=args)

    # 3. Mock groups
    groups = {"backbone": [model.backbone_param], "engram_sparse": [model.engram_param]}

    # 4. Invoke clipping
    # Need to patch EngramModel check as well
    with (
        patch("engram_peft.trainer.unwrap_model", return_value=model),
        patch("engram_peft.trainer.get_trainable_param_groups", return_value=groups),
        patch(
            "engram_peft.trainer.isinstance",
            side_effect=lambda obj, cls: True
            if cls == EngramModel
            else isinstance(obj, cls),
        ),
    ):
        trainer._clip_grad_norm(model)

    # 5. Verify results
    # Backbone should be clipped exactly to 1.0 norm
    bb_norm = torch.norm(model.backbone_param.grad, 2).item()
    assert abs(bb_norm - 1.0) < 1e-5

    # Engram should be untouched because its norm (0.0316) < 1.0
    en_norm = torch.norm(model.engram_param.grad, 2).item()
    assert abs(en_norm - 0.031622776) < 1e-5


if __name__ == "__main__":
    import sys
    from pathlib import Path

    test_group_wise_clipping_logic(Path("tmp"))
    print("Test passed!")
