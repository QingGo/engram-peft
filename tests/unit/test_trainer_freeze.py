from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from engram_peft import EngramConfig, EngramModel, EngramTrainer


def test_freeze_backbone_steps(tmp_path: Any) -> None:
    """验证 backbone_freeze_steps 会在初始步数内冻结 backbone 并在之后解冻。"""

    # Must inherit from EngramModel for EngramTrainer logic to trigger
    class MockModel(EngramModel):
        @property
        def engram_layers(self) -> nn.ModuleDict:
            return self._engram_layers

        def __init__(self, cfg: EngramConfig) -> None:
            # Skip EngramModel.__init__ but ensure nn.Module is init
            nn.Module.__init__(self)
            self.config = cfg
            self.base_model = nn.Module()
            self.backbone_param = nn.Parameter(torch.randn(8))
            self.base_model.register_parameter("p", self.backbone_param)
            # utils.get_trainable_param_groups expects this
            self._engram_layers = nn.ModuleDict()

        def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            return {"loss": torch.tensor(0.0, requires_grad=True)}

    config = EngramConfig(
        target_layers=[0],
        backbone_freeze_steps=2,
        hidden_size=8,
        enable_tokenizer_compression=False,
        engram_vocab_size_per_ngram=[100, 100],
    )
    model = MockModel(config)

    args = TrainingArguments(
        output_dir=str(tmp_path), max_steps=5, logging_steps=1, report_to="none"
    )

    # EngramTrainer.__init__ should call _handle_initial_freezing
    trainer = EngramTrainer(model=model, args=args)
    # Patch missing attribute normally set during trainer.train()
    trainer.current_gradient_accumulation_steps = 1

    # Check initial state (should be frozen because 0 < 2)
    assert model.backbone_param.requires_grad is False

    # Step 1 -> remains frozen
    trainer.state.global_step = 1
    # Check that it remains frozen
    assert model.backbone_param.requires_grad is False

    # Step 2 -> should unfreeze
    trainer.state.global_step = 2
    # In trainer.py, the unfreeze happens IN trainer.training_step()
    trainer.training_step(model, {"input_ids": torch.zeros(1, 1).long()})

    # Now it should be unfrozen
    assert model.backbone_param.requires_grad is True
