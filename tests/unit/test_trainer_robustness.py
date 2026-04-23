from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import TrainingArguments

from engram_peft.trainer import EngramTrainer


def test_trainer_clip_grad_norm_robustness():
    # Setup real model to avoid mock attribute issues
    model = nn.Linear(10, 10)

    # Setup mock trainer
    # Use positional argument to satisfy some pyright stubs
    args = TrainingArguments(output_dir="tmp_test")
    args.max_grad_norm = 1.0
    trainer = EngramTrainer(model=model, args=args)

    # Test 1: max_norm is None, should fallback to args.max_grad_norm
    with patch.object(
        EngramTrainer, "_compute_total_norm", return_value=torch.tensor(0.5)
    ):
        norm = trainer._clip_grad_norm(model, max_norm=None)
        assert norm == 0.5

    # Test 2: max_norm is None, and args.max_grad_norm is None, should fallback to 0.0 and return None
    # Use setattr to bypass strict type checking if needed
    trainer.args.max_grad_norm = None
    norm = trainer._clip_grad_norm(model, max_norm=None)
    assert norm is None

    # Test 3: total_norm is None (no gradients), should return None early
    with patch.object(EngramTrainer, "_compute_total_norm", return_value=None):
        norm = trainer._clip_grad_norm(model, max_norm=1.0)
        assert norm is None


def test_trainer_clip_grad_norm_use_per_group_consistency():
    # Setup real model
    model = nn.Linear(10, 10)

    # Setup mock trainer
    args = TrainingArguments(output_dir="tmp_test")
    args.max_grad_norm = 1.0
    trainer = EngramTrainer(model=model, args=args)

    # Mock unwrapped model to look like an EngramModel with per-group clipping enabled
    mock_engram = MagicMock()
    mock_engram.config.clip_grad_per_group = True

    with patch("engram_peft.trainer.unwrap_model", return_value=mock_engram):
        # If total_norm is None, it should return None early even if use_per_group would be True
        with patch.object(EngramTrainer, "_compute_total_norm", return_value=None):
            norm = trainer._clip_grad_norm(model, max_norm=1.0)
            assert norm is None
