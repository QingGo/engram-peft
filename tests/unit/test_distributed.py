from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from transformers import TrainingArguments

from engram_peft.model import EngramModel
from engram_peft.trainer import (
    EngramTrainer,
    _is_deepspeed_enabled,
    _warn_distributed_sparse,
)


def _make_mock_args(deepspeed_val=None, **kwargs) -> MagicMock:
    args = MagicMock(spec=TrainingArguments)
    args.deepspeed = deepspeed_val
    args.learning_rate = kwargs.pop("learning_rate", 5e-5)
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def _make_bare_trainer(**attrs) -> MagicMock:
    trainer = EngramTrainer.__new__(EngramTrainer)
    trainer.optimizer = None
    trainer.optimizer_kwargs = {}
    for k, v in attrs.items():
        setattr(trainer, k, v)
    return trainer


# --- _is_deepspeed_enabled ---


def test_is_deepspeed_enabled_false():
    args = _make_mock_args(deepspeed_val=None)
    assert not _is_deepspeed_enabled(args)


def test_is_deepspeed_enabled_true():
    args = _make_mock_args(deepspeed_val="ds_config.json")
    assert _is_deepspeed_enabled(args)


def test_is_deepspeed_enabled_none_args():
    assert not _is_deepspeed_enabled(None)
    assert not _is_deepspeed_enabled(object())


# --- _warn_distributed_sparse ---


def test_warn_distributed_sparse_no_model():
    args = _make_mock_args(deepspeed_val="ds_config.json")
    _warn_distributed_sparse(None, args)


def test_warn_distributed_sparse_no_deepspeed_no_dist():
    model = nn.Linear(10, 10)
    args = _make_mock_args(deepspeed_val=None)
    with patch("builtins.print") as mock_print:
        _warn_distributed_sparse(model, args)
        mock_print.assert_not_called()


def test_warn_distributed_sparse_non_engram_model():
    model = nn.Linear(10, 10)
    args = _make_mock_args(deepspeed_val="ds_config.json")
    with patch("builtins.print") as mock_print:
        with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
            _warn_distributed_sparse(model, args)
        mock_print.assert_not_called()


def test_warn_distributed_sparse_engram_with_sparse():
    mock_model = MagicMock(spec=EngramModel)
    mock_model.config = MagicMock()
    mock_model.config.use_sparse_embeddings = True

    args = _make_mock_args(deepspeed_val="ds_config.json")
    with patch("builtins.print") as mock_print:
        with patch("engram_peft.trainer.unwrap_model", return_value=mock_model):
            with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
                _warn_distributed_sparse(mock_model, args)
                mock_print.assert_called_once()
                text = mock_print.call_args[0][0]
                assert "DeepSpeed" in text
                assert "use_sparse_embeddings=True" in text


def test_warn_distributed_sparse_engram_without_sparse():
    mock_model = MagicMock(spec=EngramModel)
    mock_model.config = MagicMock()
    mock_model.config.use_sparse_embeddings = False

    args = _make_mock_args(deepspeed_val="ds_config.json")
    with patch("builtins.print") as mock_print:
        with patch("engram_peft.trainer.unwrap_model", return_value=mock_model):
            _warn_distributed_sparse(mock_model, args)
            mock_print.assert_not_called()


def test_config_sparse_off_no_warning():
    mock_model = MagicMock(spec=EngramModel)
    mock_model.config = MagicMock()
    mock_model.config.use_sparse_embeddings = False

    args = _make_mock_args(deepspeed_val="ds_config.json")
    with patch("builtins.print") as mock_print:
        with patch("engram_peft.trainer.unwrap_model", return_value=mock_model):
            _warn_distributed_sparse(mock_model, args)
            mock_print.assert_not_called()


# --- EngramTrainer.create_optimizer ---


def test_trainer_create_optimizer_deepspeed_skips_mixed():
    """create_optimizer should skip MixedOptimizer when DeepSpeed is enabled."""
    model = nn.Linear(10, 10)
    trainer = _make_bare_trainer(
        args=_make_mock_args(deepspeed_val="ds_config.json", learning_rate=5e-5),
        model=model,
    )

    with patch("builtins.print") as mock_print:
        with patch("engram_peft.trainer._is_deepspeed_enabled", return_value=True):
            with patch("transformers.Trainer.create_optimizer") as mock_super:
                mock_super.return_value = AdamW(model.parameters(), lr=5e-5)
                optimizer = trainer.create_optimizer()

    assert optimizer is not None
    mock_print.assert_called()
    text = mock_print.call_args[0][0]
    assert "DeepSpeed detected" in text
    assert "skipping MixedOptimizer" in text
    mock_super.assert_called_once()


def test_trainer_create_optimizer_deepspeed_with_engram():
    """When DeepSpeed is enabled and model is EngramModel, still skip MixedOptimizer."""
    mock_engram = MagicMock(spec=EngramModel)
    mock_engram.config = MagicMock()
    mock_engram.config.use_sparse_embeddings = True

    trainer = _make_bare_trainer(
        args=_make_mock_args(deepspeed_val="ds_config.json", learning_rate=5e-5),
        model=mock_engram,
    )

    with patch("builtins.print") as mock_print:
        with patch("engram_peft.trainer._is_deepspeed_enabled", return_value=True):
            with patch("transformers.Trainer.create_optimizer") as mock_super:
                mock_super.return_value = MagicMock(spec=AdamW)
                trainer.create_optimizer()

    mock_print.assert_called()
    text = mock_print.call_args[0][0]
    assert "DeepSpeed detected" in text
    mock_super.assert_called_once()


def test_trainer_create_optimizer_normal_path():
    """Non-Engram model without DeepSpeed should use super().create_optimizer()."""
    model = nn.Linear(10, 10)
    trainer = _make_bare_trainer(
        args=_make_mock_args(deepspeed_val=None, learning_rate=5e-5),
        model=model,
    )

    with patch("engram_peft.trainer._is_deepspeed_enabled", return_value=False):
        with patch("transformers.Trainer.create_optimizer") as mock_super:
            mock_super.return_value = AdamW(model.parameters(), lr=5e-5)
            with patch("engram_peft.trainer.wash_model", return_value=model):
                optimizer = trainer.create_optimizer()

    assert optimizer is not None
    mock_super.assert_called_once()


def test_trainer_create_optimizer_engram_path():
    """EngramModel without DeepSpeed should use MixedOptimizer path."""
    mock_engram = MagicMock(spec=EngramModel)
    mock_engram.create_optimizer = MagicMock(return_value=MagicMock(spec=AdamW))

    trainer = _make_bare_trainer(
        args=_make_mock_args(deepspeed_val=None, learning_rate=5e-5),
        model=mock_engram,
    )

    with patch("engram_peft.trainer._is_deepspeed_enabled", return_value=False):
        with patch("engram_peft.trainer.wash_model", return_value=mock_engram):
            optimizer = trainer.create_optimizer()

    assert optimizer is not None
    mock_engram.create_optimizer.assert_called_once()
