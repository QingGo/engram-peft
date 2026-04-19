import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import TrainingArguments

from engram_peft.config import EngramConfig
from engram_peft.model import EngramModel, get_engram_model
from engram_peft.trainer import EngramTrainer


class TestTrainerInit(unittest.TestCase):
    def test_capture_initial_weights_no_keyerror(self) -> None:
        """验证即使参数组名称变动或部分缺失，初始化也不会触发 KeyError"""

        base_model = nn.Module()
        base_model.layers = nn.ModuleList([nn.Linear(8, 8)])
        engram_config = EngramConfig(
            target_layers=[0],
            enable_telemetry=True,
            hidden_size=8,
            original_vocab_size=100,
            layer_container_path="layers",
            enable_tokenizer_compression=False,
        )
        base_model.config = type(
            "Config", (), {"vocab_size": 100, "model_type": "custom"}
        )()

        with (
            patch("engram_peft.model.NgramHashMapping", return_value=MagicMock()),
            patch("engram_peft.model.CompressedTokenizer", return_value=MagicMock()),
            patch("engram_peft.model.EngramModel.load_engram", return_value=None),
        ):
            model = get_engram_model(base_model, engram_config)

        # Scenario: Only projection exists, output is missing for some reason
        groups = {
            "backbone": [nn.Parameter(torch.randn(1))],
            "engram_dense": [nn.Parameter(torch.randn(1))],
            # engram_sparse is missing
        }

        args = TrainingArguments(output_dir="tmp_test")

        def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
            self.model = kwargs.get("model") or (args[0] if len(args) > 0 else None)
            self.args = kwargs.get("args") or (args[1] if len(args) > 1 else None)
            self._initial_weights = {}

        with (
            patch("engram_peft.trainer.unwrap_model", return_value=model),
            patch(
                "engram_peft.trainer.get_trainable_param_groups", return_value=groups
            ),
            patch(
                "engram_peft.trainer.isinstance",
                side_effect=lambda obj, cls: (
                    True if cls == EngramModel else isinstance(obj, cls)
                ),
            ),
            patch(
                "transformers.Trainer.__init__", autospec=True, side_effect=mock_init
            ),
        ):
            # This should not raise KeyError anymore
            trainer = EngramTrainer(model=model, args=args)
            trainer.state = MagicMock()
            trainer.callback_handler = MagicMock()
            # Manually trigger capture since we skipped init
            trainer._capture_initial_weights()
            self.assertEqual(len(trainer._initial_weights), 1)


if __name__ == "__main__":
    unittest.main()
