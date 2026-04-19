import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import yaml

from engram_peft.cli import apply_overrides, parse_override_value, train


class TestCLI(unittest.TestCase):
    def test_parse_override_value(self) -> None:
        self.assertEqual(parse_override_value("true"), True)
        self.assertEqual(parse_override_value("False"), False)
        self.assertEqual(parse_override_value("123"), 123)
        self.assertEqual(parse_override_value("1.23"), 1.23)
        self.assertEqual(parse_override_value("hello"), "hello")
        self.assertEqual(parse_override_value("none"), None)

    def test_apply_overrides(self) -> None:
        config: dict[str, Any] = {
            "training_args": {"learning_rate": 1e-4},
            "engram_config": {"embedding_dim": 128},
        }
        overrides = [
            "training_args.learning_rate=5e-5",
            "engram_config.embedding_dim=256",
            "engram_config.new_field=true",
            "root_val=10",
        ]
        apply_overrides(config, overrides)
        self.assertEqual(
            cast("dict[str, Any]", config["training_args"])["learning_rate"], 5e-5
        )
        self.assertEqual(
            cast("dict[str, Any]", config["engram_config"])["embedding_dim"], 256
        )
        self.assertEqual(
            cast("dict[str, Any]", config["engram_config"])["new_field"], True
        )
        self.assertEqual(config["root_val"], 10)

    @patch("engram_peft.cli.AutoTokenizer")
    @patch("engram_peft.cli.AutoModelForCausalLM")
    @patch("engram_peft.cli.get_engram_model")
    @patch("engram_peft.cli.load_dataset")
    @patch("engram_peft.cli.EngramTrainer")
    @patch("engram_peft.cli.TrainingArguments")
    def test_train_logic(
        self,
        mock_train_args: Any,
        mock_trainer: Any,
        mock_ds: Any,
        mock_get_engram: Any,
        mock_model: Any,
        mock_tokenizer: Any,
    ) -> None:
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_ds.return_value = MagicMock()

        dummy_config = {
            "model_name_or_path": "tiny-model",
            "engram_config": {"embedding_dim": 64},
            "training_args": {"output_dir": "./test_out"},
            "data_args": {"dataset_name": "dummy_ds", "text_column": "text"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(dummy_config, f)
            config_path = Path(f.name)

        def mock_get_engram_side_effect(
            model: Any, config: Any, tokenizer: Any = None
        ) -> Any:
            # Simulate ArchitectureResolver/EngramModel side effects on config
            config.compressed_vocab_size = 1000
            config.pad_id = 0
            return MagicMock()

        mock_get_engram.side_effect = mock_get_engram_side_effect

        try:
            # We wrap in a try-finally to delete the temp file
            with patch("typer.echo"), patch("typer.secho"):
                train(
                    config_path=config_path,
                    overrides=["training_args.learning_rate=2e-5"],
                )

            # Verify calls
            mock_tokenizer.from_pretrained.assert_called_with("tiny-model")
            mock_model.from_pretrained.assert_called()
            mock_get_engram.assert_called()
            mock_ds.assert_called_with("dummy_ds", None, split="train")
            mock_trainer.assert_called()

            # Check if override was applied inside the call (tricky without deep inspection,
            # but we can check if TrainingArguments was called with output_dir)
            mock_train_args.assert_called()
            args, kwargs = mock_train_args.call_args
            self.assertEqual(kwargs["output_dir"], "./test_out")
            self.assertEqual(kwargs["learning_rate"], 2e-5)

        finally:
            config_path.unlink()


if __name__ == "__main__":
    unittest.main()
