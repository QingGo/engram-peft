import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from engram_peft.cli import app, apply_overrides, parse_override_value, train

runner = CliRunner()


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
            model: Any, config: Any, tokenizer: Any = None, **kwargs: Any
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

            # Check if override was applied inside the call
            mock_train_args.assert_called()
            args, kwargs = mock_train_args.call_args
            self.assertEqual(kwargs["output_dir"], "test_out")
            self.assertEqual(kwargs["learning_rate"], 2e-5)

        finally:
            config_path.unlink()

    def test_config_template(self) -> None:
        with patch("pathlib.Path.write_text") as mock_write:
            result = runner.invoke(app, ["config-template", "--output", "test.yaml"])
            self.assertEqual(result.exit_code, 0)
            mock_write.assert_called_once()
            content = mock_write.call_args[0][0]
            self.assertIn("engram_config:", content)
            self.assertIn("training_args:", content)
            self.assertIn("ngram_sizes:", content)

    @patch("engram_peft.cli.AutoTokenizer")
    @patch("engram_peft.cli.AutoModelForCausalLM")
    @patch("engram_peft.cli.get_engram_model")
    @patch("engram_peft.cli.load_dataset")
    @patch("engram_peft.cli.EngramDataCollator")
    @patch("engram_peft.cli.EngramTrainer")
    def test_train_with_model_and_local_dataset(
        self,
        mock_trainer: Any,
        mock_collator: Any,
        mock_load_ds: Any,
        mock_get_model: Any,
        mock_model_cls: Any,
        mock_tok_cls: Any,
    ) -> None:
        # Mocking returns
        mock_tokenizer = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.pad_token = None

        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_load_ds.return_value = mock_dataset

        # We need to simulate EngramModel behavior on config for the collator check
        def mock_get_engram_side_effect(
            model: Any, config: Any, tokenizer: Any = None, **kwargs: Any
        ) -> Any:
            config.compressed_vocab_size = 1000
            config.pad_id = 0
            return MagicMock()

        mock_get_model.side_effect = mock_get_engram_side_effect

        # Test command: engram-peft train --model my-model --dataset my-data.jsonl
        with (
            patch("typer.echo"),
            patch("typer.secho"),
            patch("pathlib.Path.write_text"),
        ):
            result = runner.invoke(
                app, ["train", "--model", "my-model", "--dataset", "my-data.jsonl"]
            )

        if result.exit_code != 0:
            print(f"CLI Output:\n{result.output}")
            if result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception),
                    result.exception,
                    result.exception.__traceback__,
                )

        self.assertEqual(result.exit_code, 0)

        # Verify model loading
        mock_model_cls.from_pretrained.assert_called_once()
        # Verify dataset loading (local jsonl -> json)
        mock_load_ds.assert_called_once()
        args, kwargs = mock_load_ds.call_args
        self.assertEqual(args[0], "json")
        self.assertEqual(kwargs["data_files"], "my-data.jsonl")


if __name__ == "__main__":
    unittest.main()
