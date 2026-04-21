import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the current directory to sys.path to import from examples
sys.path.append(os.getcwd())

from examples.qwen3_engram_lora import run_example  # type: ignore


class TestQwen3Enhanced(unittest.TestCase):
    @patch("examples.qwen3_engram_lora.AutoTokenizer")
    @patch("examples.qwen3_engram_lora.AutoModelForCausalLM")
    @patch("examples.qwen3_engram_lora.load_dataset")
    @patch("examples.qwen3_engram_lora.plot_benchmark_comparison")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.is_bf16_supported", return_value=False)
    def test_enhanced_workflow(
        self,
        mock_bf16,
        mock_cuda,
        mock_plot,
        mock_load_dataset,
        mock_automodel,
        mock_tokenizer,
    ):
        # 1. Setup mocks
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.pad_token = None
        mock_tokenizer_inst.eos_token = "</s>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = 28
        mock_model.config.hidden_size = 4096
        mock_model.base_model.device = "cpu"
        mock_automodel.from_pretrained.return_value = mock_model

        # Mock Dataset
        from datasets import Dataset

        mock_ds = MagicMock(spec=Dataset)
        mock_ds.map.return_value = mock_ds
        mock_ds.train_test_split.return_value = {
            "train": MagicMock(spec=Dataset),
            "test": MagicMock(spec=Dataset),
        }
        mock_load_dataset.return_value = mock_ds

        # 2. Mock Trainer and Models
        with (
            patch("examples.qwen3_engram_lora.get_peft_model"),
            patch("examples.qwen3_engram_lora.get_engram_model") as mock_get_engram,
            patch("examples.qwen3_engram_lora.EngramTrainer") as mock_trainer_cls,
            patch("examples.qwen3_engram_lora.EngramDataCollator"),
            patch("examples.qwen3_engram_lora.BenchmarkResult") as mock_res_cls,
            patch("examples.qwen3_engram_lora.EngramModel") as mock_engram_model_cls,
        ):
            mock_trainer_inst = mock_trainer_cls.return_value
            mock_trainer_inst.evaluate.return_value = {"eval_loss": 1.5}
            mock_trainer_inst.state.log_history = [
                {"step": 1, "loss": 0.5, "eval_loss": 0.4}
            ]

            # Setup Engram model mock for saving
            mock_engram_model = MagicMock()
            mock_get_engram.return_value = mock_engram_model
            mock_engram_model.base_model = MagicMock()

            args = argparse.Namespace(
                model_id="mock/model",
                max_steps=10,
                batch_size=1,
                lr=2e-4,
                lora_r=16,
                engram_dim=128,
                load_in_4bit=True,
                load_in_8bit=False,
                eval_steps=5,
                logging_steps=1,
            )

            # 3. Run example
            run_example(args)

            # 4. Verifications
            # Check if evaluation happened (0-shot)
            self.assertTrue(mock_trainer_inst.evaluate.called)

            # Check if main_res.save() was called
            self.assertTrue(mock_res_cls.return_value.save.called)

            # Check if plot was called with two results (base + main)
            args_list = mock_plot.call_args[0][0]
            self.assertEqual(len(args_list), 2)

            # Check if EngramModel.from_pretrained was called for reloading
            self.assertTrue(mock_engram_model_cls.from_pretrained.called)


if __name__ == "__main__":
    unittest.main()
