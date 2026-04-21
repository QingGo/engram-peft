import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from engram_peft.config import EngramConfig
from engram_peft.model import EngramModel


class DummyModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(20)])

    def forward(self, input_ids, **kwargs):
        return self.layers[0](torch.randn(1, 10))


class TestHubIntegration(unittest.TestCase):
    def setUp(self):
        self.config = EngramConfig(
            target_layers=[1],
            hidden_size=10,
            compressed_vocab_size=100,
            engram_vocab_size_per_ngram=[100, 100],
            pad_id=0,
            enable_tokenizer_compression=False,
        )
        self.base_model = DummyModel(PretrainedConfig())
        self.model = EngramModel(self.base_model, self.config)

    @patch("engram_peft.model.HfApi")
    def test_push_to_hub(self, mock_hf_api):
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance

        repo_id = "test-user/test-repo"
        self.model.push_to_hub(repo_id, token="test-token")

        # Verify repo creation
        mock_api_instance.create_repo.assert_called_once_with(
            repo_id=repo_id, private=None, exist_ok=True
        )
        # Verify upload_folder call
        mock_api_instance.upload_folder.assert_called_once()
        call_kwargs = mock_api_instance.upload_folder.call_args.kwargs
        self.assertEqual(call_kwargs["repo_id"], repo_id)
        # Note: we can't check if folder_path exists because it's a temp dir already deleted

    @patch("engram_peft.model.HfApi")
    def test_push_to_hub_no_temp_dir(self, mock_hf_api):
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance

        repo_id = "test-user/test-repo"
        expected_dir = "test-repo"

        # Ensure the directory doesn't exist before
        if os.path.exists(expected_dir):
            import shutil

            shutil.rmtree(expected_dir)

        try:
            self.model.push_to_hub(repo_id, use_temp_dir=False)

            # Verify upload_folder was called with the local directory
            mock_api_instance.upload_folder.assert_called_once()
            call_kwargs = mock_api_instance.upload_folder.call_args.kwargs
            self.assertEqual(call_kwargs["folder_path"], expected_dir)
            self.assertTrue(os.path.isdir(expected_dir))
        finally:
            # Clean up
            if os.path.exists(expected_dir):
                import shutil

                shutil.rmtree(expected_dir)

    @patch("engram_peft.model.snapshot_download")
    @patch("engram_peft.config.EngramConfig.from_pretrained")
    @patch("engram_peft.model.EngramModel.load_engram")
    def test_from_pretrained_hub(
        self, mock_load_engram, mock_config_from_pretrained, mock_snapshot_download
    ):
        hub_id = "test-user/test-repo"
        mock_snapshot_download.return_value = "/mock/path"
        mock_config_from_pretrained.return_value = self.config

        # Test loading from Hub ID (path doesn't exist)
        with patch("os.path.exists", return_value=False):
            model = EngramModel.from_pretrained(
                self.base_model, hub_id, token="test-token"
            )

            mock_snapshot_download.assert_called_once_with(
                repo_id=hub_id,
                token="test-token",
                revision=None,
                library_name="engram-peft",
            )
            mock_config_from_pretrained.assert_called_once_with(
                "/mock/path", token="test-token"
            )
            # Called twice: once in __init__ (no args) and once in from_pretrained (with path)
            self.assertEqual(mock_load_engram.call_count, 2)
            mock_load_engram.assert_called_with("/mock/path")
            self.assertIsInstance(model, EngramModel)

    def test_from_pretrained_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model.save_pretrained(tmp_dir)

            # Should load from local path directly
            with patch("engram_peft.model.snapshot_download") as mock_snapshot:
                model = EngramModel.from_pretrained(self.base_model, tmp_dir)
                mock_snapshot.assert_not_called()
                self.assertIsInstance(model, EngramModel)


if __name__ == "__main__":
    unittest.main()
