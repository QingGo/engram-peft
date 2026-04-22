import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from engram_peft import EngramConfig, EngramModel, get_engram_model


class MockQuantizedLayer(nn.Module):
    def __init__(self, compute_dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(1, 1, dtype=torch.uint8), requires_grad=False
        )
        self.compute_dtype = compute_dtype

    def forward(self, x):
        return x.to(self.compute_dtype)


class TestQuantizationDtypeDetection(unittest.TestCase):
    def setUp(self):
        # Create a dummy model structure
        self.base_model = nn.Module()
        self.base_model.config = MagicMock()
        self.base_model.config.num_hidden_layers = 2
        self.base_model.dtype = torch.float16
        self.base_model.device = torch.device("cpu")

        # Mock transformer layers
        self.layer0 = MockQuantizedLayer(compute_dtype=torch.bfloat16)
        self.layer1 = nn.Linear(10, 10)  # Normal layer (float32 by default)

        # Setup as a list or ModuleList
        self.base_model.layers = nn.ModuleList([self.layer0, self.layer1])

        # Mock _find_transformer_layers to return our list
        self.patcher = patch("engram_peft.model.EngramModel._find_transformer_layers")
        self.mock_find = self.patcher.start()
        self.mock_find.return_value = [self.layer0, self.layer1]

    def tearDown(self):
        self.patcher.stop()

    def test_automatic_compute_dtype_detection(self):
        # Config targeting both layers
        config = EngramConfig(
            target_layers=[0, 1],
            embedding_dim=16,
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
        )

        # Initialize EngramModel
        model = get_engram_model(self.base_model, config)

        # Check layer 0 (Quantized)
        engram_layer0 = model.engram_layers["0"]
        # Should be bfloat16 (detected from compute_dtype)
        self.assertEqual(next(engram_layer0.parameters()).dtype, torch.bfloat16)

        # Check layer 1 (Normal)
        engram_layer1 = model.engram_layers["1"]
        # Should be float32 (default for nn.Linear in this mock)
        self.assertEqual(next(engram_layer1.parameters()).dtype, torch.float32)

    def test_explicit_engram_dtype_override(self):
        # Config with explicit float16
        config = EngramConfig(
            target_layers=[0],
            embedding_dim=16,
            n_head_per_ngram=2,
            engram_dtype="float16",
            enable_tokenizer_compression=False,
        )

        model = get_engram_model(self.base_model, config)

        engram_layer0 = model.engram_layers["0"]
        # Should be float16 even though layer0.compute_dtype is bfloat16
        self.assertEqual(next(engram_layer0.parameters()).dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
