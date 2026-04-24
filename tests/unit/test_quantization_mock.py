import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from engram_peft import EngramConfig, EngramModel, get_engram_model
from engram_peft.discovery import ArchitectureResolver


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
        """
        Tests resolve_layer_dtype directly for all three priority steps:
          Step 1: Explicit config override
          Step 2: BitsAndBytes compute_dtype detection on quantized layers
          Step 3: Parameter sampling on normal layers
        """
        # --- Step 2: Quantized layer with compute_dtype ---
        quantized_layer = MockQuantizedLayer(compute_dtype=torch.bfloat16)
        detected_dtype, source = ArchitectureResolver.resolve_layer_dtype(
            quantized_layer, config=None
        )
        self.assertEqual(
            detected_dtype,
            torch.bfloat16,
            f"Expected bfloat16 from quantized layer, got {detected_dtype} "
            f"(source: {source})",
        )
        self.assertIn("compute_dtype", source)

        # --- Step 3: Parameter sampling for normal linear layer (default float32) ---
        linear_layer = nn.Linear(10, 10)
        detected_dtype2, source2 = ArchitectureResolver.resolve_layer_dtype(
            linear_layer, config=None
        )
        self.assertEqual(
            detected_dtype2,
            torch.float32,
            f"Expected float32 from linear layer, got {detected_dtype2} "
            f"(source: {source2})",
        )
        self.assertIn("Parameter sample", source2)

        # --- Step 1: Explicit config override takes priority ---
        explicit_config = EngramConfig(
            target_layers=[0],
            embedding_dim=16,
            n_head_per_ngram=2,
            engram_dtype="float16",
            enable_tokenizer_compression=False,
        )
        detected_dtype3, source3 = ArchitectureResolver.resolve_layer_dtype(
            linear_layer, config=explicit_config
        )
        self.assertEqual(
            detected_dtype3,
            torch.float16,
            f"Expected float16 from explicit config, got {detected_dtype3} "
            f"(source: {source3})",
        )
        self.assertIn("Explicit config", source3)

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
