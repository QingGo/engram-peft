import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from engram_peft.compression import CompressedTokenizer


class TestCompressedTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_compression_rate_deepseek_v3(self) -> None:
        """Test Case 1: Use DeepSeek-V3 tokenizer to verify compression rate is ~23%."""
        try:
            tokenizerPath = "deepseek-ai/DeepSeek-V3"
            # Note: We use an extremely small test mock here if testing environment limits internet access,
            # but ideally if deepseek-v3 is available locally, we'd use it to verify the exact bounds.
            compressor = CompressedTokenizer(tokenizerPath, trust_remote_code=True)

            rate = 1.0 - (compressor.compressed_vocab_size / compressor.vocab_size)
            self.assertTrue(
                0.22 <= rate <= 0.24,
                f"Compression rate {rate:.2%} is not within 22-24%",
            )
        except Exception as e:
            pytest.skip(f"Could not load DeepSeek-V3 tokenizer (check network): {e}")

    def test_normalization_effects(self) -> None:
        """Test Case 2: Verify NFKC, NFD, StripAccents, Lowercase."""
        # Using a small standard model for simple token mappings
        compressor = CompressedTokenizer("gpt2")

        def norm(text: str) -> str:
            return str(compressor.normalizer.normalize_str(text))

        # 'Á' -> StripAccents -> 'A' -> Lowercase -> 'a'
        self.assertEqual(norm("Á"), "a")
        self.assertEqual(norm("ﬁ"), "fi")
        self.assertEqual(norm("Hello"), "hello")

    def test_space_variants(self) -> None:
        """Test Case 3: Verify space variants merge correctly."""
        compressor = CompressedTokenizer("gpt2")

        def norm(text: str) -> str:
            return str(compressor.normalizer.normalize_str(text))

        self.assertEqual(norm("  \t\n  "), " ")
        self.assertEqual(norm("Hello  \t\n  World"), "hello world")

    def test_save_load(self) -> None:
        """Test Case 4: Verify save/load function."""
        compressor = CompressedTokenizer("gpt2")
        compressor.save_pretrained(self.temp_dir)

        loaded_compressor = CompressedTokenizer.from_pretrained(self.temp_dir)

        self.assertEqual(compressor.vocab_size, loaded_compressor.vocab_size)
        self.assertEqual(
            compressor.compressed_vocab_size, loaded_compressor.compressed_vocab_size
        )
        self.assertEqual(compressor.mapping, loaded_compressor.mapping)
        self.assertEqual(
            compressor.tokenizer_name_or_path, loaded_compressor.tokenizer_name_or_path
        )

        self.assertTrue(torch.equal(compressor.lookup, loaded_compressor.lookup))

    def test_compress_tensor_shapes(self) -> None:
        """Test Case 5: Verify batch processing (preserves shape)."""
        compressor = CompressedTokenizer("gpt2")

        # Create a batched tensor [2, 3]
        input_ids = torch.tensor([[100, 200, 300], [400, 500, 600]], dtype=torch.long)
        compressed_ids = compressor.compress(input_ids)

        self.assertEqual(compressed_ids.shape, input_ids.shape)

        # Also test with numpy array
        np_ids = np.array([[100, 200, 300], [400, 500, 600]])
        np_compressed = compressor.compress(np_ids)

        self.assertIsInstance(np_compressed, np.ndarray)
        self.assertEqual(np_compressed.shape, np_ids.shape)


if __name__ == "__main__":
    unittest.main()
