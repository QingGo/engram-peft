import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
from transformers import PreTrainedTokenizer
from engram_peft.compression import TokenizerCompressor


class TestTokenizerCompressor(unittest.TestCase):
    def setUp(self) -> None:
        # Create a mock tokenizer
        self.mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)

        # Define a vocab of 1000 unique tokens
        self.vocab_size = 1000
        self.mock_tokenizer.__len__.return_value = self.vocab_size

        # Simple mapping for convert_ids_to_tokens
        # We'll make some tokens that normalize to the same thing to test grouping
        # But for the compression ratio test, we'll mostly use unique ones
        def convert_ids_to_tokens(token_id: int) -> str | None:
            if token_id < 0 or token_id >= self.vocab_size:
                return None
            return f"token_{token_id}"

        self.mock_tokenizer.convert_ids_to_tokens.side_effect = convert_ids_to_tokens
        self.mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: x[0]

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_compression_ratio(self) -> None:
        # With 1000 unique tokens and 23% ratio, it should truncate to 770
        compressor = TokenizerCompressor(self.mock_tokenizer, compression_ratio=0.23)

        # The number of unique compressed IDs should be around (1 - 0.23) * 1000 = 770
        unique_compressed_ids = len(set(compressor.mapping.values()))
        ratio = 1 - (unique_compressed_ids / self.vocab_size)

        # Verify ratio is around 23% (0.23)
        # Since we use 1000 tokens, it should be exactly 0.23
        self.assertAlmostEqual(ratio, 0.23, places=2)
        self.assertTrue(0.22 <= ratio <= 0.24)

    def test_normalization(self) -> None:
        compressor = TokenizerCompressor(self.mock_tokenizer)

        # Test NFKC and lowercasing
        # 'Á' is \u00c1. NFKC of 'Á' is 'Á'. lowercase is 'á'.
        # Wait, NFKC might decompose some things, but for 'Á' it's usually stable.
        # Let's use a more complex example if needed, but 'Á' -> 'á' is a good start.
        self.assertEqual(compressor._normalize_text("Á"), "á")
        self.assertEqual(compressor._normalize_text("ﬁ"), "fi")
        self.assertEqual(compressor._normalize_text("Hello"), "hello")

        # Test space variants
        self.assertEqual(
            compressor._normalize_text("  \t\n  "), " "
        )  # Merges to single space
        self.assertEqual(
            compressor._normalize_text("Hello  \t\n  World"), "hello world"
        )

        # Test standard normalization
        # \u212b is ANGSTROM SIGN. NFKC of \u212b is \u00c5 (LATIN CAPITAL LETTER A WITH RING ABOVE)
        # Then lowercase makes it \u00e5
        self.assertEqual(compressor._normalize_text("\u212b"), "\u00e5")

    def test_space_merging(self) -> None:
        compressor = TokenizerCompressor(self.mock_tokenizer)
        self.assertEqual(compressor._normalize_text("word1    word2"), "word1 word2")
        self.assertEqual(compressor._normalize_text("\tword1\nword2\r"), "word1 word2")

    def test_save_load(self) -> None:
        compressor = TokenizerCompressor(self.mock_tokenizer, compression_ratio=0.1)
        compressor.save_pretrained(self.temp_dir)

        # Load it back
        loaded_compressor = TokenizerCompressor.from_pretrained(
            self.temp_dir, self.mock_tokenizer
        )

        self.assertEqual(
            compressor.compression_ratio, loaded_compressor.compression_ratio
        )
        self.assertEqual(compressor.vocab_size, loaded_compressor.vocab_size)
        self.assertEqual(
            compressor.compressed_vocab_size, loaded_compressor.compressed_vocab_size
        )
        self.assertEqual(compressor.mapping, loaded_compressor.mapping)

    def test_compress_tensor(self) -> None:
        compressor = TokenizerCompressor(self.mock_tokenizer, compression_ratio=0.5)

        input_ids = torch.tensor([0, 1, 10, 100, 999], dtype=torch.long)
        compressed_ids = compressor.compress(input_ids)

        self.assertEqual(compressed_ids.shape, input_ids.shape)
        self.assertEqual(compressed_ids[0].item(), compressor.mapping[0])
        self.assertEqual(compressed_ids[4].item(), compressor.mapping[999])


if __name__ == "__main__":
    unittest.main()
