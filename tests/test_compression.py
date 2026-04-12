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

        # Define a vocab of 10 unique tokens for testing grouping
        self.vocab_size = 10
        self.mock_tokenizer.__len__.return_value = self.vocab_size

        # Simple mapping for decode and convert_ids_to_tokens
        def decode(token_ids: list[int], skip_special_tokens: bool = False) -> str:
            tid = token_ids[0]
            # Create some tokens that will normalize to the same thing
            if tid == 0:
                return "Apple"
            if tid == 1:
                return "apple"
            if tid == 2:
                return "  apple  "
            return f"token_{tid}"

        def convert_ids_to_tokens(token_id: int) -> str:
            return f"token_{token_id}"

        self.mock_tokenizer.decode.side_effect = decode
        self.mock_tokenizer.convert_ids_to_tokens.side_effect = convert_ids_to_tokens

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_compression_logic(self) -> None:
        # Test that Apple, apple, and "  apple  " all map to the same ID
        compressor = TokenizerCompressor(self.mock_tokenizer)

        # tid 0, 1, 2 should all normalize to "apple"
        id0 = compressor.mapping[0]
        id1 = compressor.mapping[1]
        id2 = compressor.mapping[2]

        self.assertEqual(id0, id1)
        self.assertEqual(id1, id2)

        # Other tokens should have different IDs
        id3 = compressor.mapping[3]
        self.assertNotEqual(id0, id3)

    def test_normalization(self) -> None:
        compressor = TokenizerCompressor(self.mock_tokenizer)

        # Test NFKC, StripAccents, and lowercasing via the internal normalizer
        def norm(text: str) -> str:
            return str(compressor.normalizer.normalize_str(text))

        # 'Á' -> StripAccents -> 'A' -> Lowercase -> 'a'
        self.assertEqual(norm("Á"), "a")
        self.assertEqual(norm("ﬁ"), "fi")
        self.assertEqual(norm("Hello"), "hello")

        # Test space variants
        self.assertEqual(norm("  \t\n  "), " ")
        self.assertEqual(norm("Hello  \t\n  World"), "hello world")

    def test_save_load(self) -> None:
        compressor = TokenizerCompressor(self.mock_tokenizer)
        compressor.save_pretrained(self.temp_dir)

        # Load it back
        loaded_compressor = TokenizerCompressor.from_pretrained(
            self.temp_dir, self.mock_tokenizer
        )

        self.assertEqual(compressor.vocab_size, loaded_compressor.vocab_size)
        self.assertEqual(
            compressor.compressed_vocab_size, loaded_compressor.compressed_vocab_size
        )
        self.assertEqual(compressor.mapping, loaded_compressor.mapping)

    def test_compress_tensor(self) -> None:
        compressor = TokenizerCompressor(self.mock_tokenizer)

        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        compressed_ids = compressor.compress(input_ids)

        self.assertEqual(compressed_ids.shape, input_ids.shape)
        # 0, 1, 2 should be the same
        self.assertEqual(compressed_ids[0].item(), compressed_ids[1].item())
        self.assertEqual(compressed_ids[1].item(), compressed_ids[2].item())
        # 3 should be different
        self.assertNotEqual(compressed_ids[0].item(), compressed_ids[3].item())


if __name__ == "__main__":
    unittest.main()
