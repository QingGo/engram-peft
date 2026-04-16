import shutil
import tempfile
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
import torch

from engram_peft.compression import CompressedTokenizer

"""
单元测试：词表压缩逻辑验证。
使用极小规模的 Mock Tokenizer，无外部依赖，执行速度极快（<0.1s）。
建议时机：本地开发过程中高频全量运行，确保核心逻辑无 Bug。
"""


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


def test_compression_logic_mock(tiny_tokenizer: Any) -> None:
    """Unit Test: Verify compression logic with a tiny mock tokenizer (Fast)."""
    compressor = CompressedTokenizer("mock/tiny", tokenizer=tiny_tokenizer)

    # Verify mapping works
    input_ids = torch.tensor([1, 2, 3])
    compressed = compressor.compress(input_ids)
    assert compressed.shape == input_ids.shape
    assert isinstance(compressed, torch.Tensor)


def test_normalization_effects(tiny_tokenizer: Any) -> None:
    """Verify NFKC, NFD, StripAccents, Lowercase using tiny_tokenizer."""
    compressor = CompressedTokenizer("mock/tiny", tokenizer=tiny_tokenizer)

    def norm(text: str) -> str:
        return str(compressor.normalizer.normalize_str(text))

    # Token 8 is 'Á'
    assert norm("Á") == "a"
    assert norm("ﬁ") == "fi"


def test_space_variants(tiny_tokenizer: Any) -> None:
    """Verify space variants merge correctly."""
    compressor = CompressedTokenizer("mock/tiny", tokenizer=tiny_tokenizer)

    def norm(text: str) -> str:
        return str(compressor.normalizer.normalize_str(text))

    assert norm("  \t\n  ") == " "


def test_save_load(temp_dir: str, tiny_tokenizer: Any) -> None:
    """Verify save/load function."""
    compressor = CompressedTokenizer("mock/tiny", tokenizer=tiny_tokenizer)
    compressor.save_pretrained(temp_dir)

    loaded_compressor = CompressedTokenizer.from_pretrained(temp_dir)

    assert compressor.vocab_size == loaded_compressor.vocab_size
    assert compressor.compressed_vocab_size == loaded_compressor.compressed_vocab_size
    assert compressor.mapping == loaded_compressor.mapping
    assert torch.equal(compressor.lookup, loaded_compressor.lookup)


def test_compress_tensor_shapes(tiny_tokenizer: Any) -> None:
    """Verify batch processing (preserves shape)."""
    compressor = CompressedTokenizer("mock/tiny", tokenizer=tiny_tokenizer)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    compressed_ids = compressor.compress(input_ids)
    assert compressed_ids.shape == input_ids.shape

    np_ids = np.array([[1, 2, 3], [4, 5, 6]])
    np_compressed = compressor.compress(np_ids)
    assert isinstance(np_compressed, np.ndarray)
    assert np_compressed.shape == np_ids.shape
