from unittest.mock import MagicMock

import numpy as np
import torch

from engram_peft.compression import CompressedTokenizer


def test_map_ids_robustness():
    # Mock tokenizer for initialization
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 100
    mock_tokenizer.decode.return_value = "token"

    # Initialize CompressedTokenizer
    compressor = CompressedTokenizer(tokenizer=mock_tokenizer)

    # Test with torch.Tensor
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
    output_tensor = compressor.map_ids(input_tensor)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (3,)

    # Test with np.ndarray
    input_np = np.array([1, 2, 3], dtype=np.int64)
    output_np = compressor.map_ids(input_np)
    assert isinstance(output_np, np.ndarray)
    assert output_np.shape == (3,)

    # Test with list (The core of the fix)
    input_list = [1, 2, 3]
    output_list = compressor.map_ids(input_list)
    assert isinstance(output_list, torch.Tensor)
    assert output_list.shape == (3,)

    # Test with nested list
    input_nested = [[1, 2], [3, 4]]
    output_nested = compressor.map_ids(input_nested)
    assert isinstance(output_nested, torch.Tensor)
    assert output_nested.shape == (2, 2)

    # Test with negative IDs (should be preserved)
    input_neg = torch.tensor([-100, 1, -100])
    output_neg = compressor.map_ids(input_neg)
    assert output_neg[0] == -100
    assert output_neg[2] == -100
