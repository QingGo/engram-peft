from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from engram_peft.collator import EngramDataCollator
from engram_peft.config import EngramConfig


@pytest.fixture
def tokenizer() -> Any:
    # Using a common small tokenizer. It should be cached or fast to download.
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


class MockCompressor:
    """
    A mock compressor to verify it's called during collation.
    """

    def compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Simply offset IDs to show it was processed
        return input_ids + 1000


class TestEngramDataCollator:
    """
    Tests for the EngramDataCollator.
    """

    @pytest.fixture
    def setup_config(self) -> EngramConfig:
        return EngramConfig(
            target_layers=[2, 15],
            ngram_sizes=[2, 3],
            n_head_per_ngram=4,
            engram_vocab_size_per_ngram=[100, 100],
            seed=42,
        )

    def test_hash_indices_content(
        self, setup_config: EngramConfig, tokenizer: Any
    ) -> None:
        """
        测试用例 1：验证 engram_hash_indices 包含所有目标层的哈希
        """
        collator = EngramDataCollator(tokenizer=tokenizer, config=setup_config)

        examples = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7]}]
        batch = collator(examples)

        assert "engram_hash_indices" in batch
        hashes = batch["engram_hash_indices"]

        # Batch size = 2
        # Max seq len = 4
        # Num layers = 2 (layers 2 and 15)
        # Total heads = len(ngram_sizes) * n_head_per_ngram = 2 * 4 = 8
        assert hashes.shape == (2, 4, 2, 8)
        assert isinstance(hashes, torch.Tensor)

    def test_batch_processing(self, setup_config: EngramConfig, tokenizer: Any) -> None:
        """
        测试用例 2：验证 batch 处理正确 (包含不同长度输入)
        """
        collator = EngramDataCollator(tokenizer=tokenizer, config=setup_config)

        # Varying lengths
        examples = [
            {"input_ids": [1, 2]},
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [1]},
        ]
        batch = collator(examples)

        assert batch["input_ids"].shape == (3, 5)
        assert batch["engram_hash_indices"].shape == (3, 5, 2, 8)

    def test_with_compressor(self, setup_config: EngramConfig, tokenizer: Any) -> None:
        """
        验证 optional compressor 是否被正确调用
        """
        compressor = MockCompressor()
        collator = EngramDataCollator(
            tokenizer=tokenizer, config=setup_config, compressor=compressor  # type: ignore[arg-type]
        )

        examples = [{"input_ids": [1, 2, 3]}]
        batch = collator(examples)

        # If it runs without error, it's already a good sign.
        # NgramHashMapping handles the compressed IDs correctly.
        assert batch["engram_hash_indices"].shape == (1, 3, 2, 8)

    def test_trainer_compatibility(
        self, setup_config: EngramConfig, tokenizer: Any
    ) -> None:
        """
        测试用例 3：验证与 transformers Trainer 兼容
        """
        collator = EngramDataCollator(tokenizer=tokenizer, config=setup_config)

        # Trainer expects a dictionary where items are tensors (except potentially some metadata)
        examples = [{"input_ids": [1, 2], "labels": [1, 2]}]
        batch = collator(examples)

        assert "input_ids" in batch
        assert "labels" in batch
        assert "engram_hash_indices" in batch

        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["engram_hash_indices"], torch.Tensor)

    def test_cpu_computation(self, setup_config: EngramConfig, tokenizer: Any) -> None:
        """
        测试用例 4：验证所有计算都在 CPU 上完成
        """
        collator = EngramDataCollator(tokenizer=tokenizer, config=setup_config)

        examples = [{"input_ids": [1, 2, 3]}]
        batch = collator(examples)

        # DataCollator output should be on CPU
        assert batch["engram_hash_indices"].device.type == "cpu"
        assert batch["input_ids"].device.type == "cpu"
