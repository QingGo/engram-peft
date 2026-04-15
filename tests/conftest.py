from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from engram_peft.compression import CompressedTokenizer
from engram_peft.config import EngramConfig


class MockTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.tokenizer_vocab_size = 129280  # from NgramHashMapping default
        self.vocab = {f"token_{i}": i for i in range(10)}

    def __len__(self) -> int:
        return 10

    def decode(self, tids: list[int], **kwargs: Any) -> str:
        return " ".join([f"token_{t}" for t in tids])

    def convert_ids_to_tokens(self, tid: int) -> str:
        return f"token_{tid}"

    def pad(
        self, examples: list[dict[str, Any]], return_tensors: str = "pt", **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        # Basic padding simulation for unit tests
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        max_len = max(len(x) for x in input_ids)
        padded_ids = []
        for x in input_ids:
            p = torch.cat(
                [
                    x,
                    torch.full(
                        (max_len - len(x),), self.pad_token_id, dtype=torch.long
                    ),
                ]
            )
            padded_ids.append(p)

        batch = {"input_ids": torch.stack(padded_ids)}
        if "labels" in examples[0]:
            batch["labels"] = batch["input_ids"].clone()
        return batch


@pytest.fixture(scope="session")
def tokenizer_gpt2() -> Any:
    """Session-scoped GPT2 tokenizer."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="session")
def tiny_tokenizer() -> MockTokenizer:
    """A minimal mock tokenizer with only 10 tokens for fast unit testing."""
    return MockTokenizer()


@pytest.fixture(scope="session")
def tiny_compressor(tiny_tokenizer: MockTokenizer) -> CompressedTokenizer:
    """A CompressedTokenizer using the tiny mock tokenizer (fast)."""
    # Create an actual CompressedTokenizer but with our mock tokenizer to skip loops
    return CompressedTokenizer("mock/tiny", tokenizer=tiny_tokenizer)


@pytest.fixture
def engram_config() -> EngramConfig:
    """Standard EngramConfig fixture."""
    return EngramConfig(
        target_layers=[2, 15],
        ngram_sizes=[2, 3],
        n_head_per_ngram=4,
        engram_vocab_size_per_ngram=[100, 100],
        seed=42,
    )
