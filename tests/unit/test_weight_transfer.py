import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from engram_peft.config import EngramConfig
from engram_peft.model import get_engram_model


class DummyModel(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()

        class Config:
            pass

        self.config = Config()
        setattr(self.config, "hidden_size", hidden_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size) for _ in range(8)
            ]  # Reduced layer count for speed
        )

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        return None


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def base_model() -> DummyModel:
    return DummyModel(hidden_size=16)


def test_layer_and_capacity_mapping(tmp_dir: Path, base_model: DummyModel) -> None:
    """Test mapping between different layers and bucket sizes."""
    src_config = EngramConfig(
        target_layers=[0],
        engram_vocab_size_per_ngram=[100, 100],  # Tiny vocab for speed
        ngram_sizes=[2, 3],
        seed=42,
        hidden_size=16,
        embedding_dim=64,
        n_head_per_ngram=2,
        enable_tokenizer_compression=False,
    )
    src_model = get_engram_model(base_model, src_config)  # type: ignore[arg-type]
    with torch.no_grad():
        for param in src_model.engram_layers.parameters():
            param.normal_()

    src_path = tmp_dir / "src"
    src_model.save_pretrained(str(src_path))

    target_config = EngramConfig(
        target_layers=[1],
        engram_vocab_size_per_ngram=[200, 200],  # Larger but still small
        ngram_sizes=[2, 3],
        seed=42,
        hidden_size=16,
        embedding_dim=64,
        n_head_per_ngram=2,
        enable_tokenizer_compression=False,
    )
    target_model = get_engram_model(base_model, target_config)  # type: ignore[arg-type]
    target_model.load_weights_flexible(
        str(src_path / "engram_weights.pt"), layer_mapping={0: 1}
    )

    src_emb = src_model.engram_layers["0"].multi_head_embedding.embedding.weight.data
    target_emb = target_model.engram_layers[
        "1"
    ].multi_head_embedding.embedding.weight.data

    src_p = src_model.hash_mapping.prime_tables[0][0][0]
    target_p = target_model.hash_mapping.prime_tables[1][0][0]
    copy_size = min(src_p, target_p)
    assert torch.allclose(src_emb[:copy_size], target_emb[:copy_size])


def test_ngram_subset_mapping(tmp_dir: Path, base_model: DummyModel) -> None:
    """Test loading [2, 3] weights into a [2] model."""
    src_config = EngramConfig(
        target_layers=[0],
        ngram_sizes=[2, 3],
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=16,
        embedding_dim=64,
        n_head_per_ngram=2,
        enable_tokenizer_compression=False,
    )
    src_model = get_engram_model(base_model, src_config)  # type: ignore[arg-type]
    src_path = tmp_dir / "src_subset"
    src_model.save_pretrained(str(src_path))

    target_config = EngramConfig(
        target_layers=[0],
        ngram_sizes=[2],
        engram_vocab_size_per_ngram=[100],
        hidden_size=16,
        embedding_dim=32,  # 1 ngram * 2 heads * 16 per head = 32
        n_head_per_ngram=2,
        enable_tokenizer_compression=False,
    )
    target_model = get_engram_model(base_model, target_config)  # type: ignore[arg-type]
    target_model.load_weights_flexible(str(src_path / "engram_weights.pt"))

    src_emb = src_model.engram_layers["0"].multi_head_embedding.embedding.weight.data
    target_emb = target_model.engram_layers[
        "0"
    ].multi_head_embedding.embedding.weight.data
    src_p_2gram_h0 = src_model.hash_mapping.prime_tables[0][0][0]
    assert torch.allclose(src_emb[:src_p_2gram_h0], target_emb[:src_p_2gram_h0])


def test_corpus_remapping(tmp_dir: Path, base_model: DummyModel) -> None:
    """Test seed remapping using a small corpus."""
    src_config = EngramConfig(
        target_layers=[0],
        seed=0,
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=16,
        embedding_dim=64,
        n_head_per_ngram=2,
        enable_tokenizer_compression=False,
    )
    src_model = get_engram_model(base_model, src_config)  # type: ignore[arg-type]
    src_path = tmp_dir / "src_seed0"
    src_model.save_pretrained(str(src_path))

    target_config = EngramConfig(
        target_layers=[0],
        seed=1,
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=16,
        embedding_dim=64,
        n_head_per_ngram=2,
        enable_tokenizer_compression=False,
    )
    target_model = get_engram_model(base_model, target_config)  # type: ignore[arg-type]

    tokens = torch.randint(0, 100, (10,))
    target_model.remap_from_corpus(tokens, str(src_path / "engram_weights.pt"))

    target_emb = target_model.engram_layers[
        "0"
    ].multi_head_embedding.embedding.weight.data
    assert (target_emb != 0).any()


def test_cross_tokenizer_remapping(tmp_dir: Path, base_model: DummyModel) -> None:
    """Test remapping between different tokenizers using raw text."""
    src_config = EngramConfig(
        target_layers=[0],
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=16,
        embedding_dim=64,
        n_head_per_ngram=1,  # simplified
        enable_tokenizer_compression=False,
        tokenizer_name_or_path="source/tokenizer",
    )
    src_model = get_engram_model(base_model, src_config)  # type: ignore[arg-type]
    with torch.no_grad():
        src_model.engram_layers["0"].multi_head_embedding.embedding.weight.fill_(1.0)

    src_path = tmp_dir / "src_cross"
    src_model.save_pretrained(str(src_path))

    target_config = EngramConfig(
        target_layers=[0],
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=16,
        embedding_dim=64,
        n_head_per_ngram=1,
        enable_tokenizer_compression=False,
        tokenizer_name_or_path="target/tokenizer",
    )
    target_model = get_engram_model(base_model, target_config)  # type: ignore[arg-type]
    with torch.no_grad():
        target_model.engram_layers["0"].multi_head_embedding.embedding.weight.zero_()

    src_mock_tokenizer = MagicMock()
    src_mock_tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": [10, 11],
        "offset_mapping": [(0, 2), (2, 5)],
    }
    target_mock_tokenizer = MagicMock()
    target_mock_tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": [20, 21],
        "offset_mapping": [(0, 3), (3, 5)],
    }

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.side_effect = lambda name, **kwargs: (
            src_mock_tokenizer if "source" in name else target_mock_tokenizer
        )
        target_model.remap_from_corpus(["Apple"], str(src_path / "engram_weights.pt"))

    target_emb = target_model.engram_layers[
        "0"
    ].multi_head_embedding.embedding.weight.data
    assert (target_emb != 0).any()
    assert torch.allclose(target_emb[target_emb != 0], torch.tensor(1.0))
