import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
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
            [nn.Linear(hidden_size, hidden_size) for _ in range(32)]
        )

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        return None


class TestWeightTransfer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.hidden_size = 16
        self.base_model = DummyModel(hidden_size=self.hidden_size)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp_dir)

    def test_layer_and_capacity_mapping(self) -> None:
        """Test mapping between different layers and bucket sizes."""
        # 1. Create Source Model
        src_config = EngramConfig(
            target_layers=[0],
            engram_vocab_size_per_ngram=[1000, 1000],
            ngram_sizes=[2, 3],
            seed=42,
            hidden_size=self.hidden_size,
            embedding_dim=128,
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
        )
        src_model = get_engram_model(self.base_model, src_config)  # type: ignore[arg-type]

        # Randomize weights
        with torch.no_grad():
            for param in src_model.engram_layers.parameters():
                param.normal_()

        src_path = Path(self.tmp_dir) / "src"
        src_model.save_pretrained(str(src_path))

        # 2. Create Target Model (Different layers, larger buckets)
        target_config = EngramConfig(
            target_layers=[1],
            engram_vocab_size_per_ngram=[2000, 2000],  # Larger
            ngram_sizes=[2, 3],
            seed=42,
            hidden_size=self.hidden_size,
            embedding_dim=128,
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
        )
        target_model = get_engram_model(self.base_model, target_config)  # type: ignore[arg-type]

        # Perform Transfer
        target_model.load_weights_flexible(
            str(src_path / "engram_weights.pt"), layer_mapping={0: 1}
        )

        # Verify
        src_emb = src_model.engram_layers[
            "0"
        ].multi_head_embedding.embedding.weight.data
        target_emb = target_model.engram_layers[
            "1"
        ].multi_head_embedding.embedding.weight.data

        # Check first N-gram head (2-gram, head 0)
        # Primes will be different because of total vocab size change,
        # but align_embedding_table handles small->large by copying min size.
        src_mapper = src_model.hash_mapping
        target_mapper = target_model.hash_mapping

        src_p = src_mapper.prime_tables[0][0][0]  # Layer 0, N-gram 0 (size 2), Head 0
        target_p = target_mapper.prime_tables[1][0][
            0
        ]  # Layer 1, N-gram 0 (size 2), Head 0

        copy_size = min(src_p, target_p)
        self.assertTrue(torch.allclose(src_emb[:copy_size], target_emb[:copy_size]))

        # Check that target tail is zero (since it's larger)
        if target_p > src_p:
            self.assertTrue(torch.all(target_emb[src_p:target_p] == 0))

    def test_ngram_subset_mapping(self) -> None:
        """Test loading [2, 3] weights into a [2] model."""
        src_config = EngramConfig(
            target_layers=[0],
            ngram_sizes=[2, 3],
            seed=42,
            hidden_size=self.hidden_size,
            embedding_dim=128,
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
        )
        src_model = get_engram_model(self.base_model, src_config)  # type: ignore[arg-type]
        src_path = Path(self.tmp_dir) / "src_subset"
        src_model.save_pretrained(str(src_path))

        target_config = EngramConfig(
            target_layers=[0],
            ngram_sizes=[2],  # Smaller set
            seed=42,
            hidden_size=self.hidden_size,
            embedding_dim=64,  # 1 head * 2 ngram = 128 total -> 1 head * 1 ngram = 64 total
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
        )
        # Note: target_dim_per_head must match.
        # src: 128 / (2*2) = 32 per head.
        # target: 64 / (1*2) = 32 per head. Matches!

        target_model = get_engram_model(self.base_model, target_config)  # type: ignore[arg-type]
        target_model.load_weights_flexible(str(src_path / "engram_weights.pt"))

        # Verify 2-gram weights are copied
        src_emb = src_model.engram_layers[
            "0"
        ].multi_head_embedding.embedding.weight.data
        target_emb = target_model.engram_layers[
            "0"
        ].multi_head_embedding.embedding.weight.data

        # 2nd n-gram (size 3) in src should be ignored.
        src_p_2gram_h0 = src_model.hash_mapping.prime_tables[0][0][0]
        self.assertTrue(
            torch.allclose(src_emb[:src_p_2gram_h0], target_emb[:src_p_2gram_h0])
        )

    def test_corpus_remapping(self) -> None:
        """Test seed remapping using a small corpus."""
        # 1. Source model with seed 0
        src_config = EngramConfig(
            target_layers=[0],
            seed=0,
            hidden_size=self.hidden_size,
            embedding_dim=128,
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
        )
        src_model = get_engram_model(self.base_model, src_config)  # type: ignore[arg-type]
        src_path = Path(self.tmp_dir) / "src_seed0"
        src_model.save_pretrained(str(src_path))

        # 2. Target model with seed 1
        target_config = EngramConfig(
            target_layers=[0],
            seed=1,
            hidden_size=self.hidden_size,
            embedding_dim=128,
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
        )
        target_model = get_engram_model(self.base_model, target_config)  # type: ignore[arg-type]

        # 3. Dummy corpus
        tokens = torch.randint(0, 1000, (100,))

        # 4. Run Remapping
        target_model.remap_from_corpus(tokens, str(src_path / "engram_weights.pt"))

        # 5. Verification: On the same text, the hash indices should now point to same weights
        # Technically hard to verify equality perfectly due to collisions, but we can check if
        # target_emb has non-zero values where tokens hashed.
        target_emb = target_model.engram_layers[
            "0"
        ].multi_head_embedding.embedding.weight.data
        self.assertTrue((target_emb != 0).any())

    def test_cross_tokenizer_remapping(self) -> None:
        """Test remapping between different tokenizers using raw text."""
        from unittest.mock import MagicMock, patch

        # 1. Source setup
        src_config = EngramConfig(
            target_layers=[0],
            seed=42,
            hidden_size=self.hidden_size,
            embedding_dim=128,
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
            tokenizer_name_or_path="source/tokenizer",
        )
        src_model = get_engram_model(self.base_model, src_config)  # type: ignore[arg-type]
        with torch.no_grad():
            src_model.engram_layers["0"].multi_head_embedding.embedding.weight.fill_(
                1.0
            )

        src_path = Path(self.tmp_dir) / "src_cross"
        src_model.save_pretrained(str(src_path))

        # 2. Target setup
        target_config = EngramConfig(
            target_layers=[0],
            seed=42,
            hidden_size=self.hidden_size,
            embedding_dim=128,
            n_head_per_ngram=2,
            enable_tokenizer_compression=False,
            tokenizer_name_or_path="target/tokenizer",
        )
        target_model = get_engram_model(self.base_model, target_config)  # type: ignore[arg-type]
        with torch.no_grad():
            target_model.engram_layers[
                "0"
            ].multi_head_embedding.embedding.weight.zero_()

        # 3. Create parallel Mock Tokenizers
        # Text: "Apple"
        # Source Tokenizer: "Ap", "ple" -> [10, 11]
        # Target Tokenizer: "App", "le" -> [20, 21]

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

        # 4. Patch AutoTokenizer to return our mocks
        from engram_peft import weight_transfer

        with patch(
            "transformers.AutoTokenizer.from_pretrained"
        ) as mock_from_pretrained:
            mock_from_pretrained.side_effect = lambda name, **kwargs: (
                src_mock_tokenizer if "source" in name else target_mock_tokenizer
            )

            # Execute Remapping
            corpus = ["Apple"]
            target_model.remap_from_corpus(corpus, str(src_path / "engram_weights.pt"))

        # 5. Verification
        target_emb = target_model.engram_layers[
            "0"
        ].multi_head_embedding.embedding.weight.data
        # Weights should be transferred (non-zero)
        self.assertTrue((target_emb != 0).any())
        self.assertTrue(torch.allclose(target_emb[target_emb != 0], torch.tensor(1.0)))


if __name__ == "__main__":
    unittest.main()
