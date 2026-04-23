import os
import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from engram_peft.config import EngramConfig
from engram_peft.layer import EngramLayer
from engram_peft.model import get_engram_model


# --- Setup Dummy Base Model ---
class DummyModel(nn.Module):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.config = SimpleNamespace()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = 32000
        self.config.pad_token_id = 0
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(12)]
        )

    def forward(
        self, input_ids: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor | None:
        return None


def main() -> None:
    base_model = DummyModel()
    save_dir = "temp_cross_tokenizer_demo"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Simulate a Source model with Tokenizer A
    src_config = EngramConfig(
        target_layers=[0],
        seed=42,
        embedding_dim=128,
        n_head_per_ngram=2,
        tokenizer_name_or_path="tokenizer_a",
        enable_tokenizer_compression=False,
    )
    src_model = get_engram_model(base_model, src_config)

    # Fill with identifiable weights
    src_layer = src_model.engram_layers["0"]
    if not isinstance(src_layer, EngramLayer):
        raise TypeError("Expected EngramLayer")
    with torch.no_grad():
        src_layer.multi_head_embedding.embedding.weight.fill_(7.7)

    src_model.save_pretrained(save_dir)
    print(f"Source model (Tokenizer A) saved to {save_dir}")

    # 2. Create Target Model with Tokenizer B
    target_config = EngramConfig(
        target_layers=[0],
        seed=42,
        embedding_dim=128,
        n_head_per_ngram=2,
        tokenizer_name_or_path="tokenizer_b",
        enable_tokenizer_compression=False,
    )
    target_model = get_engram_model(base_model, target_config)
    target_layer = target_model.engram_layers["0"]
    if not isinstance(target_layer, EngramLayer):
        raise TypeError("Expected EngramLayer")
    with torch.no_grad():
        target_layer.multi_head_embedding.embedding.weight.zero_()

    print("Target model (Tokenizer B) initialized with zero weights.")

    # Tokenizer A: "Machine", " Learning" -> [100, 101]
    # Tokenizer B: "Mach", "ine", " Learning" -> [200, 201, 202]
    src_mock = MagicMock()
    src_mock.side_effect = lambda t, **kwargs: {
        "input_ids": [100, 101],
        "offset_mapping": [(0, 7), (7, 16)],
    }

    target_mock = MagicMock()
    target_mock.side_effect = lambda t, **kwargs: {
        "input_ids": [200, 201, 202],
        "offset_mapping": [(0, 4), (4, 7), (7, 16)],
    }

    print("Executing Cross-Tokenizer Migration using parallel alignment...")

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.side_effect = lambda name, **kwargs: (
            src_mock if "a" in name else target_mock
        )

        # Use raw text corpus to bridge the two tokenizers
        # 3. Perform Cross-Tokenizer Remapping
        corpus_text = ["Machine Learning"]
        target_model.remap_from_corpus(corpus_text, save_dir)

    # 4. Verification
    target_emb = target_layer.multi_head_embedding.embedding.weight.data
    remapped_indices = torch.any(target_emb == 7.7).item()
    print(f"Weights successfully migrated across tokenizers: {remapped_indices}")

    # Cleanup
    shutil.rmtree(save_dir)


if __name__ == "__main__":
    main()
