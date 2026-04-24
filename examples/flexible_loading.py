# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none

from dotenv import load_dotenv

load_dotenv()

import os
import shutil
from types import SimpleNamespace
from typing import Any, override

import torch
import torch.nn as nn

from engram_peft.config import EngramConfig
from engram_peft.layer import EngramLayer
from engram_peft.model import get_engram_model


# --- Setup Dummy Base Model ---
class DummyModel(nn.Module):
    config: Any
    model: nn.Module

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.config = SimpleNamespace()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = 32000
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(12)]
        )

    @override
    def forward(
        self, input_ids: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor | None:
        return None


def main() -> None:
    base_model = DummyModel()
    save_dir = "temp_flexible_demo"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Simulate a Third-Party model (Source)
    # Different layers (0, 1) and small bucket size
    src_config = EngramConfig(
        target_layers=[0, 1],
        engram_vocab_size_per_ngram=[500, 500],
        ngram_sizes=[2, 3],
        seed=42,
        embedding_dim=128,
        n_head_per_ngram=2,
        enable_tokenizer_compression=False,
    )
    src_model = get_engram_model(base_model, src_config)

    # Randomize weights for demo
    for param in src_model.engram_layers.parameters():
        param.data.normal_(std=0.02)

    src_model.save_pretrained(save_dir)
    print(f"Source model (Layers 0, 1) saved to {save_dir}")

    # 2. Create Target Model with Different Config
    # New layers (5, 6), larger bucket size (1000),
    # and different branches (hc_mult: 8 instead of default 4)
    target_config = EngramConfig(
        target_layers=[5, 6],
        engram_vocab_size_per_ngram=[1000, 1000],
        ngram_sizes=[2, 3],
        seed=42,
        embedding_dim=128,
        n_head_per_ngram=2,
        hc_mult=8,
        enable_tokenizer_compression=False,
        gating_zero_init=True,
    )
    target_model = get_engram_model(base_model, target_config)
    print(
        "Target model (Layers 5, 6) initialized with larger buckets and more branches."
    )

    # 3. Perform Flexible Loading (into model with different config)
    target_model.load_weights_flexible(
        save_dir,
        layer_mapping={0: 5, 1: 6},
        reuse_structural=False,  # Gating/Conv will be zero-initialized
    )

    # 4. Verification
    src_layer0 = src_model.engram_layers["0"]
    target_layer5 = target_model.engram_layers["5"]
    if not isinstance(src_layer0, EngramLayer) or not isinstance(
        target_layer5, EngramLayer
    ):
        raise TypeError("Expected EngramLayer")

    src_emb_0 = src_layer0.multi_head_embedding.embedding.weight.data
    target_emb_5 = target_layer5.multi_head_embedding.embedding.weight.data

    # Check if weights were copied (first prime-sized block)
    src_p = src_model.hash_mapping.prime_tables[0][0][0]
    is_match = torch.allclose(src_emb_0[:src_p], target_emb_5[:src_p])
    print(f"Embedding weights transfer (Layer 0->5) successful: {is_match}")

    # Check if gating is zero-init
    gating_weight = target_layer5.gating.w_v.weight
    is_zero = torch.all(gating_weight == 0).item()
    print(f"Gating weights zero-initialized as requested: {is_zero}")

    # Cleanup
    shutil.rmtree(save_dir)


if __name__ == "__main__":
    main()
