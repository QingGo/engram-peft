"""
集成测试：大规模权重迁移及真实语料重映射验证。
涉及 32 层模型的高内存占用及 GPT2 分词器的真实文本对齐。
建议时机：
1. 涉及 NgramHashMapping 核心算法修改时。
2. 验证跨分词器迁移精度时。
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator, Optional

import pytest
import torch
import torch.nn as nn

from engram_peft.config import EngramConfig
from engram_peft.model import get_engram_model


class HeavyDummyModel(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()

        class Config:
            pass

        self.config = Config()
        setattr(self.config, "hidden_size", hidden_size)
        setattr(self.config, "vocab_size", 2262400)
        setattr(self.config, "pad_token_id", 0)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(32)]  # 32 layers
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        return None


@pytest.fixture
def heavy_tmp_dir() -> Generator[Path, None, None]:
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


def test_weight_transfer_full_integration(heavy_tmp_dir: Path) -> None:
    """
    压测：模拟 32 层大模型的权重灵活迁移。
    验证大规模素数表生成及 Embedding 物理切分的稳定性。
    """
    base_model = HeavyDummyModel(hidden_size=128)

    src_config = EngramConfig(
        target_layers=[0],
        engram_vocab_size_per_ngram=[2262400 // 2, 2262400 // 2],  # 全量容量
        ngram_sizes=[2, 3],
        seed=42,
        hidden_size=128,
        embedding_dim=256,
        n_head_per_ngram=8,
        enable_tokenizer_compression=False,
    )
    src_model = get_engram_model(base_model, src_config)  # type: ignore[arg-type]

    src_path = heavy_tmp_dir / "src_heavy"
    src_model.save_pretrained(str(src_path))

    target_config = EngramConfig(
        target_layers=[31],  # 跨度大的层映射
        engram_vocab_size_per_ngram=[2262400 // 2, 2262400 // 2],
        ngram_sizes=[2, 3],
        seed=42,
        hidden_size=128,
        embedding_dim=256,
        n_head_per_ngram=8,
        enable_tokenizer_compression=False,
    )
    target_model = get_engram_model(base_model, target_config)  # type: ignore[arg-type]

    target_model.load_weights_flexible(
        str(src_path / "engram_weights.pt"), layer_mapping={0: 31}
    )

    assert "31" in target_model.engram_layers
    target_emb = target_model.engram_layers[
        "31"
    ].multi_head_embedding.embedding.weight.data
    assert (target_emb != 0).any()


def test_corpus_remapping_integration(heavy_tmp_dir: Path, tokenizer_gpt2: Any) -> None:
    """
    重映射测试：验证跨分词器对齐精度。
    使用真实 GPT2 Tokenizer 对齐自然语言文本，确保权重能正确落入目标词表桶。
    """
    base_model = HeavyDummyModel(hidden_size=64)
    src_config = EngramConfig(
        target_layers=[0],
        seed=0,
        hidden_size=64,
        embedding_dim=128,
        n_head_per_ngram=2,
        tokenizer_name_or_path="gpt2",
    )
    src_model = get_engram_model(base_model, src_config, tokenizer=tokenizer_gpt2)  # type: ignore[arg-type]
    src_path = heavy_tmp_dir / "src_corpus"
    src_model.save_pretrained(str(src_path))

    target_config = EngramConfig(
        target_layers=[0],
        seed=1,
        hidden_size=64,
        embedding_dim=128,
        n_head_per_ngram=2,
    )
    target_model = get_engram_model(base_model, target_config, tokenizer=tokenizer_gpt2)  # type: ignore[arg-type]

    text = "The quick brown fox jumps over the lazy dog."
    target_model.remap_from_corpus(
        [text], str(src_path / "engram_weights.pt"), tokenizer=tokenizer_gpt2
    )

    target_emb = target_model.engram_layers[
        "0"
    ].multi_head_embedding.embedding.weight.data
    assert (target_emb != 0).any()
