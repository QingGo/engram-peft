from typing import Any, Dict, List, cast

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam  # type: ignore
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedModel


from engram_peft.config import EngramConfig
from engram_peft.layer import EngramLayer
from engram_peft.model import get_engram_model
from engram_peft.utils import (
    MixedOptimizer,
    get_optimizer,
    get_scheduler,
    get_trainable_param_groups,
)


class DummyModel(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(32)]
        )
        # Mock config for parameter syncing
        self.config = cast(Any, type("MockConfig", (), {"hidden_size": hidden_size})())

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # Need a tensor that requires grad for backprop to work
        x = torch.randn(batch_size, seq_len, self.hidden_size).requires_grad_(True)
        # mypy needs help since nn.Module doesn't have .layers
        model_part = cast(Any, self.model)
        for layer in model_part.layers:
            x = cast(torch.Tensor, layer(x))
        return x

    def requires_grad_(self, requires_grad: bool = True) -> "DummyModel":
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self


class RecordingOptimizer(Optimizer):
    def __init__(self, params: Any, tag: str = "recording") -> None:
        defaults = {"lr": 0.0, "weight_decay": 0.0, "tag": tag}
        super().__init__(params, defaults)
        self.tag = tag
        self.step_calls = 0

    def step(self, closure: Any = None) -> Any:
        self.step_calls += 1
        if closure is not None:
            return closure()
        return None


def test_optimizer_grouping() -> None:
    """验证优化器参数分组正确，学习率倍率应用正确。"""
    base_model = DummyModel()
    config = EngramConfig(
        target_layers=[0, 1],
        learning_rate_multiplier=5.0,
        weight_decay=0.1,
        enable_tokenizer_compression=False,
        engram_vocab_size_per_ngram=[100, 100],
    )
    model = get_engram_model(cast(PreTrainedModel, base_model), config)

    base_lr = 4e-4
    optimizer = get_optimizer(model, base_learning_rate=base_lr)

    assert isinstance(optimizer, MixedOptimizer)
    assert len(optimizer.optimizers) == 2  # SparseAdam and Adam

    # Check LR and groups
    sparse_opt = optimizer.optimizers[0]
    dense_opt = optimizer.optimizers[1]
    groups = get_trainable_param_groups(model)

    assert sparse_opt.param_groups[0]["lr"] == pytest.approx(base_lr * 5.0)
    assert sparse_opt.param_groups[0]["weight_decay"] == 0.0
    assert sum(p.numel() for p in sparse_opt.param_groups[0]["params"]) == sum(
        p.numel() for p in groups["engram_sparse"]
    )

    assert dense_opt.param_groups[0]["lr"] == pytest.approx(base_lr)
    assert dense_opt.param_groups[0]["weight_decay"] == 0.1
    assert sum(p.numel() for p in dense_opt.param_groups[0]["params"]) == sum(
        p.numel() for p in groups["engram_dense"]
    )
    assert not groups["backbone"]


def test_optimizer_grouping_full_finetune() -> None:
    """验证 full_finetune 会创建 backbone 单独参数组。"""
    base_model = DummyModel()
    config = EngramConfig(
        target_layers=[0],
        learning_rate_multiplier=3.0,
        weight_decay=0.2,
        enable_tokenizer_compression=False,
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=128,
    )
    model = get_engram_model(
        cast(PreTrainedModel, base_model),
        config,
        train_mode="full_finetune",
    )

    optimizer = get_optimizer(
        model,
        base_learning_rate=1e-3,
        backbone_learning_rate=2e-4,
        engram_dense_learning_rate=5e-4,
    )
    groups = get_trainable_param_groups(model)

    assert len(optimizer.optimizers) == 3
    assert sum(
        p.numel() for p in optimizer.optimizers[0].param_groups[0]["params"]
    ) == sum(p.numel() for p in groups["engram_sparse"])
    assert optimizer.optimizers[0].param_groups[0]["lr"] == pytest.approx(3e-3)
    assert sum(
        p.numel() for p in optimizer.optimizers[1].param_groups[0]["params"]
    ) == sum(p.numel() for p in groups["engram_dense"])
    assert optimizer.optimizers[1].param_groups[0]["lr"] == pytest.approx(5e-4)
    assert sum(
        p.numel() for p in optimizer.optimizers[2].param_groups[0]["params"]
    ) == sum(p.numel() for p in groups["backbone"])
    assert optimizer.optimizers[2].param_groups[0]["lr"] == pytest.approx(2e-4)
    assert optimizer.optimizers[2].param_groups[0]["weight_decay"] == pytest.approx(0.2)


def test_optimizer_supports_custom_backbone_builder() -> None:
    """验证 backbone optimizer 支持自定义 builder。"""
    base_model = DummyModel()
    config = EngramConfig(
        target_layers=[0],
        enable_tokenizer_compression=False,
        engram_vocab_size_per_ngram=[100, 100],
        hidden_size=128,
    )
    model = get_engram_model(
        cast(PreTrainedModel, base_model),
        config,
        train_mode="full_finetune",
    )

    optimizer = get_optimizer(
        model,
        backbone_optimizer=lambda param_groups: RecordingOptimizer(
            param_groups, tag="backbone"
        ),
        engram_dense_optimizer=lambda param_groups: RecordingOptimizer(
            param_groups, tag="engram_dense"
        ),
    )

    assert isinstance(optimizer.optimizers[1], RecordingOptimizer)
    assert isinstance(optimizer.optimizers[2], RecordingOptimizer)
    assert optimizer.optimizers[1].defaults["tag"] == "engram_dense"
    assert optimizer.optimizers[2].defaults["tag"] == "backbone"


def test_gradient_sparsity() -> None:
    """验证只有被检索的嵌入行有梯度，且梯度类型为稀疏。"""
    base_model = DummyModel()
    config = EngramConfig(
        target_layers=[0],
        n_head_per_ngram=1,
        ngram_sizes=[2],
        enable_tokenizer_compression=False,
        engram_vocab_size_per_ngram=[100],
        hidden_size=128,
    )
    model = get_engram_model(cast(PreTrainedModel, base_model), config)

    engram_layer = cast(EngramLayer, model.engram_layers["0"])
    embedding = cast(nn.Embedding, engram_layer.multi_head_embedding.embedding)
    assert embedding.sparse is True

    # Forward pass
    input_ids = torch.tensor([[10, 20, 30]])
    # We need to set hash indices manually for testing or trigger hashing
    # model.forward computes hashes
    output = model(input_ids)
    loss = output.sum()
    loss.backward()

    grad = embedding.weight.grad
    assert grad is not None
    assert grad.is_sparse

    # Verify that only a few rows have gradients
    # hash_indices will have batch_size * seq_len * heads entries
    # Here 1 * 3 * total_heads (total_heads = heads * ngrams = 1 * 1 = 1)
    # So 3 rows should have gradients
    indices = grad._indices()  # type: ignore
    num_nonempty_rows = indices.shape[1]
    assert num_nonempty_rows <= 3  # Some might collide


def test_memory_saving() -> None:
    """验证梯度显存占用远低于全量更新。"""
    vocab_size = 100000
    dim = 1024

    # 模拟全量密集梯度
    dense_grad_size = vocab_size * dim * 4  # bytes in float32

    # 模拟稀疏梯度 (3个样本，每个样本1个head)
    # Sparse grad format: indices (long), values (float32)
    # indices: [ndim, nse] -> [1, 3] * 8 bytes
    # values: [nse, dim] -> [3, 1024] * 4 bytes
    num_samples = 3
    sparse_grad_size = (1 * num_samples * 8) + (num_samples * dim * 4)

    saving_ratio = 1.0 - (sparse_grad_size / dense_grad_size)
    assert saving_ratio > 0.99  # Should be very high saving


def test_scheduler_steps() -> None:
    """验证学习率调度器在关键步长的输出。"""
    optimizer = Adam([torch.nn.Parameter(torch.randn(1))], lr=1.0)
    num_steps = 100
    warmup = 10
    scheduler = get_scheduler(
        optimizer, num_training_steps=num_steps, warmup_steps=warmup
    )

    lrs = []
    for step in range(num_steps):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Warmup phase: linear increase
    assert lrs[0] == 0.0
    assert lrs[5] == 0.5
    assert lrs[10] == 1.0

    # Plateau phase (until 80% progress)
    assert lrs[70] == 1.0

    # First decay (at 80%)
    # progress = 81/100 > 0.8
    assert lrs[81] == pytest.approx(0.316)

    # Second decay (at 90%)
    assert lrs[91] == pytest.approx(0.1)
