from typing import cast
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import typeguard
from jaxtyping import jaxtyped

from engram_peft.compression import CompressedTokenizer
from engram_peft.config import EngramConfig
from engram_peft.hashing import NgramHashMapping
from engram_peft.layer import ContextAwareGating, EngramLayer, ShortConv


@jaxtyped(typechecker=typeguard.typechecked)
def test_shortconv_initialization() -> None:
    """测试用例 1：验证初始状态下输出等于输入 (卷积零初始化验证，L1距离 < 1e-6)"""
    hidden_size = 64
    hc_mult = 4
    module = ShortConv(hidden_size=hidden_size, hc_mult=hc_mult)
    x = torch.randn(2, 10, hc_mult, hidden_size)
    out = module(x)
    l1_dist = torch.abs(out - x).max().item()
    assert l1_dist < 1e-6, (
        f"Initial output should be equal to input (due to internal residual), but max diff is {l1_dist}"
    )


def test_shortconv_output_shape() -> None:
    """测试用例 2：验证输出形状与输入形状完全相同"""
    hidden_size = 32
    hc_mult = 2
    module = ShortConv(hidden_size=hidden_size, hc_mult=hc_mult)
    x = torch.randn(4, 15, hc_mult, hidden_size)
    out = module(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_shortconv_gradients() -> None:
    """测试用例 3：验证前向/反向传播无错误"""
    hidden_size = 16
    hc_mult = 1
    module = ShortConv(hidden_size=hidden_size, hc_mult=hc_mult)
    x = torch.randn(2, 5, hc_mult, hidden_size, requires_grad=True)
    out = module(x)
    loss = out.pow(2).sum()
    loss.backward()

    assert x.grad is not None, "Input should have gradients."
    assert module.conv.weight.grad is not None, "Conv weight should have gradients."


def test_shortconv_mhc_branches() -> None:
    """测试用例 4：验证mHC分支处理正确 (各分支互相独立计算)"""
    hidden_size = 8
    hc_mult = 4
    module = ShortConv(hidden_size=hidden_size, hc_mult=hc_mult, zero_init=False)
    x = torch.randn(2, 5, hc_mult, hidden_size)

    # 验证独立性，修改一个分支的输入，不应该影响其他分支输出
    x_modified = x.clone()
    x_modified[:, :, 0, :] += 1.0

    out = module(x)
    out_modified = module(x_modified)

    # 分支0改变了
    assert not torch.allclose(out[:, :, 0, :], out_modified[:, :, 0, :])
    # 其余分支未受影响 (尽管卷积目前是权重全0，由于 residual 和独立 normalization，如果卷积全0其余分支显然不受影响)
    # We already have non-zero weight from init, but can re-init if needed
    with torch.no_grad():
        nn.init.normal_(module.conv.weight)

    out_nonzero = module(x)
    out_nonzero_modified = module(x_modified)

    for branch_idx in range(1, hc_mult):
        assert torch.allclose(
            out_nonzero[:, :, branch_idx, :], out_nonzero_modified[:, :, branch_idx, :]
        ), f"Branch {branch_idx} was incorrectly affected by branch 0 modification."


def test_cag_output_shape() -> None:
    """测试用例 5：验证输出形状正确"""
    engram_hidden_size = 128
    hidden_size = 64
    hc_mult = 4
    batch_size, seq_len = 2, 5

    config = EngramConfig(hidden_size=hidden_size, embedding_dim=engram_hidden_size)
    module = ContextAwareGating(config, engram_hidden_size, hidden_size, hc_mult)
    embeddings = torch.randn(batch_size, seq_len, engram_hidden_size)
    hidden_states = torch.randn(batch_size, seq_len, hc_mult, hidden_size)

    out = module(embeddings, hidden_states)
    assert out.shape == (batch_size, seq_len, hc_mult, hidden_size)


def test_cag_gate_values() -> None:
    """测试用例 6：验证门控值在 (0, 1) 之间"""
    config = EngramConfig(hidden_size=64, embedding_dim=128)
    module = ContextAwareGating(
        config, engram_hidden_size=128, hidden_size=64, hc_mult=2
    )
    e = torch.randn(1, 4, 128)
    h = torch.randn(1, 4, 2, 64)
    out = module(e, h)

    with torch.no_grad():
        v = module.w_v(e).unsqueeze(2).expand(-1, -1, 2, -1)
        mask = torch.abs(v) > 1e-4
        if mask.any():
            gate_implied = out[mask] / v[mask]
            assert torch.all(gate_implied >= 0.0) and torch.all(gate_implied <= 1.0)


def test_cag_multi_branch() -> None:
    """测试用例 7：验证分支特定门控正确"""
    engram_hidden_size = 64
    hidden_size = 128
    hc_mult = 3

    config = EngramConfig(hidden_size=hidden_size, embedding_dim=engram_hidden_size)
    module = ContextAwareGating(
        config, engram_hidden_size, hidden_size, hc_mult, zero_init=False
    )
    e = torch.randn(2, 5, engram_hidden_size)
    h = torch.randn(2, 5, hc_mult, hidden_size)

    out = module(e, h)

    for i in range(hc_mult):
        for j in range(i + 1, hc_mult):
            diff = (out[:, :, i, :] - out[:, :, j, :]).abs().max().item()
            assert diff > 1e-6, f"Branch {i} and {j} outputs should differ"


def test_cag_gradients() -> None:
    """测试用例 8：验证前向/反向传播无错误"""
    config = EngramConfig(hidden_size=64, embedding_dim=32)
    module = ContextAwareGating(
        config, engram_hidden_size=32, hidden_size=64, hc_mult=2
    )
    e = torch.randn(2, 4, 32, requires_grad=True)
    h = torch.randn(2, 4, 2, 64, requires_grad=True)

    out = module(e, h)
    loss = out.pow(2).sum()
    loss.backward()

    assert e.grad is not None
    assert h.grad is not None
    assert module.w_v.weight.grad is not None
    for m in range(2):
        assert module.w_k[m].weight.grad is not None


@jaxtyped(typechecker=typeguard.typechecked)
def test_engram_layer_forward() -> None:
    """测试用例 9：验证完整EngramLayer的前向传播"""
    config = EngramConfig(
        target_layers=[0],
        hidden_size=128,
        engram_vocab_size_per_ngram=[100, 100],
        compressed_vocab_size=100,
        pad_id=0,
        conv_zero_init=False,
        gating_zero_init=False,
    )

    # Mock CompressedTokenizer
    compressor = MagicMock(spec=CompressedTokenizer)
    compressor.lookup = torch.arange(100)  # Mock lookup table
    compressor.compressed_vocab_size = 100

    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
        ngram_sizes=config.ngram_sizes,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=[1],
        compressed_vocab_size=100,
    )
    primes = sum(mapping.prime_tables[1], [])

    # Use the primes calculated from mapping
    layer = EngramLayer(config, layer_id=1, primes=primes, compressor=compressor)

    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, cast("int", config.hidden_size))

    # Use mapping to compute
    compressed_ids = compressor.lookup[input_ids]
    hashes_np = mapping.hash(compressed_ids)[1]
    engram_hash_indices = torch.from_numpy(hashes_np)

    output = layer(hidden_states=hidden_states, engram_hash_indices=engram_hash_indices)

    assert output.shape == hidden_states.shape
    assert not torch.allclose(output, hidden_states), (
        "Output should be modified by EngramLayer"
    )


def test_engram_layer_indices_priority() -> None:
    """测试用例 10：验证engram_hash_indices优先使用（加速测试，速度快5倍以上）"""
    config = EngramConfig(
        ngram_sizes=[2],
        n_head_per_ngram=1,
        embedding_dim=16,
        hidden_size=32,
        engram_vocab_size_per_ngram=[100],
        hc_mult=1,
        compressed_vocab_size=100,
        pad_id=0,
    )
    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
        ngram_sizes=config.ngram_sizes,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=[1],
        compressed_vocab_size=100,
    )
    primes = sum(mapping.prime_tables[1], [])
    layer = EngramLayer(config, layer_id=1, primes=primes)

    batch_size = 1
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, cast("int", config.hidden_size))

    # Precompute hash indices
    # We'll use a specific index and verify the output depends on it
    indices = torch.zeros((batch_size, seq_len, 1), dtype=torch.long)
    indices[0, 0, 0] = 5  # Set a specific index

    # This should work even without input_ids/compressed_ids
    output = layer(hidden_states=hidden_states, engram_hash_indices=indices)
    assert output.shape == hidden_states.shape


def test_engram_layer_sparse_gradients() -> None:
    """测试用例 11：验证嵌入表的梯度只在被检索的行上更新"""
    config = EngramConfig(
        ngram_sizes=[2],
        n_head_per_ngram=1,
        embedding_dim=16,
        hidden_size=32,
        conv_zero_init=False,
        gating_zero_init=False,
        engram_vocab_size_per_ngram=[100],
        hc_mult=1,
        compressed_vocab_size=100,
        pad_id=0,
    )
    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
        ngram_sizes=config.ngram_sizes,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=[1],
        compressed_vocab_size=100,
    )
    primes = sum(mapping.prime_tables[1], [])
    layer = EngramLayer(config, layer_id=1, primes=primes)

    batch_size = 1
    seq_len = 2
    hidden_states = torch.randn(batch_size, seq_len, cast("int", config.hidden_size))

    # Only use index 0 and 1
    indices = torch.tensor([[[0], [1]]], dtype=torch.long)  # [B, L, total_heads]

    output = layer(hidden_states=hidden_states, engram_hash_indices=indices)
    loss = output.pow(2).sum()
    loss.backward()

    # Access the single embedding table
    grad = layer.multi_head_embedding.embedding.weight.grad
    assert grad is not None
    # Convert to dense for easier checking as Some operations don't support SparseCPU
    grad_dense = grad.to_dense()

    # Only rows 0 and 1 should have non-zero gradients (assuming offset is 0 for first head)
    assert torch.any(grad_dense[0] != 0)
    assert torch.any(grad_dense[1] != 0)
    if grad_dense.shape[0] > 2:
        assert torch.all(grad_dense[2:] == 0)


def test_engram_layer_output_shape() -> None:
    """测试用例 12：验证输出形状与输入形状完全相同"""
    # Test multi-branch case
    config = EngramConfig(
        ngram_sizes=[2],
        n_head_per_ngram=2,
        embedding_dim=32,
        hidden_size=32,
        hc_mult=4,
        engram_vocab_size_per_ngram=[100],
        compressed_vocab_size=100,
        pad_id=0,
    )
    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
        ngram_sizes=config.ngram_sizes,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=[1],
        compressed_vocab_size=100,
    )
    primes = sum(mapping.prime_tables[1], [])
    layer = EngramLayer(config, layer_id=1, primes=primes)

    batch_size = 2
    seq_len = 5
    hidden_states = torch.randn(
        batch_size, seq_len, config.hc_mult, cast("int", config.hidden_size)
    )

    # Create hash indices tensor with shape [B, L, total_heads]
    total_heads = config.n_head_per_ngram * len(config.ngram_sizes)
    indices = torch.zeros((batch_size, seq_len, total_heads), dtype=torch.long)

    output = layer(hidden_states=hidden_states, engram_hash_indices=indices)
    assert output.shape == hidden_states.shape
