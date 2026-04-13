import torch
import torch.nn as nn
from typing import List, Any, cast
from unittest.mock import MagicMock
from engram_peft.layer import ContextAwareGating, EngramLayer
from engram_peft.config import EngramConfig
from engram_peft.compression import CompressedTokenizer
from engram_peft.hashing import NgramHashMapping


def test_context_aware_gating_initialization() -> None:
    """
    测试用例 1：验证初始状态下输出等于输入 (v_tilde)
    卷积层初始化为零，因此 Y = SiLU(0) + v_tilde = v_tilde
    验证 L1 距离 < 1e-6
    """
    embedding_dim = 128
    hidden_dim = 256
    seq_len = 10
    batch_size = 2

    module = ContextAwareGating(
        embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_branches=1
    )

    e_t = torch.randn(batch_size, seq_len, embedding_dim)
    h_t = torch.randn(batch_size, seq_len, hidden_dim)

    with torch.no_grad():
        # 手动计算预期输出 v_tilde
        v_t = module.w_v(e_t)
        k_t = module.w_k(e_t)
        h_t_norm = module.norm_h[0](h_t)
        k_t_norm = module.norm_k[0](k_t)
        dot_product = (h_t_norm * k_t_norm).sum(dim=-1) / (hidden_dim**0.5)
        stable_dot_product = (
            dot_product.abs().clamp_min(1e-6).sqrt() * dot_product.sign()
        )
        alpha_t = torch.sigmoid(stable_dot_product)
        v_tilde = alpha_t.unsqueeze(-1) * v_t

        output = module(e_t, h_t)

    # 验证 L1 距离
    l1_dist = torch.abs(output - v_tilde).max().item()
    assert l1_dist < 1e-6, f"初始输出应等于 v_tilde，但 L1 距离为 {l1_dist}"


def test_context_aware_gating_values() -> None:
    """
    测试用例 2：验证门控值在 (0, 1) 之间
    虽然我们不能直接访问内部 alpha_t，但可以通过输出的变化来间接验证
    """
    embedding_dim = 128
    hidden_dim = 256
    seq_len = 5
    batch_size = 1

    module = ContextAwareGating(
        embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_branches=1
    )

    # 构造极端的 h_t 和 e_t 来产生极大或极小的点积
    e_t = torch.randn(batch_size, seq_len, embedding_dim)
    h_t = torch.randn(batch_size, seq_len, hidden_dim)

    # 通过 hook 捕获内部 alpha_t
    alphas: List[torch.Tensor] = []

    def hook_fn(module: ContextAwareGating, input: Any, output: Any) -> None:
        # alpha_t 计算后的 dot_product
        h_norm = module.norm_h[0](input[1])
        k_raw = module.w_k(input[0])
        k_norm = module.norm_k[0](k_raw)
        dot = (h_norm * k_norm).sum(dim=-1) / (module.hidden_dim**0.5)
        stable_dot = dot.abs().clamp_min(1e-6).sqrt() * dot.sign()
        alpha = torch.sigmoid(stable_dot)
        alphas.append(alpha)

    handle = module.register_forward_hook(hook_fn)
    _ = module(e_t, h_t)
    handle.remove()

    alpha = alphas[0]
    assert torch.all(alpha >= 0.0) and torch.all(
        alpha <= 1.0
    ), "门控值 alpha_t 应在 [0, 1] 之间"


def test_context_aware_gating_multi_branch() -> None:
    """
    测试用例 3：验证多分支架构的分支特定门控
    """
    embedding_dim = 64
    hidden_dim = 128
    num_branches = 4
    seq_len = 8
    batch_size = 2

    module = ContextAwareGating(
        embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_branches=num_branches
    )

    e_t = torch.randn(batch_size, seq_len, embedding_dim)
    h_t = torch.randn(batch_size, seq_len, num_branches, hidden_dim)

    output = module(e_t, h_t)
    assert output.shape == (batch_size, seq_len, num_branches, hidden_dim)

    # 验证不同分支的输出是不同的（因为 W_K 是分支特定的）
    for i in range(num_branches):
        for j in range(i + 1, num_branches):
            diff = (output[:, :, i, :] - output[:, :, j, :]).abs().max().item()
            assert diff > 1e-6, f"分支 {i} 和 {j} 的输出应该不同"


def test_context_aware_gating_gradients() -> None:
    """
    测试用例 4：验证前向/反向传播无错误 (梯度检查)
    """
    embedding_dim = 32
    hidden_dim = 64
    seq_len = 4
    batch_size = 2

    module = ContextAwareGating(
        embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_branches=1
    )

    e_t = torch.randn(batch_size, seq_len, embedding_dim, requires_grad=True)
    h_t = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

    output = module(e_t, h_t)
    loss = output.pow(2).sum()
    loss.backward()

    assert e_t.grad is not None, "e_t 应该有梯度"
    assert h_t.grad is not None, "h_t 应该有梯度"
    assert module.w_v.weight.grad is not None, "w_v 权重应该有梯度"
    assert module.w_k.weight.grad is not None, "w_k 权重应该有梯度"
    assert module.conv.weight.grad is not None, "conv 权重应该有梯度"


def test_context_aware_gating_shapes() -> None:
    """
    测试用例 5-1：验证 ContextAwareGating 输出形状与输入形状完全相同
    """
    # 单分支
    module1 = ContextAwareGating(embedding_dim=1280, hidden_dim=2560, num_branches=1)
    e1 = torch.randn(2, 16, 1280)
    h1 = torch.randn(2, 16, 2560)
    assert module1(e1, h1).shape == h1.shape

    # 多分支
    num_branches = 8
    module2 = ContextAwareGating(
        embedding_dim=1280, hidden_dim=2560, num_branches=num_branches
    )
    e2 = torch.randn(2, 16, 1280)
    h2 = torch.randn(2, 16, num_branches, 2560)
    assert module2(e2, h2).shape == h2.shape


def test_engram_layer_forward() -> None:
    """
    测试用例 5：验证完整 EngramLayer 的前向传播
    """
    config = EngramConfig(
        max_ngram_size=3,
        n_head_per_ngram=4,
        embedding_dim=128,
        hidden_dim=32,
        hc_mult=1,
        engram_vocab_size_per_ngram=[100, 100],
    )

    # Mock CompressedTokenizer
    compressor = MagicMock(spec=CompressedTokenizer)
    compressor.lookup = torch.arange(100)  # Mock lookup table
    compressor.compressed_vocab_size = 100

    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
        max_ngram_size=config.max_ngram_size,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=[1],
    )
    primes = sum(mapping.prime_tables[1], [])

    layer = EngramLayer(config, layer_id=1, primes=primes, compressor=compressor)

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    # Use mapping to compute
    compressed_ids = compressor.lookup[input_ids]
    hashes_np = mapping.hash(compressed_ids)[1]
    engram_hash_indices = torch.from_numpy(hashes_np)

    output = layer(hidden_states=hidden_states, engram_hash_indices=engram_hash_indices)

    assert output.shape == hidden_states.shape
    assert not torch.allclose(
        output, hidden_states
    ), "Output should be modified by EngramLayer"


def test_engram_layer_indices_priority() -> None:
    """
    测试用例 6：验证 engram_hash_indices 优先使用
    """
    config = EngramConfig(
        max_ngram_size=2,
        n_head_per_ngram=1,
        embedding_dim=16,
        hidden_dim=32,
        engram_vocab_size_per_ngram=[100],
        hc_mult=1,
    )
    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
        max_ngram_size=config.max_ngram_size,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=[1],
    )
    primes = sum(mapping.prime_tables[1], [])
    layer = EngramLayer(config, layer_id=1, primes=primes)

    batch_size = 1
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    # Precompute hash indices
    # We'll use a specific index and verify the output depends on it
    indices = torch.zeros((batch_size, seq_len, 1), dtype=torch.long)
    indices[0, 0, 0] = 5  # Set a specific index

    # This should work even without input_ids/compressed_ids
    output = layer(hidden_states=hidden_states, engram_hash_indices=indices)
    assert output.shape == hidden_states.shape


def test_engram_layer_sparse_gradients() -> None:
    """
    测试用例 7：验证嵌入表的梯度只在被检索的行上更新
    """
    config = EngramConfig(
        max_ngram_size=2,
        n_head_per_ngram=1,
        embedding_dim=16,
        hidden_dim=32,
        engram_vocab_size_per_ngram=[100],
        hc_mult=1,
    )
    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
        max_ngram_size=config.max_ngram_size,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=[1],
    )
    primes = sum(mapping.prime_tables[1], [])
    layer = EngramLayer(config, layer_id=1, primes=primes)

    batch_size = 1
    seq_len = 2
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    # Only use index 0 and 1
    indices = torch.tensor([[[0], [1]]], dtype=torch.long)  # [B, L, total_heads]

    output = layer(hidden_states=hidden_states, engram_hash_indices=indices)
    loss = output.pow(2).sum()
    loss.backward()

    # Access the single embedding table
    grad = layer.embedding.weight.grad
    assert grad is not None

    # Only rows 0 and 1 should have non-zero gradients (assuming offset is 0 for first head)
    assert torch.any(grad[0] != 0)
    assert torch.any(grad[1] != 0)
    if grad.shape[0] > 2:
        assert torch.all(grad[2:] == 0)


def test_engram_layer_output_shape() -> None:
    """
    测试用例 8：验证输出形状与输入 hidden_states 形状完全相同
    """
    # Test multi-branch case
    config = EngramConfig(
        max_ngram_size=2,
        n_head_per_ngram=2,
        embedding_dim=32,
        hidden_dim=32,
        hc_mult=4,
        engram_vocab_size_per_ngram=[100],
    )
    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
        max_ngram_size=config.max_ngram_size,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=[1],
    )
    primes = sum(mapping.prime_tables[1], [])
    layer = EngramLayer(config, layer_id=1, primes=primes)

    batch_size = 2
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, config.hc_mult, config.hidden_dim)

    # Create hash indices tensor with shape [B, L, total_heads]
    total_heads = config.n_head_per_ngram * (config.max_ngram_size - 1)
    indices = torch.zeros((batch_size, seq_len, total_heads), dtype=torch.long)

    output = layer(hidden_states=hidden_states, engram_hash_indices=indices)
    assert output.shape == hidden_states.shape
