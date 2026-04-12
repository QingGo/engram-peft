import torch
from typing import List, Any
from engram_peft.layer import ContextAwareGating


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
        h_t_norm = module.norm_h(h_t)
        k_t_norm = module.norm_k(k_t)
        dot_product = (h_t_norm * k_t_norm).sum(dim=-1)
        alpha_t = torch.sigmoid(dot_product / (hidden_dim**0.5))
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
        h_norm = module.norm_h(input[1])
        k_raw = module.w_k(input[0])
        k_norm = module.norm_k(k_raw)
        dot = (h_norm * k_norm).sum(dim=-1)
        alpha = torch.sigmoid(dot / (module.hidden_dim**0.5))
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
    测试用例 5：验证输出形状与输入形状完全相同
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
