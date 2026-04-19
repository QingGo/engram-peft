from typing import Any

import pytest
import torch
import torch.nn as nn

from engram_peft.utils import (
    apply_group_wise_clipping,
    compute_grad_norm,
    compute_telemetry_stats,
)


def test_compute_grad_norm_dense() -> None:
    p1 = nn.Parameter(torch.ones(10))
    p1.grad = torch.ones(10) * 0.5
    p2 = nn.Parameter(torch.ones(5))
    p2.grad = torch.ones(5) * 2.0

    # norm1 = sqrt(10 * 0.5^2) = sqrt(2.5)
    # norm2 = sqrt(5 * 2^2) = sqrt(20)
    # total = sqrt(2.5 + 20) = sqrt(22.5) approx 4.7434
    expected = torch.sqrt(torch.tensor(22.5))
    actual = compute_grad_norm([p1, p2])
    assert actual is not None
    assert torch.allclose(actual, expected)


def test_compute_grad_norm_sparse() -> None:
    # Sparse grad
    p = nn.Parameter(torch.ones(10, 2))
    indices = torch.tensor([[0, 1], [0, 1]])
    values = torch.tensor([1.0, 2.0])
    p.grad = torch.sparse_coo_tensor(indices, values, (10, 2))

    # norm = sqrt(1^2 + 2^2) = sqrt(5)
    expected = torch.sqrt(torch.tensor(5.0))
    actual = compute_grad_norm([p])
    assert actual is not None
    assert torch.allclose(actual, expected)


def test_apply_group_wise_clipping() -> None:
    p1 = nn.Parameter(torch.ones(10))
    p1.grad = torch.ones(10) * 10.0  # norm = sqrt(1000) approx 31.6

    p2 = nn.Parameter(torch.ones(10))
    p2.grad = torch.ones(10) * 0.1  # norm = sqrt(10 * 0.01) = sqrt(0.1) approx 0.316

    groups = {"group1": [p1], "group2": [p2]}

    # Clip to max_norm=1.0
    apply_group_wise_clipping(groups, max_norm=1.0)

    # group1 should be clipped
    norm1 = compute_grad_norm([p1])
    assert norm1 is not None
    assert torch.allclose(norm1, torch.tensor(1.0), atol=1e-5)

    # group2 should NOT be clipped
    norm2 = compute_grad_norm([p2])
    assert norm2 is not None
    assert torch.allclose(norm2, torch.sqrt(torch.tensor(0.1)), atol=1e-5)


def test_compute_telemetry_stats() -> None:
    p = nn.Parameter(torch.ones(10))
    p.grad = torch.ones(10) * 0.5
    initial_weights = {id(p): torch.zeros(10)}  # Drift should be sqrt(10) approx 3.16

    groups = {"engram_dense": [p]}
    model_stats = {"extra": 1.23}

    stats = compute_telemetry_stats(
        groups=groups,
        initial_weights=initial_weights,
        model_telemetry_stats=model_stats,
        last_ce_loss=2.0,
        last_entropy_loss=0.5,
    )

    assert stats["telemetry/engram_dense/param_norm"] == approx(
        torch.sqrt(torch.tensor(10.0)).item()
    )
    assert stats["telemetry/engram_dense/grad_norm"] == approx(
        torch.sqrt(torch.tensor(2.5)).item()
    )
    assert stats["telemetry/engram_dense/weight_drift"] == approx(
        torch.sqrt(torch.tensor(10.0)).item()
    )
    assert stats["telemetry/extra"] == 1.23
    assert stats["telemetry/loss/ce"] == 2.0
    assert stats["telemetry/loss/entropy_penalty"] == 0.5
    assert stats["telemetry/diagnostics/heartbeat"] == 1.0


def approx(val: float) -> Any:
    return pytest.approx(val, rel=1e-5)
