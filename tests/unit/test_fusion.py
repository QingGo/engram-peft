"""Tests for engram_peft.fusion."""

from __future__ import annotations

import numpy as np

from engram_peft.fusion import (
    calibrate_ngram_fusion,
    fuse_ngram_logits,
    mixture_distribution,
    softmax,
    weight_logit_sources,
)


def test_softmax_normalizes() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p = softmax(logits)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-6)


def test_fuse_ngram_logits_support_only() -> None:
    base = np.zeros(5, dtype=np.float32)
    dist = {1: 0.8, 3: 0.2}
    out = fuse_ngram_logits(base, dist, scale=1.0, bias=0.0)
    np.testing.assert_allclose(out[1], np.log(0.8))
    np.testing.assert_allclose(out[3], np.log(0.2))
    assert out[0] == 0.0
    assert out[2] == 0.0
    assert out[4] == 0.0


def test_fuse_ngram_logits_none_returns_copy() -> None:
    base = np.arange(3, dtype=np.float32)
    out = fuse_ngram_logits(base, None)
    np.testing.assert_array_equal(out, base)


def test_calibrate_ngram_fusion_finds_scale() -> None:
    base = np.zeros(5, dtype=np.float32)
    dist = {1: 0.9, 3: 0.1}
    scale_grid = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    bias_grid = np.array([0.0], dtype=np.float32)
    result = calibrate_ngram_fusion(
        [base], [1], [dist], scale_grid=scale_grid, bias_grid=bias_grid
    )
    assert result["best_scale"] == 1.0
    assert result["best_bias"] == 0.0


def test_weight_logit_sources() -> None:
    base = np.zeros(3, dtype=np.float32)
    src = np.ones(3, dtype=np.float32)
    out = weight_logit_sources(base, [src], weights=[2.0])
    np.testing.assert_allclose(out, 2.0)


def test_mixture_distribution_normalizes() -> None:
    d = mixture_distribution([{0: 1.0}, {1: 1.0}], weights=[1.0, 1.0])
    assert set(d.keys()) == {0, 1}
    assert abs(sum(d.values()) - 1.0) < 1e-6
