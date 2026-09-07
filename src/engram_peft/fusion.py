"""Calibrated multi-source log-linear fusion for sparse external memories.

This module is the upstream home of the logit-level n-gram fusion used by the
qwen35-ple experiments.  It is intentionally NumPy-only so it can be used in
training, evaluation, and lightweight serving paths without a full model
dependency.
"""

from __future__ import annotations

import math

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    m = float(logits.max())
    e = np.exp(logits - m)
    return e / e.sum()


def fuse_ngram_logits(
    base_logits: np.ndarray,
    ngram_probs: dict[int, float] | None,
    *,
    scale: float = 1.0,
    bias: float = 0.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Add a calibrated sparse n-gram log-prior to base logits.

    Parameters
    ----------
    temperature
        Applied to the n-gram log-probabilities before scaling: temperature > 1
        flattens the memory distribution, temperature < 1 sharpens it.
    """
    out = base_logits.copy()
    if not ngram_probs:
        return out
    inv_t = 1.0 / max(temperature, 1e-6)
    for tok, p in ngram_probs.items():
        if 0 <= tok < len(out) and p > 0:
            out[tok] += scale * (math.log(p) * inv_t) + bias
    return out


def calibrate_ngram_fusion(
    base_logits_list: list[np.ndarray],
    target_ids: list[int],
    ngram_dist_list: list[dict[int, float] | None],
    *,
    scale_grid: np.ndarray | None = None,
    bias_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Two-parameter grid calibration for ``scale`` and ``bias``.

    Returns the best ``(scale, bias)`` pair by average cross-entropy on the
    supplied samples.
    """
    if not base_logits_list:
        return {}
    if scale_grid is None:
        scale_grid = np.linspace(-5.0, 5.0, 41)
    if bias_grid is None:
        bias_grid = np.linspace(-5.0, 5.0, 41)

    best_scale = 0.0
    best_bias = 0.0
    best_nll = float("inf")
    base_nlls: list[float] = []
    for logits, target, _dist in zip(
        base_logits_list, target_ids, ngram_dist_list, strict=False
    ):
        p = softmax(logits)[target]
        base_nlls.append(-math.log(max(float(p), 1e-12)))

    for scale in scale_grid:
        for bias in bias_grid:
            total = 0.0
            for logits, target, dist in zip(
                base_logits_list, target_ids, ngram_dist_list, strict=False
            ):
                fused = fuse_ngram_logits(
                    logits, dist, scale=float(scale), bias=float(bias)
                )
                p = softmax(fused)[target]
                total -= math.log(max(float(p), 1e-12))
            nll = float(total / len(base_logits_list))
            if nll < best_nll:
                best_nll = nll
                best_scale = float(scale)
                best_bias = float(bias)

    base_nll = float(np.mean(base_nlls))
    return {
        "best_scale": best_scale,
        "best_bias": best_bias,
        "best_mean_nll": best_nll,
        "base_mean_nll": base_nll,
        "delta_nll_nats": base_nll - best_nll,
        "delta_nll_bits": (base_nll - best_nll) / math.log(2.0),
    }


def weight_logit_sources(
    base_logits: np.ndarray,
    source_logits: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Log-linear fusion of multiple logit sources."""
    if weights is None:
        weights = [1.0] * len(source_logits)
    out = base_logits.copy()
    for src, w in zip(source_logits, weights, strict=False):
        out = out + float(w) * src
    return out


def mixture_distribution(
    distributions: list[dict[int, float]],
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Mixture of sparse categorical distributions."""
    if not distributions:
        return {}
    if weights is None:
        weights = [1.0 / len(distributions)] * len(distributions)
    total_weight = sum(abs(w) for w in weights)
    if total_weight == 0:
        return {}
    out: dict[int, float] = {}
    for dist, w in zip(distributions, weights, strict=False):
        if w == 0:
            continue
        for tok, p in dist.items():
            out[tok] = out.get(tok, 0.0) + float(w) * p
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


__all__ = [
    "calibrate_ngram_fusion",
    "fuse_ngram_logits",
    "mixture_distribution",
    "softmax",
    "weight_logit_sources",
]
