"""Auditable memory-trust gates and token-level safety policies.

This module provides reusable logic for deciding when a sparse n-gram memory
should be allowed to modify the base model's next-token distribution.

* :class:`LogDensityRatioGate` gates on the expected log-density advantage of
  the memory distribution.
* :class:`TokenPlePolicy` is a small learned logistic policy trained on
  per-token observations, used to suppress harmful memory fusion during
  open-ended generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy_logits(logits: Any) -> np.ndarray:
    if hasattr(logits, "detach") and hasattr(logits, "to"):
        return logits.detach().cpu().float().numpy()
    return np.asarray(logits, dtype=np.float32)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    z = logits - m
    return z - math.log(float(np.exp(z).sum()))


def log_density_ratio(
    base_logits: Any,
    ngram_probs: dict[int, float] | None,
) -> float:
    """Estimate ``E_{p_m}[log(p_m/p_b)]`` over the memory support."""
    if not ngram_probs:
        return -math.inf
    logits = _to_numpy_logits(base_logits)
    log_pb = _log_softmax(logits)
    total = 0.0
    for tok, p in ngram_probs.items():
        if 0 <= tok < len(log_pb) and p > 0:
            total += float(p) * (math.log(float(p)) - float(log_pb[tok]))
    return float(total)


def pseudo_label_log_density_ratio(
    base_logits: Any,
    ngram_probs: dict[int, float] | None,
) -> float:
    """Log-density ratio at the base model's current argmax."""
    if not ngram_probs:
        return -math.inf
    logits = _to_numpy_logits(base_logits)
    log_pb = _log_softmax(logits)
    base_argmax = int(np.argmax(logits))
    p_m = float(ngram_probs.get(base_argmax, 0.0))
    if p_m <= 0:
        return -math.inf
    return math.log(p_m) - float(log_pb[base_argmax])


class LogDensityRatioGate:
    """Decide when calibrated n-gram fusion should be active.

    Modes:

    - ``expected_kl``: ``E_{p_m}[log(p_m/p_b)] >= threshold``.
    - ``pseudo_label``: log-density ratio at the base argmax.
    - ``memory_top1``: log-density advantage of the memory argmax.
    - ``hybrid``: expected-KL signal plus agreement with base argmax.
    """

    def __init__(
        self,
        *,
        mode: str = "expected_kl",
        threshold: float = 0.0,
    ) -> None:
        if mode not in {"expected_kl", "pseudo_label", "memory_top1", "hybrid"}:
            raise ValueError(f"unknown log-density gate mode: {mode}")
        self.mode = mode
        self.threshold = float(threshold)

    def evaluate(
        self,
        base_logits: Any,
        ngram_probs: dict[int, float] | None,
        *,
        memory_order: int | None = None,
    ) -> dict[str, Any]:
        if not ngram_probs:
            return {
                "active": False,
                "mode": self.mode,
                "threshold": self.threshold,
                "matched_order": memory_order,
                "expected_log_density_ratio": -math.inf,
                "pseudo_label_log_density_ratio": -math.inf,
                "memory_top1_log_density_ratio": -math.inf,
                "base_entropy": math.inf,
                "memory_entropy": math.inf,
                "base_top1_prob": 0.0,
                "memory_top1_prob": 0.0,
            }

        logits = _to_numpy_logits(base_logits)
        log_pb = _log_softmax(logits)
        pb = np.exp(log_pb)
        pm = {int(k): float(v) for k, v in ngram_probs.items() if int(k) >= 0}
        total = sum(pm.values())
        if total <= 0:
            pm = {}
        else:
            pm = {k: v / total for k, v in pm.items()}

        expected = log_density_ratio(logits, pm)
        pseudo = pseudo_label_log_density_ratio(logits, pm)
        mem_argmax = max(pm, key=pm.get) if pm else 0
        mem_top1 = (
            math.log(pm[mem_argmax]) - float(log_pb[mem_argmax])
            if pm and 0 <= mem_argmax < len(log_pb)
            else -math.inf
        )
        base_entropy = float(-np.sum(pb * np.log(np.maximum(pb, 1e-12))))
        memory_entropy = float(
            -sum(p * math.log(p) for p in pm.values() if p > 0)
        )

        if self.mode == "expected_kl":
            active = expected >= self.threshold
        elif self.mode == "pseudo_label":
            active = pseudo >= self.threshold
        elif self.mode == "memory_top1":
            active = mem_top1 >= self.threshold
        else:  # hybrid
            active = expected >= self.threshold and pseudo >= 0.0

        return {
            "active": active,
            "mode": self.mode,
            "threshold": self.threshold,
            "matched_order": memory_order,
            "expected_log_density_ratio": expected,
            "pseudo_label_log_density_ratio": pseudo,
            "memory_top1_log_density_ratio": mem_top1,
            "base_entropy": base_entropy,
            "memory_entropy": memory_entropy,
            "base_top1_prob": float(np.max(pb)) if len(pb) else 0.0,
            "memory_top1_prob": float(max(pm.values())) if pm else 0.0,
        }

    def decide(
        self,
        base_logits: Any,
        ngram_probs: dict[int, float] | None,
    ) -> bool:
        return bool(self.evaluate(base_logits, ngram_probs)["active"])

    def state_dict(self) -> dict[str, float | str]:
        return {"mode": self.mode, "threshold": self.threshold}


class TokenPlePolicy:
    """Small learned token-level policy for whether to apply PLE fusion."""

    def __init__(
        self,
        data: dict[str, Any] | str | Path,
    ) -> None:
        if isinstance(data, (str, Path)):
            with Path(data).open("r", encoding="utf-8") as f:
                data = json.load(f)
        self.feature_names = list(data["feature_names"])
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.std = np.asarray(data["std"], dtype=np.float64)
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        self.bias = float(data["bias"])
        self.metrics = dict(data.get("metrics", {}))

    def predict(self, features: dict[str, Any]) -> float:
        x = np.asarray(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        x = (x - self.mean) / self.std
        z = float(x @ self.weights + self.bias)
        return float(1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0))))

    def should_apply(self, features: dict[str, Any], threshold: float = 0.5) -> bool:
        return self.predict(features) >= threshold


__all__ = [
    "LogDensityRatioGate",
    "TokenPlePolicy",
    "log_density_ratio",
    "pseudo_label_log_density_ratio",
]
