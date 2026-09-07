"""Tests for engram_peft.policy."""

from __future__ import annotations

import numpy as np

from engram_peft.policy import (
    LogDensityRatioGate,
    TokenPlePolicy,
    log_density_ratio,
)


def test_log_density_ratio_positive_for_agreeing_memory() -> None:
    logits = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    dist = {1: 0.9, 0: 0.1}
    ratio = log_density_ratio(logits, dist)
    assert ratio > 0.0


def test_gate_expected_kl_activation() -> None:
    gate = LogDensityRatioGate(mode="expected_kl", threshold=0.0)
    logits = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    dist = {1: 0.9, 0: 0.1}
    assert gate.decide(logits, dist)


def test_gate_no_memory_inactive() -> None:
    gate = LogDensityRatioGate()
    assert not gate.decide(np.zeros(3, dtype=np.float32), None)


def test_token_policy_basic() -> None:
    data = {
        "feature_names": ["matched_order"],
        "mean": [3.0],
        "std": [1.0],
        "weights": [5.0],
        "bias": -4.0,
        "metrics": {},
    }
    policy = TokenPlePolicy(data)
    assert policy.predict({"matched_order": 4.0}) > 0.5
    assert not policy.should_apply({"matched_order": 2.0})
