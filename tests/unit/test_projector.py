"""Tests for engram_peft.projector."""

from __future__ import annotations

import numpy as np
import torch

from engram_peft.projector import (
    PleProjector,
    compute_memory_features,
    load_projector,
    save_projector,
)


def test_zero_init_is_inert() -> None:
    proj = PleProjector(8, hidden_dim=4, num_layers=1, zero_init=True)
    hidden = torch.zeros(1, 8)
    features = torch.zeros(1, proj.num_features)
    scale, bias = proj(hidden, features)
    assert float(scale[0].detach()) == 0.0
    assert float(bias[0].detach()) == 0.0


def test_compute_memory_features_basic() -> None:
    logits = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    features = compute_memory_features(logits, {1: 0.7, 2: 0.3}, matched_order=3)
    assert features["matched_order"] == 3.0
    assert features["memory_top1_agree_base"] == 1.0


def test_save_load_roundtrip(tmp_path) -> None:
    proj = PleProjector(8, hidden_dim=4, num_layers=1, zero_init=False)
    with torch.no_grad():
        proj.net[-1].weight.fill_(0.25)
        proj.net[-1].bias.fill_(-0.1)
    path = tmp_path / "projector.json"
    save_projector(path, proj)
    loaded = load_projector(path)
    h = torch.zeros(1, 8)
    f = torch.zeros(1, loaded.num_features)
    s1, b1 = proj(h, f)
    s2, b2 = loaded(h, f)
    np.testing.assert_allclose(float(s1[0].detach()), float(s2[0].detach()), atol=1e-6)
    np.testing.assert_allclose(float(b1[0].detach()), float(b2[0].detach()), atol=1e-6)
