"""Learned logit-space projector for sparse external memories.

The projector maps a frozen backbone hidden state plus lexical memory features
to a small ``(scale, bias)`` correction applied to an n-gram log-prior:

    fused = base_logits + scale * log p_memory + bias

The final linear head is zero-initialised so a freshly initialised projector is
inert; training gradually turns memory fusion on where it helps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from engram_peft.fusion import fuse_ngram_logits

SCHEMA = "ple-projector-v1"

TASK_NAMES = ["code", "name", "number", "general"]
TASK_FEATURE_NAMES = [f"task_{name}" for name in TASK_NAMES]

FEATURE_NAMES = [
    "matched_order",
    "base_entropy",
    "memory_entropy",
    "density_ratio",
    "base_top1_prob",
    "memory_top1_prob",
    "memory_top1_agree_base",
] + TASK_FEATURE_NAMES


def add_task_features(
    features: dict[str, Any],
    task: str | None,
) -> dict[str, Any]:
    """Return a copy of ``features`` with one-hot task indicators added."""
    out = dict(features)
    task = str(task or "general")
    for name in TASK_NAMES:
        out[f"task_{name}"] = 1.0 if task == name else 0.0
    return out


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    z = logits - m
    return z - math.log(float(np.exp(z).sum()))


def compute_memory_features(
    logits: Any,
    dist: dict[int, float] | None,
    *,
    matched_order: int | None = None,
) -> dict[str, float]:
    """Compute scalar memory features used by the learned projector and policy."""
    if not dist:
        return {
            "matched_order": float(matched_order or 0),
            "base_entropy": math.inf,
            "memory_entropy": math.inf,
            "density_ratio": -math.inf,
            "base_top1_prob": 0.0,
            "memory_top1_prob": 0.0,
            "memory_top1_agree_base": 0.0,
        }

    logits_np = _as_numpy(logits)
    log_pb = _log_softmax_np(logits_np)
    pb = np.exp(log_pb)
    total = sum(float(v) for v in dist.values() if v > 0)
    if total <= 0:
        pm = {}
    else:
        pm = {
            int(k): float(v) / total
            for k, v in dist.items()
            if int(k) >= 0 and v > 0
        }
    if not pm:
        return {
            "matched_order": float(matched_order or 0),
            "base_entropy": float(-np.sum(pb * np.log(np.maximum(pb, 1e-12)))),
            "memory_entropy": math.inf,
            "density_ratio": -math.inf,
            "base_top1_prob": float(np.max(pb)) if len(pb) else 0.0,
            "memory_top1_prob": 0.0,
            "memory_top1_agree_base": 0.0,
        }

    expected = sum(
        p * (math.log(p) - float(log_pb[tok]))
        for tok, p in pm.items()
        if 0 <= tok < len(log_pb) and p > 0
    )
    base_top1 = int(np.argmax(logits_np)) if len(logits_np) else 0
    mem_top1 = max(pm, key=pm.get)
    memory_entropy = float(-sum(p * math.log(p) for p in pm.values() if p > 0))
    return {
        "matched_order": float(matched_order or 0),
        "base_entropy": float(-np.sum(pb * np.log(np.maximum(pb, 1e-12)))),
        "memory_entropy": memory_entropy,
        "density_ratio": float(expected),
        "base_top1_prob": float(pb[base_top1]) if len(pb) else 0.0,
        "memory_top1_prob": float(pm[mem_top1]),
        "memory_top1_agree_base": 1.0 if mem_top1 == base_top1 else 0.0,
    }


def normalize_features(features: dict[str, Any], names: list[str]) -> np.ndarray:
    """Convert a feature dict to a normalized vector."""
    return np.asarray(
        [float(features.get(name, 0.0)) for name in names], dtype=np.float32
    )


class PleProjector(nn.Module):
    """Small MLP mapping hidden state and memory features to PLE scale/bias."""

    def __init__(
        self,
        hidden_size: int,
        *,
        feature_names: list[str] | None = None,
        hidden_dim: int = 64,
        num_layers: int = 1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.num_features = len(self.feature_names)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.register_buffer(
            "feature_mean", torch.zeros(self.num_features, dtype=torch.float32)
        )
        self.register_buffer(
            "feature_std", torch.ones(self.num_features, dtype=torch.float32)
        )

        layers: list[nn.Module] = []
        in_dim = self.hidden_size + self.num_features
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.GELU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def set_feature_stats(self, mean: Any, std: Any) -> None:
        mean_np = np.asarray(mean, dtype=np.float32).reshape(-1)
        std_np = np.asarray(std, dtype=np.float32).reshape(-1)
        if mean_np.shape[0] != self.num_features or std_np.shape[0] != self.num_features:
            raise ValueError(
                f"feature stats must match {self.num_features} features, "
                f"got mean={mean_np.shape[0]} std={std_np.shape[0]}"
            )
        self.feature_mean.copy_(torch.as_tensor(mean_np))
        self.feature_std.copy_(
            torch.as_tensor(np.where(std_np < 1e-8, 1.0, std_np))
        )

    def forward(
        self,
        hidden: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(scale, bias)`` tensors of shape ``[batch]``."""
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        feat = (features - self.feature_mean) / self.feature_std
        x = torch.cat([hidden, feat], dim=-1)
        out = self.net(x)
        return out[:, 0], out[:, 1]

    def predict_np(
        self,
        hidden: Any,
        features: dict[str, Any],
    ) -> tuple[float, float]:
        """NumPy-friendly inference returning ``(scale, bias)``."""
        hn = _as_numpy(hidden).reshape(1, -1)
        fn = normalize_features(features, self.feature_names).reshape(1, -1)
        with torch.no_grad():
            device = self.feature_mean.device
            hidden_t = torch.as_tensor(hn, dtype=torch.float32, device=device)
            feat_t = torch.as_tensor(fn, dtype=torch.float32, device=device)
            scale, bias = self(hidden_t, feat_t)
        return float(scale[0]), float(bias[0])

    def to_dict(self) -> dict[str, Any]:
        state = {}
        for key, value in self.state_dict().items():
            state[key] = value.detach().cpu().numpy().tolist()
        return {
            "schema": SCHEMA,
            "hidden_size": self.hidden_size,
            "feature_names": self.feature_names,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "feature_mean": self.feature_mean.detach().cpu().numpy().tolist(),
            "feature_std": self.feature_std.detach().cpu().numpy().tolist(),
            "state_dict": state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PleProjector:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unexpected projector schema: {data.get('schema')}")
        proj = cls(
            int(data["hidden_size"]),
            feature_names=list(data["feature_names"]),
            hidden_dim=int(data.get("hidden_dim", 64)),
            num_layers=int(data.get("num_layers", 1)),
            zero_init=False,
        )
        proj.set_feature_stats(data["feature_mean"], data["feature_std"])
        state = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in data["state_dict"].items()
        }
        proj.load_state_dict(state)
        return proj


def save_projector(path: str | Path, projector: PleProjector | dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = projector.to_dict() if isinstance(projector, PleProjector) else dict(projector)
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_projector(path: str | Path | dict[str, Any]) -> PleProjector:
    if isinstance(path, dict):
        return PleProjector.from_dict(path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PleProjector.from_dict(data)


def apply_projector_to_logits(
    logits: Any,
    dist: dict[int, float] | None,
    scale: float,
    bias: float,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """NumPy logit fusion using learned scalar scale/bias."""
    logits_np = _as_numpy(logits)
    return fuse_ngram_logits(
        logits_np,
        dist,
        scale=scale,
        bias=bias,
        temperature=temperature,
    )


__all__ = [
    "FEATURE_NAMES",
    "SCHEMA",
    "TASK_FEATURE_NAMES",
    "TASK_NAMES",
    "PleProjector",
    "add_task_features",
    "apply_projector_to_logits",
    "compute_memory_features",
    "load_projector",
    "normalize_features",
    "save_projector",
]
