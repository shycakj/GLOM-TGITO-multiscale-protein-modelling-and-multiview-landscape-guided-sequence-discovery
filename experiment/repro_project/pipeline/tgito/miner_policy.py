from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

V12_7_CONTEXT_FEATURE_NAMES = [
    "pred_self", "pred_margin", "pred_percentile",
    "ctx_true_mean_w", "ctx_true_max_w", "ctx_true_top3_w",
    "ctx_pred_mean_w", "ctx_pred_max_w", "ctx_pred_top3_w",
    "ctx_local_true_mean_w", "ctx_local_true_max_w",
    "ctx_local_pred_mean_w", "ctx_local_pred_max_w",
    "ctx_high_frac_w", "ctx_reward_mean_w", "ctx_reward_max_w",
    "ctx_dist_min_w", "ctx_dist_mean_w",
    "ctx_true_max_global", "ctx_high_frac_global", "ctx_reward_max_global",
    "support_source_count", "source_entropy",
]


class ContextMiningPolicyMLP(nn.Module):
    """The single v12.7 miner architecture used for both training and generation.

    This class is intentionally packaged inside v12.7 so the generator never imports
    a class from an older experiment directory.
    """
    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_v12_7_context_miner(path: str | Path, model: nn.Module, *, feature_names: List[str], feature_mean: np.ndarray,
                             feature_std: np.ndarray, hidden_dim: int, dropout: float, neighbor_k: int,
                             high_threshold: float, sources: List[str], metrics: Dict[str, Any] | None = None,
                             extra: Dict[str, Any] | None = None) -> None:
    ckpt = {
        "schema": "v12_7_context",
        "state_dict": model.state_dict(),
        "feature_names": list(feature_names),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "neighbor_k": int(neighbor_k),
        "high_threshold": float(high_threshold),
        "sources": list(sources),
        "metrics": metrics or {},
        "version": "v12.7-standalone",
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, str(path))


@dataclass
class LoadedContextMiner:
    path: str
    model: ContextMiningPolicyMLP
    feature_names: List[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    neighbor_k: int
    high_threshold: float
    sources: List[str]
    metrics: Dict[str, Any]

    def score(self, features: torch.Tensor, logit_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        z = (features - self.feature_mean.to(features.device)) / self.feature_std.to(features.device).clamp_min(1e-6)
        z = torch.nan_to_num(z, nan=0.0, posinf=6.0, neginf=-6.0)
        logits = self.model(z) / max(float(logit_scale), 1e-6)
        return torch.sigmoid(logits), logits


def load_v12_7_context_miner(path: str | Path, device: str = "cuda") -> LoadedContextMiner:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    schema = str(ckpt.get("schema", ""))
    if schema != "v12_7_context":
        raise ValueError(
            f"{path} is not a v12.7 standalone miner checkpoint. schema={schema!r}.\n"
            "Do not reuse v12.4/v12.5 checkpoints with v12.7. Re-train miner using train_v12_7_miner_sweep.py."
        )
    feature_names = [str(x) for x in ckpt["feature_names"]]
    if feature_names != V12_7_CONTEXT_FEATURE_NAMES:
        raise ValueError(
            f"v12.7 miner feature schema mismatch.\n"
            f"ckpt={feature_names}\nexpected={V12_7_CONTEXT_FEATURE_NAMES}"
        )
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    if mean.shape[0] != len(feature_names) or std.shape[0] != len(feature_names):
        raise ValueError(f"Scaler dim mismatch: names={len(feature_names)} mean={mean.shape} std={std.shape}")
    hidden = int(ckpt.get("hidden_dim", 96))
    dropout = float(ckpt.get("dropout", 0.10))
    model = ContextMiningPolicyMLP(len(feature_names), hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return LoadedContextMiner(
        path=str(path),
        model=model,
        feature_names=feature_names,
        feature_mean=torch.as_tensor(mean, dtype=torch.float32, device=device),
        feature_std=torch.as_tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6),
        neighbor_k=int(ckpt.get("neighbor_k", 4)),
        high_threshold=float(ckpt.get("high_threshold", 4.2)),
        sources=[str(x) for x in ckpt.get("sources", [])],
        metrics=ckpt.get("metrics", {}),
    )

# -----------------------------
# v12.8 credibility-calibrated miner
# -----------------------------
V12_8_REALISM_FEATURE_NAMES = [
    "embedding_realism",
    "embedding_realism_combo",
    "embedding_realism_saprot",
    "embedding_realism_p4",
    "embedding_realism_weight_sum",
    "embedding_realism_active_sites",
    "embedding_realism_used_global_frac",
    "realism_x_pred_margin",
    "realism_x_ctx_high_frac",
]

V12_8_CREDIBLE_FEATURE_NAMES = V12_7_CONTEXT_FEATURE_NAMES + V12_8_REALISM_FEATURE_NAMES


def save_v12_8_credible_miner(path: str | Path, model: nn.Module, *, feature_names: List[str], feature_mean: np.ndarray,
                              feature_std: np.ndarray, hidden_dim: int, dropout: float, neighbor_k: int,
                              high_threshold: float, sources: List[str], metrics: Dict[str, Any] | None = None,
                              extra: Dict[str, Any] | None = None) -> None:
    ckpt = {
        "schema": "v12_8_credible",
        "state_dict": model.state_dict(),
        "feature_names": list(feature_names),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "neighbor_k": int(neighbor_k),
        "high_threshold": float(high_threshold),
        "sources": list(sources),
        "metrics": metrics or {},
        "version": "v12.8-credible-standalone",
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, str(path))


def load_v12_8_credible_miner(path: str | Path, device: str = "cuda") -> LoadedContextMiner:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    schema = str(ckpt.get("schema", ""))
    if schema != "v12_8_credible":
        raise ValueError(f"{path} is not a v12.8 credible miner checkpoint. schema={schema!r}")
    feature_names = [str(x) for x in ckpt["feature_names"]]
    if feature_names != V12_8_CREDIBLE_FEATURE_NAMES:
        raise ValueError(
            "v12.8 credible miner feature schema mismatch.\n"
            f"ckpt={feature_names}\nexpected={V12_8_CREDIBLE_FEATURE_NAMES}"
        )
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    hidden = int(ckpt.get("hidden_dim", 96))
    dropout = float(ckpt.get("dropout", 0.10))
    model = ContextMiningPolicyMLP(len(feature_names), hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return LoadedContextMiner(
        path=str(path),
        model=model,
        feature_names=feature_names,
        feature_mean=torch.as_tensor(mean, dtype=torch.float32, device=device),
        feature_std=torch.as_tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6),
        neighbor_k=int(ckpt.get("neighbor_k", 4)),
        high_threshold=float(ckpt.get("high_threshold", 4.2)),
        sources=[str(x) for x in ckpt.get("sources", [])],
        metrics=ckpt.get("metrics", {}),
    )


def load_v12_standalone_miner(path: str | Path, device: str = "cuda") -> LoadedContextMiner:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    schema = str(ckpt.get("schema", ""))
    if schema == "v12_7_context":
        return load_v12_7_context_miner(path, device=device)
    if schema == "v12_8_credible":
        return load_v12_8_credible_miner(path, device=device)
    raise ValueError(
        f"{path} has unsupported miner schema={schema!r}. v12.8 accepts only v12_7_context or v12_8_credible produced inside this package."
    )
