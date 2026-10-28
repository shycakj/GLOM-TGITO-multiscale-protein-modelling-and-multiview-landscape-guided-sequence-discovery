from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class SourceTerrain:
    name: str
    rep: torch.Tensor
    xy: torch.Tensor
    brightness_true: torch.Tensor
    brightness_pred: torch.Tensor
    local_z_true: torch.Tensor
    local_z_pred: torch.Tensor
    sample_idx: torch.Tensor
    # v13.1: local gate/risk metadata. These are calibrated only from visible train/val rows.
    gate_support: torch.Tensor | None = None
    gate_risk: torch.Tensor | None = None
    source_quality: float = 1.0


class TerrainMemoryBank:
    def __init__(self, sources: Dict[str, SourceTerrain], device: str = "cuda", sequence_projection: torch.Tensor | None = None):
        self.sources = sources
        self.device = device
        # v13.1: optional deterministic projection for differentiable sequence UMAP source.
        # Shape: [L*20, Dseq]. TGITOGenerator uses this to map soft AA probabilities to rep_sequence.
        self.sequence_projection = sequence_projection

    @property
    def available_sources(self) -> List[str]:
        return sorted(self.sources.keys())

    @classmethod
    def from_npz(cls, path: str | Path, device: str = "cuda", sources: Optional[List[str]] = None) -> "TerrainMemoryBank":
        arr = np.load(str(path), allow_pickle=True)
        if "brightness_true" not in arr.files:
            raise ValueError(f"{path} lacks brightness_true")
        btrue = np.asarray(arr["brightness_true"], dtype=np.float32)
        if "brightness_pred_model" in arr.files:
            bpred = np.asarray(arr["brightness_pred_model"], dtype=np.float32)
        elif "brightness_pred_csv" in arr.files:
            bpred = np.asarray(arr["brightness_pred_csv"], dtype=np.float32)
        else:
            bpred = btrue.copy()
        if "sample_idx" in arr.files:
            sample_idx = np.asarray(arr["sample_idx"], dtype=np.int64)
        elif "masked_keep_rows" in arr.files:
            sample_idx = np.asarray(arr["masked_keep_rows"], dtype=np.int64)
        else:
            sample_idx = np.arange(len(btrue), dtype=np.int64)
        all_sources = sorted(k[3:] for k in arr.files if k.startswith("xy_"))
        if sources is not None and "all" not in sources:
            wanted = set(sources)
            all_sources = [s for s in all_sources if s in wanted]
        sequence_projection = None
        if "sequence_projection" in arr.files:
            sequence_projection = torch.as_tensor(np.asarray(arr["sequence_projection"], dtype=np.float32), dtype=torch.float32, device=device)
            print(f"[TerrainMemory] loaded sequence_projection shape={tuple(sequence_projection.shape)}")

        out: Dict[str, SourceTerrain] = {}
        for s in all_sources:
            rep_key = f"rep_{s}"
            xy_key = f"xy_{s}"
            if rep_key not in arr.files or xy_key not in arr.files:
                # sequence maps often have no high-dimensional rep; skip for differentiable potential.
                continue
            rep = np.asarray(arr[rep_key], dtype=np.float32)
            xy = np.asarray(arr[xy_key], dtype=np.float32)
            if len(rep) != len(btrue) or len(xy) != len(btrue):
                print(f"[TerrainMemory] skip {s}: row count mismatch rep={len(rep)} xy={len(xy)} N={len(btrue)}")
                continue
            zt = np.asarray(arr[f"local_z_{s}_true"], dtype=np.float32) if f"local_z_{s}_true" in arr.files else np.zeros(len(btrue), dtype=np.float32)
            zp = np.asarray(arr[f"local_z_{s}_pred"], dtype=np.float32) if f"local_z_{s}_pred" in arr.files else zt.copy()
            # v13.1 local gate metadata. If absent, fall back to neutral support/risk.
            gs = np.asarray(arr[f"gate_support_{s}"], dtype=np.float32) if f"gate_support_{s}" in arr.files else np.zeros(len(btrue), dtype=np.float32)
            gr = np.asarray(arr[f"gate_risk_{s}"], dtype=np.float32) if f"gate_risk_{s}" in arr.files else np.zeros(len(btrue), dtype=np.float32)
            sq = float(np.asarray(arr[f"source_quality_{s}"], dtype=np.float32).reshape(-1)[0]) if f"source_quality_{s}" in arr.files else 1.0
            t = SourceTerrain(
                name=s,
                rep=torch.as_tensor(rep, dtype=torch.float32, device=device),
                xy=torch.as_tensor(xy, dtype=torch.float32, device=device),
                brightness_true=torch.as_tensor(btrue, dtype=torch.float32, device=device),
                brightness_pred=torch.as_tensor(bpred, dtype=torch.float32, device=device),
                local_z_true=torch.as_tensor(zt, dtype=torch.float32, device=device),
                local_z_pred=torch.as_tensor(zp, dtype=torch.float32, device=device),
                sample_idx=torch.as_tensor(sample_idx, dtype=torch.long, device=device),
                gate_support=torch.as_tensor(gs, dtype=torch.float32, device=device),
                gate_risk=torch.as_tensor(gr, dtype=torch.float32, device=device),
                source_quality=sq,
            )
            # Normalize reps once for cosine KNN.
            t.rep = t.rep / t.rep.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            out[s] = t
        if not out:
            raise ValueError(f"No differentiable terrain sources loaded from {path}. Need rep_<source> and xy_<source> arrays.")
        print(f"[TerrainMemory] loaded {len(out)} sources: {sorted(out)}")
        return cls(out, device=device, sequence_projection=sequence_projection)
