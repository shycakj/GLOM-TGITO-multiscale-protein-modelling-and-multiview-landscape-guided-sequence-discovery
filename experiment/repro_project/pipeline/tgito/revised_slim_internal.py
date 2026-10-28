from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .revised_slim_model import RevisedSlimModel


def _looks_state(x: Any) -> bool:
    return isinstance(x, dict) and any(isinstance(v, torch.Tensor) for v in x.values())


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if _looks_state(obj):
        return obj
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(obj)}")
    for k in ["model", "state_dict", "model_state", "model_state_dict", "net", "module"]:
        if k in obj:
            try:
                return extract_state_dict(obj[k])
            except Exception:
                pass
    stack = [obj]
    seen = set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen or not isinstance(cur, dict):
            continue
        seen.add(id(cur))
        if _looks_state(cur):
            return cur
        for v in cur.values():
            if isinstance(v, dict):
                stack.append(v)
    raise ValueError("Could not extract model state_dict from checkpoint")


def strip_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    prefixes = ["module.", "model."]
    for k, v in state.items():
        kk = str(k)
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p):]
                    changed = True
        out[kk] = v
    return out


def _infer_y_stats(ckpt: Any) -> tuple[float, float]:
    if isinstance(ckpt, dict):
        for mkey, skey in [("y_mean", "y_std"), ("target_mean", "target_std"), ("mean", "std")]:
            if mkey in ckpt and skey in ckpt:
                return float(ckpt[mkey]), float(ckpt[skey])
        if "stats" in ckpt and isinstance(ckpt["stats"], dict):
            st = ckpt["stats"]
            for mkey, skey in [("y_mean", "y_std"), ("target_mean", "target_std"), ("mean", "std")]:
                if mkey in st and skey in st:
                    return float(st[mkey]), float(st[skey])
    # revised_slim checkpoints in earlier package used normalized output; these defaults keep old behavior if stats are absent.
    return 0.0, 1.0


def infer_input_dim_from_state(state: Dict[str, torch.Tensor]) -> int:
    for k, v in state.items():
        kk = str(k)
        if kk.endswith("backbone.res_proj.0.weight") or kk.endswith("res_proj.0.weight"):
            if "region_res_proj" not in kk and "key_res_proj" not in kk and isinstance(v, torch.Tensor) and v.ndim == 2:
                return int(v.shape[1])
    return 1280


def load_revised_slim(checkpoint: str | Path, device: str = "cuda") -> Tuple[RevisedSlimModel, float, float]:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = strip_prefixes(extract_state_dict(ckpt))
    D = infer_input_dim_from_state(state)
    model = RevisedSlimModel(D=D)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[revised_slim] missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[revised_slim] unexpected keys ignored: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
    y_mean, y_std = _infer_y_stats(ckpt)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[revised_slim] loaded D={D} y_mean={y_mean:.4f} y_std={y_std:.4f}")
    return model, y_mean, y_std
