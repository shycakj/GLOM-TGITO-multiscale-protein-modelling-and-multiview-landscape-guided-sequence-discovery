from __future__ import annotations

import os
from pathlib import Path
import numpy as np

DEFAULT_TERRAIN_DIRS = [
    "/vol/intermediate_terrains_all",
    "/vol/pplu_intermediate_terrains_all",
    "/vol/tgito_terrains/intermediate_terrains_all",
    "/vol/tgito_terrains",
]


def terrain_candidates(name: str = "intermediate_terrain_arrays.npz") -> list[Path]:
    out = []
    env_npz = os.environ.get("TGITO_TERRAIN_NPZ", "").strip()
    if env_npz:
        out.append(Path(env_npz))
    for d in DEFAULT_TERRAIN_DIRS:
        out.append(Path(d) / name)
    # keep stable unique order
    seen = set(); uniq = []
    for p in out:
        sp = str(p)
        if sp not in seen:
            uniq.append(p); seen.add(sp)
    return uniq


def resolve_existing_terrain(requested: str = "auto", require_rep: bool = False) -> str:
    requested = str(requested or "auto")
    candidates = []
    if requested.lower() not in {"", "auto", "none"}:
        p = Path(requested)
        candidates.append(p)
        # If user passed old path, also try same basename in new standard dirs.
        for d in DEFAULT_TERRAIN_DIRS:
            candidates.append(Path(d) / p.name)
    else:
        candidates.extend(terrain_candidates())
    checked = []
    for p in candidates:
        checked.append(str(p))
        if p.exists():
            if require_rep:
                rep_sources = get_rep_sources(p)
                if not rep_sources:
                    continue
            return str(p)
    msg = "Cannot find a usable terrain npz. Checked:\n  " + "\n  ".join(checked)
    if require_rep:
        msg += "\nFound files without rep_<source> do not count. Run `modal run modal_make_intermediate_feature_terrains.py` first to build differentiable terrain memory."
    raise FileNotFoundError(msg)


def default_terrain_out_dir() -> str:
    env = os.environ.get("TGITO_TERRAIN_DIR", "").strip()
    return env or "/vol/intermediate_terrains_all"


def get_rep_sources(npz_path: str | Path) -> list[str]:
    arr = np.load(str(npz_path), allow_pickle=True)
    xy = {k[3:] for k in arr.files if k.startswith("xy_")}
    rep = {k[4:] for k in arr.files if k.startswith("rep_")}
    return sorted(xy & rep)


def describe_npz(npz_path: str | Path) -> dict:
    arr = np.load(str(npz_path), allow_pickle=True)
    xy_sources = sorted(k[3:] for k in arr.files if k.startswith("xy_"))
    rep_sources = sorted(k[4:] for k in arr.files if k.startswith("rep_"))
    n = None
    for k in ["brightness_true", "sample_idx", "masked_keep_rows"]:
        if k in arr.files:
            n = int(len(arr[k])); break
    return {
        "path": str(npz_path),
        "N": n,
        "n_arrays": len(arr.files),
        "xy_sources": xy_sources,
        "rep_sources": rep_sources,
        "differentiable_sources": sorted(set(xy_sources) & set(rep_sources)),
        "missing_rep_sources": sorted(set(xy_sources) - set(rep_sources)),
    }
