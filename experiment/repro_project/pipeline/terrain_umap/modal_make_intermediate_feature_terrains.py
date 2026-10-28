"""
Generate smoothed 3D terrain maps for revised_slim intermediate features.

This script extends the previous terrain script. It generates terrain maps for:

Base representations
--------------------
1. sequence
   - x-y from AA sequence pairwise distance

2. h32
   - x-y from the last hidden representation h32

Requested intermediate features
-------------------------------
3. global_mean_pool
   - mean pool over Hf: B x 256

4. global_max_pool
   - max pool over Hf: B x 256

5. global_attn_pool
   - attention pool over Hf: B x 256

6. pos_proj_flat
   - self.pos_proj(P4): B x 237 x 24, flattened to B x 5688 for distance layout

7. per-region all/mutation attention pools
   - for each region:
       r_all: B x 256
       r_mut: B x 256
       concat: B x 512
   - outputs:
       region_beta2_all_mut512
       region_beta5_all_mut512
       ...
       region_CRO_all_mut512

8. region9_mean_pool
   - mean pool over 9 region tokens R2: B x 256

9. region9_max_pool
   - max pool over 9 region tokens R2: B x 256

10. region9_attn_pool
   - attention pool over 9 region tokens R2: B x 256

11. keysite_K_flat
   - K = Hk[:, KEY_SITES_1B - 1]: B x 15 x 256, flattened to B x 3840

For every representation source, two terrains are generated:

  terrain_<source>_true.html
  terrain_<source>_pred.html

where x-y is from pairwise feature distance, z is brightness_true or
brightness_pred, and color is brightness.

Special anomaly marking
-----------------------
The script marks two types of "special high points" on both true and predicted
terrains:

  local_z = (z_i - mean(z_neighbors)) / std(z_neighbors)

1. True-local anomalies:
   - proteins that are much brighter than their neighbors in the TRUE terrain.
   - marked with purple x markers on BOTH true and predicted terrains.

2. Pred-local anomalies:
   - proteins that are much brighter than their neighbors in the PREDICTED terrain.
   - marked with bright magenta square markers on BOTH true and predicted terrains.

If local_z >= --anomaly-z, the protein is marked as an anomaly. To avoid
overcrowding, only the top --top-anomalies points are displayed per source.

This specifically helps find AA sequences whose brightness is unusually high
relative to nearby variants in the current landscape.

Run examples
------------
Small test:

    GPU_TYPE=L4 GFP_VOLUME=gfp-revised-slim modal run modal_make_intermediate_feature_terrains.py \\
      --data-dir /vol/revised_slim_package/dataset \\
      --checkpoint /vol/revised_slim_package/weights/best_model.pt \\
      --predictions /vol/modal_output/routeB_refit_from_scratch_seed0/revised_slim_seed0/predictions.csv \\
      --out-dir /vol/revised_slim_package/results/intermediate_terrains_test \\
      --sample-size 5000 \\
      --sources all \\
      --layout-method umap \\
      --grid-size 220 \\
      --sigma 2.2 \\
      --point-k 25 \\
      --anomaly-z 2.5 \\
      --top-anomalies 200

Full run:

    GPU_TYPE=L4 GFP_VOLUME=gfp-revised-slim modal run modal_make_intermediate_feature_terrains.py \\
      --data-dir /vol/revised_slim_package/dataset \\
      --checkpoint /vol/revised_slim_package/weights/best_model.pt \\
      --predictions /vol/modal_output/routeB_refit_from_scratch_seed0/revised_slim_seed0/predictions.csv \\
      --out-dir /vol/revised_slim_package/results/intermediate_terrains_all \\
      --sample-size 0 \\
      --sources all \\
      --layout-method umap \\
      --grid-size 260 \\
      --sigma 2.5 \\
      --point-k 35

Download:

    modal volume get gfp-revised-slim revised_slim_package/results/intermediate_terrains_test ./intermediate_terrains_test
"""

from __future__ import annotations

import gc
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import modal


VOLUME_NAME = os.environ.get("GFP_VOLUME", "gfp-revised-slim")
GPU_TYPE = os.environ.get("GPU_TYPE", "L4")
GPU_CONFIG = None if GPU_TYPE.lower() in {"", "none", "cpu", "false", "0"} else GPU_TYPE

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
data_volume = modal.Volume.from_name("gfp20000-data", create_if_missing=False)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "tqdm",
        "plotly",
        "umap-learn",
    )
)

app = modal.App("gfp-intermediate-feature-terrains", image=image)


KEY_SITES_1B = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 93, 109, 110, 111, 120]
KEY_SITES_0B = [p - 1 for p in KEY_SITES_1B]

REGIONS_1B = {
    "beta2": [17, 27],
    "beta5": [53, 63],
    "beta7": [72, 82],
    "beta8": [83, 91],
    "beta9": [92, 102],
    "beta11": [115, 129],
    "beta13": [158, 171],
    "beta17": [212, 223],
    "CRO": [62, 70],
}
RIDX = [list(range(v[0] - 1, v[1])) for v in REGIONS_1B.values()]
REGION_NAMES = list(REGIONS_1B.keys())


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_col(df, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_to_real = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]
    if required:
        raise ValueError(f"Cannot find any of {candidates}. Existing columns={list(df.columns)}")
    return None


def parse_sources(sources: str) -> List[str]:
    all_sources = [
        "sequence",
        "h32",
        "global_mean_pool",
        "global_max_pool",
        "global_attn_pool",
        "pos_proj_flat",
        *[f"region_{name}_all_mut512" for name in REGION_NAMES],
        "region9_mean_pool",
        "region9_max_pool",
        "region9_attn_pool",
        "keysite_K_flat",
    ]
    if sources.strip().lower() == "all":
        return all_sources
    requested = [s.strip() for s in sources.split(",") if s.strip()]
    bad = set(requested) - set(all_sources)
    if bad:
        raise ValueError(f"Unknown sources={bad}. Allowed={all_sources}")
    return requested


def aa_encode_sequences(seqs: List[str]):
    import numpy as np

    alphabet = "ACDEFGHIKLMNPQRSTVWYBXZUOJ*-_.?"
    table = {aa: i + 1 for i, aa in enumerate(alphabet)}
    gap = 0
    L = max(len(str(s)) for s in seqs)
    arr = np.zeros((len(seqs), L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        s = str(s).upper()
        for j, ch in enumerate(s):
            arr[i, j] = table.get(ch, table["?"])
        if len(s) < L:
            arr[i, len(s) :] = gap
    return arr


def pairwise_hamming_distance_uint8(encoded, chunk_size: int = 128):
    import numpy as np
    from tqdm import tqdm

    n, _ = encoded.shape
    D = np.empty((n, n), dtype=np.float32)
    for start in tqdm(range(0, n, chunk_size), desc="pairwise sequence distance"):
        end = min(start + chunk_size, n)
        diff = encoded[start:end, None, :] != encoded[None, :, :]
        D[start:end] = diff.mean(axis=2, dtype=np.float32)
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return D


def pairwise_cosine_distance(x, chunk_size: int = 512):
    import numpy as np
    from tqdm import tqdm

    x = np.asarray(x, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    n = x.shape[0]
    D = np.empty((n, n), dtype=np.float32)
    for start in tqdm(range(0, n, chunk_size), desc=f"pairwise cosine distance dim={x.shape[1]}"):
        end = min(start + chunk_size, n)
        sim = x[start:end] @ x.T
        D[start:end] = 1.0 - sim
    D = np.clip((D + D.T) * 0.5, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return D


def layout_from_distance(D, seed: int, method: str = "umap"):
    if method == "mds":
        from sklearn.manifold import MDS

        xy = MDS(
            n_components=2,
            metric=True,
            dissimilarity="precomputed",
            random_state=seed,
            n_init=1,
            max_iter=300,
            normalized_stress="auto",
        ).fit_transform(D)
        return xy.astype("float32"), "MDS"

    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=30,
        min_dist=0.08,
        random_state=seed,
    )
    xy = reducer.fit_transform(D)
    return xy.astype("float32"), "UMAP_precomputed_distance"


def build_model_class():
    import torch
    import torch.nn as nn

    class MaskedAttentionPool(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = nn.Linear(dim, 1)

        def forward(self, x, mask=None, weights=None):
            b, l, _ = x.shape
            logits = self.attn(x).squeeze(-1).float()
            valid = torch.ones((b, l), dtype=torch.bool, device=x.device) if mask is None else mask.bool().to(x.device)
            if weights is not None:
                w = weights.float().to(x.device)
                valid = valid & (w > 1e-8)
                logits = logits + torch.log(w.clamp_min(1e-8))
            has_valid = valid.any(dim=1, keepdim=True)
            attn = torch.softmax(logits.masked_fill(~valid, -1e9), dim=1)
            attn = attn * valid.float()
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
            out = torch.sum(x * attn.to(x.dtype).unsqueeze(-1), dim=1)
            return torch.where(has_valid, out, torch.zeros_like(out))

    class RevisedSlimBackbone(nn.Module):
        def __init__(self, D=1280, pos_dim=4, hid=256, use_message=True):
            super().__init__()
            self.use_message = bool(use_message)

            self.res_proj = nn.Sequential(
                nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(512, hid), nn.GELU(),
            )
            self.pos_proj = nn.Sequential(
                nn.Linear(pos_dim, 24), nn.GELU(), nn.Linear(24, 24), nn.GELU(),
            )
            self.fuse = nn.Sequential(nn.Linear(hid + 24, hid), nn.GELU(), nn.LayerNorm(hid))
            self.seq_attn_pool = MaskedAttentionPool(hid)
            self.glob_compress = nn.Sequential(
                nn.Linear(hid * 3, 768), nn.GELU(), nn.Dropout(0.2),
                nn.LayerNorm(768), nn.Linear(768, hid),
            )

            self.region_res_proj = nn.Sequential(
                nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512, hid), nn.GELU(),
            )
            self.region_attn_pool = MaskedAttentionPool(hid)
            self.reg_compress = nn.Sequential(
                nn.Linear(hid * 2, 512), nn.GELU(), nn.Dropout(0.12),
                nn.LayerNorm(512), nn.Linear(512, hid),
            )

            self.key_res_proj = nn.Sequential(
                nn.Linear(D, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.05),
                nn.Linear(256, hid), nn.GELU(),
            )

            self.xattn_k = nn.MultiheadAttention(hid, 4, batch_first=True)
            self.xattn_r = nn.MultiheadAttention(hid, 4, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(hid, 512), nn.GELU(), nn.Dropout(0.15), nn.Linear(512, hid))
            self.ln_k1 = nn.LayerNorm(hid)
            self.ln_k2 = nn.LayerNorm(hid)
            self.ln_r1 = nn.LayerNorm(hid)
            self.ln_r2 = nn.LayerNorm(hid)

            self.k_compress = nn.Sequential(nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid))
            self.r_compress = nn.Sequential(nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid))

        def pool3_parts(self, x):
            return x.mean(dim=1), x.max(dim=1)[0], self.seq_attn_pool(x)

        def _regions_with_parts(self, Hr, P4):
            mut_mask = P4[..., 0] > 0.5
            outs = []
            parts = {}
            for name, idxs in zip(REGION_NAMES, RIDX):
                idx = torch.as_tensor(idxs, device=Hr.device, dtype=torch.long)
                r = Hr.index_select(1, idx)
                rm = mut_mask.index_select(1, idx)
                r_all = self.region_attn_pool(r)
                r_mut = self.region_attn_pool(r, mask=rm)
                concat = torch.cat([r_all, r_mut], dim=-1)
                outs.append(concat)
                parts[f"region_{name}_all_mut512"] = concat
            R = self.reg_compress(torch.stack(outs, dim=1))
            return R, parts

        def forward_intermediates(self, H, P4):
            out = {}

            H_global = self.res_proj(H)
            P4_embed = self.pos_proj(P4)
            Hf = self.fuse(torch.cat([H_global, P4_embed], dim=-1))

            g_mean, g_max, g_attn = self.pool3_parts(Hf)
            out["global_mean_pool"] = g_mean
            out["global_max_pool"] = g_max
            out["global_attn_pool"] = g_attn
            out["pos_proj_flat"] = P4_embed.reshape(P4_embed.shape[0], -1)

            Hr = self.region_res_proj(H)
            R, region_parts = self._regions_with_parts(Hr, P4)
            out.update(region_parts)

            Hk = self.key_res_proj(H)
            K = Hk[:, KEY_SITES_0B]
            out["keysite_K_flat"] = K.reshape(K.shape[0], -1)

            if self.use_message:
                K2a, _ = self.xattn_k(K, R, R, need_weights=False)
                R2a, _ = self.xattn_r(R, K, K, need_weights=False)
            else:
                K2a = torch.zeros_like(K)
                R2a = torch.zeros_like(R)

            K2 = self.ln_k1(K + K2a)
            K2 = self.ln_k2(K2 + self.ffn(K2))
            R2 = self.ln_r1(R + R2a)
            R2 = self.ln_r2(R2 + self.ffn(R2))

            r_mean, r_max, r_attn = self.pool3_parts(R2)
            out["region9_mean_pool"] = r_mean
            out["region9_max_pool"] = r_max
            out["region9_attn_pool"] = r_attn

            G = self.glob_compress(torch.cat([g_mean, g_max, g_attn], dim=-1))
            Rf = self.r_compress(torch.cat([r_mean, r_max, r_attn], dim=-1))

            k_mean, k_max, k_attn = self.pool3_parts(K2)
            Kf = self.k_compress(torch.cat([k_mean, k_max, k_attn], dim=-1))

            return out, {"G": G, "Rf": Rf, "Kf": Kf}

    class RevisedSlimModel(nn.Module):
        def __init__(self, D=1280, hid=256, use_message=True):
            super().__init__()
            self.backbone = RevisedSlimBackbone(D=D, pos_dim=4, hid=hid, use_message=use_message)
            self.narrow = nn.Sequential(
                nn.Linear(hid * 3, 512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.25),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(128, 64), nn.GELU(),
                nn.Linear(64, 32), nn.GELU(),
                nn.Linear(32, 1),
            )

        def forward_all(self, H, P):
            P4 = P[:, :, [0, 7, 8, 9]]
            inter, b = self.backbone.forward_intermediates(H, P4)
            z768 = torch.cat([b["G"], b["Rf"], b["Kf"]], dim=-1)
            h512 = self.narrow[1](self.narrow[0](z768))
            h512d = self.narrow[2](h512)
            h256 = self.narrow[4](self.narrow[3](h512d))
            h256d = self.narrow[5](h256)
            h128 = self.narrow[7](self.narrow[6](h256d))
            h128d = self.narrow[8](h128)
            h64 = self.narrow[10](self.narrow[9](h128d))
            h32 = self.narrow[12](self.narrow[11](h64))
            y_norm = self.narrow[13](h32).squeeze(-1)
            inter["h32"] = h32
            inter["h128"] = h128
            inter["h256"] = h256
            inter["z768"] = z768
            inter["y_norm"] = y_norm
            return inter

        def forward(self, H, P):
            return self.forward_all(H, P)["y_norm"]

    return RevisedSlimModel


def extract_requested_features(
    H_mmap,
    P_mmap,
    sample_idx,
    checkpoint: Path,
    batch_size: int,
    device: str,
    requested: List[str],
):
    import numpy as np
    import torch
    from tqdm import tqdm

    non_model = {"sequence"}
    model_sources = [s for s in requested if s not in non_model]
    if not model_sources:
        return {}, None

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    D = int(ckpt.get("D", H_mmap.shape[-1]))
    hid = int(ckpt.get("hid", 256))
    use_message = bool(ckpt.get("use_message", True))
    y_mean = float(ckpt.get("y_mean", 0.0))
    y_std = float(ckpt.get("y_std", 1.0))

    RevisedSlimModel = build_model_class()
    model = RevisedSlimModel(D=D, hid=hid, use_message=use_message).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    buckets: Dict[str, List] = {k: [] for k in model_sources}
    y_preds = []

    for start in tqdm(range(0, len(sample_idx), batch_size), desc="extract requested intermediate features"):
        idx = sample_idx[start : start + batch_size]
        H = torch.from_numpy(np.asarray(H_mmap[idx], dtype=np.float32)).to(device, non_blocking=True)
        P = torch.from_numpy(np.asarray(P_mmap[idx], dtype=np.float32)).to(device, non_blocking=True)
        with torch.no_grad():
            out = model.forward_all(H, P)
            for source in model_sources:
                buckets[source].append(out[source].detach().float().cpu().numpy())
            y_pred = out["y_norm"].detach().float().cpu().numpy() * y_std + y_mean
            y_preds.append(y_pred[:, None])

    reps = {k: np.concatenate(v, axis=0).astype("float32") for k, v in buckets.items()}
    pred = np.concatenate(y_preds, axis=0).reshape(-1).astype("float32")
    return reps, pred


def robust_clip(z, low_pct: float, high_pct: float):
    import numpy as np

    z = np.asarray(z, dtype=np.float32)
    if low_pct <= 0 and high_pct >= 100:
        return z
    lo = np.nanpercentile(z, low_pct)
    hi = np.nanpercentile(z, high_pct)
    return np.clip(z, lo, hi).astype("float32")


def knn_smooth_points(xy, z, k: int = 25, bandwidth_scale: float = 1.0):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    xy = np.asarray(xy, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    n = len(z)
    if k <= 1 or n <= 3:
        return z
    kk = min(k, n)
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean", algorithm="auto")
    dist, idx = nn.fit(xy).kneighbors(xy, return_distance=True)
    bw = max(float(np.median(dist[:, -1])) * float(bandwidth_scale), 1e-8)
    w = np.exp(-(dist ** 2) / (2.0 * bw * bw)).astype(np.float32)
    w = w / w.sum(axis=1, keepdims=True).clip(1e-8)
    return np.sum(w * z[idx], axis=1).astype("float32")


def make_smooth_grid(
    xy,
    z,
    grid_size: int = 220,
    point_k: int = 25,
    sigma: float = 2.2,
    interpolation: str = "linear",
    clip_low: float = 0.0,
    clip_high: float = 100.0,
    bandwidth_scale: float = 1.0,
):
    import numpy as np
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    xy = np.asarray(xy, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    z0 = robust_clip(z, clip_low, clip_high)
    z1 = knn_smooth_points(xy, z0, k=point_k, bandwidth_scale=bandwidth_scale)

    x = xy[:, 0]
    y = xy[:, 1]
    pad_x = 0.03 * max(float(x.max() - x.min()), 1e-6)
    pad_y = 0.03 * max(float(y.max() - y.min()), 1e-6)
    gx = np.linspace(float(x.min() - pad_x), float(x.max() + pad_x), grid_size)
    gy = np.linspace(float(y.min() - pad_y), float(y.max() + pad_y), grid_size)
    Xg, Yg = np.meshgrid(gx, gy)

    Z_linear = griddata(xy, z1, (Xg, Yg), method=interpolation)
    Z_nearest = griddata(xy, z1, (Xg, Yg), method="nearest")
    valid = np.isfinite(Z_linear).astype(np.float32)
    Z_filled = np.where(np.isfinite(Z_linear), Z_linear, Z_nearest).astype(np.float32)

    if sigma > 0:
        numerator = gaussian_filter(Z_filled * np.maximum(valid, 1e-3), sigma=sigma)
        denominator = gaussian_filter(np.maximum(valid, 1e-3), sigma=sigma)
        Z_smooth = numerator / np.maximum(denominator, 1e-8)
    else:
        Z_smooth = Z_filled

    Z_smooth = np.clip(Z_smooth, float(np.nanmin(z0)), float(np.nanmax(z0)))
    return Xg.astype("float32"), Yg.astype("float32"), Z_smooth.astype("float32")


def compute_high_anomalies(
    xy,
    z,
    k: int = 30,
    z_threshold: float = 2.5,
    top_n: int = 200,
):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    xy = np.asarray(xy, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    n = len(z)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean", algorithm="auto")
    idx = nn.fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    neigh = z[idx]
    mu = neigh.mean(axis=1)
    sd = neigh.std(axis=1) + 1e-6
    local_z = (z - mu) / sd
    candidates = np.where(local_z >= z_threshold)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int), local_z.astype("float32")
    order = candidates[np.argsort(-local_z[candidates])]
    if top_n > 0:
        order = order[:top_n]
    return order.astype(int), local_z.astype("float32")


def save_terrain_html(
    xy,
    z,
    out_html: Path,
    title: str,
    sample_idx,
    ids=None,
    seq_ids=None,
    seqs=None,
    brightness_true=None,
    brightness_pred=None,
    grid_size: int = 220,
    point_k: int = 25,
    sigma: float = 2.2,
    interpolation: str = "linear",
    clip_low: float = 0.0,
    clip_high: float = 100.0,
    bandwidth_scale: float = 1.0,
    show_points: bool = False,
    true_anom_idx=None,
    pred_anom_idx=None,
    true_local_z=None,
    pred_local_z=None,
):
    import numpy as np
    import plotly.graph_objects as go

    Xg, Yg, Zg = make_smooth_grid(
        xy=xy,
        z=z,
        grid_size=grid_size,
        point_k=point_k,
        sigma=sigma,
        interpolation=interpolation,
        clip_low=clip_low,
        clip_high=clip_high,
        bandwidth_scale=bandwidth_scale,
    )
    cmin = float(np.nanmin(z))
    cmax = float(np.nanmax(z))

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=Xg,
            y=Yg,
            z=Zg,
            surfacecolor=Zg,
            colorscale="RdBu_r",
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title="brightness"),
            opacity=0.97,
            contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
        )
    )

    if show_points:
        fig.add_trace(
            go.Scatter3d(
                x=xy[:, 0],
                y=xy[:, 1],
                z=z,
                mode="markers",
                marker=dict(size=1.6, color=z, colorscale="RdBu_r", cmin=cmin, cmax=cmax, opacity=0.35),
                name="all observed variants",
            )
        )

    z_lift = 0.018 * max(cmax - cmin, 1e-6)

    def add_marker_trace(anom_idx, local_z_values, origin_label, marker_color, marker_symbol, marker_size):
        if anom_idx is None or len(anom_idx) == 0:
            return
        hover = []
        for ii in anom_idx:
            parts = [
                f"anomaly_origin={origin_label}",
                f"row_index={int(sample_idx[ii])}",
                f"current_plot_z={float(z[ii]):.4f}",
            ]
            if local_z_values is not None:
                parts.append(f"{origin_label}_local_z={float(local_z_values[ii]):.2f}")
            if ids is not None:
                parts.append(f"id={ids[ii]}")
            if seq_ids is not None:
                parts.append(f"seq_id={seq_ids[ii]}")
            if brightness_true is not None:
                parts.append(f"true={float(brightness_true[ii]):.4f}")
            if brightness_pred is not None:
                parts.append(f"pred={float(brightness_pred[ii]):.4f}")
            if seqs is not None:
                s = str(seqs[ii])
                parts.append(f"seq={s[:80]}{'...' if len(s) > 80 else ''}")
            hover.append("<br>".join(parts))
        fig.add_trace(
            go.Scatter3d(
                x=xy[anom_idx, 0],
                y=xy[anom_idx, 1],
                z=z[anom_idx] + z_lift,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    symbol=marker_symbol,
                    opacity=0.98,
                    line=dict(color="white", width=1.5),
                ),
                text=hover,
                hoverinfo="text",
                name=origin_label,
            )
        )

    # true-local anomalies are shown on both true and pred terrains
    # purple x = observed experimentally high relative to local neighbors
    add_marker_trace(
        true_anom_idx,
        true_local_z,
        origin_label="true_local_high",
        marker_color="#7B2CBF",
        marker_symbol="x",
        marker_size=7.5,
    )

    # pred-local anomalies are shown on both true and pred terrains
    # bright magenta square = simulator/model thinks it is locally high
    add_marker_trace(
        pred_anom_idx,
        pred_local_z,
        origin_label="pred_local_high",
        marker_color="#FF1493",
        marker_symbol="square",
        marker_size=6.8,
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x: distance-layout coordinate",
            yaxis_title="y: distance-layout coordinate",
            zaxis_title="brightness",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.05)),
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        template="plotly_white",
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)

    return None


def save_layout_csv(out_csv: Path, xy, sample_idx, brightness_true, brightness_pred, ids=None, seq_ids=None):
    import pandas as pd

    df = pd.DataFrame(
        {
            "row_index": sample_idx.astype(int),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "brightness_true": brightness_true,
            "brightness_pred": brightness_pred,
        }
    )
    if ids is not None:
        df["id"] = ids
    if seq_ids is not None:
        df["seq_id"] = seq_ids
    df.to_csv(out_csv, index=False)


def load_predictions_and_master(data_dir: Path, predictions_path: Optional[Path]):
    import pandas as pd

    master_path = data_dir / "master_20000.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master file: {master_path}")
    master = pd.read_csv(master_path).reset_index(drop=True)

    if predictions_path is None or not predictions_path.exists():
        possible = [data_dir.parent / "results" / "predictions.csv", data_dir / "predictions.csv"]
        predictions_path = next((p for p in possible if p.exists()), None)
    if predictions_path is None:
        raise FileNotFoundError("Cannot find predictions.csv. Please pass --predictions /vol/.../predictions.csv")

    pred = pd.read_csv(predictions_path)
    required = {"row_index", "brightness_true", "brightness_pred"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"predictions.csv missing columns: {missing}")
    pred = pred.drop_duplicates("row_index").set_index("row_index", drop=False)
    return master, pred, predictions_path


@app.function(
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 12,
    memory=131072,
    volumes={"/vol": volume, "/data_src": data_volume},
)
def make_intermediate_feature_terrains(
    data_dir: str = "/vol/revised_slim_package/dataset",
    checkpoint: str = "/vol/revised_slim_package/weights/best_model.pt",
    predictions: str = "",
    out_dir: str = "/vol/revised_slim_package/results/intermediate_feature_terrains",
    sample_size: int = 5000,
    seed: int = 0,
    batch_size: int = 64,
    distance_chunk_size: int = 128,
    layout_method: str = "umap",
    split: str = "all",
    sources: str = "all",
    grid_size: int = 220,
    point_k: int = 25,
    sigma: float = 2.2,
    interpolation: str = "linear",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    bandwidth_scale: float = 1.0,
    show_points: bool = False,
    anomaly_k: int = 30,
    anomaly_z: float = 2.5,
    top_anomalies: int = 200,
) -> str:
    import numpy as np
    import pandas as pd
    import torch

    set_seed(seed)
    data_dir_p = Path(data_dir)
    checkpoint_p = Path(checkpoint)
    predictions_p = Path(predictions) if predictions else None
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    data_dir_p.mkdir(parents=True, exist_ok=True)
    for f in ["pred3di_saprot_residue_embeddings.npy", "pos_feats.npy"]:
        src = Path("/data_src") / f
        dst = data_dir_p / f
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
            print(f"[link] {dst} -> {src}")

    H_path = data_dir_p / "pred3di_saprot_residue_embeddings.npy"
    P_path = data_dir_p / "pos_feats.npy"
    split_path = data_dir_p / "split_indices.npz"
    if not H_path.exists():
        raise FileNotFoundError(f"Missing H embeddings: {H_path}")
    if not P_path.exists():
        raise FileNotFoundError(f"Missing P features: {P_path}")
    if not checkpoint_p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_p}")

    requested = parse_sources(sources)
    print(f"[INFO] requested sources ({len(requested)}): {requested}")

    master, pred, used_predictions_path = load_predictions_and_master(data_dir_p, predictions_p)
    seq_col = find_col(master, ["seq", "sequence", "aa_sequence", "protein_sequence", "variant_seq", "mutant_sequence"])
    id_col = find_col(master, ["id", "name"], required=False)
    seq_id_col = find_col(master, ["seq_id", "variant_id"], required=False)

    H = np.load(H_path, mmap_mode="r")
    P = np.load(P_path, mmap_mode="r")
    n_total = H.shape[0]

    valid_rows = np.array(sorted(set(pred.index.astype(int)).intersection(set(range(n_total)))), dtype=np.int64)
    if split != "all":
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        spl = np.load(split_path)
        key = f"{split}_idx"
        if key not in spl:
            raise ValueError("--split must be all/train/val/test")
        valid_rows = np.array(sorted(set(valid_rows.tolist()).intersection(set(np.asarray(spl[key], dtype=int).tolist()))), dtype=np.int64)

    rng = np.random.default_rng(seed)
    if sample_size <= 0 or sample_size >= len(valid_rows):
        sample_idx = valid_rows.copy()
    else:
        sample_idx = np.sort(rng.choice(valid_rows, size=sample_size, replace=False))

    brightness_true = pred.loc[sample_idx, "brightness_true"].to_numpy(dtype=np.float32)
    brightness_pred_csv = pred.loc[sample_idx, "brightness_pred"].to_numpy(dtype=np.float32)
    ids = master.iloc[sample_idx][id_col].astype(str).to_numpy() if id_col else None
    seq_ids = master.iloc[sample_idx][seq_id_col].astype(str).to_numpy() if seq_id_col else None
    seqs = master.iloc[sample_idx][seq_col].astype(str).to_numpy()

    print(f"[INFO] sample={len(sample_idx)} split={split}")
    print(f"[INFO] GPU_CONFIG={GPU_CONFIG}; torch_cuda={torch.cuda.is_available()}")
    print(f"[INFO] layout={layout_method}; grid={grid_size}; sigma={sigma}; point_k={point_k}")

    # Extract all requested model-side features once.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps, brightness_pred_model = extract_requested_features(
        H_mmap=H,
        P_mmap=P,
        sample_idx=sample_idx,
        checkpoint=checkpoint_p,
        batch_size=batch_size,
        device=device,
        requested=requested,
    )

    all_anomaly_rows = []
    arrays_to_save = {
        "sample_idx": sample_idx,
        "brightness_true": brightness_true,
        "brightness_pred_csv": brightness_pred_csv,
    }
    if brightness_pred_model is not None:
        arrays_to_save["brightness_pred_model"] = brightness_pred_model

    # Process each source sequentially to avoid holding many distance matrices.
    for source in requested:
        print(f"\n[SOURCE] {source}")
        if source == "sequence":
            encoded = aa_encode_sequences(seqs.tolist())
            D = pairwise_hamming_distance_uint8(encoded, chunk_size=distance_chunk_size)
        else:
            rep = reps[source]
            D = pairwise_cosine_distance(rep, chunk_size=512)
            arrays_to_save[f"rep_{source}"] = rep

        xy, layout_name = layout_from_distance(D, seed=seed, method=layout_method)
        arrays_to_save[f"xy_{source}"] = xy
        save_layout_csv(out_dir_p / f"layout_{source}_xy.csv", xy, sample_idx, brightness_true, brightness_pred_csv, ids, seq_ids)

        # Compute two anomaly sets ONCE per source:
        # 1) true-local-high: high compared with neighbors on the TRUE brightness landscape
        # 2) pred-local-high: high compared with neighbors on the PREDICTED/simulator landscape
        # Both sets are drawn on BOTH true and pred terrain HTML files.
        true_anom_idx, true_local_z = compute_high_anomalies(
            xy,
            brightness_true,
            k=anomaly_k,
            z_threshold=anomaly_z,
            top_n=top_anomalies,
        )
        pred_anom_idx, pred_local_z = compute_high_anomalies(
            xy,
            brightness_pred_csv,
            k=anomaly_k,
            z_threshold=anomaly_z,
            top_n=top_anomalies,
        )
        arrays_to_save[f"local_z_{source}_true"] = true_local_z
        arrays_to_save[f"local_z_{source}_pred"] = pred_local_z

        print(
            f"[ANOMALY] {source}: true_local_high={len(true_anom_idx)} "
            f"pred_local_high={len(pred_anom_idx)}"
        )

        for anomaly_type, anom_idx, local_z_values, marker_color, marker_symbol in [
            ("true_local_high", true_anom_idx, true_local_z, "purple", "x"),
            ("pred_local_high", pred_anom_idx, pred_local_z, "bright_magenta", "square"),
        ]:
            for ii in anom_idx:
                all_anomaly_rows.append(
                    {
                        "source": source,
                        "anomaly_type": anomaly_type,
                        "marker_color": marker_color,
                        "marker_symbol": marker_symbol,
                        "row_index": int(sample_idx[ii]),
                        "local_z": float(local_z_values[ii]),
                        "brightness_true": float(brightness_true[ii]),
                        "brightness_pred": float(brightness_pred_csv[ii]),
                        "id": str(ids[ii]) if ids is not None else "",
                        "seq_id": str(seq_ids[ii]) if seq_ids is not None else "",
                        "seq": str(seqs[ii]),
                    }
                )

        for label, z in [("true", brightness_true), ("pred", brightness_pred_csv)]:
            html_path = out_dir_p / f"terrain_{source}_{label}.html"
            save_terrain_html(
                xy=xy,
                z=z,
                out_html=html_path,
                title=f"{source} terrain: z={label} brightness ({layout_name})",
                sample_idx=sample_idx,
                ids=ids,
                seq_ids=seq_ids,
                seqs=seqs,
                brightness_true=brightness_true,
                brightness_pred=brightness_pred_csv,
                grid_size=grid_size,
                point_k=point_k,
                sigma=sigma,
                interpolation=interpolation,
                clip_low=clip_low,
                clip_high=clip_high,
                bandwidth_scale=bandwidth_scale,
                show_points=show_points,
                true_anom_idx=true_anom_idx,
                pred_anom_idx=pred_anom_idx,
                true_local_z=true_local_z,
                pred_local_z=pred_local_z,
            )
            print(
                f"[WRITE] {html_path.name}; marked true_local_high={len(true_anom_idx)}, "
                f"pred_local_high={len(pred_anom_idx)}"
            )

        del D
        gc.collect()

    np.savez_compressed(out_dir_p / "intermediate_terrain_arrays.npz", **arrays_to_save)
    pd.DataFrame(all_anomaly_rows).to_csv(out_dir_p / "marked_high_local_anomalies.csv", index=False)

    config = {
        "volume_name": VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "gpu_config": str(GPU_CONFIG),
        "data_dir": data_dir,
        "checkpoint": checkpoint,
        "predictions": str(used_predictions_path),
        "out_dir": out_dir,
        "sample_size": int(len(sample_idx)),
        "seed": seed,
        "split": split,
        "sources": requested,
        "layout_method": layout_method,
        "grid_size": grid_size,
        "point_k": point_k,
        "sigma": sigma,
        "interpolation": interpolation,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "anomaly_definition": "local_z = (z_i - mean(z_neighbors)) / std(z_neighbors) in x-y layout space",
        "anomaly_k": anomaly_k,
        "anomaly_z": anomaly_z,
        "top_anomalies": top_anomalies,
    }
    (out_dir_p / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    readme = [
        "# GFP intermediate feature terrains",
        "",
        "Each HTML is a smoothed 3D terrain surface, not only a point cloud.",
        "",
        "x-y: UMAP/MDS layout from pairwise feature distances.",
        "z: true or predicted brightness.",
        "color: brightness.",
        "purple x markers: true-local-high proteins, also copied onto predicted terrains.",
        "bright magenta square markers: pred-local-high proteins, also copied onto true terrains.",
        "",
        "Sources:",
    ]
    for s in requested:
        readme.append(f"- {s}: `terrain_{s}_true.html`, `terrain_{s}_pred.html`")
    (out_dir_p / "README.md").write_text("\n".join(readme), encoding="utf-8")

    volume.commit()
    print(f"[DONE] Saved all outputs to {out_dir_p}")
    return str(out_dir_p)


@app.local_entrypoint()
def main(
    data_dir: str = "/vol/revised_slim_package/dataset",
    checkpoint: str = "/vol/revised_slim_package/weights/best_model.pt",
    predictions: str = "",
    out_dir: str = "/vol/revised_slim_package/results/intermediate_feature_terrains",
    sample_size: int = 5000,
    seed: int = 0,
    batch_size: int = 64,
    distance_chunk_size: int = 128,
    layout_method: str = "umap",
    split: str = "all",
    sources: str = "all",
    grid_size: int = 220,
    point_k: int = 25,
    sigma: float = 2.2,
    interpolation: str = "linear",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    bandwidth_scale: float = 1.0,
    show_points: bool = False,
    anomaly_k: int = 30,
    anomaly_z: float = 2.5,
    top_anomalies: int = 200,
):
    result_path = make_intermediate_feature_terrains.remote(
        data_dir=data_dir,
        checkpoint=checkpoint,
        predictions=predictions,
        out_dir=out_dir,
        sample_size=sample_size,
        seed=seed,
        batch_size=batch_size,
        distance_chunk_size=distance_chunk_size,
        layout_method=layout_method,
        split=split,
        sources=sources,
        grid_size=grid_size,
        point_k=point_k,
        sigma=sigma,
        interpolation=interpolation,
        clip_low=clip_low,
        clip_high=clip_high,
        bandwidth_scale=bandwidth_scale,
        show_points=show_points,
        anomaly_k=anomaly_k,
        anomaly_z=anomaly_z,
        top_anomalies=top_anomalies,
    )
    print(f"\nRemote result path: {result_path}")
    print("\nDownload example:")
    rel = result_path.replace('/vol/', '').lstrip('/')
    print(f"modal volume get {VOLUME_NAME} {rel} ./intermediate_feature_terrains")
