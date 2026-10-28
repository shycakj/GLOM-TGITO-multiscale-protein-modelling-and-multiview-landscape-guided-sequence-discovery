"""
Replot existing GFP intermediate terrain results with colored dot anomaly markers.

This script is for post-processing existing results only.
It does NOT rerun model inference, pairwise distance calculation, or UMAP.

Input directory should contain:

    intermediate_terrain_arrays.npz

from the previous script:

    modal_make_intermediate_feature_terrains.py

The script reads existing:

    xy_<source>
    brightness_true
    brightness_pred_csv
    local_z_<source>_true
    local_z_<source>_pred

and regenerates HTML terrain plots with colored dot markers:

    true-local-high anomalies -> purple small dots
    pred-local-high anomalies -> orange/red small dots
    overlap points            -> cyan small dots
    true-brightness terrain peaks -> blue-violet small dots, shown on BOTH true/pred plots
    pred-brightness terrain peaks -> pale-yellow small dots, shown on BOTH true/pred plots

No black markers, no x, no square.

Example on Windows PowerShell:

    python replot_existing_terrains_with_colored_dots.py `
      --in-dir "D:/桌面/GFP/intermediate_terrains_all" `
      --out-dir "D:/桌面/GFP/intermediate_terrains_all_dotmarks" `
      --anomaly-z 2.8 `
      --top-anomalies 80 `
      --top-peaks 60 `
      --grid-size 240 `
      --sigma 2.5 `
      --point-k 35

If still too messy:

    python replot_existing_terrains_with_colored_dots.py `
      --in-dir "D:/桌面/GFP/intermediate_terrains_all" `
      --out-dir "D:/桌面/GFP/intermediate_terrains_all_dotmarks_clean" `
      --anomaly-z 3.2 `
      --top-anomalies 40 `
      --top-peaks 30 `
      --marker-size 2.8 `
      --marker-opacity 0.75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def robust_clip(z, low_pct: float, high_pct: float):
    z = np.asarray(z, dtype=np.float32)
    if low_pct <= 0 and high_pct >= 100:
        return z
    lo = np.nanpercentile(z, low_pct)
    hi = np.nanpercentile(z, high_pct)
    return np.clip(z, lo, hi).astype("float32")


def knn_smooth_points(xy, z, k: int = 35, bandwidth_scale: float = 1.0):
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
    grid_size: int = 240,
    point_k: int = 35,
    sigma: float = 2.5,
    interpolation: str = "linear",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    bandwidth_scale: float = 1.0,
):
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


def pick_top_anomalies(local_z, threshold: float, top_n: int):
    local_z = np.asarray(local_z, dtype=np.float32)
    idx = np.where(local_z >= threshold)[0]
    if len(idx) == 0:
        return np.array([], dtype=int)
    idx = idx[np.argsort(-local_z[idx])]
    if top_n > 0:
        idx = idx[:top_n]
    return idx.astype(int)


def split_marker_sets(true_idx, pred_idx):
    true_set = set(map(int, true_idx))
    pred_set = set(map(int, pred_idx))
    overlap = np.array(sorted(true_set & pred_set), dtype=int)
    true_only = np.array(sorted(true_set - pred_set), dtype=int)
    pred_only = np.array(sorted(pred_set - true_set), dtype=int)
    return true_only, pred_only, overlap


def add_dot_trace(fig, xy, z, idx, name, color, size, opacity, local_z=None, sample_idx=None):
    import plotly.graph_objects as go

    if idx is None or len(idx) == 0:
        return

    hover = []
    for ii in idx:
        parts = [
            f"type={name}",
            f"array_index={int(ii)}",
            f"z={float(z[ii]):.4f}",
        ]
        if sample_idx is not None:
            parts.append(f"row_index={int(sample_idx[ii])}")
        if local_z is not None:
            parts.append(f"local_z={float(local_z[ii]):.2f}")
        hover.append("<br>".join(parts))

    # Lift markers slightly above the surface so dots are visible.
    z_lift = 0.018 * max(float(np.nanmax(z) - np.nanmin(z)), 1e-6)

    fig.add_trace(
        go.Scatter3d(
            x=xy[idx, 0],
            y=xy[idx, 1],
            z=z[idx] + z_lift,
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                symbol="circle",
                opacity=opacity,
                line=dict(color="white", width=0.6),
            ),
            text=hover,
            hoverinfo="text",
            name=name,
        )
    )


def compute_point_local_peaks(xy, z, k: int = 35, top_n: int = 60, min_prominence: float = 0.0):
    """
    Detect local terrain peak tops among observed points.

    A point is a local peak if its z is at least as high as all k nearest
    neighbors in the x-y layout. Peaks are ranked by prominence:

        prominence = z_i - mean(z_neighbors)

    This finds the visible highest point on local mountains, not global-only top.
    """
    from sklearn.neighbors import NearestNeighbors

    xy = np.asarray(xy, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    n = len(z)
    if n <= 3:
        return np.arange(n, dtype=int), np.zeros(n, dtype=np.float32)

    kk = min(k + 1, n)
    idx = NearestNeighbors(n_neighbors=kk, metric="euclidean").fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    neigh_z = z[idx]
    neigh_max = neigh_z.max(axis=1)
    neigh_mean = neigh_z.mean(axis=1)
    prominence = z - neigh_mean

    # tiny tolerance avoids losing peaks due to float equality noise
    candidates = np.where((z >= neigh_max - 1e-8) & (prominence >= min_prominence))[0]
    if len(candidates) == 0:
        return np.array([], dtype=int), prominence.astype("float32")

    # Rank first by prominence, then by actual height.
    order = candidates[np.lexsort((-z[candidates], -prominence[candidates]))]
    if top_n > 0:
        order = order[:top_n]
    return order.astype(int), prominence.astype("float32")


def compute_grid_local_peaks(
    Zg,
    top_n: int = 60,
    peak_window: int = 11,
    min_prominence: float = 0.0,
    min_peak_distance: int = 8,
    exclude_border: int | None = None,
):
    """
    Detect local peak tops on the FINAL SMOOTHED TERRAIN GRID.

    This is the correct peak definition for the visible terrain surface.
    It does not search raw scattered protein points. It searches Zg after
    interpolation and Gaussian smoothing, so the blue-violet/yellow markers
    land on the actual visible mountain tops.
    """
    from scipy.ndimage import maximum_filter, uniform_filter

    Zg = np.asarray(Zg, dtype=np.float32)
    if peak_window < 3:
        peak_window = 3
    if peak_window % 2 == 0:
        peak_window += 1
    if exclude_border is None:
        exclude_border = max(2, peak_window // 2)

    local_max = maximum_filter(Zg, size=peak_window, mode="nearest")
    local_mean = uniform_filter(Zg, size=peak_window, mode="nearest")
    prominence = Zg - local_mean

    mask = np.isfinite(Zg) & (Zg >= local_max - 1e-8) & (prominence >= min_prominence)
    if exclude_border > 0 and Zg.shape[0] > 2 * exclude_border and Zg.shape[1] > 2 * exclude_border:
        mask[:exclude_border, :] = False
        mask[-exclude_border:, :] = False
        mask[:, :exclude_border] = False
        mask[:, -exclude_border:] = False

    rr, cc = np.where(mask)
    if len(rr) == 0:
        return np.empty((0, 2), dtype=int), prominence.astype("float32")

    # Rank by actual terrain height first, then local prominence.
    order = np.lexsort((-prominence[rr, cc], -Zg[rr, cc]))
    coords = np.stack([rr[order], cc[order]], axis=1)

    # Greedy non-maximum suppression, so one mountain does not get many markers.
    kept = []
    min_d2 = float(min_peak_distance * min_peak_distance)
    for r, c in coords:
        if all((float(r - kr) ** 2 + float(c - kc) ** 2) >= min_d2 for kr, kc in kept):
            kept.append((int(r), int(c)))
            if top_n > 0 and len(kept) >= top_n:
                break

    return np.asarray(kept, dtype=int), prominence.astype("float32")


def add_grid_peak_trace(
    fig,
    Xg,
    Yg,
    Zcurrent,
    peak_rc,
    prominence,
    name: str,
    color: str,
    size: float,
    opacity: float,
):
    import plotly.graph_objects as go

    if peak_rc is None or len(peak_rc) == 0:
        return

    z_lift = 0.026 * max(float(np.nanmax(Zcurrent) - np.nanmin(Zcurrent)), 1e-6)
    rr = peak_rc[:, 0]
    cc = peak_rc[:, 1]
    hover = []
    for r, c in zip(rr, cc):
        parts = [
            f"type={name}",
            f"grid_row={int(r)}",
            f"grid_col={int(c)}",
            f"current_surface_z={float(Zcurrent[r, c]):.4f}",
            f"peak_prominence={float(prominence[r, c]):.4f}",
        ]
        hover.append("<br>".join(parts))

    fig.add_trace(
        go.Scatter3d(
            x=Xg[rr, cc],
            y=Yg[rr, cc],
            z=Zcurrent[rr, cc] + z_lift,
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                symbol="circle",
                opacity=opacity,
                line=dict(color="white", width=0.8),
            ),
            text=hover,
            hoverinfo="text",
            name=name,
        )
    )


def apply_visibility_buttons(fig):
    """
    Add buttons so the HTML can hide/show point categories.
    The legend remains clickable as well.
    """
    names = [getattr(trace, "name", "") or "" for trace in fig.data]

    def visible_for(mode: str):
        vis = []
        for name in names:
            if name == "terrain_surface":
                vis.append(True)
            elif mode == "all":
                vis.append(True)
            elif mode == "hide_dots":
                vis.append(False)
            elif mode == "anomalies":
                vis.append(name in {
                    "true_local_high_only",
                    "pred_local_high_only",
                    "both_true_and_pred_local_high",
                })
            elif mode == "peaks":
                vis.append(name in {
                    "true_brightness_local_peak",
                    "pred_brightness_local_peak",
                })
            elif mode == "true_peak":
                vis.append(name == "true_brightness_local_peak")
            elif mode == "pred_peak":
                vis.append(name == "pred_brightness_local_peak")
            else:
                vis.append(True)
        return vis

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.02,
                y=1.08,
                xanchor="left",
                yanchor="top",
                showactive=True,
                buttons=[
                    dict(label="All dots", method="update", args=[{"visible": visible_for("all")}]),
                    dict(label="Hide dots", method="update", args=[{"visible": visible_for("hide_dots")}]),
                    dict(label="Only anomalies", method="update", args=[{"visible": visible_for("anomalies")}]),
                    dict(label="Only peaks", method="update", args=[{"visible": visible_for("peaks")}]),
                    dict(label="True peaks", method="update", args=[{"visible": visible_for("true_peak")}]),
                    dict(label="Pred peaks", method="update", args=[{"visible": visible_for("pred_peak")}]),
                ],
            )
        ]
    )


def save_surface_with_dotmarks(
    xy,
    z,
    surface_grid,
    source: str,
    label: str,
    out_html: Path,
    sample_idx,
    true_idx,
    pred_idx,
    true_local_z,
    pred_local_z,
    true_peak_rc,
    pred_peak_rc,
    true_peak_prominence,
    pred_peak_prominence,
    marker_size: float,
    marker_opacity: float,
    peak_marker_size: float,
    peak_marker_opacity: float,
):
    import plotly.graph_objects as go

    Xg, Yg, Zg = surface_grid

    cmin = float(np.nanmin(z))
    cmax = float(np.nanmax(z))
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=Xg,
            y=Yg,
            z=Zg,
            name="terrain_surface",
            surfacecolor=Zg,
            colorscale="RdBu_r",
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title="brightness"),
            opacity=0.97,
            contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
        )
    )

    true_only, pred_only, overlap = split_marker_sets(true_idx, pred_idx)

    # No black. No x/s. Only colored dots.
    # True-local-high = purple dot.
    # Pred-local-high = orange/red dot.
    # Both true and pred anomaly = cyan dot.
    add_dot_trace(
        fig, xy, z, true_only,
        name="true_local_high_only",
        color="#7B2CBF",
        size=marker_size,
        opacity=marker_opacity,
        local_z=true_local_z,
        sample_idx=sample_idx,
    )

    # Blue-violet points = local peak tops found from TRUE brightness TERRAIN only.
    # They are shown on BOTH true and pred plots, with z following the current plot.
    add_grid_peak_trace(
        fig=fig,
        Xg=Xg,
        Yg=Yg,
        Zcurrent=Zg,
        peak_rc=true_peak_rc,
        prominence=true_peak_prominence,
        name="true_brightness_local_peak",
        color="#5A4BFF",
        size=peak_marker_size,
        opacity=peak_marker_opacity,
    )

    # Pale-yellow points = local peak tops found from PREDICTED brightness TERRAIN only.
    # They are also shown on BOTH true and pred plots for direct comparison.
    add_grid_peak_trace(
        fig=fig,
        Xg=Xg,
        Yg=Yg,
        Zcurrent=Zg,
        peak_rc=pred_peak_rc,
        prominence=pred_peak_prominence,
        name="pred_brightness_local_peak",
        color="#FFE66D",
        size=peak_marker_size,
        opacity=peak_marker_opacity,
    )
    add_dot_trace(
        fig, xy, z, pred_only,
        name="pred_local_high_only",
        color="#FF4D00",
        size=marker_size,
        opacity=marker_opacity,
        local_z=pred_local_z,
        sample_idx=sample_idx,
    )
    add_dot_trace(
        fig, xy, z, overlap,
        name="both_true_and_pred_local_high",
        color="#00D1FF",
        size=marker_size + 1.0,
        opacity=min(1.0, marker_opacity + 0.1),
        local_z=true_local_z,
        sample_idx=sample_idx,
    )

    apply_visibility_buttons(fig)

    fig.update_layout(
        title=f"{source} terrain: z={label} brightness, colored dot anomaly markers",
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

    return {
        "source": source,
        "label": label,
        "true_only": int(len(true_only)),
        "pred_only": int(len(pred_only)),
        "overlap": int(len(overlap)),
        "true_brightness_local_peaks": int(len(true_peak_rc)),
        "pred_brightness_local_peaks": int(len(pred_peak_rc)),
        "total_markers": int(len(true_only) + len(pred_only) + len(overlap)),
    }


def infer_sources(data):
    sources = []
    for key in data.files:
        if key.startswith("xy_"):
            sources.append(key[3:])
    return sorted(sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sources", default="all", help="all or comma-separated source names")
    parser.add_argument("--anomaly-z", type=float, default=2.8)
    parser.add_argument("--top-anomalies", type=int, default=80)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--point-k", type=int, default=35)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--interpolation", default="linear")
    parser.add_argument("--clip-low", type=float, default=1.0)
    parser.add_argument("--clip-high", type=float, default=99.0)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--marker-size", type=float, default=3.0)
    parser.add_argument("--marker-opacity", type=float, default=0.78)
    parser.add_argument("--peak-k", type=int, default=35)
    parser.add_argument("--top-peaks", type=int, default=60)
    parser.add_argument("--peak-window", type=int, default=13)
    parser.add_argument("--min-peak-distance", type=int, default=10)
    parser.add_argument("--peak-min-prominence", type=float, default=0.0)
    parser.add_argument("--peak-marker-size", type=float, default=3.4)
    parser.add_argument("--peak-marker-opacity", type=float, default=0.9)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = in_dir / "intermediate_terrain_arrays.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find {npz_path}")

    data = np.load(npz_path)
    sample_idx = data["sample_idx"] if "sample_idx" in data.files else None
    brightness_true = data["brightness_true"]
    brightness_pred = data["brightness_pred_csv"]

    available_sources = infer_sources(data)
    if args.sources.strip().lower() == "all":
        sources = available_sources
    else:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        missing = [s for s in sources if s not in available_sources]
        if missing:
            raise ValueError(f"Missing sources {missing}. Available sources: {available_sources}")

    print(f"[INFO] sources={sources}")
    print(f"[INFO] anomaly_z={args.anomaly_z}, top_anomalies={args.top_anomalies}")

    summary = []
    for source in sources:
        xy_key = f"xy_{source}"
        true_lz_key = f"local_z_{source}_true"
        pred_lz_key = f"local_z_{source}_pred"

        if xy_key not in data.files:
            print(f"[SKIP] {source}: missing {xy_key}")
            continue
        if true_lz_key not in data.files or pred_lz_key not in data.files:
            print(f"[SKIP] {source}: missing local_z keys")
            continue

        xy = data[xy_key]
        true_local_z = data[true_lz_key]
        pred_local_z = data[pred_lz_key]
        true_idx = pick_top_anomalies(true_local_z, args.anomaly_z, args.top_anomalies)
        pred_idx = pick_top_anomalies(pred_local_z, args.anomaly_z, args.top_anomalies)

        true_grid = make_smooth_grid(
            xy=xy,
            z=brightness_true,
            grid_size=args.grid_size,
            point_k=args.point_k,
            sigma=args.sigma,
            interpolation=args.interpolation,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            bandwidth_scale=args.bandwidth_scale,
        )
        pred_grid = make_smooth_grid(
            xy=xy,
            z=brightness_pred,
            grid_size=args.grid_size,
            point_k=args.point_k,
            sigma=args.sigma,
            interpolation=args.interpolation,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            bandwidth_scale=args.bandwidth_scale,
        )

        true_peak_rc, true_peak_prominence = compute_grid_local_peaks(
            Zg=true_grid[2],
            top_n=args.top_peaks,
            peak_window=args.peak_window,
            min_prominence=args.peak_min_prominence,
            min_peak_distance=args.min_peak_distance,
        )
        pred_peak_rc, pred_peak_prominence = compute_grid_local_peaks(
            Zg=pred_grid[2],
            top_n=args.top_peaks,
            peak_window=args.peak_window,
            min_prominence=args.peak_min_prominence,
            min_peak_distance=args.min_peak_distance,
        )

        for label, z, surface_grid in [("true", brightness_true, true_grid), ("pred", brightness_pred, pred_grid)]:
            out_html = out_dir / f"dotmark_terrain_{source}_{label}.html"
            stat = save_surface_with_dotmarks(
                xy=xy,
                z=z,
                surface_grid=surface_grid,
                source=source,
                label=label,
                out_html=out_html,
                sample_idx=sample_idx,
                true_idx=true_idx,
                pred_idx=pred_idx,
                true_local_z=true_local_z,
                pred_local_z=pred_local_z,
                true_peak_rc=true_peak_rc,
                pred_peak_rc=pred_peak_rc,
                true_peak_prominence=true_peak_prominence,
                pred_peak_prominence=pred_peak_prominence,
                marker_size=args.marker_size,
                marker_opacity=args.marker_opacity,
                peak_marker_size=args.peak_marker_size,
                peak_marker_opacity=args.peak_marker_opacity,
            )
            summary.append(stat)
            print(
                f"[WRITE] {out_html} anomaly_markers={stat['total_markers']} "
                f"true_peaks={stat['true_brightness_local_peaks']} "
                f"pred_peaks={stat['pred_brightness_local_peaks']}"
            )

    config = vars(args)
    config["input_npz"] = str(npz_path)
    config["available_sources"] = available_sources
    config["rendered_sources"] = sources
    config["marker_legend"] = {
        "true_local_high_only": "purple circle dot (#7B2CBF)",
        "pred_local_high_only": "orange/red circle dot (#FF4D00)",
        "both_true_and_pred_local_high": "cyan circle dot (#00D1FF)",
        "true_brightness_local_peak": "blue-violet circle dot (#5A4BFF), detected from true brightness and shown on both true/pred plots",
        "pred_brightness_local_peak": "pale-yellow circle dot (#FFE66D), detected from predicted brightness and shown on both true/pred plots",
    }
    (out_dir / "dotmark_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import pandas as pd

        pd.DataFrame(summary).to_csv(out_dir / "dotmark_summary.csv", index=False)
    except Exception:
        pass

    (out_dir / "README.md").write_text(
        "# Colored-dot terrain replot\n\n"
        "This directory was generated from existing `intermediate_terrain_arrays.npz`.\n\n"
        "No model inference, pairwise distance calculation, or UMAP was rerun.\n\n"
        "Markers:\n\n"
        "- Purple dot: true-local-high only.\n"
        "- Orange/red dot: pred-local-high only.\n"
        "- Cyan dot: both true-local-high and pred-local-high.\n\n"
        "- Blue-violet dot: local peak top detected from true brightness; copied onto both true and pred plots.\n"
        "- Pale-yellow dot: local peak top detected from predicted brightness; copied onto both true and pred plots.\n\n"
        "Buttons at the top can show/hide all dots, anomaly dots, peak dots, true peaks, or pred peaks.\n\n"
        "All markers are circle dots; no black markers, no x/s symbols.\n",
        encoding="utf-8",
    )

    print(f"[DONE] Output: {out_dir}")


if __name__ == "__main__":
    main()
