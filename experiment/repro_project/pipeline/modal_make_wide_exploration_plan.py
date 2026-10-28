#!/usr/bin/env python3
"""v13.5 wide exploration plan maker.

Purpose:
  Expand v13.2/v13.4 all-high exploration from a small, over-deduplicated
  mountain set (~64 mountains) to a wide mining plan (~900 mountains, up to
  2700 TG-ITO starts) while keeping the validated v13.2 all-high miner/gate
  and 12-source terrain.

Key design:
  - Use TRUE-brightness UMAP peaks, not predicted peaks, for final mining.
  - Keep many local mountains. Only merge peaks when their local row sets are
    almost identical AND their true-brightness distributions are very similar.
  - Do NOT merge simply because the summit row is the same.
  - Select up to target_mountains (default 900) with source-balanced round-robin + score fill.
  - For each mountain, generate summit/ridge/edge seeds.
"""
from __future__ import annotations

import os
from pathlib import Path
import modal

VOLUME_NAME = os.environ.get("GFP_VOLUME", "gfp-revised-slim")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pandas", "scikit-learn", "scipy")
app = modal.App("pplu-v13-5-wide-exploration-plan", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

SOURCES_12 = [
    "global_attn_pool", "global_mean_pool", "global_max_pool",
    "region9_max_pool", "region9_mean_pool",
    "region_win_165_174_all_mut512", "region_win_77_86_all_mut512",
    "keysite_K_flat", "region_win_97_106_all_mut512", "region_win_27_36_all_mut512",
    "region_win_55_64_all_mut512", "sequence",
]

# Source priors: combine your v13.1/v13.2 learned gate preference and irreplacable-peak value.
# These are only used for mining budget allocation; the actual TG-ITO step still uses dynamic risk top5.
SOURCE_BUDGET_PRIOR = {
    "sequence": 1.25,
    "global_mean_pool": 1.20,
    "global_attn_pool": 1.15,
    "region9_max_pool": 1.10,
    "region_win_165_174_all_mut512": 1.05,
    "region9_mean_pool": 0.95,
    "region_win_77_86_all_mut512": 0.90,
    "region_win_27_36_all_mut512": 0.90,
    "global_max_pool": 0.85,
    "keysite_K_flat": 0.85,
    "region_win_97_106_all_mut512": 0.80,
    "region_win_55_64_all_mut512": 0.75,
}

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def first_col(df, names, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if str(n).lower() in lower:
            return lower[str(n).lower()]
    if required:
        raise ValueError(f"Cannot find columns {names}; got {list(df.columns)}")
    return None


def load_rows(path: Path):
    import pandas as pd
    df = pd.read_csv(path)
    col = first_col(df, ["row_index", "row", "index"], required=False)
    if col is None:
        return set(map(int, df.index.tolist()))
    return set(pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist())


def clean_seq(s: str) -> str:
    return ''.join(ch for ch in str(s).upper() if ch in AA20)


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


@app.function(timeout=60 * 60 * 3, memory=65536, volumes={"/vol": volume})
def make_plan(
    out_csv: str = "/vol/tgito_outputs/v13_5_wide_exploration_plan/v13_5_wide_explore_plan.csv",
    out_peaks_csv: str = "/vol/tgito_outputs/v13_5_wide_exploration_plan/v13_5_true_peaks_retained.csv",
    out_mountains_csv: str = "/vol/tgito_outputs/v13_5_wide_exploration_plan/v13_5_wide_mountain_groups.csv",
    umap_dir: str = "/vol/revised_slim_package/results/pplu_terrains",
    quality_csv: str = "/vol/tgito_outputs/v13_2_scores/v13_2_allhigh_source_quality.csv",
    master_csv: str = "/vol/revised_slim_package/results/pplu_terrains/pplu_31k_predictions.csv",
    terrain_npz: str = "/vol/intermediate_terrains_all/v13_2_pplu_12source_allhigh_fullterrain.npz",
    target_mountains: int = 900,
    max_runs: int = 2700,
    per_source_cap: int = 120,
    min_per_source: int = 45,
    steps: int = 90,
    proposal_candidates: int = 64,
    num_restarts: int = 1,
    grid_size: int = 50,
    th_mean: float = 3.90,
    th_max: float = 4.20,
    top_n: int = 200,
    # strict micro-dedup thresholds; intentionally much stricter than v13.2/v13.4
    jaccard_thr: float = 0.92,
    mean_diff_thr: float = 0.035,
    q90_diff_thr: float = 0.050,
    frac_diff_thr: float = 0.060,
    smoke: bool = False,
):
    import json
    import numpy as np
    import pandas as pd

    out_p = Path(out_csv); out_p.parent.mkdir(parents=True, exist_ok=True)
    peaks_p = Path(out_peaks_csv); peaks_p.parent.mkdir(parents=True, exist_ok=True)
    mountains_p = Path(out_mountains_csv); mountains_p.parent.mkdir(parents=True, exist_ok=True)

    train = load_rows(Path("/vol/intermediate_terrains_all/v12_9_pplu_high42_module_split.released_train.csv"))
    val = load_rows(Path("/vol/intermediate_terrains_all/v12_9_pplu_high42_module_split.released_val.csv"))
    hidden = load_rows(Path("/vol/intermediate_terrains_all/v12_9_pplu_high42_module_split.hidden_test.csv"))
    all_high = sorted(train | val | hidden)
    high_set = set(all_high)
    print(f"[split] train={len(train)} val={len(val)} hidden_in_final={len(hidden)} all_high={len(all_high)}")

    master = pd.read_csv(master_csv)
    seq_col = first_col(master, ["seq", "sequence", "aa_sequence", "protein_sequence", "variant_seq", "mutant_sequence"])
    seqs = master[seq_col].astype(str).map(clean_seq).tolist()

    ref = pd.read_csv(Path(umap_dir) / "umap_global_mean_pool.csv")
    btrue = pd.to_numeric(ref[first_col(ref, ["brightness_true", "Brightness", "true"])], errors="coerce").to_numpy(float)
    bpred = pd.to_numeric(ref[first_col(ref, ["brightness_pred", "pred", "prediction"])], errors="coerce").to_numpy(float)
    top_set = set(np.argsort(-btrue)[:int(top_n)].astype(int).tolist())

    if Path(quality_csv).exists():
        q = pd.read_csv(quality_csv)
    else:
        q = pd.DataFrame({"source": SOURCES_12, "learned_quality": [0.7] * len(SOURCES_12)})
    qmap = {}
    rmap = {}
    smap = {}
    for _, r in q.iterrows():
        src = str(r.get("source"))
        def get_float(keys, default):
            for k in keys:
                if k in r and str(r[k]) not in {"?", "nan", "None", ""}:
                    try:
                        return float(r[k])
                    except Exception:
                        pass
            return default
        qmap[src] = get_float(["learned_quality", "quality", "source_quality"], 0.70)
        rmap[src] = get_float(["risk_mean_allhigh", "risk", "risk_mean"], 0.35)
        smap[src] = get_float(["support_mean_allhigh", "support", "support_mean"], 0.0)

    def grid_assign(xy):
        xb = np.linspace(xy[:, 0].min() - 0.01, xy[:, 0].max() + 0.01, int(grid_size) + 1)
        yb = np.linspace(xy[:, 1].min() - 0.01, xy[:, 1].max() + 0.01, int(grid_size) + 1)
        xi = np.clip(np.digitize(xy[:, 0], xb) - 1, 0, int(grid_size) - 1)
        yi = np.clip(np.digitize(xy[:, 1], yb) - 1, 0, int(grid_size) - 1)
        return xi, yi

    def local_rows_from_cell(xi, yi, cx, cy):
        mask = np.zeros_like(xi, dtype=bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                mask |= ((xi == cx + dx) & (yi == cy + dy))
        return np.where(mask)[0].astype(int)

    def find_true_peaks(xy, true_vals):
        xi, yi = grid_assign(xy)
        cell_max = {}
        for i, tv in enumerate(true_vals):
            if not np.isfinite(tv):
                continue
            c = (int(xi[i]), int(yi[i]))
            if c not in cell_max or tv > cell_max[c][1]:
                cell_max[c] = (i, float(tv))
        peaks = []
        for (cx, cy), (pi, pb) in cell_max.items():
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in cell_max and cell_max[nb][1] >= pb:
                        ok = False; break
                if not ok:
                    break
            if ok:
                peaks.append((int(pi), float(pb), int(cx), int(cy)))
        return peaks, xi, yi

    raw_peaks = []
    per_source_summary = []
    for s in SOURCES_12:
        csv_path = Path(umap_dir) / f"umap_{s}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing UMAP CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        xy = df[[first_col(df, ["u", "x"]), first_col(df, ["v", "y"])]].to_numpy(float)
        peaks, xi, yi = find_true_peaks(xy, btrue)
        kept_for_s = []
        for peak_row, peak_true, cx, cy in peaks:
            loc_rows = local_rows_from_cell(xi, yi, cx, cy)
            loc_true = btrue[loc_rows]
            finite = np.isfinite(loc_true)
            loc_rows = loc_rows[finite]
            loc_true = loc_true[finite]
            if len(loc_rows) == 0:
                continue
            loc_pred = bpred[loc_rows]
            has_top = bool((peak_row in top_set) or any(int(r) in top_set for r in loc_rows.tolist()))
            mean_true = float(np.mean(loc_true))
            max_true = float(np.max(loc_true))
            # Wide900 filter: remove only if mean_true <= 3.9 AND max_true < 4.2 AND no global top200 in local region.
            if ((mean_true <= float(th_mean)) and (max_true < float(th_max)) and (not has_top)):
                continue
            high_loc = [int(r) for r in loc_rows.tolist() if int(r) in high_set]
            if not high_loc:
                continue
            q10, q50, q90 = [float(x) for x in np.quantile(loc_true, [0.10, 0.50, 0.90])]
            source_quality = float(qmap.get(s, 0.70))
            low_risk = float(1.0 / (1.0 + rmap.get(s, 0.35)))
            local_support = float(np.mean(loc_true >= 4.30))
            high435 = float(np.mean(loc_true >= 4.35))
            source_prior = float(SOURCE_BUDGET_PRIOR.get(s, 0.8))
            score = (
                0.22 * source_quality
                + 0.18 * local_support
                + 0.16 * np.clip((max_true - 4.2) / 0.25, 0, 1)
                + 0.12 * high435
                + 0.12 * low_risk
                + 0.10 * source_prior
                + 0.06 * bool(has_top)
                + 0.04 * np.clip((float(btrue[peak_row]) - 4.2) / 0.25, 0, 1)
            )
            kept_for_s.append({
                "peak_id": len(raw_peaks),
                "source": s,
                "row": int(peak_row),
                "summit_row": int(peak_row),
                "true": float(btrue[peak_row]),
                "pred": float(bpred[peak_row]),
                "mean_true": mean_true,
                "std_true": float(np.std(loc_true)),
                "max_true": max_true,
                "q10_true": q10,
                "q50_true": q50,
                "q90_true": q90,
                "mean_pred": float(np.mean(loc_pred)),
                "max_pred": float(np.max(loc_pred)),
                "frac_ge420": float(np.mean(loc_true >= 4.20)),
                "frac_ge430": local_support,
                "frac_ge435": high435,
                "frac_ge440": float(np.mean(loc_true >= 4.40)),
                "has_topN": has_top,
                "n_local_rows": int(len(loc_rows)),
                "n_high_local": int(len(high_loc)),
                "source_quality": source_quality,
                "source_risk_mean": float(rmap.get(s, 0.35)),
                "source_support_mean": float(smap.get(s, 0.0)),
                "source_budget_prior": source_prior,
                "mountain_score": float(score),
                "rowset": set(int(r) for r in loc_rows.tolist()),
                "high_rows": high_loc,
                "xy_source": xy,
            })
        # Optional per-source cap after quality sorting. Keep enough to reach ~600 total.
        kept_for_s = sorted(kept_for_s, key=lambda p: p["mountain_score"], reverse=True)
        if per_source_cap and int(per_source_cap) > 0:
            cap = max(int(min_per_source), int(per_source_cap))
            kept_for_s = kept_for_s[:cap]
        for p in kept_for_s:
            p["peak_id"] = len(raw_peaks)
            raw_peaks.append(p)
        per_source_summary.append((s, len(peaks), len(kept_for_s)))

    print("[peaks] per-source raw/kept after wide filter:")
    for s, raw_n, kept_n in per_source_summary:
        print(f"  {s:38s} raw={raw_n:4d} kept={kept_n:4d}")
    print(f"[peaks] retained_before_micro_dedup={len(raw_peaks)}")
    if not raw_peaks:
        raise RuntimeError("No retained peaks found")

    # Very conservative micro-dedup: only collapse near-identical local regions.
    parent = list(range(len(raw_peaks)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    def row_jaccard(a, b):
        inter = len(a["rowset"] & b["rowset"])
        uni = len(a["rowset"] | b["rowset"])
        return inter / max(1, uni)
    def dist_sim(a, b):
        return (
            abs(a["mean_true"] - b["mean_true"]) <= float(mean_diff_thr)
            and abs(a["q90_true"] - b["q90_true"]) <= float(q90_diff_thr)
            and abs(a["frac_ge435"] - b["frac_ge435"]) <= float(frac_diff_thr)
        )
    merge_pairs = 0
    for i in range(len(raw_peaks)):
        ai = raw_peaks[i]
        for j in range(i + 1, len(raw_peaks)):
            bj = raw_peaks[j]
            # Do not merge just because summit is same. Need almost same local region and distribution.
            if row_jaccard(ai, bj) >= float(jaccard_thr) and dist_sim(ai, bj):
                union(i, j); merge_pairs += 1

    groups = {}
    for i, p in enumerate(raw_peaks):
        groups.setdefault(find(i), []).append(p)

    mountain_rows = []
    seed_rows = []
    for mid0, members in enumerate(groups.values()):
        all_sources = sorted(set(m["source"] for m in members))
        union_rows = sorted(set().union(*(m["rowset"] for m in members)))
        union_high = sorted(set(r for m in members for r in m["high_rows"]))
        if not union_high:
            continue
        loc_true = btrue[union_rows]
        source_count = len(all_sources)
        rep = max(members, key=lambda m: m["mountain_score"])
        quality = float(np.mean([m["source_quality"] for m in members]))
        low_risk = float(1.0 / (1.0 + np.mean([m["source_risk_mean"] for m in members])))
        local_support = float(np.mean(loc_true >= 4.30)) if len(loc_true) else 0.0
        peak_brightness = float(np.max(loc_true)) if len(loc_true) else 0.0
        high435 = float(np.mean(loc_true >= 4.35)) if len(loc_true) else 0.0
        source_prior = float(np.mean([m["source_budget_prior"] for m in members]))
        unique_bonus = 1.0 if source_count == 1 else min(0.6, 0.10 * source_count)
        score = (
            0.22 * quality + 0.18 * local_support
            + 0.16 * np.clip((peak_brightness - 4.2) / 0.25, 0, 1)
            + 0.12 * high435 + 0.12 * low_risk
            + 0.10 * source_prior + 0.06 * unique_bonus
            + 0.04 * min(1.0, source_count / 4.0)
        )
        mountain_id = f"W{mid0:04d}"
        row = {
            "mountain_id": mountain_id,
            "representative_source": rep["source"],
            "representative_peak_id": int(rep["peak_id"]),
            "summit_row": int(rep["summit_row"]),
            "summit_true_brightness": float(btrue[int(rep["summit_row"])]),
            "summit_pred_brightness": float(bpred[int(rep["summit_row"])]),
            "n_member_peaks": int(len(members)),
            "n_sources": int(source_count),
            "all_sources": ",".join(all_sources),
            "n_union_rows": int(len(union_rows)),
            "n_high_rows": int(len(union_high)),
            "mean_true": float(np.mean(loc_true)) if len(loc_true) else 0.0,
            "max_true": peak_brightness,
            "frac_ge420": float(np.mean(loc_true >= 4.20)) if len(loc_true) else 0.0,
            "frac_ge430": local_support,
            "frac_ge435": high435,
            "frac_ge440": float(np.mean(loc_true >= 4.40)) if len(loc_true) else 0.0,
            "source_quality_mean": quality,
            "low_risk_score": low_risk,
            "source_budget_prior": source_prior,
            "uniqueness_bonus": unique_bonus,
            "mountain_score": float(score),
        }
        mountain_rows.append(row)

        high_arr = np.asarray(union_high, dtype=int)
        summit_seed = int(high_arr[np.argmax(btrue[high_arr])])
        xy_rep = rep["xy_source"]
        center = xy_rep[summit_seed]
        dists = np.linalg.norm(xy_rep[high_arr] - center[None, :], axis=1)
        # ridge: bright, not identical to summit, moderate distance.
        bright_norm = np.clip((btrue[high_arr] - 4.2) / 0.25, 0, 1)
        dist_norm = dists / (np.percentile(dists, 90) + 1e-6)
        ridge_score = bright_norm + 0.30 * dist_norm
        ridge_seed = summit_seed
        for oi in np.argsort(-ridge_score):
            cand = int(high_arr[oi])
            if cand != summit_seed:
                ridge_seed = cand; break
        # edge: farthest reasonably bright high row.
        thresh = max(4.20, min(4.35, np.percentile(btrue[high_arr], 50)))
        edge_pool = high_arr[btrue[high_arr] >= thresh]
        if len(edge_pool) == 0:
            edge_pool = high_arr
        edge_d = np.linalg.norm(xy_rep[edge_pool] - center[None, :], axis=1)
        edge_seed = int(edge_pool[np.argmax(edge_d)]) if len(edge_pool) else summit_seed
        roles = [("summit", summit_seed), ("ridge", ridge_seed), ("edge", edge_seed)]
        used = set()
        for role, seed_row in roles:
            # Keep at most one identical seed role within the same mountain; allow same row from other mountains/sources.
            if (role, int(seed_row)) in used:
                continue
            used.add((role, int(seed_row)))
            split = "train" if seed_row in train else "val" if seed_row in val else "hidden_in_final"
            seed_rows.append({
                **row,
                "source": rep["source"],
                "seed_row": int(seed_row),
                "seed_role": role,
                "split_role": split,
                "seed_true_brightness": float(btrue[seed_row]),
                "seed_pred_brightness": float(bpred[seed_row]),
                "seed_score": float(score + 0.25 * np.clip((btrue[seed_row] - 4.2) / 0.25, 0, 1)),
            })

    peaks_export = []
    for p in raw_peaks:
        x = {k: v for k, v in p.items() if k not in {"rowset", "high_rows", "xy_source"}}
        peaks_export.append(x)
    pd.DataFrame(peaks_export).sort_values(["mountain_score", "true"], ascending=False).to_csv(peaks_p, index=False)

    mdf = pd.DataFrame(mountain_rows)
    if mdf.empty:
        raise RuntimeError("No mountain groups after micro-dedup")
    mdf = mdf.sort_values(["mountain_score", "summit_true_brightness"], ascending=False).reset_index(drop=True)

    # Source-balanced mountain selection: first round-robin by representative source, then score-fill.
    selected_ids = []
    by_source = {s: g.reset_index(drop=True) for s, g in mdf.groupby("representative_source")}
    curs = {s: 0 for s in by_source}
    target = int(target_mountains)
    while len(selected_ids) < target and any(curs[s] < len(by_source[s]) for s in by_source):
        # dynamic order by source prior, then current best mountain score
        order = sorted(by_source.keys(), key=lambda s: (SOURCE_BUDGET_PRIOR.get(s, 0.8), by_source[s].iloc[curs[s]]["mountain_score"] if curs[s] < len(by_source[s]) else -1), reverse=True)
        made = False
        for s in order:
            if len(selected_ids) >= target:
                break
            if curs[s] >= len(by_source[s]):
                continue
            mid = by_source[s].iloc[curs[s]]["mountain_id"]
            curs[s] += 1
            if mid not in selected_ids:
                selected_ids.append(mid); made = True
        if not made:
            break
    # Fill any remaining from global score.
    if len(selected_ids) < target:
        for mid in mdf["mountain_id"].tolist():
            if len(selected_ids) >= target:
                break
            if mid not in selected_ids:
                selected_ids.append(mid)

    selected_set = set(selected_ids[:target])
    mdf["selected_for_plan"] = mdf["mountain_id"].isin(selected_set)
    mdf.to_csv(mountains_p, index=False)

    seeds = pd.DataFrame(seed_rows)
    seeds = seeds[seeds["mountain_id"].isin(selected_set)].copy()
    if seeds.empty:
        raise RuntimeError("No seeds produced for selected mountains")
    # Sort within selected mountains, then output all three roles up to max_runs.
    role_order = {"summit": 0, "ridge": 1, "edge": 2}
    seeds["role_order"] = seeds["seed_role"].map(role_order).fillna(9)
    seeds = seeds.sort_values(["mountain_score", "role_order", "seed_score"], ascending=[False, True, False]).reset_index(drop=True)
    max_total = 1 if bool(smoke) else int(max_runs)
    selected_seed_rows = seeds.head(max_total).to_dict("records")

    rows = []
    fixed_sources = "global_attn_pool,region9_max_pool,region9_mean_pool,global_mean_pool,region_win_165_174_all_mut512"
    for i, r in enumerate(selected_seed_rows):
        out_name = f"v13_5_wide_explore/s{i:04d}_{r['mountain_id']}_{r['source']}_{r['seed_role']}_row{int(r['seed_row'])}"
        rows.append({
            **r,
            "method": "v13_5_wide_true_mountain_risk_top5",
            "experiment_tag": "v13_5_wide_allhigh_true_mountain",
            "out_name": out_name,
            "terrain_npz": terrain_npz,
            "revised_ckpt": "/vol/revised_slim_package/weights/best_model_pplu_31k.pt",
            "bridge_ckpt": "/data/warm_best_model.pt",
            "master_csv": master_csv,
            "reference_row": "pplu_wt",
            "length": "auto",
            "mutable": "all",
            "sources": ",".join(SOURCES_12),
            "steps": int(steps),
            "num_restarts": int(num_restarts),
            "lr": 0.05,
            "terrain_weight": 0.03,
            "mutation_budget": 2,
            "budget_relative": "seed",
            "mut_mask_mode": "changed",
            "brightness_floor_weight": 12,
            "proposal_candidates": int(proposal_candidates),
            "proposal_order": 2,
            "proposal_every": 6,
            "sample_every": 10,
            "terrain_topk": 5,
            "terrain_gate_mode": "risk",
            "terrain_fixed_sources": fixed_sources,
            "terrain_gate_temp": 0.20,
            "terrain_support_weight": 0.55,
            "terrain_risk_weight": 0.55,
            "terrain_pred_gap_weight": 0.18,
            "terrain_source_quality_power": 0.80,
            "true_high_weight": 0.0,
            "embedding_realism_weight": 0.0,
            "embedding_repair_weight": 0.0,
            "miner_weight": 0.0,
            "seed_explore_weight": 0.20,
            "seed_explore_min_change": 1.0,
            "accept_by_objective": True,
            "select_by_objective": True,
            "amp_dtype": "bf16",
        })
    plan = pd.DataFrame(rows)
    plan.to_csv(out_p, index=False)
    meta = {
        "version": "v13.5_wide900_exploration",
        "plan_csv": out_csv,
        "peaks_csv": out_peaks_csv,
        "mountains_csv": out_mountains_csv,
        "true_peak_detection": True,
        "micro_dedup_only": True,
        "dedup_note": "merge only near-identical rowsets and true-brightness distributions; same summit alone not enough",
        "grid_size": int(grid_size),
        "retained_peak_count": int(len(raw_peaks)),
        "mountain_group_count": int(len(mdf)),
        "selected_mountain_count": int(len(selected_set)),
        "plan_rows": int(len(plan)),
        "target_mountains": int(target_mountains),
        "max_runs": int(max_runs),
        "smoke": bool(smoke),
        "thresholds": {
            "th_mean": float(th_mean), "th_max": float(th_max), "top_n": int(top_n),
            "jaccard_thr": float(jaccard_thr), "mean_diff_thr": float(mean_diff_thr),
            "q90_diff_thr": float(q90_diff_thr), "frac_diff_thr": float(frac_diff_thr),
        },
    }
    Path(str(out_p) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    volume.commit()
    print(f"[dedup] retained_peaks={len(raw_peaks)} micro_dedup_mountains={len(mdf)} merge_pairs={merge_pairs}")
    print(f"[select] selected_mountains={len(selected_set)} plan_rows={len(plan)} smoke={bool(smoke)}")
    print("[source counts in plan]")
    print(plan["source"].value_counts().to_string())
    print("[role counts in plan]")
    print(plan["seed_role"].value_counts().to_string())
    print(f"[DONE] wrote {out_p}")
    return str(out_p)


@app.local_entrypoint()
def main(
    target_mountains: int = 900,
    max_runs: int = 2700,
    per_source_cap: int = 120,
    min_per_source: int = 45,
    steps: int = 90,
    proposal_candidates: int = 64,
    num_restarts: int = 1,
    smoke: bool = False,
):
    print(make_plan.remote(
        target_mountains=target_mountains,
        max_runs=max_runs,
        per_source_cap=per_source_cap,
        min_per_source=min_per_source,
        steps=steps,
        proposal_candidates=proposal_candidates,
        num_restarts=num_restarts,
        smoke=smoke,
    ))
