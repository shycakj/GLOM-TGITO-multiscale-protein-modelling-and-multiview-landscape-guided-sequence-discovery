#!/usr/bin/env python3
"""Aggregate v13.2 all-high candidates and compute confidence/novelty/diversity tables."""
from __future__ import annotations

import os
from pathlib import Path
import modal

VOLUME_NAME = os.environ.get("GFP_VOLUME", "gfp-revised-slim")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pandas", "scipy")
app = modal.App("pplu-v13-2-candidate-confidence", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(s):
    return ''.join(ch for ch in str(s).upper() if ch in AA20)


def hamming(a, b):
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def first_col(df, names, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if str(n).lower() in lower:
            return lower[str(n).lower()]
    if required:
        raise ValueError(f"Cannot find {names}; columns={list(df.columns)}")
    return None


def norm01(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@app.function(timeout=60*60*2, memory=65536, volumes={"/vol": volume})
def summarize(
    plan_csv: str = "/vol/tgito_outputs/v13_2_allhigh_exploration_plan/v13_2_allhigh_explore_plan.csv",
    out_dir: str = "/vol/tgito_outputs/v13_2_scores",
    master_csv: str = "/vol/revised_slim_package/results/pplu_terrains/pplu_31k_predictions.csv",
    top6_min_pairwise_hamming: int = 2,
):
    import numpy as np
    import pandas as pd

    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(plan_csv)
    cand_rows = []
    for _, pr in plan.iterrows():
        fc = Path("/vol/tgito_outputs") / str(pr["out_name"]) / "final_candidates.csv"
        if not fc.exists():
            continue
        df = pd.read_csv(fc)
        if df.empty:
            continue
        # keep all restarts; usually 1, but robust.
        for _, r in df.iterrows():
            row = r.to_dict()
            row.update({f"plan_{k}": pr[k] for k in plan.columns if k not in row})
            row["candidate_file"] = str(fc)
            cand_rows.append(row)
    if not cand_rows:
        raise RuntimeError("No candidates found from plan outputs")
    df = pd.DataFrame(cand_rows)
    seq_col = "seq" if "seq" in df.columns else first_col(df, ["final_seq", "sequence"])
    df["seq_clean"] = df[seq_col].map(clean_seq)
    pred_col = "final_pred_brightness" if "final_pred_brightness" in df.columns else "hard_argmax_pred_brightness" if "hard_argmax_pred_brightness" in df.columns else "pred_brightness"
    if pred_col not in df.columns:
        pred_col = first_col(df, ["pred_brightness", "final_pred", "hard_argmax_pred_brightness"])
    df["pred"] = pd.to_numeric(df[pred_col], errors="coerce")
    support = pd.to_numeric(df.get("terrain_selected_support_wmean", 0.0), errors="coerce").fillna(0.0)
    risk = pd.to_numeric(df.get("terrain_selected_risk_wmean", 0.0), errors="coerce").fillna(0.0)
    gap = pd.to_numeric(df.get("terrain_selected_pred_gap_wmean", 0.0), errors="coerce").fillna(0.0)
    terrain = pd.to_numeric(df.get("terrain", 0.0), errors="coerce").fillna(0.0)
    entropy = pd.to_numeric(df.get("terrain_weight_entropy", 0.0), errors="coerce").fillna(0.0)

    master = pd.read_csv(master_csv)
    mseq_col = first_col(master, ["seq", "sequence", "aa_sequence", "protein_sequence", "variant_seq", "mutant_sequence"])
    known = set(master[mseq_col].astype(str).map(clean_seq).tolist())
    df["exact_in_31k"] = df["seq_clean"].isin(known)
    # nearest distance to 31K: exact and nearest-small is enough; full O(N*M) for small candidates is fine.
    known_list = list(known)
    nearest = []
    for s in df["seq_clean"].tolist():
        if s in known:
            nearest.append(0)
        else:
            # cheap early bound: compute all because candidate count is small.
            nearest.append(min(hamming(s, k) for k in known_list if len(k) == len(s)) if known_list else 999)
    df["nearest_31k_hamming"] = nearest

    df["conf_pred"] = np.clip((df["pred"].fillna(0.0).to_numpy() - 4.20) / 0.25, 0, 1)
    df["conf_support"] = norm01(support)
    df["conf_terrain"] = norm01(terrain)
    df["conf_lowrisk"] = 1.0 - norm01(risk + 0.5 * gap)
    df["conf_entropy"] = norm01(entropy)
    df["confidence_score"] = (
        0.35 * df["conf_pred"]
        + 0.25 * df["conf_support"]
        + 0.20 * df["conf_terrain"]
        + 0.15 * df["conf_lowrisk"]
        + 0.05 * df["conf_entropy"]
        - 0.50 * df["exact_in_31k"].astype(float)
    )
    df = df.sort_values("confidence_score", ascending=False)
    # Dedup by sequence, keeping highest confidence.
    ranked = df.drop_duplicates("seq_clean", keep="first").copy()
    ranked.to_csv(outp / "v13_2_all_candidates_ranked_with_confidence.csv", index=False)

    selected = []
    for _, r in ranked.iterrows():
        if bool(r["exact_in_31k"]):
            continue
        if float(r["pred"]) < 4.31:
            continue
        s = r["seq_clean"]
        if all(hamming(s, x["seq_clean"]) >= int(top6_min_pairwise_hamming) for x in selected):
            selected.append(r.to_dict())
        if len(selected) >= 6:
            break
    top6 = pd.DataFrame(selected)
    backup = ranked[~ranked["seq_clean"].isin(top6["seq_clean"].tolist() if not top6.empty else [])].head(20)
    top6.to_csv(outp / "v13_2_final_top6_submission_candidates.csv", index=False)
    backup.to_csv(outp / "v13_2_final_top20_backup_candidates.csv", index=False)

    # Mean selected source weights for confidence report.
    weight_cols = [c for c in ranked.columns if c.startswith("selected_weight_")]
    if weight_cols:
        sw = []
        for c in weight_cols:
            sw.append({"source": c.replace("selected_weight_", ""), "mean_weight": float(pd.to_numeric(ranked[c], errors="coerce").fillna(0).mean())})
        pd.DataFrame(sw).sort_values("mean_weight", ascending=False).to_csv(outp / "v13_2_mean_selected_source_weights.csv", index=False)
    volume.commit()
    print("[DONE] candidates", len(ranked))
    print("[TOP6]")
    cols = [c for c in ["pred", "confidence_score", "exact_in_31k", "nearest_31k_hamming", "terrain_selected_support_wmean", "terrain_selected_risk_wmean", "seq_clean"] if c in top6.columns]
    if not top6.empty:
        print(top6[cols].to_string(index=False))
    return str(outp)


@app.local_entrypoint()
def main(plan_csv: str = "/vol/tgito_outputs/v13_2_allhigh_exploration_plan/v13_2_allhigh_explore_plan.csv",
         out_dir: str = "/vol/tgito_outputs/v13_2_scores",
         top6_min_pairwise_hamming: int = 2):
    print(summarize.remote(plan_csv=plan_csv, out_dir=out_dir, top6_min_pairwise_hamming=top6_min_pairwise_hamming))
