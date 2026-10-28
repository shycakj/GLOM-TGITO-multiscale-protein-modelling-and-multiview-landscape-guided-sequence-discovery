from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score

from .common import hamming, load_config


def prediction_metrics(y: np.ndarray, p: np.ndarray, ks: list[int], high_threshold: float) -> dict:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
    }
    relevance = y - np.nanmin(y)
    for k in ks:
        if k > len(y):
            continue
        true_top = set(np.argsort(-y)[:k].tolist())
        pred_top = set(np.argsort(-p)[:k].tolist())
        out[f"top{k}_overlap_recall"] = float(len(true_top & pred_top) / k)
        out[f"ndcg@{k}"] = float(ndcg_score(relevance.reshape(1, -1), p.reshape(1, -1), k=k))
        high_idx = set(np.where(y >= high_threshold)[0].tolist())
        out[f"high_recall@{k}"] = float(len(high_idx & pred_top) / max(1, len(high_idx)))
    return out


def score_prediction_file(pred_path: Path, truth_path: Path, cfg: dict, output: Path | None = None) -> dict:
    pred = pd.read_csv(pred_path)
    truth = pd.read_csv(truth_path)
    pred_col = next((c for c in ["prediction", "y_pred", "pred", "brightness_pred"] if c in pred.columns), None)
    if pred_col is None or "sequence" not in pred.columns:
        raise ValueError(f"Prediction file needs sequence and a prediction column; found {list(pred.columns)}")
    m = truth[["sequence", "brightness"]].merge(pred[["sequence", pred_col]], on="sequence", how="inner", validate="one_to_one")
    if len(m) != len(truth):
        raise ValueError(f"Prediction coverage mismatch: matched {len(m)} of {len(truth)} test rows")
    result = prediction_metrics(m.brightness.to_numpy(), m[pred_col].to_numpy(), cfg["prediction_task"]["topk"], float(cfg["prediction_task"]["high_threshold"]))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def score_search_file(cand_path: Path, hidden_path: Path, observed_path: Path, near_hamming: int, topks=(10, 50, 100, 200, 500, 1000), seed_pool_path: Path | None = None) -> tuple[dict, pd.DataFrame]:
    c = pd.read_csv(cand_path)
    h = pd.read_csv(hidden_path)
    o = pd.read_csv(observed_path)
    seq_col = next((x for x in ["sequence", "seq", "final_seq"] if x in c.columns), None)
    score_col = next((x for x in ["common_oracle_score", "model_score", "final_pred_brightness", "prediction", "score"] if x in c.columns), None)
    if seq_col is None:
        raise ValueError(f"No sequence column in candidates: {list(c.columns)}")
    c = c.copy(); c["sequence"] = c[seq_col].astype(str)
    c = c.drop_duplicates("sequence", keep="first")
    observed = set(o.sequence.astype(str))
    hidden_seqs = h.sequence.astype(str).tolist()
    hidden_set = set(hidden_seqs)
    c["is_observed_duplicate"] = c.sequence.isin(observed)
    c["exact_hidden"] = c.sequence.isin(hidden_set)
    # This is final evaluation only; hidden sequences are deliberately not exposed to search scripts.
    min_d, near_idx = [], []
    for s in c.sequence:
        ds = [hamming(s, t) for t in hidden_seqs]
        j = int(np.argmin(ds)); min_d.append(int(ds[j])); near_idx.append(j)
    c["min_hamming_to_hidden"] = min_d
    c["nearest_hidden_index"] = near_idx
    c["near_hidden"] = c.min_hamming_to_hidden <= int(near_hamming)
    if score_col is not None:
        c = c.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    else:
        c = c.reset_index(drop=True)
    novel = c[~c.is_observed_duplicate].copy()

    def hidden_coverage(df: pd.DataFrame, max_h: int) -> int:
        covered = set()
        for s in df.sequence:
            for i, hs in enumerate(hidden_seqs):
                if hamming(s, hs) <= max_h:
                    covered.add(i)
        return len(covered)

    result = {
        "n_candidates_unique": int(len(c)),
        "n_candidates_novel_vs_observed": int(len(novel)),
        "exact_hidden_unique_hits_all": int(novel.exact_hidden.sum()),
        f"hidden_coverage_hamming_le_{near_hamming}_all": int(hidden_coverage(novel, near_hamming)),
        "hidden_total": int(len(h)),
    }
    if seed_pool_path is not None:
        seeds = pd.read_csv(seed_pool_path).sequence.astype(str).tolist()
        reachable = sum(min(hamming(hs, ss) for ss in seeds) <= 2 for hs in hidden_seqs)
        result["hidden_reachable_from_common_seed_pool_hamming_le_2"] = int(reachable)
        result["reachable_fraction"] = float(reachable / max(1, len(hidden_seqs)))
    for k in topks:
        if k > len(novel):
            continue
        sub = novel.iloc[:k]
        result[f"exact_hidden_hits@{k}"] = int(sub.exact_hidden.sum())
        result[f"exact_hidden_recall@{k}"] = float(sub.exact_hidden.sum() / len(h))
        result[f"hidden_coverage_hamming_le_{near_hamming}@{k}"] = int(hidden_coverage(sub, near_hamming))
    return result, c


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    pp = sub.add_parser("prediction")
    pp.add_argument("--predictions", required=True); pp.add_argument("--truth", required=True); pp.add_argument("--out", required=True); pp.add_argument("--config", default=None)
    sp = sub.add_parser("search")
    sp.add_argument("--candidates", required=True); sp.add_argument("--hidden", required=True); sp.add_argument("--observed", required=True); sp.add_argument("--seed-pool", default=None); sp.add_argument("--out", required=True); sp.add_argument("--annotated", required=True); sp.add_argument("--config", default=None)
    args = ap.parse_args(); cfg = load_config(args.config)
    if args.mode == "prediction":
        r = score_prediction_file(Path(args.predictions), Path(args.truth), cfg, Path(args.out)); print(json.dumps(r, indent=2))
    else:
        r, ann = score_search_file(Path(args.candidates), Path(args.hidden), Path(args.observed), int(cfg["search_task"]["hidden_near_hit_hamming"]), seed_pool_path=(Path(args.seed_pool) if args.seed_pool else None))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(r, indent=2), encoding="utf-8"); ann.to_csv(args.annotated, index=False); print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
