from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_write_json, load_config, mutation_count_from_reference, sha256_file, sha256_text, validate_sequence

import re


def has_beyond_window_mutation(mut: object, length: int) -> bool:
    """True if the mutation annotation contains any position beyond the modeled window."""
    if pd.isna(mut):
        return False
    for tok in str(mut).strip().replace(",", ";").replace(";", ":").split(":"):
        tok = tok.strip()
        if not tok:
            continue
        m = re.search(r"(\d+)", tok)
        if m and int(m.group(1)) > int(length):
            return True
    return False


def canonical_split(x: object, aliases: dict) -> str:
    s = str(x).strip().lower()
    for canonical, vals in aliases.items():
        if s in {str(v).strip().lower() for v in vals}:
            return canonical
    raise ValueError(f"Unrecognized split value: {x!r}; expected train/val/test aliases from config")


def infer_reference_sequence(df: pd.DataFrame, seq_col: str, mut_col: str, length: int) -> tuple[str, str]:
    raw_mut = df[mut_col]
    blank = raw_mut.isna() | raw_mut.astype(str).str.strip().str.lower().isin({"", "wt", "wildtype", "wild-type", "none", "nan", "0"})
    if blank.any():
        counts = df.loc[blank, seq_col].astype(str).str.upper().value_counts()
        for s in counts.index:
            try:
                return validate_sequence(s, length), "row_with_no_reported_mutation"
            except ValueError:
                continue
    seqs = [validate_sequence(s, length) for s in df[seq_col].tolist()]
    arr = np.asarray([list(s) for s in seqs], dtype="U1")
    ref = []
    for j in range(length):
        vals, cnt = np.unique(arr[:, j], return_counts=True)
        ref.append(str(vals[int(np.argmax(cnt))]))
    return "".join(ref), "per_position_consensus_fallback"


def choose_groupwise_validation(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy().reset_index(drop=True)
    q = min(20, max(2, len(work) // 50))
    work["_b_bin"] = pd.qcut(work["brightness"], q=q, duplicates="drop", labels=False)
    work["_m_bin"] = work["n_mutations"].clip(upper=8)
    work["_group"] = work["_b_bin"].astype(str) + "_" + work["_m_bin"].astype(str)
    val_indices: list[int] = []
    for g_i, (_, g) in enumerate(work.groupby("_group", sort=True)):
        n = len(g)
        if n <= 2:
            continue
        n_val = min(max(1, int(round(n * fraction))), n - 1)
        rng = np.random.default_rng(seed + 1000 + g_i)
        val_indices.extend(int(i) for i in rng.choice(g.index.to_numpy(), size=n_val, replace=False))
    val_mask = work.index.isin(val_indices)
    drops = ["_b_bin", "_m_bin", "_group"]
    train = work.loc[~val_mask].drop(columns=drops).reset_index(drop=True)
    val = work.loc[val_mask].drop(columns=drops).reset_index(drop=True)
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError("Could not build non-empty GLOM train/validation split")
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare leakage-safe ppluGFP prediction and search benchmark splits")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None, help="WSL directory containing the two 31480-row CSV files")
    ap.add_argument("--out-root", default=None, help="Output data root; default <repo>/data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = Path(__file__).resolve().parents[2]
    root = Path(args.out_root) if args.out_root else repo / "data"
    processed, sealed, audit_dir = root / "processed", root / "sealed", root / "audit"
    for p in (processed, sealed, audit_dir):
        p.mkdir(parents=True, exist_ok=True)

    dc = cfg["data"]
    data_dir = Path(args.data_dir or dc["wsl_dir"])
    seq_path = data_dir / dc["sequence_file"]
    pred_path = data_dir / dc["prediction_file"]
    if not seq_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {seq_path}")

    df0 = pd.read_csv(seq_path)
    required = [dc["sequence_column"], dc["brightness_column"], dc["split_column"], dc["mutations_column"]]
    missing = [c for c in required if c not in df0.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; found {list(df0.columns)}")
    if len(df0) != int(dc["expected_rows"]):
        raise ValueError(f"Expected {dc['expected_rows']} rows, found {len(df0)}")

    seq_col, y_col, split_col, mut_col = dc["sequence_column"], dc["brightness_column"], dc["split_column"], dc["mutations_column"]
    L = int(dc["sequence_length"])
    reference, ref_method = infer_reference_sequence(df0, seq_col, mut_col, L)
    (processed / "reference_sequence.txt").write_text(reference + "\n", encoding="utf-8")

    rows = []
    for i, r in df0.iterrows():
        seq = validate_sequence(r[seq_col], L)
        y = pd.to_numeric(pd.Series([r[y_col]]), errors="coerce").iloc[0]
        if pd.isna(y):
            raise ValueError(f"Non-numeric brightness at source row {i}: {r[y_col]!r}")
        rows.append({
            "row_id": int(i),
            "sequence": seq,
            "sequence_sha256": sha256_text(seq),
            "brightness": float(y),
            "source_split": canonical_split(r[split_col], cfg["prediction_task"]["allowed_split_names"]),
            "aaMutations": "" if pd.isna(r[mut_col]) else str(r[mut_col]),
            "n_mutations": mutation_count_from_reference(seq, reference),
        })
    df = pd.DataFrame(rows)

    # Source CSV contains 46 "twin" rows whose mutation annotation differs only
    # at position > L (e.g. `R8C` vs `R8C:*222G`); mut2seq collapses both to the
    # same 221-mer. Per dataset-owner decision these twin rows are dropped, keeping
    # the base row with its original label. Unique sequences carrying a
    # beyond-window annotation (e.g. `P181T:*222G` with no base twin) are kept.
    df["_beyond_window"] = df.aaMutations.map(lambda m: has_beyond_window_mutation(m, L))
    dup_seq = df[df.duplicated("sequence", keep=False)].sequence.unique()
    twin_drop = df.sequence.isin(dup_seq) & df._beyond_window
    n_dropped_twins = int(twin_drop.sum())
    if n_dropped_twins:
        print(f"[data] dropping {n_dropped_twins} beyond-window twin row(s) sharing a 221-mer with a base row")
        df = df.loc[~twin_drop].reset_index(drop=True)
    df = df.drop(columns=["_beyond_window"])
    # Safety net: after twin removal the protocol still requires one unique row per sequence.
    dup = df[df.duplicated("sequence", keep=False)]
    if len(dup):
        conflicting = dup.groupby("sequence")["brightness"].nunique()
        if (conflicting > 1).any():
            raise ValueError("Duplicate sequences with conflicting brightness labels found")
        raise ValueError(f"Found {dup['sequence'].nunique()} duplicate sequence(s); protocol requires one unique row per sequence")

    cross = {"available": False}
    if pred_path.is_file():
        p = pd.read_csv(pred_path)
        if "sequence" not in p.columns:
            raise ValueError(f"{pred_path.name} lacks sequence column")
        p["sequence"] = p["sequence"].map(lambda x: validate_sequence(x, L))
        # Apply the same beyond-window twin filter to the complementary CSV.
        if "aaMutations" in p.columns:
            p["_bw"] = p.aaMutations.map(lambda m: has_beyond_window_mutation(m, L))
            p = p[~(p.sequence.isin(dup_seq) & p._bw)].drop(columns=["_bw"]).reset_index(drop=True)
        if len(p) != len(df) or p.sequence.nunique() != len(df):
            raise ValueError("Complementary prediction CSV does not contain the same unique sequence set")
        merged = df[["sequence", "brightness"]].merge(p, on="sequence", how="left", validate="one_to_one")
        diffs = {}
        for c in [c for c in ["brightness_true", "Brightness", "medianBrightness"] if c in p.columns]:
            yy = pd.to_numeric(merged[c], errors="coerce").to_numpy(float)
            diffs[c] = float(np.nanmax(np.abs(yy - merged.brightness.to_numpy(float))))
        cross = {"available": True, "rows": len(p), "source_sha256": sha256_file(pred_path), "max_abs_label_diffs": diffs, "has_brightness_pred": "brightness_pred" in p.columns}

    # Prediction task: preserve the source train/val/test assignments exactly.
    pred_dir, pred_sealed = processed / "prediction", sealed / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True); pred_sealed.mkdir(parents=True, exist_ok=True)
    split_dfs = {s: df[df.source_split == s].copy() for s in ("train", "val", "test")}
    split_fallback = False
    if any(len(v) == 0 for v in split_dfs.values()):
        # Source CSV carries no usable train/val/test assignment (all rows share one
        # value). Deterministic fallback: 80/10/10 stratified by brightness decile,
        # seeded with the protocol master seed (mirrors the package's own synthetic
        # 80/10/10 convention).
        print("[data] no usable source train/val/test assignment; building seeded stratified 80/10/10 split")
        rng = np.random.default_rng(int(cfg["master_seed"]))
        q = min(10, max(2, len(df) // 500))
        bins = pd.qcut(df.brightness, q=q, labels=False, duplicates="drop")
        test_idx, val_idx = [], []
        for b in range(int(bins.nunique())):
            g = df.index[bins == b].to_numpy()
            g = rng.permutation(g)
            n_test = max(1, int(round(len(g) * 0.1)))
            n_val = max(1, int(round(len(g) * 0.1)))
            test_idx.extend(int(i) for i in g[:n_test])
            val_idx.extend(int(i) for i in g[n_test:n_test + n_val])
        split_arr = np.full(len(df), "train", dtype=object)
        split_arr[np.asarray(test_idx)] = "test"
        split_arr[np.asarray(val_idx)] = "val"
        df = df.copy()
        df["source_split"] = split_arr
        split_dfs = {s: df[df.source_split == s].copy() for s in ("train", "val", "test")}
        split_fallback = True
    if any(len(v) == 0 for v in split_dfs.values()):
        raise ValueError({k: len(v) for k, v in split_dfs.items()})
    split_dfs["train"].to_csv(pred_dir / "train.csv", index=False)
    split_dfs["val"].to_csv(pred_dir / "val.csv", index=False)
    split_dfs["test"][["row_id", "sequence", "sequence_sha256", "n_mutations", "aaMutations"]].to_csv(pred_dir / "test_sequences_only.csv", index=False)
    split_dfs["test"].to_csv(pred_sealed / "test_with_labels.csv", index=False)

    # μFormer adapter: test file contains sequence only. μFormer will fill dummy score internally; final metrics are external.
    mu = processed / "prediction_adapters" / "muformer"; mu.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", split_dfs["train"]), ("valid", split_dfs["val"])]:
        d[["sequence", "brightness"]].rename(columns={"sequence": "mutated_sequence"}).to_csv(mu / f"{name}.tsv", sep="\t", index=False)
    split_dfs["test"][["sequence"]].rename(columns={"sequence": "mutated_sequence"}).to_csv(mu / "test_unlabeled.tsv", sep="\t", index=False)
    (mu / "pplu_reference.fasta").write_text(f">pplu_reference\n{reference}\n", encoding="utf-8")

    # ProteinNPT adapter: test DMS_score is NaN, so labels never enter the tool. Folds 0=train,1=val,2=test.
    pn = processed / "prediction_adapters" / "proteinnpt"; pn.mkdir(parents=True, exist_ok=True)
    a = []
    for fold, d in [(0, split_dfs["train"]), (1, split_dfs["val"]), (2, split_dfs["test"])]:
        x = pd.DataFrame({"mutated_sequence": d.sequence.astype(str), "DMS_score": d.brightness.to_numpy(float) if fold < 2 else np.nan, "benchmark_fold": fold})
        a.append(x)
    pd.concat(a, ignore_index=True).to_csv(pn / "pplu_assay_blind_test.csv", index=False)
    (pn / "reference_sequence.txt").write_text(reference + "\n", encoding="utf-8")

    # Search task: Top1500 -> 20 equal rank strata; each stratum 20 hidden + 40 GLOM-high + 15 TGITO-tune.
    sc = cfg["search_task"]
    top_n, strata = int(sc["top_n"]), int(sc["num_strata"])
    if top_n % strata:
        raise ValueError("top_n must be divisible by num_strata")
    stratum_size = top_n // strata
    expected_per = int(sc["hidden_per_stratum"]) + int(sc["glom_high_per_stratum"]) + int(sc["tgito_tune_per_stratum"])
    if expected_per != stratum_size:
        raise ValueError(f"Allocation invalid: per stratum {expected_per} != {stratum_size}. With Top1500: 1500-400-800=300, not 700.")

    ranked = df.sort_values(["brightness", "sequence_sha256"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    ranked["brightness_rank"] = np.arange(1, len(ranked) + 1)
    top, ordinary = ranked.iloc[:top_n].copy(), ranked.iloc[top_n:].copy()
    top["rank_stratum"] = ((top.brightness_rank - 1) // stratum_size).astype(int)
    master_seed = int(cfg["master_seed"])
    hidden_parts, glom_parts, tune_parts, summaries = [], [], [], []
    for s in range(strata):
        g = top[top.rank_stratum == s].copy()
        if len(g) != stratum_size:
            raise RuntimeError(f"stratum {s} has {len(g)}, expected {stratum_size}")
        g = g.iloc[np.random.default_rng(master_seed + 20000 + s).permutation(len(g))].reset_index(drop=True)
        a = int(sc["hidden_per_stratum"]); b = a + int(sc["glom_high_per_stratum"])
        h, gh, tt = g.iloc[:a].copy(), g.iloc[a:b].copy(), g.iloc[b:].copy()
        hidden_parts.append(h); glom_parts.append(gh); tune_parts.append(tt)
        summaries.append({"rank_stratum": s, "rank_min": s * stratum_size + 1, "rank_max": (s + 1) * stratum_size, "hidden": len(h), "glom_high": len(gh), "tgito_tune": len(tt), "brightness_min": float(top[top.rank_stratum == s].brightness.min()), "brightness_max": float(top[top.rank_stratum == s].brightness.max())})
    hidden = pd.concat(hidden_parts, ignore_index=True).sort_values("brightness_rank")
    glom_high = pd.concat(glom_parts, ignore_index=True).sort_values("brightness_rank")
    tune = pd.concat(tune_parts, ignore_index=True).sort_values("brightness_rank")
    glom_fit_pool = pd.concat([ordinary, glom_high], ignore_index=True)
    # glom_fit_pool = unique_rows - top1500 + glom_high(800). 31480 -> 30780 (synthetic),
    # 31434 -> 30734 (real data after the 46 beyond-window twin rows are dropped).
    expected_fit_pool = len(df) - top_n + len(glom_high)
    if (len(hidden), len(glom_high), len(tune), len(glom_fit_pool)) != (400, 800, 300, expected_fit_pool):
        raise RuntimeError((len(hidden), len(glom_high), len(tune), len(glom_fit_pool)))
    glom_train, glom_val = choose_groupwise_validation(glom_fit_pool, float(sc["glom_validation_fraction"]), master_seed)
    observed_all = pd.concat([glom_fit_pool, tune], ignore_index=True)

    seed_parts = []
    for s in range(strata):
        g = glom_high[glom_high.rank_stratum == s].copy()
        idx = np.random.default_rng(master_seed + 30000 + s).choice(g.index.to_numpy(), size=int(sc["seed_pool_per_stratum"]), replace=False)
        seed_parts.append(glom_high.loc[idx])
    seed_pool = pd.concat(seed_parts, ignore_index=True).sort_values(["rank_stratum", "brightness_rank"])

    sd, ss = processed / "search", sealed / "search"
    sd.mkdir(parents=True, exist_ok=True); ss.mkdir(parents=True, exist_ok=True)
    glom_fit_pool.to_csv(sd / "glom_fit_pool.csv", index=False)
    glom_train.to_csv(sd / "glom_train.csv", index=False)
    glom_val.to_csv(sd / "glom_val.csv", index=False)
    glom_high.to_csv(sd / "glom_high_800.csv", index=False)
    tune.to_csv(sd / "tgito_tune_300.csv", index=False)
    observed_all.to_csv(sd / "observed_all.csv", index=False)
    seed_pool.to_csv(sd / "common_seed_pool.csv", index=False)
    hidden.to_csv(ss / "hidden_top1500_400_with_labels.csv", index=False)
    # Hash-only embargo list is safe to expose to development-data preparation scripts.
    # Tuning code uses it only to remove any legacy row that overlaps the final hidden400;
    # it never receives final hidden sequences or labels.
    (ss / "hidden_sequence_hashes.txt").write_text("\n".join(sorted(hidden.sequence_sha256.astype(str))) + "\n", encoding="utf-8")
    pd.DataFrame(summaries).to_csv(sd / "rank_strata_summary.csv", index=False)

    protocol = {
        "protocol_version": cfg["protocol_version"], "master_seed": master_seed,
        "source_dataset": str(seq_path), "source_dataset_sha256": sha256_file(seq_path),
        "source_rows": int(len(df0)),
        "beyond_window_twin_rows_dropped": n_dropped_twins,
        "final_unique_rows": int(len(df)),
        "reference_sequence_method": ref_method, "reference_sequence_sha256": sha256_text(reference),
        "prediction_split_source": "seeded_stratified_80_10_10_fallback" if split_fallback else "source_csv_existing",
        "complementary_prediction_csv": cross,
        "prediction_counts": {k: len(v) for k, v in split_dfs.items()},
        "search_counts": {"top1500": top_n, "hidden": len(hidden), "glom_high": len(glom_high), "tgito_tune": len(tune), "ordinary_outside_top1500": len(ordinary), "glom_fit_pool": len(glom_fit_pool), "glom_train": len(glom_train), "glom_val": len(glom_val), "observed_all": len(observed_all), "common_seed_pool": len(seed_pool)},
        "rank_stratification": {"num_strata": strata, "stratum_size": stratum_size, "hidden_per_stratum": int(sc["hidden_per_stratum"]), "glom_high_per_stratum": int(sc["glom_high_per_stratum"]), "tgito_tune_per_stratum": int(sc["tgito_tune_per_stratum"])},
        "safety": ["Search hidden sequences+labels only exist under data/sealed/search.", "Prediction test labels only exist under data/sealed/prediction.", "No public row-level manifest containing hidden sequences is generated."],
    }
    atomic_write_json(protocol, audit_dir / "protocol_manifest.json")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(f"Prepared data at {root}")
    print("DO NOT pass data/sealed/* to training/search code.")


if __name__ == "__main__":
    main()
