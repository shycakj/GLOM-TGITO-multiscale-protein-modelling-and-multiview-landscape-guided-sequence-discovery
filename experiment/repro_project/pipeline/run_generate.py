#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from tgito.aa import clean_seq
from tgito.pplu_config import PPLU_WT, PPLU_LENGTH, clean_pplu_wt
from tgito.esm_bridge import DifferentiableESMBridge, load_bridge
from tgito.optimizer import GeneratorConfig, TGITOGenerator, normalize_mutation_signature
from tgito.revised_slim_internal import load_revised_slim
from tgito.terrain_memory import TerrainMemoryBank
from tgito.terrain_potential import TerrainPotential, TerrainPotentialConfig
from tgito.embedding_realism import EmbeddingRealismBank
from tgito.embedding_repair import EmbeddingRepairBank


def find_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Cannot find any of {candidates}; columns={list(df.columns)}")
    return None


def apply_seed_mutations(seq: str, mutation_spec: str | None) -> str:
    if not mutation_spec:
        return seq
    aa20 = set("ACDEFGHIKLMNPQRSTVWY")
    arr = list(str(seq).upper())
    for raw in str(mutation_spec).replace(",", ";").split(";"):
        m = raw.strip()
        if not m:
            continue
        old, new = m[0].upper(), m[-1].upper()
        pos = int(m[1:-1])
        if old not in aa20 or new not in aa20 or not (1 <= pos <= len(arr)):
            raise ValueError(f"Bad mutation spec {m}; expected E110V")
        if arr[pos - 1] != old:
            raise ValueError(f"Mutation {m} does not match seed: position {pos} is {arr[pos-1]}, not {old}")
        arr[pos - 1] = new
    return "".join(arr)


def load_seed_sequence(args):
    if args.seed_seq:
        return apply_seed_mutations(args.seed_seq, args.seed_mutations)
    df = pd.read_csv(args.master_csv)
    seq_col = args.seq_col or find_col(df, ["seq", "sequence", "aa_sequence", "protein_sequence"])
    seq = str(df.iloc[int(args.seed_row)][seq_col])
    return apply_seed_mutations(seq, args.seed_mutations)


def find_reference_sequence(args, df, seq_col):
    """Resolve reference sequence for PPLU.

    Priority:
      1. --reference-seq
      2. --reference-row explicit integer
      3. --reference-row pplu_wt/wt/reference -> bundled PPLU WT
      4. auto: dataset n_mut==0 row, then exact PPLU_WT sequence match, then bundled PPLU WT

    We do not fall back to seed row for PPLU discovery, because P must be encoded
    relative to the PPLU WT/reference, not relative to an arbitrary high-brightness seed.
    """
    if args.reference_seq:
        return str(args.reference_seq), -1, "--reference-seq"
    rr = str(args.reference_row).strip().lower() if args.reference_row is not None else "auto"
    if rr in {"pplu_wt", "pplu", "wt", "reference", "ref"}:
        return clean_pplu_wt(), -1, "bundled:pplu_wt"
    if rr not in {"", "auto", "none"}:
        ridx = int(args.reference_row)
        return str(df.iloc[ridx][seq_col]), ridx, "--reference-row"

    nmut_col = find_col(df, ["n_mut", "num_mut", "mutation_count", "n_mutations"], required=False)
    if nmut_col is not None:
        vals = pd.to_numeric(df[nmut_col], errors="coerce")
        hit = df.index[vals.fillna(1e9).eq(0)].tolist()
        if hit:
            ridx = int(hit[0])
            return str(df.iloc[ridx][seq_col]), ridx, f"auto:{nmut_col}==0"

    wt = clean_pplu_wt()
    seqs = df[seq_col].astype(str).map(lambda x: ''.join(ch for ch in x.upper() if ch in 'ACDEFGHIKLMNPQRSTVWY')[:PPLU_LENGTH])
    hit = df.index[seqs.eq(wt)].tolist()
    if hit:
        ridx = int(hit[0])
        return str(df.iloc[ridx][seq_col]), ridx, "auto:exact_pplu_wt_sequence"
    return wt, -1, "auto:bundled_pplu_wt"

def load_known_sequences(master_csv, seq_col=None):
    if not master_csv:
        return set()
    df = pd.read_csv(master_csv)
    col = seq_col or find_col(df, ["seq", "sequence", "aa_sequence", "protein_sequence"])
    return set(df[col].astype(str).str.upper())


def load_archive_sequences(path: str | None):
    if not path:
        return []
    df = pd.read_csv(path)
    for col in ["final_seq", "seq", "sequence", "aa_sequence"]:
        if col in df.columns:
            return df[col].astype(str).str.upper().tolist()
    return []


def parse_sources(s: str):
    vals = [x.strip() for x in str(s).split(",") if x.strip()]
    return vals or ["all"]


def main():
    p = argparse.ArgumentParser(description="TG-ITO v12.7 standalone PPLU generator")
    p.add_argument("--revised-ckpt", required=True)
    p.add_argument("--bridge-ckpt", required=True)
    p.add_argument("--terrain-npz", required=True)
    p.add_argument("--master-csv", required=True)
    p.add_argument("--seed-row", type=int, default=0)
    p.add_argument("--seed-seq", default=None)
    p.add_argument("--seed-mutations", default="")
    p.add_argument("--seq-col", default=None)
    p.add_argument("--reference-row", default="pplu_wt")
    p.add_argument("--reference-seq", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mutable", default="all")
    p.add_argument("--sources", default="all")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--num-restarts", type=int, default=4)
    p.add_argument("--esm-model", default="auto")
    p.add_argument("--bridge-output", default="auto", choices=["auto", "soft_struct", "saprot_hat"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--length", default="auto", help="Sequence/model length. PPLU model expects 221; auto resolves to seed/reference length, usually 221.")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--terrain-weight", type=float, default=0.03)
    p.add_argument("--brightness-weight", type=float, default=1.0)
    p.add_argument("--mutation-budget", type=float, default=4.0)
    p.add_argument("--budget-relative", choices=["seed", "reference"], default="seed")
    p.add_argument("--mut-mask-mode", choices=["changed", "allowed"], default="changed")
    p.add_argument("--brightness-floor-weight", type=float, default=12.0)
    p.add_argument("--init-mode", choices=["relaxed", "hard"], default="relaxed")
    p.add_argument("--mutable-anchor-logit", type=float, default=5.0)
    p.add_argument("--mutable-alt-logit", type=float, default=0.0)
    p.add_argument("--temp-start", type=float, default=1.2)
    p.add_argument("--temp-end", type=float, default=0.8)
    p.add_argument("--bridge-temp", type=float, default=1.0)
    p.add_argument("--proposal-every", type=int, default=6)
    p.add_argument("--proposal-candidates", type=int, default=64)
    p.add_argument("--proposal-order", type=int, default=2, choices=[1, 2])
    p.add_argument("--proposal-accept-margin", type=float, default=0.0)
    p.add_argument("--line-search-trials", type=int, default=1)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--noise-scale", type=float, default=0.15)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ban-signatures", default="")
    p.add_argument("--archive-csv", default="")
    p.add_argument("--novelty-min-hamming", type=int, default=0)
    # v12.7 standalone gate path.
    p.add_argument("--v12-true-high-gate-npz", default="", help="NPZ from train_v12_true_high_gate.py")
    p.add_argument("--true-high-weight", type=float, default=0.0, help="Extra terrain reward for proximity to released true-high prototypes")
    p.add_argument("--true-high-topk", type=int, default=24)
    p.add_argument("--true-high-temp", type=float, default=0.12)
    p.add_argument("--source-gate-power", type=float, default=1.0)
    p.add_argument("--min-source-support", type=int, default=1)
    p.add_argument("--terrain-topk", type=int, default=4)
    # v13.1: cos0 local top-k gate. These do not re-enable embedding cos realism/repair.
    p.add_argument("--terrain-gate-mode", choices=["authority", "fixed", "rule", "risk"], default="authority")
    p.add_argument("--terrain-fixed-sources", default="")
    p.add_argument("--terrain-gate-temp", type=float, default=0.20)
    p.add_argument("--terrain-support-weight", type=float, default=0.45)
    p.add_argument("--terrain-risk-weight", type=float, default=0.65)
    p.add_argument("--terrain-pred-gap-weight", type=float, default=0.20)
    p.add_argument("--terrain-source-quality-power", type=float, default=0.60)
    p.add_argument("--v12-miner-policy-pt", default="", help="Frozen learned mining policy from train_v12_mining_policy.py")
    p.add_argument("--miner-weight", type=float, default=0.0, help="Reward weight for learned mining policy score")
    p.add_argument("--miner-logit-scale", type=float, default=1.0)
    p.add_argument("--context-neighbor-k", type=int, default=0, help="For v12.7 ctx_mlp_k4/k8 miner: 0 uses checkpoint neighbor_k")
    p.add_argument("--context-top-sources", type=int, default=4, help="For v12.7 ctx_mlp miner live feature aggregation")
    p.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off", help="Autocast ESM2/Bridge/revised_slim for speed on CUDA")
    # v12.7: differentiable mutation-site embedding realism prior.
    # This keeps the main TG-ITO path: reverse AA optimization + multi-terrain guidance,
    # but adds a realism reward from original SaProt H and Pmutation P4 tokens of real existing high rows.
    p.add_argument("--embedding-realism-bank-npz", default="", help="NPZ from build_v12_7_embedding_realism_bank.py")
    p.add_argument("--embedding-realism-weight", type=float, default=0.0)
    p.add_argument("--embedding-realism-topk", type=int, default=16)
    p.add_argument("--embedding-realism-temp", type=float, default=0.08)
    p.add_argument("--embedding-realism-h-alpha", type=float, default=0.80, help="H/P4 blend; 0.80 means mostly original SaProt H")
    p.add_argument("--embedding-realism-reward-scale", type=float, default=0.10, help="Small brightness reward bias inside the real token bank")
    p.add_argument("--embedding-realism-min-mut-prob", type=float, default=1e-4)
    p.add_argument("--embedding-realism-fallback-global-weight", type=float, default=0.35)
    # v12.8.1: denoising-residual embedding repair prior. This directly repairs the
    # v12.8 credible miner embedding branch instead of only lowering cosine thresholds.
    p.add_argument("--embedding-repair-bank-npz", default="", help="Usually the same NPZ as embedding-realism bank")
    p.add_argument("--embedding-repair-model-pt", default="", help="PT from train_v12_8_1_embedding_repair_model.py")
    p.add_argument("--embedding-repair-weight", type=float, default=0.0)
    p.add_argument("--embedding-repair-topk", type=int, default=16)
    p.add_argument("--embedding-repair-temp", type=float, default=0.07)
    p.add_argument("--embedding-repair-h-alpha", type=float, default=0.80)
    p.add_argument("--embedding-repair-target-combo", type=float, default=0.63)
    p.add_argument("--embedding-repair-pull-weight", type=float, default=1.0)
    p.add_argument("--embedding-repair-gap-weight", type=float, default=1.5)
    p.add_argument("--embedding-repair-denoised-bonus-weight", type=float, default=0.0)
    p.add_argument("--embedding-repair-residual-penalty", type=float, default=0.03)
    p.add_argument("--embedding-repair-fallback-global-weight", type=float, default=0.25)
    p.add_argument("--embedding-repair-residual-scale", type=float, default=1.0)
    p.add_argument("--embedding-repair-mode", default="directional", choices=["directional", "legacy"])
    p.add_argument("--embedding-repair-soft-mutation-floor", type=float, default=1e-4)
    p.add_argument("--embedding-repair-mutation-weight-power", type=float, default=1.0)
    p.add_argument("--embedding-repair-site-topk", type=int, default=16)
    p.add_argument("--embedding-repair-min-mutation-count", type=float, default=1.25)
    p.add_argument("--embedding-repair-min-mutation-weight", type=float, default=0.80)
    # v12.8.4: smooth seed-divergence and objective-based line search.
    p.add_argument("--seed-explore-weight", type=float, default=0.0)
    p.add_argument("--seed-explore-min-change", type=float, default=0.0)
    p.add_argument("--change-count-threshold-margin", type=float, default=0.08)
    p.add_argument("--change-count-temp", type=float, default=0.035)
    p.add_argument("--change-count-topk", type=int, default=8)
    p.add_argument("--accept-by-objective", action="store_true", default=False)
    p.add_argument("--objective-accept-drop", type=float, default=0.01)
    p.add_argument("--select-by-objective", action="store_true", default=False)
    p.add_argument("--reject-known-output", action="store_true", default=False, help="Reject final candidates already present in master-csv; publication search should enable this.")
    p.add_argument("--multi-start-file", default="", help="Resident mode: CSV with columns seed_seq,out_dir,seed. Load models once, run every row.")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_seed_seq = load_seed_sequence(args)
    master_df = pd.read_csv(args.master_csv)
    seq_col = args.seq_col or find_col(master_df, ["seq", "sequence", "aa_sequence", "protein_sequence"])
    reference_seq, ref_row, ref_reason = find_reference_sequence(args, master_df, seq_col)
    if str(args.length).lower() == "auto":
        seed_len = len("".join(ch for ch in str(raw_seed_seq).upper() if ch in "ACDEFGHIKLMNPQRSTVWY"))
        ref_len = len("".join(ch for ch in str(reference_seq).upper() if ch in "ACDEFGHIKLMNPQRSTVWY"))
        if seed_len >= PPLU_LENGTH and ref_len >= PPLU_LENGTH:
            length = PPLU_LENGTH
        else:
            length = min(seed_len, ref_len)
        if length <= 0:
            raise ValueError("Cannot infer positive sequence length from seed/reference")
    else:
        length = int(args.length)
    if length != PPLU_LENGTH:
        print(f"[WARN] PPLU package expected length={PPLU_LENGTH}, but using length={length}. Check your dataset/model if this is intentional.")
    seed_seq = clean_seq(raw_seed_seq, length)
    reference_seq = clean_seq(reference_seq, length)
    seed_id = master_df.iloc[int(args.seed_row)].get("id", "") if 0 <= int(args.seed_row) < len(master_df) else ""
    print(f"[seed] row={args.seed_row} id={seed_id} seed_mutations={args.seed_mutations} len={len(seed_seq)} length={length}")
    print(f"[reference] row={ref_row} reason={ref_reason} len={len(reference_seq)}")
    known = load_known_sequences(args.master_csv, args.seq_col)
    avoid = {normalize_mutation_signature(x) for x in str(args.ban_signatures).split("|") if normalize_mutation_signature(x)}
    archive = load_archive_sequences(args.archive_csv)

    revised, y_mean, y_std = load_revised_slim(args.revised_ckpt, device=args.device)
    bridge = load_bridge(args.bridge_ckpt, device=args.device)
    seq_encoder = DifferentiableESMBridge(bridge, esm_model_name=args.esm_model, device=args.device, bridge_output=args.bridge_output, target_h_dim=1280)
    active_sources = parse_sources(args.sources)
    bank = TerrainMemoryBank.from_npz(args.terrain_npz, device=args.device, sources=active_sources)
    terrain = TerrainPotential(
        bank,
        TerrainPotentialConfig(
            topk_sources=max(1, int(args.terrain_topk)),
            min_source_support=max(1, int(args.min_source_support)),
            true_high_gate_npz=args.v12_true_high_gate_npz,
            true_high_weight=float(args.true_high_weight),
            true_high_topk=int(args.true_high_topk),
            true_high_temp=float(args.true_high_temp),
            source_gate_power=float(args.source_gate_power),
            miner_policy_pt=args.v12_miner_policy_pt,
            miner_weight=float(args.miner_weight),
            miner_logit_scale=float(args.miner_logit_scale),
            context_neighbor_k=int(args.context_neighbor_k),
            context_top_sources=int(args.context_top_sources),
            terrain_gate_mode=str(args.terrain_gate_mode),
            terrain_fixed_sources=str(args.terrain_fixed_sources),
            terrain_gate_temp=float(args.terrain_gate_temp),
            terrain_support_weight=float(args.terrain_support_weight),
            terrain_risk_weight=float(args.terrain_risk_weight),
            terrain_pred_gap_weight=float(args.terrain_pred_gap_weight),
            terrain_source_quality_power=float(args.terrain_source_quality_power),
        ),
        sources=active_sources,
    )
    embedding_realism = None
    if str(args.embedding_realism_bank_npz).strip() and float(args.embedding_realism_weight) > 0:
        embedding_realism = EmbeddingRealismBank(args.embedding_realism_bank_npz, device=args.device)
    embedding_repair = None
    repair_bank_path = str(args.embedding_repair_bank_npz).strip() or str(args.embedding_realism_bank_npz).strip()
    if repair_bank_path and str(args.embedding_repair_model_pt).strip() and float(args.embedding_repair_weight) > 0:
        embedding_repair = EmbeddingRepairBank(repair_bank_path, args.embedding_repair_model_pt, device=args.device)
    gen = TGITOGenerator(seq_encoder, revised, terrain, y_mean=y_mean, y_std=y_std, device=args.device, embedding_realism=embedding_realism, embedding_repair=embedding_repair)
    cfg = GeneratorConfig(
        length=length, steps=args.steps, lr=args.lr, min_lr=args.min_lr, line_search_trials=args.line_search_trials,
        init_mode=args.init_mode, mutable_anchor_logit=args.mutable_anchor_logit, mutable_alt_logit=args.mutable_alt_logit,
        temp_start=args.temp_start, temp_end=args.temp_end, bridge_temp=args.bridge_temp,
        brightness_weight=args.brightness_weight, terrain_weight=args.terrain_weight,
        mutation_budget=args.mutation_budget, budget_relative=args.budget_relative,
        mut_mask_mode=args.mut_mask_mode, brightness_floor_weight=args.brightness_floor_weight,
        noise_scale=args.noise_scale, proposal_every=args.proposal_every, proposal_candidates=args.proposal_candidates,
        proposal_order=args.proposal_order, proposal_accept_margin=args.proposal_accept_margin,
        novelty_min_hamming=args.novelty_min_hamming, sample_every=args.sample_every, seed=args.seed, amp_dtype=args.amp_dtype,
        embedding_realism_weight=float(args.embedding_realism_weight),
        embedding_realism_topk=int(args.embedding_realism_topk),
        embedding_realism_temp=float(args.embedding_realism_temp),
        embedding_realism_h_alpha=float(args.embedding_realism_h_alpha),
        embedding_realism_reward_scale=float(args.embedding_realism_reward_scale),
        embedding_realism_min_mut_prob=float(args.embedding_realism_min_mut_prob),
        embedding_realism_fallback_global_weight=float(args.embedding_realism_fallback_global_weight),
        embedding_repair_weight=float(args.embedding_repair_weight),
        embedding_repair_topk=int(args.embedding_repair_topk),
        embedding_repair_temp=float(args.embedding_repair_temp),
        embedding_repair_h_alpha=float(args.embedding_repair_h_alpha),
        embedding_repair_target_combo=float(args.embedding_repair_target_combo),
        embedding_repair_pull_weight=float(args.embedding_repair_pull_weight),
        embedding_repair_gap_weight=float(args.embedding_repair_gap_weight),
        embedding_repair_denoised_bonus_weight=float(args.embedding_repair_denoised_bonus_weight),
        embedding_repair_residual_penalty=float(args.embedding_repair_residual_penalty),
        embedding_repair_fallback_global_weight=float(args.embedding_repair_fallback_global_weight),
        embedding_repair_residual_scale=float(args.embedding_repair_residual_scale),
        embedding_repair_mode=str(args.embedding_repair_mode),
        embedding_repair_soft_mutation_floor=float(args.embedding_repair_soft_mutation_floor),
        embedding_repair_mutation_weight_power=float(args.embedding_repair_mutation_weight_power),
        embedding_repair_site_topk=int(args.embedding_repair_site_topk),
        embedding_repair_min_mutation_count=float(args.embedding_repair_min_mutation_count),
        embedding_repair_min_mutation_weight=float(args.embedding_repair_min_mutation_weight),
        seed_explore_weight=float(args.seed_explore_weight),
        seed_explore_min_change=float(args.seed_explore_min_change),
        change_count_threshold_margin=float(args.change_count_threshold_margin),
        change_count_temp=float(args.change_count_temp),
        change_count_topk=int(args.change_count_topk),
        accept_by_objective=bool(args.accept_by_objective),
        objective_accept_drop=float(args.objective_accept_drop),
        select_by_objective=bool(args.select_by_objective),
        reject_known_output=bool(args.reject_known_output),
    )
    def run_one(seed_seq_i: str, out_dir_i: Path, cfg_i: GeneratorConfig):
        rows_i = []
        for r in range(int(args.num_restarts)):
            rr = out_dir_i / f"restart_{r}"
            row = gen.run(seed_seq_i, args.mutable, out_dir=rr, cfg=cfg_i, restart_id=r, known_sequences=known,
                          reference_seq=reference_seq, avoid_signatures=avoid, archive_sequences=archive)
            row["restart"] = r
            rows_i.append(row)
        df_i = pd.DataFrame(rows_i).sort_values(["final_pred_brightness", "final_objective"], ascending=False)
        df_i.to_csv(out_dir_i / "final_candidates.csv", index=False)
        return df_i

    if args.multi_start_file:
        import dataclasses
        ms = pd.read_csv(args.multi_start_file)
        (out_dir / "multi_start_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        for i, r in ms.iterrows():
            od_i = Path(r.out_dir); od_i.mkdir(parents=True, exist_ok=True)
            if (od_i / "final_candidates.csv").is_file():
                print(f"[resident] {i+1}/{len(ms)} skip (exists)", flush=True)
                continue
            seed_seq_i = clean_seq(str(r.seed_seq), length)
            cfg_i = dataclasses.replace(cfg, seed=int(r.seed))
            run_one(seed_seq_i, od_i, cfg_i)
            print(f"[resident] {i+1}/{len(ms)} {od_i.name} done", flush=True)
        print(f"[resident] ALL {len(ms)} DONE", flush=True)
        return

    run_one(seed_seq, out_dir, cfg)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    df = pd.read_csv(out_dir / "final_candidates.csv")
    print(df[["restart", "final_pred_brightness", "final_mutations_vs_reference", "is_known"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
