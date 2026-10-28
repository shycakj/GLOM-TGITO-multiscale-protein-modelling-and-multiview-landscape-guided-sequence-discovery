from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F

from .embedding_realism import EmbeddingRealismBank, EmbeddingRealismConfig
from .embedding_repair import EmbeddingRepairBank, EmbeddingRepairConfig
from .aa import (
    AA20,
    aa_bits_tensor,
    apply_mutation_mask,
    clean_seq,
    ids0_to_seq,
    make_logits_from_seq,
    make_relaxed_logits_from_ids,
    make_relaxed_logits_from_seq,
    mutation_count_soft,
    parse_positions,
    seq_to_ids0,
    soft_pos_feats,
)


def format_mutations(anchor: str, seq: str) -> str:
    muts = []
    for i, (a, b) in enumerate(zip(anchor, seq), start=1):
        if a != b:
            muts.append(f"{a}{i}{b}")
    return ";".join(muts)


def normalize_mutation_signature(muts: str) -> str:
    parts = [x.strip() for x in str(muts).replace(",", ";").split(";") if x.strip()]
    def key(m: str):
        digits = "".join(ch for ch in m if ch.isdigit())
        return (int(digits) if digits else 10**9, m)
    return ";".join(sorted(parts, key=key))


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(str(a), str(b))) + abs(len(str(a)) - len(str(b)))


@dataclass
class GeneratorConfig:
    length: int = 221
    steps: int = 180
    lr: float = 0.05
    min_lr: float = 1e-4
    line_search_trials: int = 1
    init_mode: str = "relaxed"
    init_scale: float = 9.0
    mutable_anchor_logit: float = 5.0
    mutable_alt_logit: float = 0.0
    lock_scale: float = 30.0
    temp_start: float = 1.2
    temp_end: float = 0.8
    bridge_temp: float = 1.0
    brightness_weight: float = 1.0
    terrain_weight: float = 0.03
    mutation_budget: float = 4.0
    budget_relative: str = "reference"  # seed|reference
    budget_weight: float = 0.25
    entropy_weight: float = 0.001
    brightness_floor_weight: float = 12.0
    anchor_drop: float = 0.05
    accept_drop: float = 0.03
    mut_mask_mode: str = "changed"
    noise_scale: float = 0.15
    proposal_every: int = 6
    proposal_candidates: int = 64
    proposal_order: int = 2
    proposal_alts_per_pos: int = 4
    proposal_accept_margin: float = 0.0
    proposal_reset_logits: bool = True
    novelty_min_hamming: int = 0
    sample_every: int = 10
    seed: int = 0
    amp_dtype: str = "off"  # off|fp16|bf16; speeds ESM2/Bridge on CUDA
    # v12.6: mutation-site embedding realism prior. This does not learn forced AA conversions.
    # It rewards live mutated sites whose original SaProt H + Pmutation P4 tokens are close
    # to real existing released-train high-brightness mutation-site tokens.
    embedding_realism_weight: float = 0.0
    embedding_realism_topk: int = 16
    embedding_realism_temp: float = 0.08
    embedding_realism_h_alpha: float = 0.80
    embedding_realism_reward_scale: float = 0.10
    embedding_realism_min_mut_prob: float = 1e-4
    embedding_realism_fallback_global_weight: float = 0.35
    # v12.8.1: denoising-residual embedding repair prior. Unlike raw cosine scoring,
    # this learns a residual vector field from corrupted true-high mutation tokens back
    # to the released true-high mutation-token manifold.
    embedding_repair_weight: float = 0.0
    embedding_repair_topk: int = 16
    embedding_repair_temp: float = 0.07
    embedding_repair_h_alpha: float = 0.80
    embedding_repair_target_combo: float = 0.63
    embedding_repair_pull_weight: float = 1.0
    embedding_repair_gap_weight: float = 1.5
    embedding_repair_denoised_bonus_weight: float = 0.0
    embedding_repair_residual_penalty: float = 0.03
    embedding_repair_min_mut_prob: float = 1e-4
    embedding_repair_fallback_global_weight: float = 0.25
    embedding_repair_residual_scale: float = 1.0
    # v12.8.2 directional repair: soft look-ahead, not hard active-site gating.
    embedding_repair_mode: str = "directional"
    embedding_repair_soft_mutation_floor: float = 1e-4
    embedding_repair_mutation_weight_power: float = 1.0
    embedding_repair_site_topk: int = 16
    embedding_repair_min_mutation_count: float = 1.25
    embedding_repair_min_mutation_weight: float = 0.80
    # v12.8.4: smooth seed-divergence floor. This is independent of repair and
    # prevents peak-seeded runs from simply returning the input seed unchanged.
    seed_explore_weight: float = 0.0
    seed_explore_min_change: float = 0.0
    # v12.8.4: relaxed logits leak non-anchor probability even when argmax is unchanged.
    # Count only confident soft changes above the initialization leakage baseline.
    change_count_threshold_margin: float = 0.08
    change_count_temp: float = 0.035
    change_count_topk: int = 8
    # Accept relaxed line-search steps by full objective when enabled.
    accept_by_objective: bool = False
    objective_accept_drop: float = 0.01
    select_by_objective: bool = False
    # Publication benchmark: final output must satisfy the hard mutation budget and,
    # when requested, must not be an already-observed sequence.
    reject_known_output: bool = False


class TGITOGenerator:
    def __init__(self, seq_encoder, revised, terrain, y_mean: float = 0.0, y_std: float = 1.0, device: str = "cuda",
                 embedding_realism: Optional[EmbeddingRealismBank] = None,
                 embedding_repair: Optional[EmbeddingRepairBank] = None):
        self.seq_encoder = seq_encoder
        self.revised = revised
        self.terrain = terrain
        self.embedding_realism = embedding_realism
        self.embedding_repair = embedding_repair
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.device = device
        self.aa_bits = aa_bits_tensor(device=device)
        self._anchor_pred = -1e9

    def _hard_probs_from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return F.one_hot(ids.to(self.device).long(), num_classes=20).float()[None]

    def _probs_from_logits(self, logits: torch.Tensor, base_ids: torch.Tensor, mutable_mask: torch.Tensor, cfg: GeneratorConfig) -> torch.Tensor:
        temp = float(cfg.temp_end)
        probs = torch.softmax(logits / max(temp, 1e-6), dim=-1)
        locked = F.one_hot(base_ids.to(logits.device).long(), num_classes=20).float()
        return torch.where(mutable_mask.to(logits.device).bool()[:, None], probs, locked)[None]

    def _init_leakage_prob(self, cfg: GeneratorConfig) -> float:
        """Expected non-anchor probability at an unchanged mutable site under relaxed init.

        With anchor_logit=5 and alt_logit=0 this is about 0.113, so a naive
        soft mutation count over many mutable sites falsely reports several
        mutations even when argmax has not changed.
        """
        import math
        a = float(cfg.mutable_anchor_logit)
        b = float(cfg.mutable_alt_logit)
        temp = max(float(cfg.temp_end), 1e-6)
        ea = math.exp(a / temp)
        eb = math.exp(b / temp)
        p_anchor = ea / max(ea + 19.0 * eb, 1e-12)
        return float(max(0.0, min(1.0, 1.0 - p_anchor)))

    def _confident_change_mass(self, probs: torch.Tensor, ids0: torch.Tensor, mutable_mask: torch.Tensor, cfg: GeneratorConfig, topk: Optional[int] = None) -> torch.Tensor:
        """Differentiable hard-like mutation count with relaxed-logit leakage removed.

        This is the key v12.8.4 fix. `mutation_count_soft` treats the initial
        relaxed distribution as already mutated because all 19 non-anchor AAs have
        small probability. Here we count only probability above that baseline,
        using softplus so gradients still exist before the argmax flips.
        """
        if probs.ndim == 2:
            probs = probs[None]
        ids = ids0.to(probs.device).long()
        one = F.one_hot(ids, num_classes=20).to(dtype=probs.dtype)
        same = (probs * one[None]).sum(dim=-1)
        p_change = (1.0 - same).clamp(0.0, 1.0)
        mm = mutable_mask.to(probs.device).bool().view(1, -1)
        p_change = torch.where(mm, p_change, torch.zeros_like(p_change))
        thr = self._init_leakage_prob(cfg) + float(getattr(cfg, "change_count_threshold_margin", 0.08))
        thr = float(max(0.05, min(0.75, thr)))
        tau = float(max(getattr(cfg, "change_count_temp", 0.035), 1e-4))
        site = F.softplus((p_change - thr) / tau) * tau / max(1.0 - thr, 1e-6)
        if topk is None:
            topk = int(getattr(cfg, "change_count_topk", 0))
        if topk and int(topk) > 0 and int(topk) < site.shape[1]:
            site = torch.topk(site, k=int(topk), dim=1).values
        return site.sum(dim=1)

    def _forward_from_probs(self, probs: torch.Tensor, anchor_ids: torch.Tensor, reference_ids: torch.Tensor, mutable_mask: torch.Tensor, cfg: GeneratorConfig):
        use_amp = (str(getattr(cfg, "amp_dtype", "off")).lower() != "off") and torch.cuda.is_available() and str(self.device).startswith("cuda")
        dtype = torch.bfloat16 if str(getattr(cfg, "amp_dtype", "off")).lower() == "bf16" else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp)
        with ctx:
            enc = self.seq_encoder(probs, bridge_temp=cfg.bridge_temp)
            H = enc["H"]
            P = soft_pos_feats(probs, reference_ids, self.aa_bits)
            if cfg.mut_mask_mode == "allowed":
                P = P.clone()
                P[0, :, 0] = mutable_mask.to(P.device, dtype=P.dtype)
            out = self.revised.forward_all(H, P)
            y = out["y_norm"] * self.y_std + self.y_mean
            # v12.8: compute mutation-site embedding realism BEFORE TerrainPotential
            # so the credible miner can use the same live-computable realism features
            # that are optimized in the objective. This fixes the earlier issue where
            # embedding-aware learning existed only as a post-hoc benchmark.
            z = y.new_zeros(y.shape)
            real = {
                "embedding_realism": z,
                "embedding_realism_combo": z,
                "embedding_realism_saprot": z,
                "embedding_realism_p4": z,
                "embedding_realism_weight_sum": z,
                "embedding_realism_active_sites": z,
                "embedding_realism_used_global_sites": z,
            }
            if self.embedding_realism is not None and float(getattr(cfg, "embedding_realism_weight", 0.0)) > 0:
                rcfg = EmbeddingRealismConfig(
                    bank_npz=getattr(self.embedding_realism, "path", ""),
                    weight=float(cfg.embedding_realism_weight),
                    topk=int(cfg.embedding_realism_topk),
                    temp=float(cfg.embedding_realism_temp),
                    h_alpha=float(cfg.embedding_realism_h_alpha),
                    reward_scale=float(cfg.embedding_realism_reward_scale),
                    min_mut_prob=float(cfg.embedding_realism_min_mut_prob),
                    fallback_global_weight=float(cfg.embedding_realism_fallback_global_weight),
                )
                real = self.embedding_realism(H, P, mutable_mask=mutable_mask, cfg=rcfg)
            repair = {
                "embedding_repair_score": z,
                "embedding_repair_raw_combo": z,
                "embedding_repair_raw_saprot": z,
                "embedding_repair_raw_p4": z,
                "embedding_repair_denoised_combo": z,
                "embedding_repair_constraint_gap": z,
                "embedding_repair_residual_norm": z,
                "embedding_repair_weight_sum": z,
                "embedding_repair_active_sites": z,
                "embedding_repair_used_global_sites": z,
            }
            if self.embedding_repair is not None and float(getattr(cfg, "embedding_repair_weight", 0.0)) > 0:
                pcfg = EmbeddingRepairConfig(
                    bank_npz=getattr(self.embedding_repair, "bank_path", ""),
                    model_pt=getattr(self.embedding_repair, "model_path", ""),
                    weight=float(cfg.embedding_repair_weight),
                    topk=int(cfg.embedding_repair_topk),
                    temp=float(cfg.embedding_repair_temp),
                    h_alpha=float(cfg.embedding_repair_h_alpha),
                    target_combo=float(cfg.embedding_repair_target_combo),
                    pull_weight=float(cfg.embedding_repair_pull_weight),
                    gap_weight=float(cfg.embedding_repair_gap_weight),
                    denoised_bonus_weight=float(cfg.embedding_repair_denoised_bonus_weight),
                    residual_penalty=float(cfg.embedding_repair_residual_penalty),
                    min_mut_prob=float(cfg.embedding_repair_min_mut_prob),
                    fallback_global_weight=float(cfg.embedding_repair_fallback_global_weight),
                    residual_scale=float(cfg.embedding_repair_residual_scale),
                    mode=str(getattr(cfg, "embedding_repair_mode", "directional")),
                    soft_mutation_floor=float(getattr(cfg, "embedding_repair_soft_mutation_floor", 1e-4)),
                    mutation_weight_power=float(getattr(cfg, "embedding_repair_mutation_weight_power", 1.0)),
                    site_topk=int(getattr(cfg, "embedding_repair_site_topk", 16)),
                )
                repair = self.embedding_repair(H, P, mutable_mask=mutable_mask, cfg=pcfg)
            out_for_terrain = dict(out)
            out_for_terrain["pred_brightness_live"] = y.float()
            # v13.1: differentiable sequence source for UMAP_sequence.
            # The 12-source NPZ stores sequence_projection [221*20, Dseq].
            # Live soft AA probabilities are projected with the same matrix.
            try:
                seq_proj = getattr(getattr(self.terrain, "bank", None), "sequence_projection", None)
                if seq_proj is not None and "sequence" in getattr(self.terrain, "sources", []):
                    flat_probs = probs.reshape(probs.shape[0], -1).float()
                    out_for_terrain["sequence"] = flat_probs @ seq_proj.to(flat_probs.device, dtype=flat_probs.dtype)
            except Exception as e:
                print(f"[WARN] v13.1 sequence source disabled: {e}")
            out_for_terrain.update(real)
            out_for_terrain.update(repair)
            terr = dict(self.terrain(out_for_terrain))
            terr.update(real)
            terr.update(repair)
        return y.float(), terr, out, enc

    def predict_hard_sequence(self, seq: str, cfg: GeneratorConfig, reference_seq: Optional[str] = None, mutable_mask: Optional[torch.Tensor] = None) -> float:
        seq_clean = clean_seq(seq, cfg.length)
        ids = seq_to_ids0(seq_clean, cfg.length, self.device)
        reference_seq = reference_seq or seq_clean
        ref_ids = seq_to_ids0(clean_seq(reference_seq, cfg.length), cfg.length, self.device)
        if mutable_mask is None:
            mutable_mask = torch.zeros(cfg.length, dtype=torch.bool, device=self.device)
        probs = self._hard_probs_from_ids(ids)
        with torch.no_grad():
            y, _, _, _ = self._forward_from_probs(probs, ids, ref_ids, mutable_mask, cfg)
        return float(y.mean().detach().cpu())

    def _predict_hard_ids_batch(self, ids_batch: torch.Tensor, reference_ids: torch.Tensor, mutable_mask: torch.Tensor, cfg: GeneratorConfig) -> torch.Tensor:
        probs = F.one_hot(ids_batch.to(self.device).long(), num_classes=20).float()
        with torch.no_grad():
            y, _, _, _ = self._forward_from_probs(probs, ids_batch[0].to(self.device), reference_ids, mutable_mask, cfg)
        return y.reshape(-1)

    def _loss_and_diag(self, logits: torch.Tensor, base_ids: torch.Tensor, reference_ids: torch.Tensor, mutable_mask: torch.Tensor, cfg: GeneratorConfig):
        probs = self._probs_from_logits(logits, base_ids, mutable_mask, cfg)
        y, terr, out, enc = self._forward_from_probs(probs, base_ids, reference_ids, mutable_mask, cfg)
        budget_ids = reference_ids if cfg.budget_relative == "reference" else base_ids
        mcnt_soft_raw = mutation_count_soft(probs, budget_ids)
        anchor_mcnt_soft_raw = mutation_count_soft(probs, base_ids)
        mcnt = self._confident_change_mass(probs, budget_ids, mutable_mask, cfg, topk=0)
        anchor_mcnt = self._confident_change_mass(probs, base_ids, mutable_mask, cfg, topk=int(getattr(cfg, "change_count_topk", 8)))
        budget = torch.relu(mcnt - float(cfg.mutation_budget)).pow(2)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=-1)
        floor = torch.relu((self._anchor_pred - float(cfg.anchor_drop)) - y).pow(2)
        repair_w = float(getattr(cfg, "embedding_repair_weight", 0.0))
        repair_min_mut = torch.relu(float(getattr(cfg, "embedding_repair_min_mutation_count", 1.25)) - mcnt).pow(2)
        seed_explore_w = float(getattr(cfg, "seed_explore_weight", 0.0))
        seed_explore_penalty = torch.relu(float(getattr(cfg, "seed_explore_min_change", 0.0)) - anchor_mcnt).pow(2)
        objective = (
            float(cfg.brightness_weight) * y
            + float(cfg.terrain_weight) * terr["terrain"]
            + float(getattr(cfg, "embedding_realism_weight", 0.0)) * terr.get("embedding_realism", torch.zeros_like(y))
            + repair_w * terr.get("embedding_repair_score", torch.zeros_like(y))
            - repair_w * float(getattr(cfg, "embedding_repair_min_mutation_weight", 0.80)) * repair_min_mut
            - seed_explore_w * seed_explore_penalty
            - float(cfg.budget_weight) * budget
            + float(cfg.entropy_weight) * entropy
            - float(cfg.brightness_floor_weight) * floor
        )
        loss = -objective.mean()
        diag = {
            "pred_brightness": float(y.mean().detach().cpu()),
            "terrain": float(terr["terrain"].mean().detach().cpu()),
            "mutation_count_soft": float(mcnt.mean().detach().cpu()),
            "anchor_change_count_soft": float(anchor_mcnt.mean().detach().cpu()),
            "mutation_count_soft_raw": float(mcnt_soft_raw.mean().detach().cpu()),
            "anchor_change_count_soft_raw": float(anchor_mcnt_soft_raw.mean().detach().cpu()),
            "change_count_leakage_baseline": float(self._init_leakage_prob(cfg)),
            "budget_penalty": float(budget.mean().detach().cpu()),
            "embedding_repair_min_mut_penalty": float(repair_min_mut.mean().detach().cpu()),
            "seed_explore_penalty": float(seed_explore_penalty.mean().detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
            "objective": float(objective.mean().detach().cpu()),
            "anchor_pred_brightness": float(self._anchor_pred),
        }
        # Preserve terrain/gate evidence in history and final_candidates. This is critical for v12 credibility:
        # a candidate should report not only predicted brightness, but which maps supported it and whether it
        # was close to released true-high prototypes.
        for k, v in terr.items():
            if k == "terrain":
                continue
            try:
                if torch.is_tensor(v):
                    vv = v.detach().float().mean().cpu().item()
                    if isinstance(vv, float):
                        diag[str(k)] = float(vv)
                elif isinstance(v, (float, int)):
                    diag[str(k)] = float(v)
            except Exception:
                pass
        with torch.no_grad():
            p0 = probs[0]
            m = mutable_mask.bool().to(p0.device)
            if m.any():
                base_p = p0[m].gather(1, base_ids[m, None]).squeeze(1)
                alt = p0[m].clone(); alt.scatter_(1, base_ids[m, None], -1.0)
                diag["mutable_anchor_prob_mean"] = float(base_p.mean().cpu())
                diag["mutable_max_alt_prob_mean"] = float(alt.max(dim=1).values.mean().cpu())
                diag["mutable_changed_argmax_count"] = float((p0[m].argmax(dim=1) != base_ids[m]).float().sum().cpu())
        return loss, objective.detach(), diag, probs, out, enc

    def _record_current(self, logits, base_ids, reference_ids, mutable_mask, cfg, anchor, reference, known_sequences):
        with torch.no_grad():
            _, _, diag, probs, _, _ = self._loss_and_diag(logits, base_ids, reference_ids, mutable_mask, cfg)
            ids = probs[0].argmax(dim=-1)
            seq = ids0_to_seq(ids)
            hard_pred = self.predict_hard_sequence(seq, cfg, reference_seq=reference, mutable_mask=mutable_mask)
            mut_seed = format_mutations(anchor, seq)
            mut_ref = format_mutations(reference, seq)
            row = {
                **diag,
                "changed_vs_seed_count": hamming(anchor, seq),
                "changed_vs_reference_count": hamming(reference, seq),
                "is_seed_unchanged": bool(seq == anchor),
                "is_reference_unchanged": bool(seq == reference),
                "hard_argmax_pred_brightness": hard_pred,
                "seq": seq,
                "mutations": mut_seed,
                "mutations_vs_reference": mut_ref,
                "is_known": bool(seq in known_sequences) if known_sequences is not None else False,
            }
            return row, seq

    def _score_hard_ids_batch(self, ids_batch: torch.Tensor, base_ids: torch.Tensor, reference_ids: torch.Tensor, mutable_mask: torch.Tensor, cfg: GeneratorConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """Return brightness and full frozen-objective score for hard proposal sequences.

        v12.6 uses the full objective, including embedding_realism, so hard proposals do not
        jump to high predicted brightness but unrealistic mutation-site embeddings.
        """
        probs = F.one_hot(ids_batch.to(self.device).long(), num_classes=20).float()
        with torch.no_grad():
            y, terr, _, _ = self._forward_from_probs(probs, ids_batch[0].to(self.device), reference_ids, mutable_mask, cfg)
            budget_ids = reference_ids if cfg.budget_relative == "reference" else base_ids
            hard_ids = ids_batch.to(self.device).long()
            mcnt = (hard_ids != budget_ids[None, :].to(self.device).long()).float().sum(dim=1)
            anchor_mcnt = (hard_ids != base_ids[None, :].to(self.device).long()).float().sum(dim=1)
            budget = torch.relu(mcnt - float(cfg.mutation_budget)).pow(2)
            floor = torch.relu((self._anchor_pred - float(cfg.anchor_drop)) - y).pow(2)
            repair_w = float(getattr(cfg, "embedding_repair_weight", 0.0))
            repair_min_mut = torch.relu(float(getattr(cfg, "embedding_repair_min_mutation_count", 1.25)) - mcnt).pow(2)
            seed_explore_w = float(getattr(cfg, "seed_explore_weight", 0.0))
            seed_explore_penalty = torch.relu(float(getattr(cfg, "seed_explore_min_change", 0.0)) - anchor_mcnt).pow(2)
            score = (
                float(cfg.brightness_weight) * y.reshape(-1)
                + float(cfg.terrain_weight) * terr["terrain"].reshape(-1)
                + float(getattr(cfg, "embedding_realism_weight", 0.0)) * terr.get("embedding_realism", torch.zeros_like(y)).reshape(-1)
                + repair_w * terr.get("embedding_repair_score", torch.zeros_like(y)).reshape(-1)
                - repair_w * float(getattr(cfg, "embedding_repair_min_mutation_weight", 0.80)) * repair_min_mut.reshape(-1)
                - seed_explore_w * seed_explore_penalty.reshape(-1)
                - float(cfg.budget_weight) * budget.reshape(-1)
                - float(cfg.brightness_floor_weight) * floor.reshape(-1)
            )
        return y.reshape(-1), score.reshape(-1)

    def _try_discrete_proposals(self, logits, grad, base_ids, reference_ids, mutable_mask, cfg, current_hard_pred, current_objective=None, reference_seq="", avoid_signatures=None, archive_sequences=None):
        if int(cfg.proposal_candidates) <= 0:
            return logits, False, current_hard_pred, ""
        avoid_signatures = avoid_signatures or set()
        archive_sequences = archive_sequences or []
        with torch.no_grad():
            cur_ids = logits.argmax(dim=-1).detach()
            mpos = torch.where(mutable_mask.bool().to(logits.device))[0]
            if mpos.numel() == 0:
                return logits, False, current_hard_pred, ""
            cand_logits = logits - float(cfg.lr) * grad
            budget_ids = reference_ids if cfg.budget_relative == "reference" else base_ids

            def ok(test_ids):
                if (test_ids != budget_ids).float().sum().item() > float(cfg.mutation_budget) + 1e-6:
                    return False
                seq = ids0_to_seq(test_ids)
                sig = normalize_mutation_signature(format_mutations(reference_seq, seq))
                if sig in avoid_signatures:
                    return False
                if int(cfg.novelty_min_hamming) > 0 and archive_sequences:
                    if min(hamming(seq, a) for a in archive_sequences) < int(cfg.novelty_min_hamming):
                        return False
                return True

            singles = []
            for i in mpos.tolist():
                cur = int(cur_ids[i].item())
                scores = cand_logits[i].detach().clone(); scores[cur] = -1e9
                topv, topaa = torch.topk(scores, k=min(max(1, int(cfg.proposal_alts_per_pos)), 20))
                for val, aa in zip(topv.tolist(), topaa.tolist()):
                    test = cur_ids.clone(); test[i] = int(aa)
                    if ok(test):
                        singles.append((float(val), [(int(i), cur, int(aa))], test))
            proposals = list(singles)
            if int(cfg.proposal_order) >= 2 and len(singles) >= 2:
                pool = sorted(singles, key=lambda x: x[0], reverse=True)[:64]
                seen = {tuple(p[-1].detach().cpu().tolist()) for p in proposals}
                for a in range(len(pool)):
                    for b in range(a + 1, len(pool)):
                        ma, mb = pool[a][1], pool[b][1]
                        if ma[0][0] == mb[0][0]:
                            continue
                        test = cur_ids.clone()
                        for pos, _, aa in ma + mb:
                            test[pos] = aa
                        key = tuple(test.detach().cpu().tolist())
                        if key in seen:
                            continue
                        seen.add(key)
                        if ok(test):
                            proposals.append((float(pool[a][0] + pool[b][0]), ma + mb, test))
            if not proposals:
                return logits, False, current_hard_pred, ""
            proposals.sort(key=lambda x: x[0], reverse=True)
            proposals = proposals[:int(cfg.proposal_candidates)]
            ids_batch = torch.stack([p[-1] for p in proposals], dim=0)
            preds, scores = self._score_hard_ids_batch(ids_batch, base_ids, reference_ids, mutable_mask, cfg)
            best_idx = int(torch.argmax(scores).item())
            best_pred = float(preds[best_idx].detach().cpu())
            best_score = float(scores[best_idx].detach().cpu())
            _, best_muts, best_ids = proposals[best_idx]
            # Keep a brightness floor, but rank/accept proposals by full objective so realism matters.
            accept = best_pred >= float(self._anchor_pred) - float(cfg.anchor_drop)
            if current_objective is not None:
                accept = accept and best_score >= float(current_objective) + float(cfg.proposal_accept_margin)
            else:
                accept = accept and best_pred >= float(current_hard_pred) + float(cfg.proposal_accept_margin)
            if not accept:
                return logits, False, current_hard_pred, ""
            new_logits = make_relaxed_logits_from_ids(best_ids, cfg.length, mutable_mask, cfg.mutable_anchor_logit, cfg.mutable_alt_logit, cfg.lock_scale, self.device)
            mut = ";".join(f"{pos+1}:{old}->{new}" for pos, old, new in best_muts)
            return new_logits.detach(), True, best_pred, mut

    def run(self, anchor_seq: str, mutable_positions: str, out_dir: str | Path, cfg: GeneratorConfig, restart_id: int = 0,
            known_sequences: Optional[set[str]] = None, reference_seq: Optional[str] = None,
            avoid_signatures: Optional[set[str]] = None, archive_sequences: Optional[list[str]] = None):
        torch.manual_seed(int(cfg.seed) + int(restart_id) * 10007)
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        anchor = clean_seq(anchor_seq, cfg.length)
        reference = clean_seq(reference_seq if reference_seq is not None else anchor, cfg.length)
        base_ids = seq_to_ids0(anchor, cfg.length, self.device)
        reference_ids = seq_to_ids0(reference, cfg.length, self.device)
        mutable_mask = parse_positions(mutable_positions, cfg.length).to(self.device)
        known_sequences = known_sequences or set()
        avoid_signatures = avoid_signatures or set()
        archive_sequences = archive_sequences or []

        self._anchor_pred = self.predict_hard_sequence(anchor, cfg, reference_seq=reference, mutable_mask=mutable_mask)
        print(f"[anchor] hard_pred={self._anchor_pred:.4f} mutable_sites={int(mutable_mask.sum())} budget={cfg.mutation_budget} budget_relative={cfg.budget_relative} muts_vs_ref={format_mutations(reference, anchor)}")
        if cfg.init_mode == "hard":
            logits = make_logits_from_seq(anchor, cfg.length, cfg.init_scale, self.device)
            logits = apply_mutation_mask(logits, base_ids, mutable_mask, cfg.lock_scale)
        else:
            logits = make_relaxed_logits_from_seq(anchor, cfg.length, mutable_mask, cfg.mutable_anchor_logit, cfg.mutable_alt_logit, cfg.lock_scale, self.device)
        if cfg.noise_scale > 0:
            logits = logits + torch.randn_like(logits) * float(cfg.noise_scale) * mutable_mask[:, None].float()
        logits = apply_mutation_mask(logits, base_ids, mutable_mask, cfg.lock_scale).detach().requires_grad_(True)

        row0, _ = self._record_current(logits, base_ids, reference_ids, mutable_mask, cfg, anchor, reference, known_sequences)
        first_row = dict(step=0, restart=restart_id, **row0, accepted=True, proposal_accepted=False, proposal_mutation="", used_lr=0.0)
        def hard_valid(row):
            cnt = float(row["changed_vs_reference_count"] if cfg.budget_relative == "reference" else row["changed_vs_seed_count"])
            if cnt > float(cfg.mutation_budget) + 1e-6:
                return False
            if int(getattr(cfg, "novelty_min_hamming", 0)) > 0 and float(row["changed_vs_seed_count"]) < int(cfg.novelty_min_hamming):
                return False
            if bool(getattr(cfg, "reject_known_output", False)) and bool(row.get("is_known", False)):
                return False
            return True
        first_row["hard_valid_output"] = bool(hard_valid(first_row))
        best_row = dict(first_row) if first_row["hard_valid_output"] else None
        best_hard = float(first_row["hard_argmax_pred_brightness"]) if best_row is not None else -float("inf")
        best_obj = float(first_row["objective"]) if best_row is not None else -float("inf")
        best_logits = logits.detach().clone() if best_row is not None else None
        rows = [first_row]
        prev_pred = float(row0["pred_brightness"])

        for t in range(1, int(cfg.steps) + 1):
            if logits.grad is not None:
                logits.grad = None
            loss, _, diag, _, _, _ = self._loss_and_diag(logits, base_ids, reference_ids, mutable_mask, cfg)
            grad = torch.autograd.grad(loss, logits, retain_graph=False, allow_unused=False)[0]
            if not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                raise RuntimeError("Invalid or zero gradient through AA logits")
            old_logits = logits.detach().clone(); old_pred = prev_pred
            # Current objective baseline for objective-aware line search.
            old_obj = float(diag.get("objective", old_pred))
            accepted = False; used_lr = float(cfg.lr)
            # Speed mode: line_search_trials <= 0 skips extra full ESM/Bridge forward passes.
            # The brightness floor and budget penalties remain inside the differentiable objective.
            if int(cfg.line_search_trials) <= 0:
                cand = (old_logits - float(cfg.lr) * grad).detach()
                logits = apply_mutation_mask(cand, base_ids, mutable_mask, cfg.lock_scale).requires_grad_(True)
                accepted = True; used_lr = float(cfg.lr); prev_pred = old_pred
            else:
                for ls in range(int(cfg.line_search_trials)):
                    lr = max(float(cfg.lr) * (0.5 ** ls), float(cfg.min_lr))
                    cand = (old_logits - lr * grad).detach()
                    cand = apply_mutation_mask(cand, base_ids, mutable_mask, cfg.lock_scale).requires_grad_(True)
                    with torch.no_grad():
                        _, _, cand_diag, _, _, _ = self._loss_and_diag(cand, base_ids, reference_ids, mutable_mask, cfg)
                    brightness_ok = cand_diag["pred_brightness"] >= self._anchor_pred - float(cfg.anchor_drop)
                    if bool(getattr(cfg, "accept_by_objective", False)):
                        improve_ok = float(cand_diag.get("objective", cand_diag["pred_brightness"])) >= old_obj - float(getattr(cfg, "objective_accept_drop", 0.01))
                    else:
                        improve_ok = cand_diag["pred_brightness"] >= old_pred - float(cfg.accept_drop)
                    if brightness_ok and improve_ok:
                        logits = cand.detach().requires_grad_(True)
                        prev_pred = float(cand_diag["pred_brightness"])
                        accepted = True; used_lr = lr
                        break
                if not accepted:
                    logits = old_logits.detach().requires_grad_(True); used_lr = 0.0; prev_pred = old_pred
            proposal_accepted = False; proposal_mutation = ""
            if int(cfg.proposal_every) > 0 and t % int(cfg.proposal_every) == 0:
                current_row_tmp, _ = self._record_current(logits, base_ids, reference_ids, mutable_mask, cfg, anchor, reference, known_sequences)
                current_hard = float(current_row_tmp["hard_argmax_pred_brightness"])
                prop_logits, prop_ok, prop_pred, prop_mut = self._try_discrete_proposals(
                    logits.detach(), grad.detach(), base_ids, reference_ids, mutable_mask, cfg, current_hard,
                    current_objective=float(current_row_tmp.get("objective", current_hard)),
                    reference_seq=reference, avoid_signatures=avoid_signatures, archive_sequences=archive_sequences,
                )
                if prop_ok:
                    logits = prop_logits.detach().requires_grad_(True)
                    proposal_accepted = True; proposal_mutation = prop_mut; prev_pred = max(prev_pred, prop_pred)
            if (t % max(1, int(cfg.sample_every)) == 0) or proposal_accepted or t == int(cfg.steps):
                row, _ = self._record_current(logits, base_ids, reference_ids, mutable_mask, cfg, anchor, reference, known_sequences)
                row.update(step=t, restart=restart_id, accepted=accepted, proposal_accepted=proposal_accepted,
                           proposal_mutation=proposal_mutation, used_lr=used_lr, grad_norm=float(grad.norm().detach().cpu()),
                           mutable_grad_norm=float(grad[mutable_mask.bool()].norm().detach().cpu()) if mutable_mask.any() else float(grad.norm().detach().cpu()))
                row["hard_valid_output"] = bool(hard_valid(row))
                rows.append(row)
                rh = float(row["hard_argmax_pred_brightness"]); ro = float(row["objective"])
                if row["hard_valid_output"]:
                    if bool(getattr(cfg, "select_by_objective", False)):
                        is_better = (rh >= float(self._anchor_pred) - float(cfg.anchor_drop)) and (ro > best_obj + 1e-6)
                    else:
                        is_better = rh > best_hard + 1e-6 or (rh >= best_hard - 1e-6 and ro > best_obj)
                    if is_better:
                        best_hard = rh; best_obj = ro; best_row = dict(row); best_logits = logits.detach().clone()
                if t % max(10, int(cfg.sample_every) * 4) == 0 or proposal_accepted:
                    print(f"[r{restart_id} step {t}] hard={rh:.4f} soft={row['pred_brightness']:.4f} terrain={row['terrain']:.3f} prop={proposal_accepted} muts={row['mutations_vs_reference']}")
        pd.DataFrame(rows).to_csv(out_dir / f"history_restart{restart_id}.csv", index=False)
        if best_row is None or best_logits is None:
            # Keep the run auditable instead of silently returning an invalid sequence.
            fallback = dict(first_row)
            fallback["final_seq"] = anchor
            fallback["final_pred_brightness"] = float(first_row["hard_argmax_pred_brightness"])
            fallback["final_objective"] = float(first_row["objective"])
            fallback["final_mutations"] = ""
            fallback["final_mutations_vs_seed"] = ""
            fallback["final_mutations_vs_reference"] = str(first_row.get("mutations_vs_reference", ""))
            fallback["valid_output"] = False
            fallback["invalid_reason"] = "no_novel_hard_budget_valid_candidate"
            return fallback
        final_ids = best_logits.argmax(dim=-1)
        final_seq = ids0_to_seq(final_ids)
        final_cnt = hamming(reference if cfg.budget_relative == "reference" else anchor, final_seq)
        if final_cnt > float(cfg.mutation_budget) + 1e-6:
            raise RuntimeError(f"Internal error: final candidate violates hard mutation budget: {final_cnt}>{cfg.mutation_budget}")
        if bool(getattr(cfg, "reject_known_output", False)) and final_seq in known_sequences:
            raise RuntimeError("Internal error: final candidate is an observed/known sequence despite reject_known_output")
        best_row["valid_output"] = True
        best_row["invalid_reason"] = ""
        best_row["final_seq"] = final_seq
        best_row["final_pred_brightness"] = float(best_row["hard_argmax_pred_brightness"])
        best_row["final_objective"] = float(best_row["objective"])
        best_row["final_mutations"] = str(best_row.get("mutations", ""))
        best_row["final_mutations_vs_seed"] = str(best_row.get("mutations", ""))
        best_row["final_mutations_vs_reference"] = str(best_row.get("mutations_vs_reference", ""))
        return best_row
