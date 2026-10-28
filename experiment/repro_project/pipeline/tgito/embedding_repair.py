from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmbeddingRepairConfig:
    """Denoising-residual mutation-site prior for v12.8.1.

    This is not a hotspot AA rule. It is a learned vector-field / projection prior:
    if a live mutated residue token is outside the released true-high mutation-token
    manifold, a frozen denoising residual model predicts how to move the token back
    toward that manifold. The optimizer still changes AA logits through the frozen
    ESM/Bridge/revised_slim path.
    """
    bank_npz: str = ""
    model_pt: str = ""
    weight: float = 0.0
    topk: int = 16
    temp: float = 0.07
    h_alpha: float = 0.80
    target_combo: float = 0.63
    pull_weight: float = 1.0
    gap_weight: float = 1.5
    denoised_bonus_weight: float = 0.35
    residual_penalty: float = 0.03
    min_mut_prob: float = 1e-4
    fallback_global_weight: float = 0.25
    residual_scale: float = 1.0
    # v12.8.2 directional mode: no hard "already-mutated-only" switch.
    # All mutable sites receive a small differentiable look-ahead weight so the
    # repair field can guide which positions should become mutations before the
    # argmax sequence has changed.
    mode: str = "directional"  # directional|legacy
    # A very small floor keeps look-ahead differentiable without letting 221
    # non-mutated sites dominate the score. site_topk keeps the repair field local.
    soft_mutation_floor: float = 1e-4
    mutation_weight_power: float = 1.0
    site_topk: int = 16


class DenoiseResidualMLP(nn.Module):
    def __init__(self, h_dim: int = 1280, p_dim: int = 4, pos_dim: int = 16, hidden: int = 512, dropout: float = 0.05):
        super().__init__()
        self.h_dim = int(h_dim)
        self.p_dim = int(p_dim)
        self.pos_dim = int(pos_dim)
        self.net = nn.Sequential(
            nn.Linear(self.h_dim + self.p_dim + self.pos_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.h_dim + self.p_dim),
        )
        # Zero-init makes the first denoising map identity. Training then learns only
        # residual corrections that improve manifold similarity, avoiding early collapse.
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, h: torch.Tensor, p4: torch.Tensor, pos_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([h.float(), p4.float(), pos_feat.float()], dim=-1)
        r = self.net(x)
        return r[..., : self.h_dim], r[..., self.h_dim :]


def position_features(pos: torch.Tensor, length: int = 221, dim: int = 16) -> torch.Tensor:
    """Sin/cos position code. pos is 0-based, shape [B] or scalar."""
    pos = pos.float().reshape(-1, 1) / max(float(length - 1), 1.0)
    half = max(1, int(dim) // 2)
    freqs = torch.pow(2.0, torch.arange(half, device=pos.device, dtype=pos.dtype)).reshape(1, -1)
    ang = 2.0 * math.pi * pos * freqs
    feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if feat.shape[-1] < dim:
        feat = F.pad(feat, (0, dim - feat.shape[-1]))
    return feat[..., :dim]


class EmbeddingRepairBank(nn.Module):
    """Frozen denoising-residual prior + true-high mutation token bank.

    Given live H/P4 from the current differentiable AA sequence, this module:
      1. scores raw mutation-site similarity to released true-high tokens;
      2. predicts a residual denoising step toward the true-high token manifold;
      3. builds a differentiable target from nearest true-high tokens;
      4. returns a reward that penalizes being below a calibrated target similarity
         such as 0.63.

    v12.8.2 note: generation no longer uses a hard "already mutated" switch.
    The bank is evaluated on all mutable candidate positions with a soft look-ahead
    weight derived from mutation probability. WT collapse is handled in the optimizer
    by a smooth minimum-mutation floor, not by disabling the repair field.
    """
    def __init__(self, bank_npz: str | Path, model_pt: str | Path = "", device: str = "cuda"):
        super().__init__()
        self.bank_path = str(bank_npz)
        self.model_path = str(model_pt) if model_pt else ""
        arr = np.load(self.bank_path, allow_pickle=True)
        if "positions" not in arr.files:
            raise ValueError(f"{bank_npz} lacks positions; expected v12.8 embedding realism bank")
        self.length = int(np.asarray(arr.get("length", np.array([221]))).reshape(-1)[0])
        self.device_name = str(device)
        self.H_by_pos: Dict[int, torch.Tensor] = {}
        self.P_by_pos: Dict[int, torch.Tensor] = {}
        self.R_by_pos: Dict[int, torch.Tensor] = {}
        for p in [int(x) for x in np.asarray(arr["positions"], dtype=np.int64).tolist()]:
            hk, pk, rk = f"H_pos_{p}", f"P4_pos_{p}", f"reward_pos_{p}"
            if hk not in arr.files or pk not in arr.files:
                continue
            H = np.asarray(arr[hk], dtype=np.float32)
            P = np.asarray(arr[pk], dtype=np.float32)
            if H.ndim != 2 or P.ndim != 2 or H.shape[0] == 0 or P.shape[0] != H.shape[0]:
                continue
            R = np.asarray(arr[rk], dtype=np.float32) if rk in arr.files else np.zeros((H.shape[0],), dtype=np.float32)
            Ht = torch.as_tensor(H, dtype=torch.float32, device=device)
            Pt = torch.as_tensor(P, dtype=torch.float32, device=device)
            self.H_by_pos[p] = F.normalize(Ht, dim=-1)
            self.P_by_pos[p] = F.normalize(Pt, dim=-1)
            self.R_by_pos[p] = torch.as_tensor(R, dtype=torch.float32, device=device)
        self.global_H: Optional[torch.Tensor] = None
        self.global_P: Optional[torch.Tensor] = None
        self.global_R: Optional[torch.Tensor] = None
        if "global_H" in arr.files and "global_P4" in arr.files:
            gH = np.asarray(arr["global_H"], dtype=np.float32)
            gP = np.asarray(arr["global_P4"], dtype=np.float32)
            if gH.ndim == 2 and gP.ndim == 2 and gH.shape[0] > 0 and gP.shape[0] == gH.shape[0]:
                gR = np.asarray(arr["global_reward"], dtype=np.float32) if "global_reward" in arr.files else np.zeros((gH.shape[0],), dtype=np.float32)
                self.global_H = F.normalize(torch.as_tensor(gH, dtype=torch.float32, device=device), dim=-1)
                self.global_P = F.normalize(torch.as_tensor(gP, dtype=torch.float32, device=device), dim=-1)
                self.global_R = torch.as_tensor(gR, dtype=torch.float32, device=device)
        if not self.H_by_pos and self.global_H is None:
            raise ValueError(f"{bank_npz} loaded no usable token banks")
        self.model: Optional[DenoiseResidualMLP] = None
        self.h_alpha_default = 0.80
        self.target_combo_default = 0.63
        if self.model_path:
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            schema = str(ckpt.get("schema", ""))
            if schema not in {"v12_8_1_embedding_repair", "v12_8_2_directional_repair"}:
                raise ValueError(f"{model_pt} schema={ckpt.get('schema')!r}; expected v12_8_1_embedding_repair or v12_8_2_directional_repair")
            self.h_alpha_default = float(ckpt.get("h_alpha", 0.80))
            self.target_combo_default = float(ckpt.get("target_combo", 0.63))
            self.model = DenoiseResidualMLP(
                h_dim=int(ckpt.get("h_dim", 1280)),
                p_dim=int(ckpt.get("p_dim", 4)),
                pos_dim=int(ckpt.get("pos_dim", 16)),
                hidden=int(ckpt.get("hidden", 512)),
                dropout=float(ckpt.get("dropout", 0.05)),
            ).to(device)
            self.model.load_state_dict(ckpt["state_dict"], strict=True)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        print(f"[EmbeddingRepairBank] site_banks={len(self.H_by_pos)} global={self.global_H is not None} model={bool(self.model_path)}")

    @staticmethod
    def _extract_p4(P: torch.Tensor) -> torch.Tensor:
        if P.shape[-1] == 4:
            return P
        if P.shape[-1] >= 10:
            return P[..., [0, 7, 8, 9]]
        raise ValueError(f"P/P4 last dim must be 4 or >=10, got {P.shape}")

    def _select_bank(self, pos: int, fallback_global_weight: float):
        if pos in self.H_by_pos:
            return self.H_by_pos[pos], self.P_by_pos[pos], self.R_by_pos[pos], 1.0, False
        if self.global_H is not None and float(fallback_global_weight) > 0:
            return self.global_H, self.global_P, self.global_R, float(fallback_global_weight), True
        return None, None, None, 0.0, False

    def _site_score(
        self,
        h_live: torch.Tensor,
        p_live: torch.Tensor,
        pos: int,
        bank_H: torch.Tensor,
        bank_P: torch.Tensor,
        bank_R: torch.Tensor,
        cfg: EmbeddingRepairConfig,
    ) -> Dict[str, torch.Tensor]:
        h0 = F.normalize(h_live.float(), dim=-1)
        p0 = F.normalize(p_live.float(), dim=-1)
        if self.model is not None:
            pf = position_features(torch.full((h0.shape[0],), int(pos), device=h0.device), length=self.length, dim=self.model.pos_dim)
            dh, dp = self.model(h0, p0, pf)
            h_den = F.normalize(h0 + float(cfg.residual_scale) * dh, dim=-1)
            p_den = F.normalize(p0 + float(cfg.residual_scale) * dp, dim=-1)
            residual_norm = torch.sqrt(dh.pow(2).mean(dim=-1) + dp.pow(2).mean(dim=-1).clamp_min(0.0))
        else:
            h_den, p_den = h0, p0
            residual_norm = h0.new_zeros((h0.shape[0],))
        alpha = float(cfg.h_alpha)
        sim_sel_h = h_den @ bank_H.T
        sim_sel_p = p_den @ bank_P.T
        sim_sel = alpha * sim_sel_h + (1.0 - alpha) * sim_sel_p
        k = min(max(1, int(cfg.topk)), sim_sel.shape[-1])
        vals, idx = torch.topk(sim_sel, k=k, dim=-1)
        r = bank_R.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        logits = (vals + 0.05 * r) / max(float(cfg.temp), 1e-6)
        w = torch.softmax(logits, dim=-1)
        target_H = F.normalize((w[..., None] * bank_H.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], -1)).sum(dim=1), dim=-1)
        target_P = F.normalize((w[..., None] * bank_P.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], -1)).sum(dim=1), dim=-1)
        raw_h = (h0 * target_H).sum(dim=-1)
        raw_p = (p0 * target_P).sum(dim=-1)
        raw_combo = alpha * raw_h + (1.0 - alpha) * raw_p
        den_h = (h_den * target_H).sum(dim=-1)
        den_p = (p_den * target_P).sum(dim=-1)
        den_combo = alpha * den_h + (1.0 - alpha) * den_p
        gap = torch.relu(float(cfg.target_combo) - raw_combo)
        # v12.8.2 directional repair:
        #   legacy mode rewards the denoised cosine itself. That produced excellent
        #   denoised metrics but collapsed generation back to WT. Directional mode
        #   uses the denoiser only to pick a plausible manifold target/direction,
        #   while the differentiable reward is paid to the *raw live* H/P4 token.
        #   The denoised value is kept as a diagnostic, not as the main reward.
        if str(getattr(cfg, "mode", "directional")).lower() == "legacy":
            score = (
                float(cfg.pull_weight) * raw_combo
                + float(cfg.denoised_bonus_weight) * torch.relu(den_combo - raw_combo)
                - float(cfg.gap_weight) * gap.pow(2)
                - float(cfg.residual_penalty) * residual_norm
            )
        else:
            score = (
                float(cfg.pull_weight) * raw_combo
                - float(cfg.gap_weight) * gap.pow(2)
                - float(cfg.residual_penalty) * residual_norm
            )
        return {
            "score": score,
            "raw_combo": raw_combo,
            "raw_saprot": raw_h,
            "raw_p4": raw_p,
            "denoised_combo": den_combo,
            "constraint_gap": gap,
            "residual_norm": residual_norm,
        }

    def forward(
        self,
        H_live: torch.Tensor,
        P_live: torch.Tensor,
        *,
        mutable_mask: Optional[torch.Tensor] = None,
        cfg: Optional[EmbeddingRepairConfig] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = cfg or EmbeddingRepairConfig(bank_npz=self.bank_path, model_pt=self.model_path)
        if not self.model_path and cfg.model_pt:
            raise ValueError("EmbeddingRepairBank was constructed without model_pt; reload with --embedding-repair-model-pt")
        P4 = self._extract_p4(P_live)
        if H_live.ndim != 3 or P4.ndim != 3:
            raise ValueError(f"Expected H[B,L,D] and P4[B,L,4], got H={H_live.shape}, P4={P4.shape}")
        B, L, _ = H_live.shape
        mprob = P4[..., 0].float().clamp(0.0, 1.0)
        if mutable_mask is not None:
            mm = mutable_mask.to(H_live.device).bool().view(1, L)
        else:
            mm = torch.ones((1, L), dtype=torch.bool, device=H_live.device)
        mprob = torch.where(mm, mprob, torch.zeros_like(mprob))
        names = ["score", "raw_combo", "raw_saprot", "raw_p4", "denoised_combo", "constraint_gap", "residual_norm"]
        totals = {n: H_live.new_zeros((B,), dtype=torch.float32) for n in names}
        wsum = H_live.new_zeros((B,), dtype=torch.float32)
        active = H_live.new_zeros((B,), dtype=torch.float32)
        used_global = H_live.new_zeros((B,), dtype=torch.float32)
        # Soft look-ahead weighting. This is not a hard active-site switch, but v12.8.2
        # used a 0.02 floor over every mutable site, so hundreds of near-WT positions
        # dominated the score. v12.8.4 keeps the floor tiny and evaluates only the
        # highest-probability potential mutation sites per sequence.
        floor = max(float(getattr(cfg, "soft_mutation_floor", 1e-4)), 0.0)
        power = max(float(getattr(cfg, "mutation_weight_power", 1.0)), 1e-6)
        soft_w_all = torch.where(mm, (mprob + floor).clamp_min(0.0).pow(power), torch.zeros_like(mprob))
        site_topk = int(max(1, min(int(getattr(cfg, "site_topk", 16)), min(L, self.length))))
        # Union of top-k positions across the batch. This remains differentiable w.r.t.
        # the selected weights and avoids averaging over all 221 residues.
        top_pos = torch.topk(soft_w_all, k=site_topk, dim=1).indices.detach().reshape(-1).unique().tolist()
        for pos in [int(x) for x in top_pos if int(x) < min(L, self.length)]:
            sw = soft_w_all[:, pos]
            if torch.max(sw).detach().item() <= float(cfg.min_mut_prob):
                continue
            bank_H, bank_P, bank_R, mult, is_global = self._select_bank(pos, float(cfg.fallback_global_weight))
            if bank_H is None:
                continue
            site = self._site_score(H_live[:, pos, :], P4[:, pos, :], pos, bank_H, bank_P, bank_R, cfg)
            w = sw * float(mult)
            for n in names:
                totals[n] = totals[n] + w * site[n]
            wsum = wsum + w
            active = active + sw
            if is_global:
                used_global = used_global + sw
        denom = wsum.clamp_min(1.0)
        return {
            "embedding_repair_score": totals["score"] / denom,
            "embedding_repair_raw_combo": totals["raw_combo"] / denom,
            "embedding_repair_raw_saprot": totals["raw_saprot"] / denom,
            "embedding_repair_raw_p4": totals["raw_p4"] / denom,
            "embedding_repair_denoised_combo": totals["denoised_combo"] / denom,
            "embedding_repair_constraint_gap": totals["constraint_gap"] / denom,
            "embedding_repair_residual_norm": totals["residual_norm"] / denom,
            "embedding_repair_weight_sum": wsum,
            "embedding_repair_active_sites": active,
            "embedding_repair_used_global_sites": used_global,
            "embedding_repair_soft_mut_floor": H_live.new_full((B,), float(getattr(cfg, "soft_mutation_floor", 1e-4))),
            "embedding_repair_site_topk": H_live.new_full((B,), float(getattr(cfg, "site_topk", 16))),
        }
