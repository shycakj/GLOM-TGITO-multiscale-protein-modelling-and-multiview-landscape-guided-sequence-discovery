from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EmbeddingRealismConfig:
    """Runtime config for mutation-site embedding realism prior.

    The prior does not learn a forced AA conversion. It stores real high-brightness
    mutation-site tokens from released_train rows and rewards a live candidate when
    its *mutated positions* are close to those real tokens in original SaProt H and
    current Pmutation P4 space.
    """
    bank_npz: str = ""
    weight: float = 0.0
    topk: int = 16
    temp: float = 0.08
    h_alpha: float = 0.80
    reward_scale: float = 0.10
    min_mut_prob: float = 1e-4
    fallback_global_weight: float = 0.35


class EmbeddingRealismBank(nn.Module):
    """Differentiable mutation-site realism bank.

    Bank layout is created by build_v12_7_embedding_realism_bank.py.
    Per position p, it may contain:
        H_pos_{p}:      [M, 1280] normalized or raw SaProt embeddings
        P4_pos_{p}:     [M, 4]    current Pmutation features
        reward_pos_{p}: [M]       brightness-derived reward in roughly [0, 1]

    At generation time, the live sequence provides H_live[B,L,1280] and P_live[B,L,10]
    or P4_live[B,L,4]. Only positions with non-trivial mutation probability are scored.
    Non-mutated positions do not dominate the score.
    """
    def __init__(self, path: str | Path, device: str = "cuda"):
        super().__init__()
        self.path = str(path)
        arr = np.load(self.path, allow_pickle=True)
        if "positions" not in arr.files:
            raise ValueError(f"{path} lacks 'positions'; expected v12.7 embedding realism bank")
        self.length = int(np.asarray(arr.get("length", np.array([221]))).reshape(-1)[0])
        self.h_alpha = float(np.asarray(arr.get("h_alpha_default", np.array([0.80]))).reshape(-1)[0])
        self.device_name = device
        positions = [int(x) for x in np.asarray(arr["positions"], dtype=np.int64).tolist()]
        self.positions = positions
        self.H_by_pos: Dict[int, torch.Tensor] = {}
        self.P_by_pos: Dict[int, torch.Tensor] = {}
        self.R_by_pos: Dict[int, torch.Tensor] = {}
        self.count_by_pos: Dict[int, int] = {}
        for p in positions:
            hk = f"H_pos_{p}"
            pk = f"P4_pos_{p}"
            rk = f"reward_pos_{p}"
            if hk not in arr.files or pk not in arr.files:
                continue
            H = np.asarray(arr[hk], dtype=np.float32)
            P = np.asarray(arr[pk], dtype=np.float32)
            if H.ndim != 2 or P.ndim != 2 or H.shape[0] == 0 or P.shape[0] != H.shape[0]:
                continue
            R = np.asarray(arr[rk], dtype=np.float32) if rk in arr.files else np.zeros((H.shape[0],), dtype=np.float32)
            Ht = torch.as_tensor(H, dtype=torch.float32, device=device)
            Pt = torch.as_tensor(P, dtype=torch.float32, device=device)
            Rt = torch.as_tensor(R, dtype=torch.float32, device=device)
            Ht = Ht / Ht.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            Pt = Pt / Pt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.H_by_pos[p] = Ht
            self.P_by_pos[p] = Pt
            self.R_by_pos[p] = Rt
            self.count_by_pos[p] = int(H.shape[0])
        self.global_H: Optional[torch.Tensor] = None
        self.global_P: Optional[torch.Tensor] = None
        self.global_R: Optional[torch.Tensor] = None
        if "global_H" in arr.files and "global_P4" in arr.files:
            gH = np.asarray(arr["global_H"], dtype=np.float32)
            gP = np.asarray(arr["global_P4"], dtype=np.float32)
            if gH.ndim == 2 and gP.ndim == 2 and gH.shape[0] > 0 and gP.shape[0] == gH.shape[0]:
                gR = np.asarray(arr["global_reward"], dtype=np.float32) if "global_reward" in arr.files else np.zeros((gH.shape[0],), dtype=np.float32)
                self.global_H = torch.as_tensor(gH, dtype=torch.float32, device=device)
                self.global_H = self.global_H / self.global_H.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                self.global_P = torch.as_tensor(gP, dtype=torch.float32, device=device)
                self.global_P = self.global_P / self.global_P.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                self.global_R = torch.as_tensor(gR, dtype=torch.float32, device=device)
        if not self.H_by_pos and self.global_H is None:
            raise ValueError(f"{path} loaded no usable site or global realism tokens")
        print(f"[EmbeddingRealismBank] loaded {len(self.H_by_pos)} site banks from {path}; global={self.global_H is not None}")

    @staticmethod
    def _extract_p4(P: torch.Tensor) -> torch.Tensor:
        if P.shape[-1] == 4:
            return P
        if P.shape[-1] >= 10:
            return P[..., [0, 7, 8, 9]]
        raise ValueError(f"P/P4 last dim must be 4 or >=10, got {P.shape}")

    def _site_score(
        self,
        h_live: torch.Tensor,
        p_live: torch.Tensor,
        bank_H: torch.Tensor,
        bank_P: torch.Tensor,
        bank_R: torch.Tensor,
        *,
        topk: int,
        temp: float,
        h_alpha: float,
        reward_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # h_live: [B,1280], p_live: [B,4]
        h = h_live.float() / h_live.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        p = p_live.float() / p_live.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sim_h = h @ bank_H.T
        sim_p = p @ bank_P.T
        combo = float(h_alpha) * sim_h + (1.0 - float(h_alpha)) * sim_p
        k = min(max(1, int(topk)), combo.shape[-1])
        vals, idx = torch.topk(combo, k=k, dim=-1)
        r = bank_R.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        logits = (vals + float(reward_scale) * r) / max(float(temp), 1e-6)
        w = torch.softmax(logits, dim=-1)
        combo_score = (w * vals).sum(dim=-1)
        h_score = (w * sim_h.gather(1, idx)).sum(dim=-1)
        p_score = (w * sim_p.gather(1, idx)).sum(dim=-1)
        reward_score = (w * r).sum(dim=-1)
        site_score = combo_score + float(reward_scale) * reward_score
        return site_score, combo_score, h_score, p_score

    def forward(
        self,
        H_live: torch.Tensor,
        P_live: torch.Tensor,
        *,
        mutable_mask: Optional[torch.Tensor] = None,
        cfg: Optional[EmbeddingRealismConfig] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = cfg or EmbeddingRealismConfig(bank_npz=self.path)
        P4 = self._extract_p4(P_live)
        if H_live.ndim != 3 or P4.ndim != 3:
            raise ValueError(f"Expected H[B,L,D] and P4[B,L,4], got H={H_live.shape}, P4={P4.shape}")
        B, L, _ = H_live.shape
        mprob = P4[..., 0].float().clamp(0.0, 1.0)
        if mutable_mask is not None:
            mm = mutable_mask.to(H_live.device).bool().view(1, L)
            mprob = torch.where(mm, mprob, torch.zeros_like(mprob))
        min_mp = float(cfg.min_mut_prob)
        total_score = H_live.new_zeros((B,), dtype=torch.float32)
        total_combo = H_live.new_zeros((B,), dtype=torch.float32)
        total_h = H_live.new_zeros((B,), dtype=torch.float32)
        total_p = H_live.new_zeros((B,), dtype=torch.float32)
        total_weight = H_live.new_zeros((B,), dtype=torch.float32)
        active_sites = H_live.new_zeros((B,), dtype=torch.float32)
        used_global = H_live.new_zeros((B,), dtype=torch.float32)
        for pos in range(min(L, self.length)):
            mp = mprob[:, pos]
            # Keep differentiability for small mutation probabilities, but avoid pure numerical noise.
            if torch.max(mp).detach().item() <= min_mp:
                continue
            if pos in self.H_by_pos:
                bank_H = self.H_by_pos[pos]
                bank_P = self.P_by_pos[pos]
                bank_R = self.R_by_pos[pos]
                site_multiplier = 1.0
            elif self.global_H is not None and float(cfg.fallback_global_weight) > 0:
                bank_H = self.global_H
                bank_P = self.global_P
                bank_R = self.global_R
                site_multiplier = float(cfg.fallback_global_weight)
                used_global = used_global + (mp > min_mp).float()
            else:
                continue
            site_score, combo_score, h_score, p_score = self._site_score(
                H_live[:, pos, :], P4[:, pos, :], bank_H, bank_P, bank_R,
                topk=int(cfg.topk), temp=float(cfg.temp), h_alpha=float(cfg.h_alpha), reward_scale=float(cfg.reward_scale),
            )
            w = mp * site_multiplier
            total_score = total_score + w * site_score
            total_combo = total_combo + w * combo_score
            total_h = total_h + w * h_score
            total_p = total_p + w * p_score
            total_weight = total_weight + w
            active_sites = active_sites + (mp > min_mp).float()
        denom = total_weight.clamp_min(1.0)
        score = total_score / denom
        combo = total_combo / denom
        hmean = total_h / denom
        pmean = total_p / denom
        return {
            "embedding_realism": score,
            "embedding_realism_combo": combo,
            "embedding_realism_saprot": hmean,
            "embedding_realism_p4": pmean,
            "embedding_realism_weight_sum": total_weight,
            "embedding_realism_active_sites": active_sites,
            "embedding_realism_used_global_sites": used_global,
        }
