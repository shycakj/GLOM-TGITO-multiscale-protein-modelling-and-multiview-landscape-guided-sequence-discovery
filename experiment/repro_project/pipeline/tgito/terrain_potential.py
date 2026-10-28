from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .terrain_memory import TerrainMemoryBank
from .miner_policy import V12_7_CONTEXT_FEATURE_NAMES, V12_8_CREDIBLE_FEATURE_NAMES, load_v12_standalone_miner


MINER_FEATURE_NAMES = V12_8_CREDIBLE_FEATURE_NAMES


@dataclass
class TerrainPotentialConfig:
    k_feature: int = 32
    k_xy: int = 48
    feature_temp: float = 0.15
    true_weight: float = 0.7
    pred_weight: float = 0.3
    local_z_scale: float = 1.0
    brightness_scale: float = 0.35
    topk_sources: int = 4
    min_source_support: int = 1
    # v12 released true-high prototype reward/gate.
    true_high_gate_npz: str = ""
    true_high_weight: float = 0.0
    true_high_temp: float = 0.12
    true_high_topk: int = 24
    source_gate_power: float = 1.0
    # v12.3/v12.7 learned mining policy. v12.6.2 supports both checkpoint schemas:
    #   legacy 17-dim terrain aggregate miner
    #   v12.7  23-dim context-neighbor ctx_mlp_k4/k8 miner
    miner_policy_pt: str = ""
    miner_weight: float = 0.0
    miner_logit_scale: float = 1.0
    # Context-miner live feature reconstruction.
    context_neighbor_k: int = 0       # 0 => use checkpoint neighbor_k if present
    context_top_sources: int = 4
    context_reward_scale: float = 0.12
    context_percentile_temp: float = 0.05
    # v13.1 cos0-top5 local gate. Keeps cos-sim realism/repair disabled; only changes terrain source weighting.
    # gate_mode:
    #   authority = original v12.9 behavior: top-k by authority.
    #   fixed     = use terrain_fixed_sources as fixed top-k, with source/local weights.
    #   rule      = dynamic top-k by local support + source quality, no risk penalty.
    #   risk      = dynamic top-k by rule score minus local pred-only risk. This is the final merged mode.
    terrain_gate_mode: str = "authority"
    terrain_fixed_sources: str = ""
    terrain_gate_temp: float = 0.20
    terrain_support_weight: float = 0.45
    terrain_risk_weight: float = 0.65
    terrain_pred_gap_weight: float = 0.20
    terrain_source_quality_power: float = 0.60


class TrueHighGateMemory:
    """Per-source released true-high prototypes."""
    def __init__(self, path: str | Path, device: str = "cuda"):
        self.path = str(path)
        arr = np.load(str(path), allow_pickle=True)
        if "v12_gate_sources" not in arr.files:
            raise ValueError(f"{path} lacks v12_gate_sources; expected output from train_v12_true_high_gate.py")
        self.sources = [str(x) for x in arr["v12_gate_sources"].tolist()]
        self.device = device
        self.rep: Dict[str, torch.Tensor] = {}
        self.xy: Dict[str, torch.Tensor] = {}
        self.reward: Dict[str, torch.Tensor] = {}
        self.weight: Dict[str, torch.Tensor] = {}
        self.rows: Dict[str, torch.Tensor] = {}
        for s in self.sources:
            rkey = f"v12_proto_rep__{s}"
            wkey = f"v12_source_weight__{s}"
            ykey = f"v12_proto_reward__{s}"
            if rkey not in arr.files or ykey not in arr.files or wkey not in arr.files:
                continue
            rep = np.asarray(arr[rkey], dtype=np.float32)
            if rep.ndim != 2 or rep.shape[0] == 0:
                continue
            rep_t = torch.as_tensor(rep, dtype=torch.float32, device=device)
            rep_t = rep_t / rep_t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.rep[s] = rep_t
            self.reward[s] = torch.as_tensor(np.asarray(arr[ykey], dtype=np.float32), dtype=torch.float32, device=device)
            self.weight[s] = torch.as_tensor(float(np.asarray(arr[wkey]).reshape(-1)[0]), dtype=torch.float32, device=device)
            xkey = f"v12_proto_xy__{s}"
            if xkey in arr.files:
                self.xy[s] = torch.as_tensor(np.asarray(arr[xkey], dtype=np.float32), dtype=torch.float32, device=device)
            row_key = f"v12_proto_rows__{s}"
            if row_key in arr.files:
                self.rows[s] = torch.as_tensor(np.asarray(arr[row_key], dtype=np.int64), dtype=torch.long, device=device)
        if not self.rep:
            raise ValueError(f"{path} loaded no usable v12 prototypes")
        print(f"[TrueHighGate] loaded {len(self.rep)} sources from {path}: {sorted(self.rep)}")

    def has_source(self, s: str) -> bool:
        return s in self.rep and s in self.reward and s in self.weight


class MiningPolicyMemory:
    """Frozen v12.7 standalone mining policy.

    v12.7 deliberately refuses older v12.3/v12.4/v12.5 checkpoints. Re-train
    the miner with train_v12_7_miner_sweep.py so generator and checkpoint schema
    are from the same version package.
    """
    def __init__(self, path: str | Path, device: str = "cuda"):
        self.loaded = load_v12_standalone_miner(path, device=device)
        self.path = self.loaded.path
        self.schema = str(getattr(self.loaded, "metrics", {}).get("schema", "")) or ("v12_8_credible" if len(self.loaded.feature_names) == len(V12_8_CREDIBLE_FEATURE_NAMES) else "v12_7_context")
        self.feature_names = self.loaded.feature_names
        self.feature_mean = self.loaded.feature_mean
        self.feature_std = self.loaded.feature_std
        self.neighbor_k = self.loaded.neighbor_k
        self.high_threshold = self.loaded.high_threshold
        self.sources = self.loaded.sources
        self.metrics = self.loaded.metrics
        self.device = device
        print(f"[MiningPolicy:v12.8] loaded {self.path}; features={len(self.feature_names)} neighbor_k={self.neighbor_k} metrics={self.metrics}")

    def score(self, features: torch.Tensor, logit_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.loaded.score(features, logit_scale=logit_scale)


class TerrainPotential(nn.Module):
    """Differentiable potential from live revised_slim features to fixed PPLU UMAP terrains."""
    def __init__(self, bank: TerrainMemoryBank, cfg: TerrainPotentialConfig | None = None, sources: Optional[List[str]] = None):
        super().__init__()
        self.bank = bank
        self.cfg = cfg or TerrainPotentialConfig()
        if sources is None or "all" in sources:
            self.sources = bank.available_sources
        else:
            self.sources = [s for s in sources if s in bank.sources]
        if not self.sources:
            raise ValueError("No valid sources selected for TerrainPotential")
        self.true_high_gate: Optional[TrueHighGateMemory] = None
        if str(self.cfg.true_high_gate_npz).strip():
            self.true_high_gate = TrueHighGateMemory(self.cfg.true_high_gate_npz, device=bank.device)
        self.miner_policy: Optional[MiningPolicyMemory] = None
        if str(self.cfg.miner_policy_pt).strip():
            self.miner_policy = MiningPolicyMemory(self.cfg.miner_policy_pt, device=bank.device)
        print(f"[TerrainPotential] active sources={self.sources}")

    def _true_high_bonus(self, name: str, x_norm: torch.Tensor):
        if self.true_high_gate is None or float(self.cfg.true_high_weight) <= 0 or not self.true_high_gate.has_source(name):
            z = x_norm.new_zeros((x_norm.shape[0],))
            return z, z
        proto = self.true_high_gate.rep[name]
        rewards = self.true_high_gate.reward[name]
        sim = x_norm @ proto.T
        k = min(max(1, int(self.cfg.true_high_topk)), sim.shape[-1])
        vals, idx = torch.topk(sim, k=k, dim=-1)
        w = torch.softmax(vals / max(float(self.cfg.true_high_temp), 1e-6), dim=-1)
        r = rewards.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        raw_bonus = (w * r).sum(dim=1)
        src_w = self.true_high_gate.weight[name].clamp_min(0)
        if float(self.cfg.source_gate_power) != 1.0:
            src_w = src_w.pow(float(self.cfg.source_gate_power))
        bonus = src_w * raw_bonus
        authority = (src_w * raw_bonus).detach()
        return bonus, authority

    def _one_source(self, name: str, live: torch.Tensor):
        mem = self.bank.sources[name]
        x = live.float()
        if x.ndim > 2:
            x = x.flatten(1)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sim = x @ mem.rep.T
        k = min(int(self.cfg.k_feature), sim.shape[-1])
        vals, idx = torch.topk(sim, k=k, dim=-1)
        w = torch.softmax(vals / max(float(self.cfg.feature_temp), 1e-6), dim=-1)
        zt = mem.local_z_true.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        zp = mem.local_z_pred.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        bt = mem.brightness_true.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        bp = mem.brightness_pred.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        xy = mem.xy.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], 2)
        xy_hat = (w.unsqueeze(-1) * xy).sum(dim=1)

        local_true = (w * torch.relu(zt)).sum(dim=1)
        local_pred = (w * torch.relu(zp)).sum(dim=1)
        bright_true = (w * bt).sum(dim=1)
        bright_pred = (w * bp).sum(dim=1)
        local = float(self.cfg.true_weight) * torch.relu(zt) + float(self.cfg.pred_weight) * torch.relu(zp)
        bright = float(self.cfg.true_weight) * bt + float(self.cfg.pred_weight) * bp
        base_pot = float(self.cfg.local_z_scale) * (w * local).sum(dim=1) + float(self.cfg.brightness_scale) * (w * bright).sum(dim=1)
        base_auth = (w * torch.relu(local)).sum(dim=1).detach()
        th_bonus, th_auth = self._true_high_bonus(name, x)
        pot = base_pot + float(self.cfg.true_high_weight) * th_bonus
        authority = base_auth + float(self.cfg.true_high_weight) * th_auth
        sim_top = vals[:, 0]
        sim_wmean = (w * vals).sum(dim=1)
        metrics = {
            "local_true": local_true,
            "local_pred": local_pred,
            "bright_true": bright_true,
            "bright_pred": bright_pred,
            "true_high_bonus": th_bonus,
            "sim_top": sim_top,
            "sim_wmean": sim_wmean,
            "base_pot": base_pot,
            "authority": authority,
            # v13.1 local-gate statistics; neutral if missing in NPZ.
            "gate_support": (w * mem.gate_support.index_select(0, idx.reshape(-1)).reshape(idx.shape)).sum(dim=1) if mem.gate_support is not None else torch.zeros_like(base_pot),
            "gate_risk": (w * mem.gate_risk.index_select(0, idx.reshape(-1)).reshape(idx.shape)).sum(dim=1) if mem.gate_risk is not None else torch.zeros_like(base_pot),
            "source_quality": torch.full_like(base_pot, float(getattr(mem, "source_quality", 1.0))),
        }
        return pot, authority, xy_hat, base_pot, th_bonus, metrics

    @staticmethod
    def _wmean(weights: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        return (weights * vals).sum(dim=1)

    @staticmethod
    def _masked_max(vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (vals * mask + (-1e6) * (1 - mask)).max(dim=1).values

    def _build_legacy_miner_features(self, terrain: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor,
                                     P: torch.Tensor, A: torch.Tensor, M: Dict[str, torch.Tensor]):
        w = weights
        active = mask.sum(dim=1)
        entropy = -(w * w.clamp_min(1e-8).log()).sum(dim=1)
        feats = {
            "terrain_agg": terrain,
            "authority_wmean": self._wmean(w, A),
            "authority_max": self._masked_max(A, mask),
            "local_true_wmean": self._wmean(w, M["local_true"]),
            "local_true_max": self._masked_max(M["local_true"], mask),
            "local_pred_wmean": self._wmean(w, M["local_pred"]),
            "local_pred_max": self._masked_max(M["local_pred"], mask),
            "bright_true_wmean": self._wmean(w, M["bright_true"]),
            "bright_true_max": self._masked_max(M["bright_true"], mask),
            "bright_pred_wmean": self._wmean(w, M["bright_pred"]),
            "bright_pred_max": self._masked_max(M["bright_pred"], mask),
            "true_high_bonus_wmean": self._wmean(w, M["true_high_bonus"]),
            "true_high_bonus_max": self._masked_max(M["true_high_bonus"], mask),
            "sim_top_wmean": self._wmean(w, M["sim_top"]),
            "sim_top_max": self._masked_max(M["sim_top"], mask),
            "active_source_count": active,
            "terrain_weight_entropy": entropy,
        }
        names = self.miner_policy.feature_names if self.miner_policy is not None else LEGACY_MINER_FEATURE_NAMES
        return torch.stack([feats[n] for n in names], dim=1), feats

    def _soft_pred_percentile(self, pred_self: torch.Tensor) -> torch.Tensor:
        # Smooth rank among terrain predicted brightness values. This replaces v12.7's numpy searchsorted
        # so the live feature remains differentiable with respect to predicted brightness.
        first_source = next((s for s in self.sources if s in self.bank.sources), None)
        if first_source is None:
            return torch.full_like(pred_self, 0.5)
        ref = self.bank.sources[first_source].brightness_pred.to(pred_self.device).float()
        if ref.numel() > 8192:
            # fixed deterministic subsample for speed, still enough to estimate rank
            idx = torch.linspace(0, ref.numel() - 1, steps=8192, device=ref.device).long()
            ref = ref.index_select(0, idx)
        temp = max(float(self.cfg.context_percentile_temp), 1e-4)
        return torch.sigmoid((pred_self[:, None] - ref[None, :]) / temp).mean(dim=1)

    def _ctx_top3_mean(self, values: torch.Tensor) -> torch.Tensor:
        k = min(3, values.shape[1])
        return torch.topk(values, k=k, dim=1).values.mean(dim=1)

    def _add_v12_8_realism_features(self, feats: Dict[str, torch.Tensor], model_outputs: Dict[str, torch.Tensor]) -> None:
        """Append live-computable embedding-realism features for v12.8 credible miners.

        These fields come from EmbeddingRealismBank, computed before TerrainPotential
        in TGITOGenerator. For v12.7 context miners they are harmless extras; for
        v12.8 credible miners they are required by the checkpoint feature schema.
        """
        ref = feats.get("pred_self", None)
        if ref is None:
            ref = next(iter(model_outputs.values())).new_zeros((next(iter(model_outputs.values())).shape[0],))
        def get(name: str) -> torch.Tensor:
            v = model_outputs.get(name, None)
            if v is None:
                return torch.zeros_like(ref)
            return v.reshape(-1).to(ref.device).float()
        er = get("embedding_realism")
        ec = get("embedding_realism_combo")
        eh = get("embedding_realism_saprot")
        ep = get("embedding_realism_p4")
        ew = get("embedding_realism_weight_sum")
        ea = get("embedding_realism_active_sites")
        eg = get("embedding_realism_used_global_sites")
        feats["embedding_realism"] = er
        feats["embedding_realism_combo"] = ec
        feats["embedding_realism_saprot"] = eh
        feats["embedding_realism_p4"] = ep
        feats["embedding_realism_weight_sum"] = ew
        feats["embedding_realism_active_sites"] = ea
        feats["embedding_realism_used_global_frac"] = eg / ea.clamp_min(1.0)
        feats["realism_x_pred_margin"] = er * feats.get("pred_margin", torch.zeros_like(ref))
        feats["realism_x_ctx_high_frac"] = er * feats.get("ctx_high_frac_w", torch.zeros_like(ref))

    def _build_context_miner_features(self, model_outputs: Dict[str, torch.Tensor], used: List[str], xy_hats: List[torch.Tensor]):
        if self.miner_policy is None:
            raise RuntimeError("_build_context_miner_features requires miner_policy")
        if "pred_brightness_live" not in model_outputs:
            raise ValueError(
                "v12.7 ctx_mlp miner requires model_outputs['pred_brightness_live']. "
                "Use v12.6.2 run_generate.py, which passes live predicted brightness into TerrainPotential."
            )
        pred_self = model_outputs["pred_brightness_live"].reshape(-1).float()
        B = pred_self.shape[0]
        threshold = float(getattr(self.miner_policy, "high_threshold", 4.2))
        neighbor_k = int(getattr(self.miner_policy, "neighbor_k", 8))
        if int(self.cfg.context_neighbor_k) > 0:
            neighbor_k = int(self.cfg.context_neighbor_k)
        top_sources = max(1, int(self.cfg.context_top_sources))
        reward_scale = max(float(self.cfg.context_reward_scale), 1e-6)

        per: Dict[str, List[torch.Tensor]] = {k: [] for k in [
            "auth", "true_mean", "true_max", "true_top3",
            "pred_mean", "pred_max", "pred_top3",
            "lt_mean", "lt_max", "lp_mean", "lp_max",
            "high_frac", "reward_mean", "reward_max",
            "dist_min", "dist_mean",
        ]}
        for s, xy_hat in zip(used, xy_hats):
            if s not in self.bank.sources:
                continue
            mem = self.bank.sources[s]
            xy = mem.xy.to(xy_hat.device).float()
            # B x N Euclidean distance in the fixed UMAP plane. Top-k indices are discrete, but
            # dist_min/dist_mean remain differentiable through xy_hat. This reproduces v12.7's
            # context-neighbor schema for live generated sequences.
            d = torch.cdist(xy_hat.float(), xy)
            k = min(max(1, neighbor_k), d.shape[1])
            dist, idx = torch.topk(d, k=k, dim=1, largest=False)
            flat = idx.reshape(-1)
            bt = mem.brightness_true.to(xy_hat.device).float().index_select(0, flat).reshape(B, k)
            bp = mem.brightness_pred.to(xy_hat.device).float().index_select(0, flat).reshape(B, k)
            lzt = torch.relu(mem.local_z_true.to(xy_hat.device).float().index_select(0, flat).reshape(B, k))
            lzp = torch.relu(mem.local_z_pred.to(xy_hat.device).float().index_select(0, flat).reshape(B, k))
            reward = torch.clamp(torch.relu(bt - threshold) / reward_scale, 0.0, 4.0)
            high_frac = (bt >= threshold).float().mean(dim=1)
            auth = 0.40 * lzt.max(dim=1).values + 0.20 * lzp.max(dim=1).values + 0.30 * high_frac + 0.10 * torch.clamp(reward.max(dim=1).values, 0, 4) / 4.0
            auth = auth.clamp_min(1e-6)
            vals = {
                "auth": auth,
                "true_mean": bt.mean(dim=1),
                "true_max": bt.max(dim=1).values,
                "true_top3": self._ctx_top3_mean(bt),
                "pred_mean": bp.mean(dim=1),
                "pred_max": bp.max(dim=1).values,
                "pred_top3": self._ctx_top3_mean(bp),
                "lt_mean": lzt.mean(dim=1),
                "lt_max": lzt.max(dim=1).values,
                "lp_mean": lzp.mean(dim=1),
                "lp_max": lzp.max(dim=1).values,
                "high_frac": high_frac,
                "reward_mean": reward.mean(dim=1),
                "reward_max": reward.max(dim=1).values,
                "dist_min": dist.min(dim=1).values,
                "dist_mean": dist.mean(dim=1),
            }
            for key, val in vals.items():
                per[key].append(val)

        if not per["auth"]:
            z = pred_self.new_zeros((B,))
            feats = {n: z for n in V12_7_CONTEXT_FEATURE_NAMES}
            feats["pred_self"] = pred_self
            feats["pred_margin"] = pred_self - threshold
            feats["pred_percentile"] = self._soft_pred_percentile(pred_self)
            self._add_v12_8_realism_features(feats, model_outputs)
            names = self.miner_policy.feature_names
            return torch.stack([feats[n] for n in names], dim=1), feats

        S = len(per["auth"])
        T = {k: torch.stack(v, dim=1) for k, v in per.items()}
        auth = T["auth"]
        topm = min(top_sources, S)
        _, top_idx = torch.topk(auth, k=topm, dim=1)
        src_mask = torch.zeros_like(auth).scatter_(1, top_idx, 1.0)
        w = auth.clamp_min(1e-6) * src_mask
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        def wmean(key: str) -> torch.Tensor:
            return (w * T[key]).sum(dim=1)
        def globmax(key: str) -> torch.Tensor:
            return T[key].max(dim=1).values
        high_support = ((T["high_frac"] >= 0.25) | (T["reward_max"] > 0)).float().sum(dim=1)
        ent = -(w * w.clamp_min(1e-8).log()).sum(dim=1)
        feats = {
            "pred_self": pred_self,
            "pred_margin": pred_self - threshold,
            "pred_percentile": self._soft_pred_percentile(pred_self),
            "ctx_true_mean_w": wmean("true_mean"),
            "ctx_true_max_w": wmean("true_max"),
            "ctx_true_top3_w": wmean("true_top3"),
            "ctx_pred_mean_w": wmean("pred_mean"),
            "ctx_pred_max_w": wmean("pred_max"),
            "ctx_pred_top3_w": wmean("pred_top3"),
            "ctx_local_true_mean_w": wmean("lt_mean"),
            "ctx_local_true_max_w": wmean("lt_max"),
            "ctx_local_pred_mean_w": wmean("lp_mean"),
            "ctx_local_pred_max_w": wmean("lp_max"),
            "ctx_high_frac_w": wmean("high_frac"),
            "ctx_reward_mean_w": wmean("reward_mean"),
            "ctx_reward_max_w": wmean("reward_max"),
            "ctx_dist_min_w": wmean("dist_min"),
            "ctx_dist_mean_w": wmean("dist_mean"),
            "ctx_true_max_global": globmax("true_max"),
            "ctx_high_frac_global": globmax("high_frac"),
            "ctx_reward_max_global": globmax("reward_max"),
            "support_source_count": high_support,
            "source_entropy": ent,
        }
        self._add_v12_8_realism_features(feats, model_outputs)
        names = self.miner_policy.feature_names
        return torch.stack([torch.nan_to_num(feats[n], nan=0.0, posinf=1e6, neginf=-1e6) for n in names], dim=1), feats

    def _build_miner_features(self, model_outputs: Dict[str, torch.Tensor], base_terrain: torch.Tensor,
                              weights: torch.Tensor, mask: torch.Tensor, P: torch.Tensor, A: torch.Tensor,
                              M: Dict[str, torch.Tensor], used: List[str], xy_hats: List[torch.Tensor]):
        if self.miner_policy is None:
            # No learned miner requested. Return a zero feature placeholder; caller will not score it.
            z = base_terrain.new_zeros((base_terrain.shape[0], len(MINER_FEATURE_NAMES)))
            return z, {}
        return self._build_context_miner_features(model_outputs, used, xy_hats)


    def _parse_fixed_sources(self, used: List[str]) -> List[str]:
        raw = str(getattr(self.cfg, "terrain_fixed_sources", "") or "")
        vals = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
        return [x for x in vals if x in used]

    def _aggregate_sources(self, P: torch.Tensor, gate: torch.Tensor, topk: int, mode_name: str,
                           used: List[str], fixed_sources: Optional[List[str]] = None):
        """Return weighted terrain, source weights, and selected mask for a gate score matrix."""
        B, S = P.shape
        topk = min(max(1, int(topk)), S)
        if fixed_sources:
            mask = torch.zeros_like(gate)
            idxs = [used.index(x) for x in fixed_sources if x in used]
            if idxs:
                idx = torch.as_tensor(idxs[:topk], dtype=torch.long, device=gate.device)
                mask[:, idx] = 1.0
            # If fewer than topk fixed sources exist, fill by gate score.
            fill = topk - int(mask[0].sum().item())
            if fill > 0:
                fill_gate = gate.masked_fill(mask.bool(), -1e6)
                _, extra = torch.topk(fill_gate, k=min(fill, S), dim=1)
                mask = mask.scatter(1, extra, 1.0)
        else:
            _, idx = torch.topk(gate, k=topk, dim=1)
            mask = torch.zeros_like(gate).scatter_(1, idx, 1.0)
        temp = max(float(getattr(self.cfg, "terrain_gate_temp", 0.20)), 1e-6)
        # Stable softmax inside the selected top-k. Gate may include penalties; selected inactive sources get zero.
        masked_gate = gate.masked_fill(mask <= 0, -1e6)
        weights = torch.softmax(masked_gate / temp, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        terrain = (weights * P).sum(dim=1)
        return terrain, weights, mask

    def _source_quality_tensor(self, used: List[str], ref: torch.Tensor) -> torch.Tensor:
        qs = []
        for s in used:
            try:
                q = float(getattr(self.bank.sources[s], "source_quality", 1.0))
            except Exception:
                q = 1.0
            qs.append(max(q, 1e-4))
        q = torch.tensor(qs, dtype=ref.dtype, device=ref.device)[None, :].expand(ref.shape[0], -1)
        return q

    def forward(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pots, auths, xy_hats = [], [], []
        metric_lists: Dict[str, List[torch.Tensor]] = {k: [] for k in ["local_true", "local_pred", "bright_true", "bright_pred", "true_high_bonus", "sim_top", "sim_wmean", "base_pot", "authority", "gate_support", "gate_risk", "source_quality"]}
        details = {}
        used = []
        for s in self.sources:
            if s not in model_outputs:
                continue
            p, a, xy, base_p, th_b, metrics = self._one_source(s, model_outputs[s])
            pots.append(p)
            auths.append(a)
            xy_hats.append(xy)
            used.append(s)
            details[f"terrain_{s}"] = p.detach()
            details[f"terrain_base_{s}"] = base_p.detach()
            details[f"true_high_bonus_{s}"] = th_b.detach()
            details[f"authority_{s}"] = a.detach()
            details[f"xy_{s}_x"] = xy[:, 0].detach()
            details[f"xy_{s}_y"] = xy[:, 1].detach()
            for k, v in metrics.items():
                metric_lists[k].append(v)
        if not pots:
            zero = next(iter(model_outputs.values())).new_zeros((next(iter(model_outputs.values())).shape[0],))
            return {"terrain": zero, "source_count": torch.tensor(0.0, device=zero.device), **details}
        P = torch.stack(pots, dim=1)
        A = torch.stack(auths, dim=1)
        M = {k: torch.stack(v, dim=1) for k, v in metric_lists.items() if v}
        topk = min(max(int(self.cfg.min_source_support), int(self.cfg.topk_sources)), P.shape[1])
        source_quality = M.get("source_quality", self._source_quality_tensor(used, A)).clamp_min(1e-4)
        support = M.get("gate_support", torch.zeros_like(A)).clamp(0.0, 4.0)
        risk = M.get("gate_risk", torch.zeros_like(A)).clamp(0.0, 4.0)
        pred_gap = torch.relu(M.get("local_pred", torch.zeros_like(A)) - M.get("local_true", torch.zeros_like(A)))
        prior = source_quality.pow(float(getattr(self.cfg, "terrain_source_quality_power", 0.60)))
        authority_gate = A.clamp_min(1e-6)
        rule_gate = authority_gate * prior * (1.0 + float(getattr(self.cfg, "terrain_support_weight", 0.45)) * support)
        risk_gate = rule_gate - float(getattr(self.cfg, "terrain_risk_weight", 0.65)) * risk - float(getattr(self.cfg, "terrain_pred_gap_weight", 0.20)) * pred_gap

        fixed_sources = self._parse_fixed_sources(used)
        fixed_probe, fixed_w, fixed_mask = self._aggregate_sources(P, rule_gate, topk, "fixed", used, fixed_sources=fixed_sources)
        rule_probe, rule_w, rule_mask = self._aggregate_sources(P, rule_gate, topk, "rule", used)
        risk_probe, risk_w, risk_mask = self._aggregate_sources(P, risk_gate, topk, "risk", used)
        auth_probe, auth_w, auth_mask = self._aggregate_sources(P, authority_gate, topk, "authority", used)

        mode = str(getattr(self.cfg, "terrain_gate_mode", "authority")).lower().strip()
        if mode == "fixed":
            base_terrain, weights, mask = fixed_probe, fixed_w, fixed_mask
        elif mode == "rule":
            base_terrain, weights, mask = rule_probe, rule_w, rule_mask
        elif mode == "risk":
            base_terrain, weights, mask = risk_probe, risk_w, risk_mask
        else:
            base_terrain, weights, mask = auth_probe, auth_w, auth_mask
        terrain = base_terrain

        details["terrain_fixed_top5_probe"] = fixed_probe.detach()
        details["terrain_rule_top5_probe"] = rule_probe.detach()
        details["terrain_risk_top5_probe"] = risk_probe.detach()
        details["terrain_authority_top5_probe"] = auth_probe.detach()
        details["terrain_rule_minus_fixed_probe"] = (rule_probe - fixed_probe).detach()
        details["terrain_risk_minus_rule_probe"] = (risk_probe - rule_probe).detach()
        details["terrain_selected_support_wmean"] = (weights * support).sum(dim=1).detach()
        details["terrain_selected_risk_wmean"] = (weights * risk).sum(dim=1).detach()
        details["terrain_selected_pred_gap_wmean"] = (weights * pred_gap).sum(dim=1).detach()
        details["terrain_weight_entropy"] = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1).detach()
        details["active_source_count"] = mask.sum(dim=1).detach()
        for j, s in enumerate(used):
            details[f"selected_weight_{s}"] = weights[:, j].detach()
            details[f"rule_weight_{s}"] = rule_w[:, j].detach()
            details[f"risk_weight_{s}"] = risk_w[:, j].detach()
            details[f"gate_support_{s}"] = support[:, j].detach()
            details[f"gate_risk_{s}"] = risk[:, j].detach()
            details[f"source_quality_{s}"] = source_quality[:, j].detach()
        if self.miner_policy is not None and float(self.cfg.miner_weight) > 0:
            miner_features, named_feats = self._build_miner_features(model_outputs, base_terrain, weights, mask, P, A, M, used, xy_hats)
            miner_prob, miner_logit = self.miner_policy.score(miner_features, logit_scale=float(self.cfg.miner_logit_scale))
            terrain = terrain + float(self.cfg.miner_weight) * miner_prob
            details["miner_score"] = miner_prob.detach()
            details["miner_logit"] = miner_logit.detach()
            details["miner_schema_v12_8_or_v12_7"] = torch.ones_like(miner_prob).detach()
            for name, tensor in named_feats.items():
                try:
                    details[f"miner_feat_{name}"] = tensor.detach()
                except Exception:
                    pass
        return {"terrain": terrain, "source_count": mask.sum(dim=1).detach(), **details}
