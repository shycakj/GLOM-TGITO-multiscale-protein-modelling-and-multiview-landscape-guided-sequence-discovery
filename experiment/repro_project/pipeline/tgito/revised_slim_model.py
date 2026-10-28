"""ppluGFP revised_slim brightness predictor with intermediate outputs.

This module is PPLU-specific. It uses:
  H [B,221,1280] pred3Di→SaProt embeddings
  P [B,221,10]   position features; the model uses P[:, :, [0,7,8,9]]

The constants match the user's ppluGFP model: 221 AA, PPLU key residues, and
PPLU ridge/sliding-window regions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pplu_config import KEY_SITES_1B, KEY_SITES_0B, REGIONS_1B, REGION_NAMES, RIDX, PPLU_WT, PPLU_LENGTH

L = PPLU_LENGTH


class MaskedAttentionPool(nn.Module):
    """Masked learnable attention pool."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None, weights=None):
        b, l, _ = x.shape
        logits = self.attn(x).squeeze(-1).float()
        valid = torch.ones((b, l), dtype=torch.bool, device=x.device) if mask is None else mask.bool().to(x.device)
        if weights is not None:
            w = weights.float().to(x.device)
            valid = valid & (w > 1e-8)
            logits = logits + torch.log(w.clamp_min(1e-8))
        has_valid = valid.any(dim=1, keepdim=True)
        attn = torch.softmax(logits.masked_fill(~valid, -1e9), dim=1)
        attn = attn * valid.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        out = torch.sum(x * attn.to(x.dtype).unsqueeze(-1), dim=1)
        return torch.where(has_valid, out, torch.zeros_like(out))


class RevisedSlimBackbone(nn.Module):
    """Multi-source backbone adapted for ppluGFP (221 AA)."""
    def __init__(self, D=1280, pos_dim=4, hid=256, use_message=True):
        super().__init__()
        self.use_message = bool(use_message)

        self.res_proj = nn.Sequential(
            nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, hid), nn.GELU(),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(pos_dim, 24), nn.GELU(), nn.Linear(24, 24), nn.GELU(),
        )
        self.fuse = nn.Sequential(nn.Linear(hid + 24, hid), nn.GELU(), nn.LayerNorm(hid))
        self.seq_attn_pool = MaskedAttentionPool(hid)
        self.glob_compress = nn.Sequential(
            nn.Linear(hid * 3, 768), nn.GELU(), nn.Dropout(0.2),
            nn.LayerNorm(768), nn.Linear(768, hid),
        )

        self.region_res_proj = nn.Sequential(
            nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, hid), nn.GELU(),
        )
        self.region_attn_pool = MaskedAttentionPool(hid)
        self.reg_compress = nn.Sequential(
            nn.Linear(hid * 2, 512), nn.GELU(), nn.Dropout(0.12),
            nn.LayerNorm(512), nn.Linear(512, hid),
        )

        self.key_res_proj = nn.Sequential(
            nn.Linear(D, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(256, hid), nn.GELU(),
        )

        self.xattn_k = nn.MultiheadAttention(hid, 4, batch_first=True)
        self.xattn_r = nn.MultiheadAttention(hid, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hid, 512), nn.GELU(), nn.Dropout(0.15), nn.Linear(512, hid))
        self.ln_k1 = nn.LayerNorm(hid); self.ln_k2 = nn.LayerNorm(hid)
        self.ln_r1 = nn.LayerNorm(hid); self.ln_r2 = nn.LayerNorm(hid)

        self.k_compress = nn.Sequential(nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid))
        self.r_compress = nn.Sequential(nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid))

    def _pool3(self, x):
        return torch.cat([x.mean(dim=1), x.max(dim=1)[0], self.seq_attn_pool(x)], dim=-1)

    def _pool3_parts(self, x):
        return x.mean(dim=1), x.max(dim=1)[0], self.seq_attn_pool(x)

    def _regions_tokens_and_raw(self, Hr, P4):
        mut_mask = P4[..., 0] > 0.5
        raw512 = []
        for idxs in RIDX:
            idx = torch.as_tensor(idxs, device=Hr.device, dtype=torch.long)
            r = Hr.index_select(1, idx)
            rm = mut_mask.index_select(1, idx)
            raw512.append(torch.cat([self.region_attn_pool(r), self.region_attn_pool(r, mask=rm)], dim=-1))
        raw = torch.stack(raw512, dim=1)
        return self.reg_compress(raw), raw

    def forward_intermediates(self, H, P4):
        feats = {}
        H_global = self.res_proj(H)
        Pproj = self.pos_proj(P4)
        Hf = self.fuse(torch.cat([H_global, Pproj], dim=-1))
        g_mean, g_max, g_attn = self._pool3_parts(Hf)
        feats["global_mean_pool"] = g_mean
        feats["global_max_pool"] = g_max
        feats["global_attn_pool"] = g_attn
        feats["pos_proj_flat"] = Pproj.flatten(1)
        G = self.glob_compress(torch.cat([g_mean, g_max, g_attn], dim=-1))

        Hr = self.region_res_proj(H)
        R, raw_region = self._regions_tokens_and_raw(Hr, P4)
        for j, name in enumerate(REGION_NAMES):
            feats[f"region_{name}_all_mut512"] = raw_region[:, j, :]

        Hk = self.key_res_proj(H)
        K = Hk[:, KEY_SITES_0B]
        feats["keysite_K_flat"] = K.flatten(1)

        if self.use_message:
            K2a, _ = self.xattn_k(K, R, R, need_weights=False)
            R2a, _ = self.xattn_r(R, K, K, need_weights=False)
        else:
            K2a = torch.zeros_like(K); R2a = torch.zeros_like(R)
        K2 = self.ln_k1(K + K2a); K2 = self.ln_k2(K2 + self.ffn(K2))
        R2 = self.ln_r1(R + R2a); R2 = self.ln_r2(R2 + self.ffn(R2))

        r_mean, r_max, r_attn = self._pool3_parts(R2)
        feats["region9_mean_pool"] = r_mean
        feats["region9_max_pool"] = r_max
        feats["region9_attn_pool"] = r_attn

        Rf = self.r_compress(torch.cat([r_mean, r_max, r_attn], dim=-1))
        k_mean, k_max, k_attn = self._pool3_parts(K2)
        Kf = self.k_compress(torch.cat([k_mean, k_max, k_attn], dim=-1))
        feats["G"] = G
        feats["Rf"] = Rf
        feats["Kf"] = Kf
        return {"G": G, "Rf": Rf, "Kf": Kf}, feats

    def forward(self, H, P4):
        b, _ = self.forward_intermediates(H, P4)
        return b


class RevisedSlimModel(nn.Module):
    """ppluGFP brightness predictor.

    Input:
        H  [B, 221, 1280]
        P  [B, 221, 10], uses columns [0,7,8,9]
    Output:
        y_hat [B] normalized or raw depending on checkpoint stats.
    """
    def __init__(self, D=1280, hid=256, use_message=True):
        super().__init__()
        self.backbone = RevisedSlimBackbone(D=D, pos_dim=4, hid=hid, use_message=use_message)
        self.narrow = nn.Sequential(
            nn.Linear(hid * 3, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, H, P):
        P4 = P[:, :, [0, 7, 8, 9]]
        b = self.backbone(H, P4)
        return self.narrow(torch.cat([b["G"], b["Rf"], b["Kf"]], dim=-1)).squeeze(-1)

    def forward_all(self, H, P):
        P4 = P[:, :, [0, 7, 8, 9]]
        b, feats = self.backbone.forward_intermediates(H, P4)
        hcat = torch.cat([b["G"], b["Rf"], b["Kf"]], dim=-1)
        x = hcat
        h512 = h256 = h128 = h64 = h32 = None
        linear_seen = 0
        # Capture hidden layers by width; keep robust to same sequential architecture.
        for layer in self.narrow:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                linear_seen += 1
                if layer.out_features == 512: h512 = x
                elif layer.out_features == 256: h256 = x
                elif layer.out_features == 128: h128 = x
                elif layer.out_features == 64: h64 = x
                elif layer.out_features == 32: h32 = x
        y = x.squeeze(-1)
        feats["z768"] = hcat
        if h512 is not None: feats["h512"] = h512
        if h256 is not None: feats["h256"] = h256
        if h128 is not None: feats["h128"] = h128
        if h64 is not None: feats["h64"] = h64
        feats["h32"] = h32 if h32 is not None else hcat
        return {"y_norm": y, **feats}


# Backward-compatible aliases for checkpoint loading and scripts.
PpluRevisedSlimBackbone = RevisedSlimBackbone
PpluRevisedSlimModel = RevisedSlimModel
