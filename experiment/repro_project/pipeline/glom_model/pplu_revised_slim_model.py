"""ppluGFP revised_slim brightness prediction model — standalone."""
import torch
import torch.nn as nn

# ── ppluGFP-specific constants ──
PPLU_WT = "PAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA"
L = 221  # ppluGFP sequence length

# Top 15 key residues (Ridge analysis, 1-indexed)
KEY_SITES_1B = [24, 33, 35, 36, 52, 53, 55, 56, 57, 58, 59, 62, 84, 86, 100]

# Top 9 region windows (Ridge sliding-window analysis)
REGIONS_1B = {
    "win_53_62":   [53, 62],
    "win_55_64":   [55, 64],
    "win_51_60":   [51, 60],
    "win_207_216": [207, 216],
    "win_203_212": [203, 212],
    "win_77_86":   [77, 86],
    "win_27_36":   [27, 36],
    "win_165_174": [165, 174],
    "win_97_106":  [97, 106],
}
RIDX = [list(range(v[0] - 1, v[1])) for v in REGIONS_1B.values()]


class MaskedAttentionPool(nn.Module):
    """Masked learnable attention pool."""
    def __init__(self, dim):
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


class PpluRevisedSlimBackbone(nn.Module):
    """Multi-source backbone adapted for ppluGFP (221 AA)."""

    def __init__(self, D=1280, pos_dim=4, hid=256, use_message=True):
        super().__init__()
        self.use_message = bool(use_message)

        # Global branch
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

        # Region branch (ppluGFP windows)
        self.region_res_proj = nn.Sequential(
            nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, hid), nn.GELU(),
        )
        self.region_attn_pool = MaskedAttentionPool(hid)
        self.reg_compress = nn.Sequential(
            nn.Linear(hid * 2, 512), nn.GELU(), nn.Dropout(0.12),
            nn.LayerNorm(512), nn.Linear(512, hid),
        )

        # Key-site branch (ppluGFP top 15)
        self.key_res_proj = nn.Sequential(
            nn.Linear(D, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(256, hid), nn.GELU(),
        )

        # Cross-attention
        self.xattn_k = nn.MultiheadAttention(hid, 4, batch_first=True)
        self.xattn_r = nn.MultiheadAttention(hid, 4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hid, 512), nn.GELU(), nn.Dropout(0.15), nn.Linear(512, hid),
        )
        self.ln_k1 = nn.LayerNorm(hid); self.ln_k2 = nn.LayerNorm(hid)
        self.ln_r1 = nn.LayerNorm(hid); self.ln_r2 = nn.LayerNorm(hid)

        self.k_compress = nn.Sequential(
            nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid),
        )
        self.r_compress = nn.Sequential(
            nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid),
        )

    def _pool3(self, x):
        return torch.cat([x.mean(dim=1), x.max(dim=1)[0], self.seq_attn_pool(x)], dim=-1)

    def _regions(self, Hr, P4):
        mut_mask = P4[..., 0] > 0.5
        outs = []
        for idxs in RIDX:
            idx = torch.as_tensor(idxs, device=Hr.device, dtype=torch.long)
            r = Hr.index_select(1, idx)
            rm = mut_mask.index_select(1, idx)
            outs.append(torch.cat([
                self.region_attn_pool(r),
                self.region_attn_pool(r, mask=rm),
            ], dim=-1))
        return self.reg_compress(torch.stack(outs, dim=1))

    def forward(self, H, P4):
        # G: global path
        H_global = self.res_proj(H)
        Hf = self.fuse(torch.cat([H_global, self.pos_proj(P4)], dim=-1))
        G = self.glob_compress(self._pool3(Hf))

        # Rf: region path
        Hr = self.region_res_proj(H)
        R = self._regions(Hr, P4)

        # Kf: key-site path
        Hk = self.key_res_proj(H)
        K = Hk[:, [p - 1 for p in KEY_SITES_1B]]

        # Cross-attention
        if self.use_message:
            K2a, _ = self.xattn_k(K, R, R, need_weights=False)
            R2a, _ = self.xattn_r(R, K, K, need_weights=False)
        else:
            K2a = torch.zeros_like(K); R2a = torch.zeros_like(R)

        K2 = self.ln_k1(K + K2a); K2 = self.ln_k2(K2 + self.ffn(K2))
        R2 = self.ln_r1(R + R2a); R2 = self.ln_r2(R2 + self.ffn(R2))

        return {
            "G": G,
            "Rf": self.r_compress(self._pool3(R2)),
            "Kf": self.k_compress(self._pool3(K2)),
        }


class PpluRevisedSlimModel(nn.Module):
    """ppluGFP brightness predictor.

    Input:
        H  [B, 221, 1280]   pred3Di→SaProt embeddings
        P  [B, 221, 10]     position features (uses cols [0,7,8,9])
    Output:
        y_hat  [B]          predicted log-brightness
    """
    def __init__(self, D=1280, hid=256, use_message=True):
        super().__init__()
        self.backbone = PpluRevisedSlimBackbone(D=D, pos_dim=4, hid=hid, use_message=use_message)
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


# ── Usage ──
if __name__ == "__main__":
    # Load trained checkpoint
    ckpt = torch.load("best_model_pplu_31k.pt", map_location="cpu", weights_only=False)
    model = PpluRevisedSlimModel(D=1280, hid=256)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Predict
    H = torch.randn(2, 221, 1280)  # SaProt embeddings
    P = torch.randn(2, 221, 10)    # position features
    with torch.no_grad():
        y_norm = model(H, P)
        brightness = y_norm * ckpt["y_std"] + ckpt["y_mean"]
    print(f"Predicted: {brightness.tolist()}")
