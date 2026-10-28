"""
revised_slim — Full Pipeline (Model + Data + Train + Eval + Inference)

Ablation-guided GFP brightness prediction model.
All components in one file: architecture, dataset, training, evaluation, inference.

Requirements:
  pip install torch numpy pandas scipy scikit-learn tqdm

Data files needed (local or mount paths):
  master_20000.csv            — 19925 variant metadata with 'brightness' column
  split_indices.npz            — {'train_idx','val_idx','test_idx'} arrays
  pred3di_saprot_residue_embeddings.npy  — [19925, 237, 1280] float32
  pos_feats.npy                — [19925, 237, 10] float32

Usage:
  # Train from scratch
  python revised_slim_full_pipeline.py train \\
      --data-dir ./data --out-dir ./output --epochs 120 --batch 32

  # Resume / finetune from checkpoint
  python revised_slim_full_pipeline.py train \\
      --data-dir ./data --out-dir ./output --epochs 120 --resume ./weights/best_model.pt

  # Eval only (load checkpoint)
  python revised_slim_full_pipeline.py eval \\
      --data-dir ./data --checkpoint ./weights/best_model.pt

  # Predict single variant (from H/P arrays)
  python revised_slim_full_pipeline.py predict \\
      --checkpoint ./weights/best_model.pt --H ./emb.npy --P ./pos.npy
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

KEY_SITES_1B = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 93, 109, 110, 111, 120]

REGIONS_1B = {
    "beta2":  [17, 27],   "beta5":  [53, 63],
    "beta7":  [72, 82],   "beta8":  [83, 91],
    "beta9":  [92, 102],  "beta11": [115, 129],
    "beta13": [158, 171], "beta17": [212, 223],
    "CRO":    [62, 70],
}
RIDX = [list(range(v[0] - 1, v[1])) for v in REGIONS_1B.values()]


# ── Model ──────────────────────────────────────────────────────────────────────

class MaskedAttentionPool(nn.Module):
    """Masked learnable attention pool over a variable-length set."""
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


class RevisedSlimBackbone(nn.Module):
    """Ablation-guided multi-source backbone.

    Sources:
      G  — global SaProt pool (all 237 positions) + P4 fused
      Rf — 9 region tokens with mutation-aware pooling (all + mut only)
      Kf — 15 key-site tokens with INDEPENDENT key_res_proj (NOT shared He)

    Key changes from old_full:
      - K uses independent key_res_proj from raw SaProt H
      - P keeps only 4 columns: [0]=mutation_flag, [7,8,9]=mutant_AA_class_bits
      - Region pooling: all-region + mutation only (no near-mutation pool)
      - Mf (mutation MLP) branch removed
    """

    def __init__(self, D=1280, pos_dim=4, hid=256, use_message=True):
        super().__init__()
        self.use_message = bool(use_message)

        # ── Global branch ──
        self.res_proj = nn.Sequential(
            nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, hid), nn.GELU(),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(pos_dim, 24), nn.GELU(), nn.Linear(24, 24), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hid + 24, hid), nn.GELU(), nn.LayerNorm(hid),
        )
        self.seq_attn_pool = MaskedAttentionPool(hid)
        self.glob_compress = nn.Sequential(
            nn.Linear(hid * 3, 768), nn.GELU(), nn.Dropout(0.2),
            nn.LayerNorm(768), nn.Linear(768, hid),
        )

        # ── Region branch (independent projection) ──
        self.region_res_proj = nn.Sequential(
            nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, hid), nn.GELU(),
        )
        self.region_attn_pool = MaskedAttentionPool(hid)
        self.reg_compress = nn.Sequential(
            nn.Linear(hid * 2, 512), nn.GELU(), nn.Dropout(0.12),
            nn.LayerNorm(512), nn.Linear(512, hid),
        )

        # ── Key-site branch (INDEPENDENT projection — key ablation finding) ──
        self.key_res_proj = nn.Sequential(
            nn.Linear(D, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(256, hid), nn.GELU(),
        )

        # ── Cross-attention K↔R (optional, kept for safety) ──
        self.xattn_k = nn.MultiheadAttention(hid, 4, batch_first=True)
        self.xattn_r = nn.MultiheadAttention(hid, 4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hid, 512), nn.GELU(), nn.Dropout(0.15), nn.Linear(512, hid),
        )
        self.ln_k1 = nn.LayerNorm(hid)
        self.ln_k2 = nn.LayerNorm(hid)
        self.ln_r1 = nn.LayerNorm(hid)
        self.ln_r2 = nn.LayerNorm(hid)

        # ── Output compressors ──
        self.k_compress = nn.Sequential(
            nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid),
        )
        self.r_compress = nn.Sequential(
            nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid),
        )

    def _pool3(self, x):
        return torch.cat([x.mean(dim=1), x.max(dim=1)[0], self.seq_attn_pool(x)], dim=-1)

    def _regions(self, Hr, P4):
        """9-region pooling: all-residue + mutation-only pools per region."""
        mut_mask = P4[..., 0] > 0.5
        outs = []
        for idxs in RIDX:
            idx = torch.as_tensor(idxs, device=Hr.device, dtype=torch.long)
            r = Hr.index_select(1, idx)
            rm = mut_mask.index_select(1, idx)
            outs.append(torch.cat([
                self.region_attn_pool(r),           # all residues
                self.region_attn_pool(r, mask=rm),  # mutation positions only
            ], dim=-1))
        return self.reg_compress(torch.stack(outs, dim=1))

    def forward(self, H, P4):
        # G: global path
        H_global = self.res_proj(H)
        Hf = self.fuse(torch.cat([H_global, self.pos_proj(P4)], dim=-1))
        G = self.glob_compress(self._pool3(Hf))

        # Rf: region path (independent region_res_proj)
        Hr = self.region_res_proj(H)
        R = self._regions(Hr, P4)

        # Kf: key-site path (independent key_res_proj on raw H)
        Hk = self.key_res_proj(H)
        K = Hk[:, [p - 1 for p in KEY_SITES_1B]]

        # Cross-attention
        if self.use_message:
            K2a, _ = self.xattn_k(K, R, R, need_weights=False)
            R2a, _ = self.xattn_r(R, K, K, need_weights=False)
        else:
            K2a = torch.zeros_like(K)
            R2a = torch.zeros_like(R)

        K2 = self.ln_k1(K + K2a)
        K2 = self.ln_k2(K2 + self.ffn(K2))
        R2 = self.ln_r1(R + R2a)
        R2 = self.ln_r2(R2 + self.ffn(R2))

        return {
            "G": G,
            "Rf": self.r_compress(self._pool3(R2)),
            "Kf": self.k_compress(self._pool3(K2)),
        }


class RevisedSlimModel(nn.Module):
    """Top-level brightness predictor.

    Input:
      H [B, 237, D]  — SaProt pred3Di embeddings (D=1280)
      P [B, 237, 10] — position features (only cols [0,7,8,9] used)

    Output:
      y_hat [B]  — predicted log-brightness
    """
    def __init__(self, D=1280, hid=256, use_message=True):
        super().__init__()
        self.backbone = RevisedSlimBackbone(D=D, pos_dim=4, hid=hid, use_message=use_message)
        self.narrow = nn.Sequential(
            nn.Linear(hid * 3, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, 32),   nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, H, P):
        P4 = P[:, :, [0, 7, 8, 9]]
        b = self.backbone(H, P4)
        return self.narrow(torch.cat([b["G"], b["Rf"], b["Kf"]], dim=-1)).squeeze(-1)


# ── Dataset ────────────────────────────────────────────────────────────────────

class BrightDS(Dataset):
    """Brightness dataset from pre-extracted embeddings."""
    def __init__(self, H, P, y, idx):
        self.H, self.P, self.y = H, P, y
        self.idx = np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        idx = int(self.idx[i])
        return {
            "idx": idx,
            "H": torch.from_numpy(np.asarray(self.H[idx], dtype=np.float32)),
            "P": torch.from_numpy(np.asarray(self.P[idx], dtype=np.float32)),
            "y": torch.tensor(float(self.y[idx]), dtype=torch.float32),
        }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(yt, yp):
    """Compute all regression and ranking metrics."""
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    n = len(yt)
    order_true = np.argsort(-yt)
    order_pred = np.argsort(-yp)
    out = {
        "n": int(n),
        "spearman": float(spearmanr(yt, yp)[0]),
        "pearson":  float(pearsonr(yt, yp)[0]),
        "rmse":     float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mae":      float(mean_absolute_error(yt, yp)),
        "r2":       float(r2_score(yt, yp)) if n > 1 else float("nan"),
    }
    for k in [5, 10, 20, 50, 100]:
        kk = min(k, n)
        out[f"top{k}"] = int(len(set(order_true[:kk]) & set(order_pred[:kk])))
    for pct in [1, 5]:
        kk = max(1, int(n * pct / 100))
        out[f"top{pct}pct"] = int(len(set(order_true[:kk]) & set(order_pred[:kk])))
    return out


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, y_mean, y_std, device="cuda"):
    """Run inference and return (indices, true_y, pred_y)."""
    model.eval()
    idxs, ys, ps = [], [], []
    for batch in loader:
        H = batch["H"].to(device)
        P = batch["P"].to(device)
        pred = model(H, P).detach().cpu().numpy() * y_std + y_mean
        idxs.extend(batch["idx"].numpy().tolist())
        ys.extend(batch["y"].numpy().tolist())
        ps.extend(pred.tolist())
    return np.asarray(idxs), np.asarray(ys), np.asarray(ps)


# ── Training ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    data_dir, out_dir,
    epochs=120, batch_size=32, lr=5e-5, seed=0,
    use_message=True, resume=None, device="cuda",
):
    """Train revised_slim from scratch (or resume from checkpoint)."""
    set_seed(seed)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    df = pd.read_csv(data_dir / "master_20000.csv").reset_index(drop=True)
    y = df["brightness"].to_numpy(dtype=np.float32)
    split = np.load(data_dir / "split_indices.npz")
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

    # Leakage checks
    s_tr, s_va, s_te = set(map(int, tr)), set(map(int, va)), set(map(int, te))
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        raise RuntimeError("Index leakage: train/val/test indices not disjoint")
    if len(s_tr | s_va | s_te) != len(df):
        raise RuntimeError("Split indices do not cover all rows")
    if "seq_hash" in df.columns:
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = set(df.iloc[list({"train": s_tr, "val": s_va, "test": s_te}[a])]["seq_hash"]) & \
                       set(df.iloc[list({"train": s_tr, "val": s_va, "test": s_te}[b])]["seq_hash"])
            if overlap:
                raise RuntimeError(f"Sequence leakage: {a}/{b} overlap={len(overlap)}")

    H = np.load(data_dir / "pred3di_saprot_residue_embeddings.npy", mmap_mode="r")
    P = np.load(data_dir / "pos_feats.npy", mmap_mode="r")
    if H.shape[0] != len(df) or P.shape[0] != len(df):
        raise RuntimeError(f"Row mismatch: H={H.shape[0]}, P={P.shape[0]}, df={len(df)}")
    if H.shape[:2] != P.shape[:2]:
        raise RuntimeError(f"Length mismatch: H={H.shape[1]}, P={P.shape[1]}")

    y_mean, y_std = float(y[tr].mean()), float(y[tr].std() + 1e-8)
    D = int(H.shape[-1])

    # ── Dataloaders ──
    train_loader = DataLoader(BrightDS(H, P, y, tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(BrightDS(H, P, y, va), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(BrightDS(H, P, y, te), batch_size=batch_size, shuffle=False)

    # ── Model ──
    model = RevisedSlimModel(D=D, hid=256, use_message=use_message).to(device)
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[resume] loaded from {resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)
    best = {"epoch": -1, "val_spearman": -1e9, "state": None}
    history = []

    # ── Training loop ──
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"ep {ep:3d}/{epochs}", leave=False):
            H_b = batch["H"].to(device)
            P_b = batch["P"].to(device)
            y_b = ((batch["y"].to(device) - y_mean) / y_std).float()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(H_b, P_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(y_b)
            n += len(y_b)

        _, yv, pv = predict(model, val_loader, y_mean, y_std, device)
        vm = compute_metrics(yv, pv)
        row = {"epoch": ep, "train_loss": total / max(n, 1), **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        if vm["spearman"] > best["val_spearman"]:
            best = {
                "epoch": int(ep),
                "val_spearman": float(vm["spearman"]),
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            torch.save(
                {"model": best["state"], "y_mean": y_mean, "y_std": y_std,
                 "best": {k: v for k, v in best.items() if k != "state"},
                 "use_message": use_message, "hid": 256, "D": D},
                out_dir / "best_model.pt",
            )

        if ep % max(1, epochs // 6) == 0:
            print(f"[ep {ep:3d}] val_sp={vm['spearman']:.4f} val_rmse={vm['rmse']:.4f}")

    # ── Final evaluation ──
    model.load_state_dict(best["state"])
    metrics_out = {
        "best": {k: v for k, v in best.items() if k != "state"},
        "epochs": epochs, "seed": seed, "use_message": use_message,
    }
    pred_rows = []
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        idxs, yy, pp = predict(model, loader, y_mean, y_std, device)
        metrics_out[name] = compute_metrics(yy, pp)
        for idx, yt, yp in zip(idxs, yy, pp):
            pred_rows.append({
                "row_index": int(idx), "split": name,
                "id": df.loc[int(idx), "id"],
                "seq_id": df.loc[int(idx), "seq_id"],
                "brightness_true": float(yt), "brightness_pred": float(yp),
            })
    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    tm = metrics_out["test"]
    print(f"\n[DONE] test_sp={tm['spearman']:.4f} test_r2={tm['r2']:.4f} "
          f"test_mae={tm['mae']:.4f} test_rmse={tm['rmse']:.4f} best_ep={best['epoch']}")
    return metrics_out


# ── Eval-only ──────────────────────────────────────────────────────────────────

def evaluate(data_dir, checkpoint, batch_size=32, device="cuda"):
    """Load a trained checkpoint and compute metrics on all splits."""
    data_dir = Path(data_dir)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    D = ckpt.get("D", 1280)
    use_message = ckpt.get("use_message", True)
    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]

    model = RevisedSlimModel(D=D, hid=256, use_message=use_message).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df  = pd.read_csv(data_dir / "master_20000.csv").reset_index(drop=True)
    y   = df["brightness"].to_numpy(dtype=np.float32)
    spl = np.load(data_dir / "split_indices.npz")
    H   = np.load(data_dir / "pred3di_saprot_residue_embeddings.npy", mmap_mode="r")
    P   = np.load(data_dir / "pos_feats.npy", mmap_mode="r")

    for name, idx in [("train", spl["train_idx"]), ("val", spl["val_idx"]), ("test", spl["test_idx"])]:
        loader = DataLoader(BrightDS(H, P, y, idx), batch_size=batch_size, shuffle=False)
        _, yy, pp = predict(model, loader, y_mean, y_std, device)
        m = compute_metrics(yy, pp)
        print(f"[{name:5s}] sp={m['spearman']:.4f} r2={m['r2']:.4f} mae={m['mae']:.4f} rmse={m['rmse']:.4f} top1%={m['top1pct']}")
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="revised_slim full pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # train
    p = sub.add_parser("train")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-message", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # eval
    p = sub.add_parser("eval")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # predict
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--H", required=True, help=".npy file [N, 237, 1280]")
    p.add_argument("--P", required=True, help=".npy file [N, 237, 10]")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            data_dir=args.data_dir, out_dir=args.out_dir,
            epochs=args.epochs, batch_size=args.batch, lr=args.lr, seed=args.seed,
            use_message=not args.no_message, resume=args.resume, device=args.device,
        )
    elif args.cmd == "eval":
        evaluate(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            batch_size=args.batch, device=args.device,
        )
    elif args.cmd == "predict":
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        D = ckpt.get("D", 1280)
        use_message = ckpt.get("use_message", True)
        model = RevisedSlimModel(D=D, hid=256, use_message=use_message).to(args.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        H = torch.from_numpy(np.load(args.H)).float().to(args.device)
        P = torch.from_numpy(np.load(args.P)).float().to(args.device)
        with torch.no_grad():
            y_hat = (model(H, P).cpu() * ckpt["y_std"] + ckpt["y_mean"]).numpy()
        for i, val in enumerate(y_hat):
            print(f"{i:5d}  brightness_pred={val:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
