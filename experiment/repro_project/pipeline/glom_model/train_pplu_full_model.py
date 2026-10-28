"""
Train revised_slim for ppluGFP — same architecture, adapted for 221 AA.
Key sites: top 15 from Ridge. Regions: top 9 windows from Ridge.
"""
import json, os, random, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import modal

image = modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", add_python="3.11").pip_install("numpy","pandas","scipy","scikit-learn","tqdm","transformers","openpyxl")
app = modal.App("gfp-pplu-full-train", image=image)
DATA = modal.Volume.from_name("gfp20000-data")
OUT = modal.Volume.from_name("gfp-revised-slim")

# ── ppluGFP-specific config ──
PPLU_WT = "PAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA"
L = 221

# Top 15 key residues (1-indexed, from Ridge analysis)
KEY_SITES_1B = [24, 33, 35, 36, 52, 53, 55, 56, 57, 58, 59, 62, 84, 86, 100]

# Top 9 windows as region definitions (from Ridge sliding-window analysis)
REGIONS_1B = {
    "win_53_62":  [53, 62],
    "win_55_64":  [55, 64],
    "win_51_60":  [51, 60],
    "win_207_216":[207, 216],
    "win_203_212":[203, 212],
    "win_77_86":  [77, 86],
    "win_27_36":  [27, 36],
    "win_165_174":[165, 174],
    "win_97_106": [97, 106],
}
RIDX = [list(range(v[0]-1, v[1])) for v in REGIONS_1B.values()]

AAP = {"A":"H","V":"H","I":"H","L":"H","M":"H","F":"H","W":"H","P":"H","G":"H",
       "S":"P","T":"P","N":"P","Q":"P","C":"P","Y":"P",
       "D":"C","E":"C","K":"C","R":"C","H":"C"}

@app.function(gpu="H100", volumes={"/d":DATA,"/v":OUT}, timeout=24*3600, memory=131072)
def train(epochs=100, batch_size=32, lr=3e-5):
    from transformers import EsmModel, EsmTokenizer
    set_seed = lambda s: (random.seed(s), np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s))
    set_seed(0); dev = torch.device("cuda")

    # ── Load data ──
    df_all = pd.read_excel("/v/gfp_data.xlsx")
    df = df_all[df_all["GFP type"] == "ppluGFP"].copy().reset_index(drop=True)
    y_all = df["Brightness"].to_numpy(dtype=np.float32)
    n = len(df)
    print(f"ppluGFP ALL variants: {n}")

    # ── Generate sequences ──
    def mut2seq(s):
        if pd.isna(s) or str(s).strip().upper() in ('WT',''): return PPLU_WT
        seq = list(PPLU_WT)
        for m in str(s).strip().split(':'):
            m = m.strip()
            if not m or len(m) < 3: continue
            ps = m[1:-1]
            if not ps.isdigit(): continue
            p = int(ps)
            if p < 1 or p > L: continue
            seq[p-1] = m[-1]
        return ''.join(seq)
    seqs = [mut2seq(s) for s in df["aaMutations"]]
    print(f"Generated {len(seqs)} sequences, len={len(seqs[0])}")

    # ── Position features ──
    def ab(aa): cls=AAP.get(aa.upper(),""); return (1.0,0.0,0.0) if cls=="H" else (0.0,1.0,0.0) if cls=="P" else (0.0,0.0,1.0)
    P_all = np.zeros((n, L, 10), dtype=np.float32)
    for r in tqdm(range(n), desc="PosFeats"):
        seq = seqs[r]
        for i in range(L):
            is_mut = seq[i] != PPLU_WT[i]
            P_all[r,i,0] = 1.0 if is_mut else 0.0
            P_all[r,i,1] = 0.0  # ppluGFP has different chromophore geometry
            P_all[r,i,4:7] = ab(PPLU_WT[i])
            if is_mut: P_all[r,i,7:10] = ab(seq[i])

    # ── ESM2 + Bridge ──
    print("Loading ESM2+Bridge...")
    tok = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", torch_dtype=torch.float16).to(dev).eval()
    bc = torch.load("/d/warm_best_model.pt", map_location="cpu", weights_only=False)
    class Bridge(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj=nn.Sequential(nn.Linear(640,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.1))
            e=nn.TransformerEncoderLayer(d_model=512,nhead=8,dim_feedforward=2048,dropout=0.1,activation="gelu",batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(e,num_layers=3); self.out_norm=nn.LayerNorm(512)
            self.token_head=nn.Linear(512,21); self.struct_embedding=nn.Embedding(21,128)
            self.bridge=nn.Sequential(nn.Linear(1280,1024),nn.GELU(),nn.LayerNorm(1024),nn.Dropout(0.1),nn.Linear(1024,1280))
        def forward(self,x):
            h0=self.in_proj(x); h=self.encoder(h0); h=self.out_norm(h)
            q=torch.softmax(self.token_head(h),dim=-1); ss=q@self.struct_embedding.weight
            return self.bridge(torch.cat([x,ss.to(x.dtype),h],dim=-1))
    bri = Bridge().to(dev).eval(); bri.load_state_dict(bc["model"])

    H_all = torch.zeros((n, L, 1280), dtype=torch.float16)
    esm_bs = 6
    for bs in tqdm(range(0, n, esm_bs), desc="ESM2+Bridge"):
        be = min(bs+esm_bs, n)
        t = tok([seqs[i] for i in range(bs, be)], return_tensors="pt", padding=True)
        t = {k: v.to(dev) for k, v in t.items()}
        with torch.no_grad():
            esm_out = esm(**t, output_hidden_states=True).last_hidden_state[:, 1:1+L, :].float()
            H_all[bs:be] = bri(esm_out).half().cpu()

    # ── Train/val split ──
    idx = np.random.permutation(n); n_tr = int(n*0.8)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
    y_mean = float(y_all[tr_idx].mean()); y_std = float(y_all[tr_idx].std() + 1e-8)
    print(f"y_mean={y_mean:.4f} y_std={y_std:.4f}")

    class DS(Dataset):
        def __init__(self, indices):
            self.ix = np.asarray(indices, dtype=np.int64)
        def __len__(self): return len(self.ix)
        def __getitem__(self, i):
            j = int(self.ix[i])
            return {"H": torch.from_numpy(np.array(H_all[j], dtype=np.float32, copy=True)),
                    "P": torch.from_numpy(np.array(P_all[j], dtype=np.float32, copy=True)),
                    "y": torch.tensor(float(y_all[j]), dtype=torch.float32)}

    tr_ld = DataLoader(DS(tr_idx), batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(DS(va_idx), batch_size=batch_size, shuffle=False)

    # ── Revised_slim model (same architecture, ppluGFP-adapted constants) ──
    class MaskedAttentionPool(nn.Module):
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
        def __init__(self, D=1280, pos_dim=4, hid=256, use_message=True):
            super().__init__()
            self.use_message = bool(use_message)
            self.res_proj = nn.Sequential(nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, hid), nn.GELU())
            self.pos_proj = nn.Sequential(nn.Linear(pos_dim, 24), nn.GELU(), nn.Linear(24, 24), nn.GELU())
            self.fuse = nn.Sequential(nn.Linear(hid + 24, hid), nn.GELU(), nn.LayerNorm(hid))
            self.region_res_proj = nn.Sequential(nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, hid), nn.GELU())
            self.key_res_proj = nn.Sequential(nn.Linear(D, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.05), nn.Linear(256, hid), nn.GELU())
            self.seq_attn_pool = MaskedAttentionPool(hid)
            self.region_attn_pool = MaskedAttentionPool(hid)
            self.reg_compress = nn.Sequential(nn.Linear(hid * 2, 512), nn.GELU(), nn.Dropout(0.12), nn.LayerNorm(512), nn.Linear(512, hid))
            self.glob_compress = nn.Sequential(nn.Linear(hid * 3, 768), nn.GELU(), nn.Dropout(0.2), nn.LayerNorm(768), nn.Linear(768, hid))
            self.xattn_k = nn.MultiheadAttention(hid, 4, batch_first=True)
            self.xattn_r = nn.MultiheadAttention(hid, 4, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(hid, 512), nn.GELU(), nn.Dropout(0.15), nn.Linear(512, hid))
            self.ln_k1 = nn.LayerNorm(hid); self.ln_k2 = nn.LayerNorm(hid)
            self.ln_r1 = nn.LayerNorm(hid); self.ln_r2 = nn.LayerNorm(hid)
            self.k_compress = nn.Sequential(nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid))
            self.r_compress = nn.Sequential(nn.Linear(hid * 3, 768), nn.GELU(), nn.LayerNorm(768), nn.Linear(768, hid))

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
            H_global = self.res_proj(H)
            Hf = self.fuse(torch.cat([H_global, self.pos_proj(P4)], dim=-1))
            G = self.glob_compress(self._pool3(Hf))
            Hr = self.region_res_proj(H)
            R = self._regions(Hr, P4)
            Hk = self.key_res_proj(H)
            K = Hk[:, [p - 1 for p in KEY_SITES_1B]]
            if self.use_message:
                K2a, _ = self.xattn_k(K, R, R, need_weights=False)
                R2a, _ = self.xattn_r(R, K, K, need_weights=False)
            else:
                K2a = torch.zeros_like(K); R2a = torch.zeros_like(R)
            K2 = self.ln_k1(K + K2a); K2 = self.ln_k2(K2 + self.ffn(K2))
            R2 = self.ln_r1(R + R2a); R2 = self.ln_r2(R2 + self.ffn(R2))
            return {"G": G, "Rf": self.r_compress(self._pool3(R2)), "Kf": self.k_compress(self._pool3(K2))}

    class RevisedSlimModel(nn.Module):
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

    model = RevisedSlimModel().to(dev)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    @torch.no_grad()
    def evaluate(loader):
        model.eval(); ys_list, ps_list = [], []
        for batch in loader:
            H_b, P_b, y_b = batch["H"].to(dev), batch["P"].to(dev), batch["y"]
            pred = (model(H_b, P_b).cpu() * y_std + y_mean).numpy()
            ys_list.extend(y_b.numpy().tolist()); ps_list.extend(pred.tolist())
        yt, yp = np.asarray(ys_list), np.asarray(ps_list)
        return {"sp": float(spearmanr(yt, yp)[0]), "r2": float(r2_score(yt, yp)),
                "mae": float(mean_absolute_error(yt, yp)), "rmse": float(np.sqrt(np.mean((yt - yp) ** 2)))}

    # Quick baseline: random init
    m_init = evaluate(va_ld)
    print(f"Random init: val sp={m_init['sp']:.4f} r2={m_init['r2']:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)
    best = {"epoch": -1, "val_sp": -1e9, "state": None}
    history = []

    for ep in range(1, epochs + 1):
        model.train(); total, cnt = 0.0, 0
        for batch in tqdm(tr_ld, desc=f"ep {ep:3d}/{epochs}", leave=False):
            H_b, P_b, y_b = batch["H"].to(dev), batch["P"].to(dev), batch["y"].to(dev)
            yn = ((y_b - y_mean) / y_std).float()
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(H_b, P_b), yn)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach().cpu()) * len(y_b); cnt += len(y_b)
        vm = evaluate(va_ld)
        history.append({"epoch": ep, "train_loss": total / max(cnt, 1), **{f"val_{k}": v for k, v in vm.items()}})
        if vm["sp"] > best["val_sp"]:
            best = {"epoch": ep, "val_sp": float(vm["sp"]), "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        if ep % max(1, epochs // 10) == 0:
            print(f"[ep {ep:3d}] val_sp={vm['sp']:.4f} r2={vm['r2']:.4f} rmse={vm['rmse']:.4f} (best_ep={best['epoch']})")

    model.load_state_dict(best["state"]); mf = evaluate(va_ld)

    # Full predictions
    model.eval(); preds = np.zeros(n, dtype=np.float32)
    for i in range(0, n, 128):
        end = min(i + 128, n)
        Hb = torch.from_numpy(np.array(H_all[i:end], dtype=np.float32)).to(dev)
        Pb = torch.from_numpy(np.array(P_all[i:end], dtype=np.float32)).to(dev)
        with torch.no_grad(): preds[i:end] = (model(Hb, Pb).cpu().numpy() * y_std + y_mean)

    df_out = df.copy(); df_out["brightness_pred"] = preds
    df_out.to_csv("/v/revised_slim_package/results/pplu_31480_predictions.csv", index=False)
    pd.DataFrame(history).to_csv("/v/revised_slim_package/results/pplu_31480_history.csv", index=False)
    torch.save({"model": best["state"], "y_mean": y_mean, "y_std": y_std, "best": best, "L": L, "WT_type": "ppluGFP"},
               "/v/revised_slim_package/weights/best_model_pplu_31k.pt")
    OUT.commit()

    print(f"\n{'='*60}")
    print(f"ppluGFP TRAINING COMPLETE ({L} AA, {n} variants)")
    print(f"  Random init: sp={m_init['sp']:.4f} r2={m_init['r2']:.4f}")
    print(f"  Best val:    sp={mf['sp']:.4f} r2={mf['r2']:.4f} mae={mf['mae']:.4f} rmse={mf['rmse']:.4f}")
    print(f"  Best epoch:  {best['epoch']}")
    print(f"  Model saved to best_model_pplu_31k.pt")
    return {"init": m_init, "best_val": mf, "best_epoch": best["epoch"]}

@app.local_entrypoint()
def main():
    print(train.remote(epochs=100, batch_size=32, lr=3e-5))
