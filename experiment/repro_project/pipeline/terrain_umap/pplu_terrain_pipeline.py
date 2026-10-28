"""
ppluGFP Intermediate Feature Terrain Pipeline
=============================================
Adapted from modal_make_intermediate_feature_terrains.py for ppluGFP.
- 221 AA, Ridge-derived regions + key sites
- Saves all intermediate CSVs + UMAP coords + HTML terrain maps
- Runs on L4 after training completes
"""
from __future__ import annotations

import gc, json, os, random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

import modal

VOLUME_NAME = os.environ.get("GFP_VOLUME", "gfp-revised-slim")
GPU_TYPE = os.environ.get("GPU_TYPE", "L4")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", add_python="3.11")
    .pip_install("numpy","pandas","scipy","scikit-learn","tqdm","plotly","umap-learn","openpyxl","transformers")
)

app = modal.App("gfp-pplu-terrain", image=image)

DATA_VOL = modal.Volume.from_name("gfp20000-data", create_if_missing=False)
RUNS_VOL = modal.Volume.from_name("gfp20000-runs", create_if_missing=False)
OUT_VOL = modal.Volume.from_name("gfp-revised-slim", create_if_missing=False)

# ═══════════════════════════════════════════════════════════════
# ppluGFP-specific constants
# ═══════════════════════════════════════════════════════════════

PPLU_WT = "PAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA"
L = 221

# Top 15 key residues (1-indexed, from Ridge analysis)
KEY_SITES_1B = [24, 33, 35, 36, 52, 53, 55, 56, 57, 58, 59, 62, 84, 86, 100]
KEY_SITES_0B = [p - 1 for p in KEY_SITES_1B]

# Top 9 Ridge windows as region definitions
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
RIDX = [list(range(v[0]-1, v[1])) for v in REGIONS_1B.values()]
REGION_NAMES = list(REGIONS_1B.keys())

AAP = {"A":"H","V":"H","I":"H","L":"H","M":"H","F":"H","W":"H","P":"H","G":"H",
       "S":"P","T":"P","N":"P","Q":"P","C":"P","Y":"P",
       "D":"C","E":"C","K":"C","R":"C","H":"C"}

ALL_SOURCES = [
    "sequence", "h32",
    "global_mean_pool", "global_max_pool", "global_attn_pool",
    "pos_proj_flat",
    *[f"region_{name}_all_mut512" for name in REGION_NAMES],
    "region9_mean_pool", "region9_max_pool", "region9_attn_pool",
    "keysite_K_flat",
]

# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def aa_encode_sequences(seqs):
    alphabet = "ACDEFGHIKLMNPQRSTVWYBXZUOJ*-_.?"
    table = {aa: i+1 for i,aa in enumerate(alphabet)}
    Lm = max(len(str(s)) for s in seqs)
    arr = np.zeros((len(seqs), Lm), dtype=np.uint8)
    for i,s in enumerate(seqs):
        s=str(s).upper()
        for j,ch in enumerate(s): arr[i,j] = table.get(ch, table["?"])
    return arr

def pairwise_hamming(encoded, chunk=128):
    from tqdm import tqdm
    n = encoded.shape[0]; D = np.empty((n,n), dtype=np.float32)
    for s in tqdm(range(0,n,chunk), desc="hamming dist"):
        e = min(s+chunk, n)
        D[s:e] = (encoded[s:e,None,:] != encoded[None,:,:]).mean(axis=2, dtype=np.float32)
    D = (D + D.T)*0.5; np.fill_diagonal(D, 0.0)
    return D

def pairwise_cosine(x, chunk=512):
    from tqdm import tqdm
    x = np.asarray(x, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    n = x.shape[0]; D = np.empty((n,n), dtype=np.float32)
    for s in tqdm(range(0,n,chunk), desc=f"cosine dist dim={x.shape[1]}"):
        e = min(s+chunk, n)
        D[s:e] = 1.0 - x[s:e] @ x.T
    D = np.clip((D+D.T)*0.5, 0.0, 2.0); np.fill_diagonal(D, 0.0)
    return D

def umap_layout(D, seed=42):
    import umap
    reducer = umap.UMAP(n_components=2, metric="precomputed", random_state=seed, n_jobs=1)
    xy = reducer.fit_transform(D)
    return xy.astype(np.float32)

# ═══════════════════════════════════════════════════════════════
# Model definition (ppluGFP-adapted)
# ═══════════════════════════════════════════════════════════════

def build_model():
    import torch, torch.nn as nn

    class MaskedAttentionPool(nn.Module):
        def __init__(self, dim):
            super().__init__(); self.attn = nn.Linear(dim, 1)
        def forward(self, x, mask=None, weights=None):
            b,l,_ = x.shape; lo = self.attn(x).squeeze(-1).float()
            v = torch.ones((b,l), dtype=torch.bool, device=x.device) if mask is None else mask.bool().to(x.device)
            if weights is not None:
                w = weights.float().to(x.device); v = v & (w > 1e-8); lo = lo + torch.log(w.clamp_min(1e-8))
            hv = v.any(dim=1, keepdim=True); at = torch.softmax(lo.masked_fill(~v, -1e9), dim=1)
            at = at * v.float(); at = at / at.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return torch.where(hv, torch.sum(x * at.to(x.dtype).unsqueeze(-1), dim=1), torch.zeros(x.shape[0], x.shape[2], device=x.device))

    class RevisedSlimBackbone(nn.Module):
        def __init__(self, D=1280, pos_dim=4, hid=256, use_message=True):
            super().__init__(); self.use_message = bool(use_message)
            self.res_proj = nn.Sequential(nn.Linear(D,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,hid),nn.GELU())
            self.pos_proj = nn.Sequential(nn.Linear(pos_dim,24),nn.GELU(),nn.Linear(24,24),nn.GELU())
            self.fuse = nn.Sequential(nn.Linear(hid+24,hid),nn.GELU(),nn.LayerNorm(hid))
            self.seq_attn_pool = MaskedAttentionPool(hid)
            self.glob_compress = nn.Sequential(nn.Linear(hid*3,768),nn.GELU(),nn.Dropout(0.2),nn.LayerNorm(768),nn.Linear(768,hid))
            self.region_res_proj = nn.Sequential(nn.Linear(D,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.1),nn.Linear(512,hid),nn.GELU())
            self.region_attn_pool = MaskedAttentionPool(hid)
            self.reg_compress = nn.Sequential(nn.Linear(hid*2,512),nn.GELU(),nn.Dropout(0.12),nn.LayerNorm(512),nn.Linear(512,hid))
            self.key_res_proj = nn.Sequential(nn.Linear(D,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(0.05),nn.Linear(256,hid),nn.GELU())
            self.xattn_k = nn.MultiheadAttention(hid,4,batch_first=True); self.xattn_r = nn.MultiheadAttention(hid,4,batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(hid,512),nn.GELU(),nn.Dropout(0.15),nn.Linear(512,hid))
            self.ln_k1=nn.LayerNorm(hid);self.ln_k2=nn.LayerNorm(hid);self.ln_r1=nn.LayerNorm(hid);self.ln_r2=nn.LayerNorm(hid)
            self.k_compress = nn.Sequential(nn.Linear(hid*3,768),nn.GELU(),nn.LayerNorm(768),nn.Linear(768,hid))
            self.r_compress = nn.Sequential(nn.Linear(hid*3,768),nn.GELU(),nn.LayerNorm(768),nn.Linear(768,hid))

        def pool3_parts(self, x): return x.mean(1), x.max(1)[0], self.seq_attn_pool(x)

        def _regions_with_parts(self, Hr, P4):
            mut_mask = P4[...,0] > 0.5; outs = []; parts = {}
            for name, idxs in zip(REGION_NAMES, RIDX):
                idx = torch.as_tensor(idxs, device=Hr.device, dtype=torch.long)
                r = Hr.index_select(1, idx); rm = mut_mask.index_select(1, idx)
                r_all = self.region_attn_pool(r); r_mut = self.region_attn_pool(r, mask=rm)
                concat = torch.cat([r_all, r_mut], dim=-1)
                outs.append(concat); parts[f"region_{name}_all_mut512"] = concat
            return self.reg_compress(torch.stack(outs, dim=1)), parts

        def forward_intermediates(self, H, P4):
            out = {}
            H_global = self.res_proj(H); P4_embed = self.pos_proj(P4)
            Hf = self.fuse(torch.cat([H_global, P4_embed], dim=-1))
            g_mean, g_max, g_attn = self.pool3_parts(Hf)
            out["global_mean_pool"] = g_mean; out["global_max_pool"] = g_max; out["global_attn_pool"] = g_attn
            out["pos_proj_flat"] = P4_embed.reshape(P4_embed.shape[0], -1)
            Hr = self.region_res_proj(H)
            R, region_parts = self._regions_with_parts(Hr, P4); out.update(region_parts)
            Hk = self.key_res_proj(H); K = Hk[:, KEY_SITES_0B]
            out["keysite_K_flat"] = K.reshape(K.shape[0], -1)
            K2a,_ = self.xattn_k(K,R,R,need_weights=False) if self.use_message else (torch.zeros_like(K),None)
            R2a,_ = self.xattn_r(R,K,K,need_weights=False) if self.use_message else (torch.zeros_like(R),None)
            K2=self.ln_k1(K+K2a);K2=self.ln_k2(K2+self.ffn(K2));R2=self.ln_r1(R+R2a);R2=self.ln_r2(R2+self.ffn(R2))
            r_mean,r_max,r_attn=self.pool3_parts(R2)
            out["region9_mean_pool"]=r_mean;out["region9_max_pool"]=r_max;out["region9_attn_pool"]=r_attn
            G=self.glob_compress(torch.cat([g_mean,g_max,g_attn],dim=-1))
            Rf=self.r_compress(torch.cat([r_mean,r_max,r_attn],dim=-1))
            k_mean,k_max,k_attn=self.pool3_parts(K2)
            Kf=self.k_compress(torch.cat([k_mean,k_max,k_attn],dim=-1))
            return out, {"G":G,"Rf":Rf,"Kf":Kf}

    class RevisedSlimModel(nn.Module):
        def __init__(self, D=1280, hid=256, use_message=True):
            super().__init__(); self.backbone = RevisedSlimBackbone(D=D,pos_dim=4,hid=hid,use_message=use_message)
            self.narrow = nn.Sequential(
                nn.Linear(hid*3,512),nn.GELU(),nn.Dropout(0.3),
                nn.Linear(512,256),nn.GELU(),nn.Dropout(0.25),
                nn.Linear(256,128),nn.GELU(),nn.Dropout(0.15),
                nn.Linear(128,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        def forward_all(self, H, P):
            P4 = P[:,:,[0,7,8,9]]; inter,b = self.backbone.forward_intermediates(H, P4)
            z768 = torch.cat([b["G"],b["Rf"],b["Kf"]], dim=-1)
            h512 = self.narrow[1](self.narrow[0](z768)); h512d = self.narrow[2](h512)
            h256 = self.narrow[4](self.narrow[3](h512d)); h256d = self.narrow[5](h256)
            h128 = self.narrow[7](self.narrow[6](h256d)); h128d = self.narrow[8](h128)
            h64 = self.narrow[10](self.narrow[9](h128d))
            h32 = self.narrow[12](self.narrow[11](h64))
            inter["h32"] = h32; inter["h128"] = h128; inter["h256"] = h256; inter["z768"] = z768
            inter["y_norm"] = self.narrow[13](h32).squeeze(-1)
            return inter
        def forward(self, H, P):
            P4 = P[:,:,[0,7,8,9]]
            _, b = self.backbone.forward_intermediates(H, P4)
            return self.narrow(torch.cat([b["G"],b["Rf"],b["Kf"]], dim=-1)).squeeze(-1)

    return RevisedSlimModel

# ═══════════════════════════════════════════════════════════════
# Main pipeline function
# ═══════════════════════════════════════════════════════════════

@app.function(gpu=GPU_TYPE, volumes={"/d":DATA_VOL,"/r":RUNS_VOL,"/v":OUT_VOL}, timeout=24*3600, memory=131072)
def run_terrain_pipeline(
    checkpoint_path: str = "/v/revised_slim_package/weights/best_model_pplu_31k.pt",
    out_dir: str = "/v/revised_slim_package/results/pplu_terrains",
    seed: int = 42,
):
    import torch
    from tqdm import tqdm

    set_seed(seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # ── Load ppluGFP data ──
    print("Loading ppluGFP data...")
    df = pd.read_excel("/v/gfp_data.xlsx")
    df = df[df["GFP type"] == "ppluGFP"].copy().reset_index(drop=True)
    n = len(df); y_true = df["Brightness"].to_numpy(dtype=np.float32)
    print(f"Variants: {n}")

    # ── Generate sequences + position features ──
    def ab(aa): cls=AAP.get(aa.upper(),""); return (1.0,0.0,0.0) if cls=="H" else (0.0,1.0,0.0) if cls=="P" else (0.0,0.0,1.0)
    def mut2seq(s):
        if pd.isna(s) or str(s).strip().upper() in ('WT',''): return PPLU_WT
        seq=list(PPLU_WT)
        for m in str(s).strip().split(':'):
            m=m.strip()
            if not m or len(m)<3: continue
            ps=m[1:-1]
            if not ps.isdigit(): continue
            p=int(ps)
            if p<1 or p>L: continue
            seq[p-1]=m[-1]
        return ''.join(seq)

    print("Building sequences + pos_feats...")
    seqs = [mut2seq(s) for s in df["aaMutations"]]
    P_all = np.zeros((n,L,10), dtype=np.float32)
    for r in range(n):
        seq=seqs[r]
        for i in range(L):
            is_mut=seq[i]!=PPLU_WT[i]; P_all[r,i,0]=1.0 if is_mut else 0.0
            P_all[r,i,1]=0.0; P_all[r,i,4:7]=ab(PPLU_WT[i])
            if is_mut: P_all[r,i,7:10]=ab(seq[i])

    # ── Load model ──
    print("Loading model...")
    RevisedSlimModel = build_model()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    y_mean = float(ckpt["y_mean"]); y_std = float(ckpt["y_std"])
    model = RevisedSlimModel().to(dev).eval()
    model.load_state_dict(ckpt["model"])

    # ── ESM2 + Bridge on ALL 31K — save embeddings ──
    H_cache_path = "/v/revised_slim_package/dataset/pplu_31k_H_embeddings.npy"
    if os.path.exists(H_cache_path):
        print(f"Loading cached embeddings from {H_cache_path}")
        H_all = np.memmap(H_cache_path, dtype=np.float16, mode='r', shape=(n,L,1280))
    else:
        print("Generating ESM2+Bridge embeddings for 31K...")
        from transformers import EsmModel, EsmTokenizer
        tok = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", torch_dtype=torch.float16).to(dev).eval()
        bc = torch.load("/d/warm_best_model.pt", map_location="cpu", weights_only=False)

        class Bridge(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj=torch.nn.Sequential(torch.nn.Linear(640,512),torch.nn.LayerNorm(512),torch.nn.GELU(),torch.nn.Dropout(0.1))
                e=torch.nn.TransformerEncoderLayer(d_model=512,nhead=8,dim_feedforward=2048,dropout=0.1,activation="gelu",batch_first=True,norm_first=True)
                self.encoder=torch.nn.TransformerEncoder(e,num_layers=3); self.out_norm=torch.nn.LayerNorm(512)
                self.token_head=torch.nn.Linear(512,21); self.struct_embedding=torch.nn.Embedding(21,128)
                self.bridge=torch.nn.Sequential(torch.nn.Linear(1280,1024),torch.nn.GELU(),torch.nn.LayerNorm(1024),torch.nn.Dropout(0.1),torch.nn.Linear(1024,1280))
            def forward(self,x):
                h0=self.in_proj(x); h=self.encoder(h0); h=self.out_norm(h)
                q=torch.softmax(self.token_head(h),dim=-1); ss=q@self.struct_embedding.weight
                return self.bridge(torch.cat([x,ss.to(x.dtype),h],dim=-1))
        bri = Bridge().to(dev).eval(); bri.load_state_dict(bc["model"])

        # Use memmap to save incrementally
        H_all = np.memmap(H_cache_path, dtype=np.float16, mode='w+', shape=(n,L,1280))
        esm_bs = 6
        for bs in tqdm(range(0,n,esm_bs), desc="ESM2+Bridge 31K"):
            be=min(bs+esm_bs,n); bseqs=seqs[bs:be]
            t=tok(bseqs,return_tensors="pt",padding=True); t={k:v.to(dev) for k,v in t.items()}
            with torch.no_grad():
                H_all[bs:be]=bri(esm(**t,output_hidden_states=True).last_hidden_state[:,1:1+L,:].float()).half().cpu().numpy()
        del H_all; H_all = np.memmap(H_cache_path, dtype=np.float16, mode='r', shape=(n,L,1280))
        OUT_VOL.commit()
        print(f"Embeddings saved to {H_cache_path}")

    # ── Extract ALL intermediate features ──
    print("Extracting intermediate features...")
    os.makedirs(out_dir, exist_ok=True)

    inter_data = {src: [] for src in ALL_SOURCES}
    y_pred_all = np.zeros(n, dtype=np.float32)

    feat_bs = 128
    for bs in tqdm(range(0, n, feat_bs), desc="Feature Extraction"):
        be = min(bs+feat_bs, n)
        Hb = torch.from_numpy(np.array(H_all[bs:be], dtype=np.float32)).to(dev)
        Pb = torch.from_numpy(P_all[bs:be]).to(dev)
        with torch.no_grad():
            inter = model.forward_all(Hb, Pb)
            y_pred = model.forward(Hb, Pb)  # use the standard forward for predictions
        for src in ALL_SOURCES:
            if src == "sequence":
                inter_data[src].extend([seqs[i] for i in range(bs, be)])
            elif src in inter:
                inter_data[src].append(inter[src].cpu().numpy())
        y_pred_all[bs:be] = (y_pred.cpu().numpy() * y_std + y_mean)

    # Concatenate
    for src in ALL_SOURCES:
        if src != "sequence":
            inter_data[src] = np.concatenate(inter_data[src], axis=0)

    # ── Save predictions + feature tables ──
    print("Saving predictions and feature tables...")
    df_out = df.copy()
    df_out["brightness_pred"] = y_pred_all
    df_out["brightness_true"] = y_true
    df_out.to_csv(f"{out_dir}/pplu_31k_predictions.csv", index=False)
    r2 = 1 - ((y_true-y_pred_all)**2).sum() / ((y_true-y_true.mean())**2).sum()
    from scipy.stats import spearmanr
    sp = spearmanr(y_true, y_pred_all)[0]
    print(f"Prediction quality: sp={sp:.4f} r2={r2:.4f}")

    # Save each intermediate feature table
    for src in ALL_SOURCES:
        if src == "sequence":
            pd.DataFrame({"sequence": inter_data[src], "brightness_true": y_true, "brightness_pred": y_pred_all}).to_csv(
                f"{out_dir}/table_{src}.csv", index=False)
        else:
            arr = inter_data[src]
            df_feat = pd.DataFrame(arr.reshape(arr.shape[0], -1))
            df_feat["brightness_true"] = y_true
            df_feat["brightness_pred"] = y_pred_all
            df_feat.to_csv(f"{out_dir}/table_{src}.csv", index=False)
    print(f"Saved {len(ALL_SOURCES)} feature tables to {out_dir}/")

    # ── UMAP layout + terrain HTML for each source ──
    print("Computing UMAP layouts and generating terrain maps...")
    import plotly.graph_objects as go

    # Compute sequence distance once
    encoded = aa_encode_sequences(inter_data["sequence"])
    seq_D = pairwise_hamming(encoded)

    terrain_summary = []

    for src_idx, src in enumerate(ALL_SOURCES):
        print(f"\n[{src_idx+1}/{len(ALL_SOURCES)}] {src}")

        if src == "sequence":
            D = seq_D
        else:
            D = pairwise_cosine(inter_data[src])

        # UMAP
        xy = umap_layout(D, seed=seed)
        df_xy = pd.DataFrame({"u": xy[:,0], "v": xy[:,1], "brightness_true": y_true, "brightness_pred": y_pred_all})
        df_xy.to_csv(f"{out_dir}/umap_{src}.csv", index=False)

        # Terrain HTML (true brightness)
        for z_name, z_col in [("true", "brightness_true"), ("pred", "brightness_pred")]:
            z = df_xy[z_col].to_numpy()
            fig = go.Figure(data=go.Scattergl(
                x=xy[:,0], y=xy[:,1],
                mode='markers',
                marker=dict(size=2, color=z, colorscale='Viridis', showscale=True,
                           colorbar=dict(title="Brightness")),
                text=[f"idx={i} B={z[i]:.3f}" for i in range(n)],
                hovertemplate='%{text}<extra></extra>',
            ))
            fig.update_layout(
                title=f"ppluGFP — {src} — {z_name} brightness",
                width=800, height=700,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            html_path = f"{out_dir}/terrain_{src}_{z_name}.html"
            fig.write_html(html_path)

        terrain_summary.append({"source": src, "umap_csv": f"umap_{src}.csv",
                                "terrain_true": f"terrain_{src}_true.html",
                                "terrain_pred": f"terrain_{src}_pred.html"})

    # ── Save summary ──
    pd.DataFrame(terrain_summary).to_csv(f"{out_dir}/terrain_summary.csv", index=False)
    with open(f"{out_dir}/pipeline_config.json", "w") as f:
        json.dump({"L":L, "key_sites":KEY_SITES_1B, "regions":REGIONS_1B,
                   "n_variants":n, "sources":ALL_SOURCES, "model_ckpt":checkpoint_path}, f, indent=2)
    OUT_VOL.commit()

    print(f"\n{'='*60}")
    print(f"ppluGFP TERRAIN PIPELINE COMPLETE")
    print(f"  Variants: {n}")
    print(f"  Sources:  {len(ALL_SOURCES)}")
    print(f"  Pred sp:  {sp:.4f}")
    print(f"  Pred r2:  {r2:.4f}")
    print(f"  Output:   {out_dir}/")
    return {"n": n, "sources": len(ALL_SOURCES), "sp": sp, "r2": r2}

@app.local_entrypoint()
def main():
    run_terrain_pipeline.remote()
