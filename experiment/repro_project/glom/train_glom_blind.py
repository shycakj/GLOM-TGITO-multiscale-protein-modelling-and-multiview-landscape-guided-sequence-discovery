from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pplu_repro.common import load_reference_sequence, set_all_seeds, atomic_write_json


def add_pipeline(path: Path):
    sys.path.insert(0, str(path))
    from tgito.aa import AA20, aa_bits_tensor, seq_to_ids0, soft_pos_feats
    from tgito.esm_bridge import load_bridge, DifferentiableESMBridge
    from tgito.revised_slim_model import RevisedSlimModel
    return AA20, aa_bits_tensor, seq_to_ids0, soft_pos_feats, load_bridge, DifferentiableESMBridge, RevisedSlimModel

class CachedDataset(Dataset):
    def __init__(self, idx, y, H, P): self.idx=np.asarray(idx); self.y=np.asarray(y,np.float32); self.H=H; self.P=P
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j=int(self.idx[i]); return torch.from_numpy(np.asarray(self.H[j], dtype=np.float32)), torch.from_numpy(np.asarray(self.P[j], dtype=np.float32)), torch.tensor(self.y[i], dtype=torch.float32)

def main():
    ap=argparse.ArgumentParser(description='Train a blind-safe pplu GLOM brightness predictor; never reads hidden400 or TGITO-tune300.')
    ap.add_argument('--root', default=str(ROOT)); ap.add_argument('--pipeline-root', required=True); ap.add_argument('--bridge-ckpt', required=True)
    ap.add_argument('--seed', type=int, default=20260902); ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--embed-batch', type=int, default=8); ap.add_argument('--train-batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-5); ap.add_argument('--workers', type=int, default=0); ap.add_argument('--force-reembed', action='store_true')
    args=ap.parse_args(); root=Path(args.root); pipeline=Path(args.pipeline_root); bridge_ckpt=Path(args.bridge_ckpt)
    set_all_seeds(args.seed); reference=load_reference_sequence(root)
    if not torch.cuda.is_available(): raise RuntimeError('Blind GLOM training requires CUDA GPU for ESM/Bridge embedding.')
    device='cuda'
    AA20, aa_bits_tensor, seq_to_ids0, soft_pos_feats, load_bridge, DifferentiableESMBridge, RevisedSlimModel = add_pipeline(pipeline)
    sdir=root/'data/processed/search'; train=pd.read_csv(sdir/'glom_train.csv'); val=pd.read_csv(sdir/'glom_val.csv')
    # Deliberately no path to sealed hidden or tgito_tune here.
    fit=pd.concat([train.assign(_part='train'), val.assign(_part='val')], ignore_index=True)
    # Expected fit rows come from the protocol manifest: unique_rows - 1500 + 800.
    # Real data: 31434 - 1500 + 800 = 30734 (synthetic 31480 -> 30780).
    manifest=json.loads((root/'data'/'audit'/'protocol_manifest.json').read_text(encoding='utf-8'))
    expected_fit=int(manifest['search_counts']['glom_train'])+int(manifest['search_counts']['glom_val'])
    assert len(fit)==expected_fit, f'Expected {expected_fit} GLOM fit rows, got {len(fit)}'
    out=root/'artifacts/glom_blind'; out.mkdir(parents=True, exist_ok=True); cache=out/'cache'; cache.mkdir(exist_ok=True)
    H_path=cache/'H_fp16.dat'; P_path=cache/'P_fp16.dat'; meta_path=cache/'meta.json'; N=len(fit); L=len(reference); D=1280
    H=np.memmap(H_path, mode='w+' if args.force_reembed or not H_path.exists() else 'r+', dtype='float16', shape=(N,L,D))
    P=np.memmap(P_path, mode='w+' if args.force_reembed or not P_path.exists() else 'r+', dtype='float16', shape=(N,L,10))
    need = args.force_reembed or not meta_path.exists()
    if not need:
        try:
            m=json.loads(meta_path.read_text()); need=not (m.get('complete') and m.get('N')==N and m.get('bridge_ckpt')==str(bridge_ckpt.resolve()))
        except Exception: need=True
    if need:
        bridge=load_bridge(bridge_ckpt, device=device)
        enc=DifferentiableESMBridge(bridge, esm_model_name='auto', device=device, freeze_esm=True, target_h_dim=D).to(device).eval()
        ref=seq_to_ids0(reference, length=L, device=device); bits=aa_bits_tensor(device=device)
        with torch.inference_mode():
            for st in range(0,N,args.embed_batch):
                seqs=fit.sequence.iloc[st:st+args.embed_batch].tolist()
                ids=torch.stack([seq_to_ids0(s, length=L, device=device) for s in seqs])
                probs=F.one_hot(ids, num_classes=20).float()
                HH=enc(probs)['H']; PP=soft_pos_feats(probs, ref, bits)
                H[st:st+len(seqs)]=HH.detach().cpu().numpy().astype('float16'); P[st:st+len(seqs)]=PP.detach().cpu().numpy().astype('float16')
                if st % max(args.embed_batch*50,1)==0: print(f'[embed] {st}/{N}', flush=True)
        H.flush(); P.flush(); del enc, bridge; torch.cuda.empty_cache()
        atomic_write_json({'complete':True,'N':N,'L':L,'D':D,'bridge_ckpt':str(bridge_ckpt.resolve()),'seed':args.seed}, meta_path)
    tr_idx=np.where(fit._part.to_numpy()=='train')[0]; va_idx=np.where(fit._part.to_numpy()=='val')[0]
    ytr=train.brightness.to_numpy(np.float32); yva=val.brightness.to_numpy(np.float32); y_mean=float(ytr.mean()); y_std=float(ytr.std() or 1.0)
    tr_ds=CachedDataset(tr_idx, (ytr-y_mean)/y_std, H, P); va_ds=CachedDataset(va_idx, (yva-y_mean)/y_std, H, P)
    g=torch.Generator(); g.manual_seed(args.seed)
    tr_loader=DataLoader(tr_ds,batch_size=args.train_batch,shuffle=True,generator=g,num_workers=args.workers,pin_memory=True)
    va_loader=DataLoader(va_ds,batch_size=args.train_batch*2,shuffle=False,num_workers=args.workers,pin_memory=True)
    model=RevisedSlimModel(D=D).to(device); opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best=-1e9; hist=[]; ckpt_path=out/'best_model_pplu_blind.pt'
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for hb,pb,yb in tr_loader:
            hb=hb.to(device,non_blocking=True); pb=pb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); pred=model(hb,pb); loss=F.huber_loss(pred,yb,delta=1.0); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); pp=[]
        with torch.inference_mode():
            for hb,pb,_ in va_loader:
                pp.append((model(hb.to(device),pb.to(device))*y_std+y_mean).cpu().numpy())
        pv=np.concatenate(pp); rho=float(spearmanr(yva,pv).statistic); rmse=float(np.sqrt(np.mean((yva-pv)**2)))
        rec={'epoch':epoch,'train_huber':float(np.mean(losses)),'val_spearman':rho,'val_rmse':rmse}; hist.append(rec); print(json.dumps(rec), flush=True)
        if np.isfinite(rho) and rho>best:
            best=rho; torch.save({'model':model.state_dict(),'y_mean':y_mean,'y_std':y_std,'best_val_spearman':best,'L':L,'reference_sequence':reference,'protocol':'blind_top1500_400hidden_800glom_300tgito','seed':args.seed}, ckpt_path)
    pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
    atomic_write_json({'best_val_spearman':best,'checkpoint':str(ckpt_path),'train_n':len(train),'val_n':len(val),'hidden_loaded':False,'tgito_tune_loaded':False}, out/'report.json')
    print(f'[done] {ckpt_path}')
if __name__=='__main__': main()
