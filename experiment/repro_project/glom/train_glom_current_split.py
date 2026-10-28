from __future__ import annotations
import argparse, json, math, os, random, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.tgito.revised_slim_model import RevisedSlimModel


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def load_cfg(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def p10_from_p4(p4: np.ndarray) -> np.ndarray:
    if p4.shape[-1] == 10: return p4
    if p4.shape[-1] != 4: raise ValueError(f'P feature last dim must be 4 or 10, got {p4.shape}')
    out=np.zeros((*p4.shape[:-1],10),dtype=np.float32)
    out[..., [0,7,8,9]]=p4.astype(np.float32)
    return out


class IndexedHP(Dataset):
    def __init__(self, df, H, P):
        self.df=df.reset_index(drop=True); self.H=H; self.P=P
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; j=int(r.row_id)
        h=np.asarray(self.H[j],dtype=np.float32)
        p=np.asarray(self.P[j],dtype=np.float32)
        if p.shape[-1]==4: p=p10_from_p4(p)
        return torch.from_numpy(h), torch.from_numpy(p), torch.tensor(float(r.brightness),dtype=torch.float32)


def predict(model, loader, device, mean, std):
    ys=[]; ps=[]; model.eval()
    with torch.inference_mode():
        for h,p,y in loader:
            h=h.to(device,non_blocking=True); p=p.to(device,non_blocking=True)
            z=model(h,p); pred=z*std+mean
            ys.extend(y.numpy().tolist()); ps.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(ys,float),np.asarray(ps,float)


def metrics(y,p):
    return {
        'rmse':float(np.sqrt(np.mean((y-p)**2))),
        'mae':float(mean_absolute_error(y,p)),
        'r2':float(r2_score(y,p)),
        'pearson':float(pearsonr(y,p).statistic),
        'spearman':float(spearmanr(y,p).statistic),
    }


def main():
    ap=argparse.ArgumentParser(description='Strict current-split GLOM training using the existing 31K-aligned H/P arrays. No hidden400 file is opened.')
    ap.add_argument('--config',default=str(ROOT/'configs/publication_experiment.json'))
    ap.add_argument('--root',default=str(ROOT)); ap.add_argument('--seed',type=int,default=20260902)
    ap.add_argument('--epochs',type=int,default=None); ap.add_argument('--batch-size',type=int,default=None)
    ap.add_argument('--output-dir',default='',help='Write fresh training artifacts separately from archived glom_current.'); ap.add_argument('--lr',type=float,default=None); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--init-checkpoint',default='',help='Only use a checkpoint known not to contain final hidden labels. Full31K checkpoint is refused unless --allow-leaky-warmstart.')
    ap.add_argument('--allow-leaky-warmstart',action='store_true')
    ap.add_argument('--resume',default='',help='Resume training from a saved current-split checkpoint (restores model+optimizer+scheduler+epoch).')
    args=ap.parse_args(); root=Path(args.root); cfg=load_cfg(args.config); seed_all(args.seed)
    gc=cfg['glom']; epochs=int(args.epochs or gc['epochs']); bs=int(args.batch_size or 64); lr=float(args.lr or gc['learning_rate'])
    data=cfg['data']; H_path=Path(os.getenv('PPLU_H_NPY',data['h_embeddings'])); P_path=Path(os.getenv('PPLU_P4_NPY',data['p4_features']))
    tr=pd.read_csv(root/'data/processed/search/glom_train.csv'); va=pd.read_csv(root/'data/processed/search/glom_val.csv')
    if len(set(tr.sequence)&set(va.sequence)): raise RuntimeError('train/val sequence overlap')
    # _safe_mmap: H/P are raw binary (no NUMPY header) in this deployment
    def _safe_mmap(path, shape, dtype):
        p=Path(path); ed=np.dtype(dtype); eb=int(np.prod(shape))*ed.itemsize
        with open(str(p),'rb') as f: magic=f.read(6)
        if magic==b'\x93NUMPY': return np.load(str(p),mmap_mode='r')
        if p.stat().st_size!=eb: raise ValueError(f'{path}: raw size mismatch')
        return np.memmap(str(p),mode='r',dtype=ed,shape=tuple(shape))
    N4=int(P_path.stat().st_size//(221*4*4))
    H=_safe_mmap(H_path,[N4,221,1280],np.float16)
    P=_safe_mmap(P_path,[N4,221,4],np.float32)
    if H.shape[0] < max(tr.row_id.max(),va.row_id.max())+1 or P.shape[0] < max(tr.row_id.max(),va.row_id.max())+1: raise RuntimeError('H/P arrays are not aligned to original row_id')
    if H.shape[1:] != (221,1280): raise ValueError(f'Unexpected H shape {H.shape}')
    if P.shape[1] != 221 or P.shape[-1] not in (4,10): raise ValueError(f'Unexpected P shape {P.shape}')
    ymean=float(tr.brightness.mean()); ystd=float(tr.brightness.std(ddof=0)+1e-8)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=RevisedSlimModel(D=1280).to(device)
    if args.init_checkpoint:
        ck=Path(args.init_checkpoint)
        legacy=Path(os.getenv('PPLU_LEGACY_GLOM_CKPT',data.get('legacy_glom_ckpt','')))
        if legacy and ck.resolve()==legacy.resolve() and not args.allow_leaky_warmstart:
            raise RuntimeError('Refusing supervised full31K GLOM warm-start in strict headline benchmark. Reuse H/P and architecture, not leaky weights.')
        obj=torch.load(ck,map_location='cpu',weights_only=False); sd=obj.get('model',obj.get('state_dict',obj)); model.load_state_dict(sd,strict=True)
    g=torch.Generator().manual_seed(args.seed)
    trld=DataLoader(IndexedHP(tr,H,P),batch_size=bs,shuffle=True,generator=g,num_workers=args.workers,pin_memory=True)
    vald=DataLoader(IndexedHP(va,H,P),batch_size=bs*2,shuffle=False,num_workers=args.workers,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(gc['weight_decay']))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs),eta_min=lr*0.05)
    out=Path(args.output_dir) if args.output_dir else root/'artifacts/glom_current'; out.mkdir(parents=True,exist_ok=True); best=-1e9; bad=0; history=[]; start_ep=1
    ckpt=out/'best_model_pplu_current_split.pt'; resume_pt=out/'resume_state.pt'; patience=int(gc['patience'])
    if args.resume:
        rs=Path(args.resume)
        if not rs.exists(): raise FileNotFoundError(f'--resume file missing: {rs}')
        obj=torch.load(rs,map_location='cpu',weights_only=False)
        model.load_state_dict(obj['model'],strict=True)
        opt.load_state_dict(obj['optimizer']); sched.load_state_dict(obj['scheduler'])
        best=float(obj.get('best',-1e9)); bad=int(obj.get('bad',0)); history=list(obj.get('history',[])); start_ep=int(obj.get('epoch',0))+1
        print(f'[resume] restored epoch={start_ep-1} best={best:.4f} bad={bad}',flush=True)
    for ep in range(start_ep,epochs+1):
        model.train(); losses=[]
        for h,p,y in trld:
            h=h.to(device,non_blocking=True); p=p.to(device,non_blocking=True); y=((y.to(device)-ymean)/ystd)
            opt.zero_grad(set_to_none=True); z=model(h,p); loss=F.huber_loss(z,y,delta=1.0); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        sched.step(); yv,pv=predict(model,vald,device,ymean,ystd); m=metrics(yv,pv); rec={'epoch':ep,'train_huber':float(np.mean(losses)),'lr':float(opt.param_groups[0]['lr']),**{f'val_{k}':v for k,v in m.items()}}; history.append(rec); print(json.dumps(rec),flush=True)
        score=m['spearman']
        if np.isfinite(score) and score>best+1e-6:
            best=score; bad=0
            torch.save({'model':model.state_dict(),'y_mean':ymean,'y_std':ystd,'reference_sequence':(root/'data/processed/reference_sequence.txt').read_text().strip(),'seed':args.seed,'split_protocol':cfg['protocol_version'],'train_rows':len(tr),'val_rows':len(va),'init_mode':'scratch' if not args.init_checkpoint else 'safe_checkpoint','H_path':str(H_path),'P_path':str(P_path)},ckpt)
        else:
            bad+=1
        # Resume checkpoint: full training state, saved every epoch.
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':ep,'best':best,'bad':bad,'history':history},resume_pt)
        if bad>=patience:
            print(f'[early-stop] epoch={ep} patience={patience}',flush=True); break
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    bestobj=torch.load(ckpt,map_location='cpu',weights_only=False); model.load_state_dict(bestobj['model'],strict=True); model.to(device)
    yv,pv=predict(model,vald,device,ymean,ystd); report={'checkpoint':str(ckpt),'best_val_spearman':best,'validation':metrics(yv,pv),'train_n':len(tr),'val_n':len(va),'hidden400_loaded':False,'tgito_tune300_loaded':False,'strict_headline_valid':not args.allow_leaky_warmstart}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame({'row_id':va.row_id,'sequence':va.sequence,'brightness_true':yv,'brightness_pred':pv}).to_csv(out/'validation_predictions.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
