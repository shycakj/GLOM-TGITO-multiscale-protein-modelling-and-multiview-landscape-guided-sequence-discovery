"""Shared metrics and faithful architecture ablations for the 2026-10-13 benchmark."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

def setup_project(root):
    sys.path.insert(0,str(Path(root)/'repro_project'))

def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False

def regression_metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    assert len(y)==len(p) and np.isfinite(y).all() and np.isfinite(p).all()
    out={'n':len(y),'spearman':float(spearmanr(y,p).statistic),
         'pearson':float(pearsonr(y,p).statistic),'rmse':float(np.sqrt(np.mean((y-p)**2))),
         'mae':float(np.mean(np.abs(y-p))),'r2':float(r2_score(y,p))}
    order=np.argsort(-p,kind='stable');topcut=np.quantile(y,.9)
    for k in [6,30,100]:
        ix=order[:min(k,len(y))];yy=y[ix]
        out.update({f'top{k}_true_mean':float(yy.mean()),f'top{k}_true_median':float(np.median(yy)),
                    f'top{k}_true_min':float(yy.min()),f'top{k}_high420':float((yy>=4.2).mean()),
                    f'top{k}_high435':float((yy>=4.35).mean()),
                    f'top{k}_enrichment_top10pct':float((yy>=topcut).mean()/max(1e-12,(y>=topcut).mean()))})
    return out

def open_p4(path,n):
    path=Path(path)
    with path.open('rb') as f:magic=f.read(6)
    if magic==b'\x93NUMPY':a=np.load(path,mmap_mode='r')
    else:a=np.memmap(path,mode='r',dtype='float32',shape=(n,221,4))
    assert a.shape==(n,221,4)
    return a

def create_model(method):
    import torch
    import torch.nn as nn
    from pipeline.tgito.revised_slim_model import RevisedSlimModel
    if method=='hp_transformer':
        class HPTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input=nn.Sequential(nn.Linear(1284,256),nn.LayerNorm(256),nn.GELU())
                self.position=nn.Parameter(torch.randn(1,221,256)*.02)
                block=nn.TransformerEncoderLayer(256,8,1024,.1,activation='gelu',batch_first=True,norm_first=True)
                self.encoder=nn.TransformerEncoder(block,3,enable_nested_tensor=False)
                self.attention=nn.Linear(256,1)
                self.head=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,512),nn.GELU(),nn.Dropout(.2),
                                        nn.Linear(512,128),nn.GELU(),nn.Linear(128,1))
            def forward(self,h,p):
                x=self.encoder(self.input(torch.cat([h,p[:,:,[0,7,8,9]]],-1))+self.position)
                pool=(x*self.attention(x).softmax(1)).sum(1)
                return self.head(torch.cat([x.mean(1),x.max(1).values,pool],-1)).squeeze(-1)
        return HPTransformer()
    if method=='glom':return RevisedSlimModel(D=1280)
    if method=='no_message':return RevisedSlimModel(D=1280,use_message=False)
    if method=='global_only':
        class GlobalOnly(nn.Module):
            def __init__(self):
                super().__init__();full=RevisedSlimModel(D=1280);b=full.backbone
                self.res_proj=b.res_proj;self.pos_proj=b.pos_proj;self.fuse=b.fuse
                self.seq_attn_pool=b.seq_attn_pool;self.glob_compress=b.glob_compress
                self.narrow=full.narrow;self.narrow[0]=nn.Linear(256,512)
            def forward(self,h,p):
                x=self.fuse(torch.cat([self.res_proj(h),self.pos_proj(p[:,:,[0,7,8,9]])],dim=-1))
                g=self.glob_compress(torch.cat([x.mean(1),x.max(1).values,self.seq_attn_pool(x)],dim=-1))
                return self.narrow(g).squeeze(-1)
        return GlobalOnly()
    raise ValueError(method)

def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
