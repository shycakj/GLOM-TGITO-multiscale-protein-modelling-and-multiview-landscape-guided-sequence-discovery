"""Scratch-only model training; validation selects checkpoints, test is never opened."""
from __future__ import annotations
import argparse, json, time, os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from experiment_core import setup_project,seed_all,create_model,open_p4,regression_metrics,write_json

class HP(Dataset):
    def __init__(self,frame,hpath,ppath):
        self.frame=frame.reset_index(drop=True);self.hpath=hpath;self.ppath=ppath;self.h=None;self.p=None
        self.ids=self.frame.row_id.to_numpy('int64');self.y=self.frame.brightness.to_numpy('float32')
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        if self.h is None:
            self.h=np.load(self.hpath,mmap_mode='r');self.p=open_p4(self.ppath,len(self.h))
        j=self.ids[i];h=np.array(self.h[j],dtype='float32');p=np.zeros((221,10),dtype='float32')
        p[:,[0,7,8,9]]=self.p[j]
        return torch.from_numpy(h),torch.from_numpy(p),torch.tensor(self.y[i])

def predict(model,loader,mean,std,amp=True):
    model.eval();truth=[];pred=[]
    with torch.inference_mode():
        for h,p,y in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):z=model(h.cuda(non_blocking=True),p.cuda(non_blocking=True))
            pred.append((z.float()*std+mean).cpu().numpy());truth.append(y.numpy())
    return np.concatenate(truth),np.concatenate(pred)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--method',choices=['glom','global_only','no_message','hp_transformer'],required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=120)
    ap.add_argument('--patience',type=int,default=18);ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--lr',type=float,default=3e-5);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--split',choices=['random','depth'],default='random');ap.add_argument('--max-train-rows',type=int,default=0)
    a=ap.parse_args();root=a.root.resolve();setup_project(root);seed_all(a.seed);torch.set_num_threads(4)
    if a.split=='random':tr=pd.read_csv(root/'data/train.csv');va=pd.read_csv(root/'data/validation.csv')
    else:
        d=pd.read_csv(root/'data/depth_extrapolation.csv');tr=d[d.partition.eq('train')];va=d[d.partition.eq('validation')]
    if a.max_train_rows:tr=tr.head(a.max_train_rows);va=va.head(min(256,a.max_train_rows))
    assert not set(tr.sequence)&set(va.sequence)
    run=f'{a.method}_seed{a.seed}'+('_smoke' if a.max_train_rows else '')
    out=root/'artifacts/models'/a.split/run;out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():print('ALREADY_DONE '+run,flush=True);return
    H=root/'artifacts/embeddings/H.npy';P=root/'artifacts/embeddings/P4_glom_original.npy'
    trld=DataLoader(HP(tr,H,P),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,
                    generator=torch.Generator().manual_seed(a.seed),persistent_workers=a.workers>0)
    vald=DataLoader(HP(va,H,P),batch_size=a.batch_size*2,shuffle=False,num_workers=a.workers,pin_memory=True,
                    persistent_workers=a.workers>0)
    ym=float(tr.brightness.mean());ys=float(tr.brightness.std(ddof=0)+1e-8)
    model=create_model(a.method).cuda();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr*.05)
    meta={'method':a.method,'seed':a.seed,'split':a.split,'train_n':len(tr),'validation_n':len(va),'init':'scratch',
      'epochs_limit':a.epochs,'patience':a.patience,'batch_size':a.batch_size,'lr':a.lr,'amp':'bf16',
      'parameters':sum(p.numel() for p in model.parameters()),'test_labels_read':False,
      'historical_key_sites_and_regions':'fixed architecture design; provenance retained, not reselected on test',
      'torch':torch.__version__,'cuda':torch.version.cuda}
    write_json(out/'config.json',meta);best=-np.inf;bad=0;history=[];start=1
    resume=out/'resume.pt'
    if resume.exists():
        rs=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(rs['model']);opt.load_state_dict(rs['optimizer']);sched.load_state_dict(rs['scheduler'])
        best=rs['best'];bad=rs['bad'];history=rs['history'];start=rs['epoch']+1
        torch.set_rng_state(rs['torch_rng']);torch.cuda.set_rng_state_all(rs['cuda_rng']);trld.generator.set_state(rs['loader_rng'])
        np.random.set_state(rs['numpy_rng'])
    started=time.monotonic()
    for ep in range(start,a.epochs+1):
        model.train();total=0.;count=0;epoch_start=time.monotonic()
        for h,p,y in trld:
            h=h.cuda(non_blocking=True);p=p.cuda(non_blocking=True);y=(y.cuda(non_blocking=True)-ym)/ys
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(h,p);loss=torch.nn.functional.huber_loss(z.float(),y,delta=1.)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(y);count+=len(y)
        sched.step();yv,pv=predict(model,vald,ym,ys);metrics=regression_metrics(yv,pv)
        rec={'epoch':ep,'train_huber':total/count,'seconds':time.monotonic()-epoch_start,
             'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,**metrics};history.append(rec)
        print(json.dumps({'run':run,**rec}),flush=True)
        if metrics['spearman']>best+1e-6:
            best=metrics['spearman'];bad=0
            torch.save({'model':model.state_dict(),'y_mean':ym,'y_std':ys,'method':a.method,'metadata':meta},out/'best.pt')
            pd.DataFrame({'row_id':va.row_id,'true':yv,'pred':pv}).to_csv(out/'validation_predictions.csv',index=False)
        else:bad+=1
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
               'best':best,'bad':bad,'epoch':ep,'history':history,'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':trld.generator.get_state(),'numpy_rng':np.random.get_state()}
        temp=out/'resume.tmp';torch.save(state,temp);temp.replace(resume)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if bad>=a.patience:break
    write_json(out/'DONE.json',{'best_validation_spearman':best,'epochs':len(history),'elapsed_seconds':time.monotonic()-started,**meta})

if __name__=='__main__':main()
