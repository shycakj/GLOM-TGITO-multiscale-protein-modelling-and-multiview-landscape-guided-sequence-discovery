"""Official-style SaProt CLS/LoRA recipe; project labels only; no task checkpoint transfer."""
from pathlib import Path
import argparse,json,time,os
os.environ.setdefault("OMP_NUM_THREADS","4")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from transformers import AutoTokenizer,EsmModel
from peft import LoraConfig,get_peft_model
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from experiment_core import seed_all,regression_metrics,write_json

class Regressor(nn.Module):
    def __init__(self,path,rank):
        super().__init__()
        assert rank==8
        from model import make_model
        self.backbone=make_model()
    def forward(self,ids,mask):
        return self.backbone(input_ids=ids,attention_mask=mask).logits.squeeze(-1).float()


def loader(d,tok,batch,shuffle,seed):
    cached=np.asarray(tok[d.row_id.to_numpy()],dtype="int64")
    assert cached.shape==(len(d),223)
    ids={"input_ids":torch.from_numpy(cached.copy()),"attention_mask":torch.ones((len(d),223),dtype=torch.long)}
    assert (ids["attention_mask"].sum(1)==223).all()
    data=TensorDataset(ids["input_ids"],ids["attention_mask"],torch.tensor(d.brightness.to_numpy("float32")))
    return DataLoader(data,batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))
def predict(m,dl,mean,std):
    m.eval();ys=[];ps=[]
    with torch.inference_mode():
        for ids,mask,y in dl:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                p=m(ids.cuda(non_blocking=True),mask.cuda(non_blocking=True))
            ys.append(y.numpy());ps.append((p.float()*std+mean).cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--seed",type=int,default=20261013)
    ap.add_argument("--epochs",type=int,default=25);ap.add_argument("--patience",type=int,default=6)
    ap.add_argument("--batch-size",type=int,default=8);ap.add_argument("--accumulation",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-4);ap.add_argument("--rank",type=int,default=8)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--input-mode",choices=["aa","pred3di"],required=True)
    ap.add_argument("--split",choices=["random","depth"],default="random")
    a=ap.parse_args();root=a.root.resolve();seed_all(a.seed);torch.set_num_threads(4)
    out=root/"artifacts/models"/a.split/(f"saprot650_official_{a.input_mode}_lora_seed{a.seed}"+("_smoke" if a.smoke else ""))
    out.mkdir(parents=True,exist_ok=True)
    if (out/"DONE.json").exists():print("ALREADY_DONE");return
    if a.split=="random":
        frames={r:pd.read_csv(root/f"data/{r}.csv") for r in ["train","validation"]}
    else:
        d=pd.read_csv(root/"data/depth_extrapolation.csv")
        frames={r:d[d.partition.eq(r)] for r in ["train","validation"]}
    if a.smoke:frames={k:v.head(32) for k,v in frames.items()}
    path=root/"repro_project/assets/saprot_650m_af2"
    from model import token_cache
    import hashlib
    spec=json.loads((HERE/"PROTOCOL.json").read_text())
    audit=json.loads((HERE/"INPUT_AUDIT.json").read_text())
    assert audit['status']=='PASS'
    protocol_hash=hashlib.sha256((HERE/"PROTOCOL.json").read_bytes()).hexdigest()
    assert audit['protocol_sha256']==protocol_hash
    assert a.rank==spec['lora']['rank'] and a.lr==spec['training']['lr']
    assert a.batch_size*a.accumulation==spec['training']['effective_batch']
    if not a.smoke:assert a.epochs==spec['training']['max_epochs'] and a.patience==spec['training']['patience']
    for filename,digest in spec['data_hashes'].items():
        assert hashlib.sha256((root/'data'/filename).read_bytes()).hexdigest()==digest,filename
    tok=token_cache(a.input_mode)
    tr=loader(frames["train"],tok,a.batch_size,True,a.seed)
    va=loader(frames["validation"],tok,a.batch_size*2,False,a.seed)
    m=Regressor(path,a.rank).cuda()
    trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
    config={**vars(a),"root":str(root),"parameters":sum(p.numel() for p in m.parameters()),
            "trainable_parameters":trainable,"method":"Official-style native SaProt CLS regression, trained only on project labels",
            "input_features":spec["input_modes"][a.input_mode],
            "protocol_sha256":protocol_hash,"model_revision":spec["base_revision"],
            "selection":"minimum validation MSE","loss":"raw-label MSE","amp":"bf16",
            "test_labels_used_for_training_or_selection":False,"checkpoint_origin":"general pretrained only, no external task adapters"}
    write_json(out/"config.json",config);print(json.dumps(config),flush=True)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01,betas=(.9,.98))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr*.05)
    ym=0.;ys=1.  # Official task recipe: MSE on original log-brightness labels.
    best=-float("inf");bad=0;hist=[];start=1
    train_names={n for n,p in m.named_parameters() if p.requires_grad}
    def state():return {n:p.detach().cpu() for n,p in m.named_parameters() if n in train_names}
    resume=out/"resume.pt"
    if resume.exists():
        rs=torch.load(resume,map_location="cpu",weights_only=False)
        m.load_state_dict(rs["model"],strict=False);opt.load_state_dict(rs["optimizer"]);sched.load_state_dict(rs["scheduler"])
        best,bad,hist,start=rs["best"],rs["bad"],rs["history"],rs["epoch"]+1
        torch.set_rng_state(rs["rng"]);torch.cuda.set_rng_state_all(rs["cuda_rng"]);tr.generator.set_state(rs["loader_rng"])
    for ep in range(start,a.epochs+1):
        begin=time.monotonic();m.train();total=0.;count=0;opt.zero_grad(set_to_none=True)
        for step,(ids,mask,y) in enumerate(tr):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                z=m(ids.cuda(non_blocking=True),mask.cuda(non_blocking=True))
                loss=nn.functional.mse_loss(z.float(),y.cuda(non_blocking=True))
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite LoRA training loss")
            group_start=(step//a.accumulation)*a.accumulation
            group_size=min(a.accumulation,len(tr)-group_start)
            (loss/group_size).backward()
            if (step+1)%a.accumulation==0 or step+1==len(tr):
                nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
                opt.step();opt.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(y);count+=len(y)
            if step%100==0:print(json.dumps({"epoch":ep,"step":step,"steps":len(tr),"seconds":time.monotonic()-begin}),flush=True)
        sched.step();yv,pv=predict(m,va,ym,ys);metrics=regression_metrics(yv,pv)
        rec={"epoch":ep,"train_mse":total/count,"seconds":time.monotonic()-begin,
             "peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30,**metrics};hist.append(rec);print(json.dumps(rec),flush=True)
        if -(metrics["rmse"]**2)>best+1e-6:
            best=-(metrics["rmse"]**2);bad=0
            torch.save({"model":state(),"y_mean":ym,"y_std":ys,"config":config},out/"best.pt")
        else:bad+=1
        temp=out/"resume.tmp"
        torch.save({"model":state(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"best":best,
                    "bad":bad,"history":hist,"epoch":ep,"rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all(),"loader_rng":tr.generator.get_state()},temp);temp.replace(resume)
        pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        if bad>=a.patience:break
    # All model/hyperparameter selection is complete before test labels are opened.
    ck=torch.load(out/"best.pt",map_location="cpu",weights_only=False);m.load_state_dict(ck["model"],strict=False)
    rows=[]
    for role in (["validation"] if a.smoke else ["validation","calibration","test"]):
        if a.split=="random":d=pd.read_csv(root/f"data/{role}.csv")
        else:
            full=pd.read_csv(root/"data/depth_extrapolation.csv");d=full[full.partition.eq(role)]
        if a.smoke:d=d.head(32)
        y,p=predict(m,loader(d,tok,a.batch_size*2,False,a.seed),ym,ys)
        rows.append({"partition":role,**regression_metrics(y,p)})
        pd.DataFrame({"row_id":d.row_id,"true":y,"pred":p}).to_csv(out/f"{role}_predictions.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    write_json(out/"DONE.json",{"best_validation_mse":-best,"epochs":len(hist),"config":config})
    print("PLM_LORA_COMPLETE",flush=True)
if __name__=="__main__":main()

