"""Strong pretrained embeddings with matched supervised Ridge, MLP and EVOLVEpro RF heads."""
from pathlib import Path
import argparse,json,time,os
os.environ.setdefault("OMP_NUM_THREADS","3");os.environ.setdefault("OPENBLAS_NUM_THREADS","3")
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from experiment_core import seed_all,write_json,regression_metrics
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--feature",required=True);ap.add_argument("--seed",type=int,default=20261013)
    ap.add_argument("--head",choices=["ridge","mlp","rf"],required=True)
    ap.add_argument("--split",choices=["random","depth"],default="random");a=ap.parse_args()
    root=a.root;seed_all(a.seed);threadpool_limits(3);torch.set_num_threads(3)
    out=root/"artifacts/models"/a.split/f"{a.feature}_{a.head}_seed{a.seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"DONE.json").exists():print("ALREADY_DONE");return
    state=json.loads((root/"artifacts/embeddings"/(a.feature+"_state.json")).read_text())
    assert state["completed_rows"]==state["rows"]==31480
    X=np.load(root/"artifacts/embeddings"/(a.feature+"_mean.npy"),mmap_mode="r")
    def data(role):
        if a.split=="random":d=pd.read_csv(root/f"data/{role}.csv")
        else:
            q=pd.read_csv(root/"data/depth_extrapolation.csv");d=q[q.partition.eq(role)]
        return d,np.asarray(X[d.row_id.to_numpy()],dtype="float32"),d.brightness.to_numpy("float32")
    tr,xt,yt=data("train");va,xv,yv=data("validation")
    record={"feature":a.feature,"head":a.head,"seed":a.seed,"split":a.split,
            "selection":"validation Spearman","training_test_labels_read":False}
    best=-np.inf;best_model=None;history=[]
    if a.head=="ridge":
        for alpha in [.1,1,10,100,1000]:
            m=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver="lsqr",tol=1e-5,max_iter=2000))
            m.fit(xt,yt);p=m.predict(xv);met=regression_metrics(yv,p);history.append({"alpha":alpha,**met})
            if met["spearman"]>best:best=met["spearman"];best_model=m;record["alpha"]=alpha
        def infer(x):return best_model.predict(x)
    elif a.head=="rf":
        # Matches the EVOLVEpro RF estimator; backbone/label-budget adaptation is explicit.
        m=RandomForestRegressor(n_estimators=100,criterion='friedman_mse',random_state=a.seed,n_jobs=3)
        m.fit(xt,yt);best_model=m;best=float(spearmanr(yv,m.predict(xv)).statistic)
        record.update(estimator="sklearn RandomForestRegressor",n_estimators=100,
                      reference="EVOLVEpro Science 2025",adaptation="frozen specified backbone; static full-training-set regression")
        def infer(x):return best_model.predict(x)
    else:
        scaler=StandardScaler().fit(xt)
        tx=torch.from_numpy(scaler.transform(xt)).cuda();vx=torch.from_numpy(scaler.transform(xv)).cuda()
        mean=float(yt.mean());std=float(yt.std()+1e-8);ty=torch.from_numpy((yt-mean)/std).cuda()
        m=nn.Sequential(nn.Linear(tx.shape[1],512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(.2),
                        nn.Linear(512,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1)).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01);bad=0;chosen=None
        for ep in range(1,201):
            m.train();ix=torch.randperm(len(tx),device="cuda");begin=time.monotonic()
            for idx in ix.split(256):
                opt.zero_grad(set_to_none=True);p=m(tx[idx]).squeeze(-1)
                loss=nn.functional.huber_loss(p,ty[idx]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            m.eval()
            with torch.inference_mode():p=torch.cat([m(b).squeeze(-1) for b in vx.split(512)]).cpu().numpy()*std+mean
            met=regression_metrics(yv,p);history.append({"epoch":ep,"seconds":time.monotonic()-begin,**met})
            print(json.dumps({"model":out.name,"epoch":ep,**{k:met[k] for k in ["spearman","rmse","r2"]}}),flush=True)
            if met["spearman"]>best+1e-6:
                best=met["spearman"];bad=0;chosen={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                record["best_epoch"]=ep
            else:bad+=1
            if bad>=20:break
        m.load_state_dict(chosen);m.eval()
        torch.save({"model":chosen,"mean":mean,"std":std,"scaler_mean":scaler.mean_,
                    "scaler_scale":scaler.scale_,"config":record},out/"best.pt")
        def infer(x):
            with torch.inference_mode():
                q=torch.from_numpy(scaler.transform(x)).cuda()
                return torch.cat([m(b).squeeze(-1) for b in q.split(512)]).cpu().numpy()*std+mean
    if best_model is not None:
        import joblib
        joblib.dump(best_model,out/"best.joblib",compress=1)
    results=[]
    for role in ["validation","calibration","test"]:
        d,x,y=data(role);p=infer(x)
        results.append({"partition":role,**regression_metrics(y,p)})
        pd.DataFrame({"row_id":d.row_id,"true":y,"pred":p}).to_csv(out/f"{role}_predictions.csv",index=False)
    pd.DataFrame(results).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(history).to_csv(out/"validation_search.csv",index=False)
    write_json(out/"DONE.json",{"best_validation_spearman":best,**record});print(json.dumps(results),flush=True)
if __name__=="__main__":main()

