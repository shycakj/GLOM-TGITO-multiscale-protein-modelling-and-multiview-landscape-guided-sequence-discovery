"""Evaluate original risk gate/confidence on held-out measured sequences."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault("OMP_NUM_THREADS","3")
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from experiment_core import setup_project,write_json
def norm01(x):
    x=np.asarray(x,float);lo=x.min();hi=x.max()
    return (x-lo)/(hi-lo) if hi>lo else np.zeros_like(x)
def confidence(d):
    out=d.copy()
    out["conf_pred"]=np.clip((out.pred-4.2)/.25,0,1)
    out["conf_support"]=norm01(out.terrain_selected_support_wmean)
    out["conf_terrain"]=norm01(out.terrain)
    out["conf_lowrisk"]=1-norm01(out.terrain_selected_risk_wmean+.5*out.terrain_selected_pred_gap_wmean)
    out["conf_entropy"]=norm01(out.terrain_weight_entropy)
    out["confidence_score"]=.35*out.conf_pred+.25*out.conf_support+.20*out.conf_terrain+.15*out.conf_lowrisk+.05*out.conf_entropy
    out["confidence_without_final_risk"]=out.confidence_score-.15*out.conf_lowrisk
    return out
def selection(y,s,k):
    idx=np.argsort(-np.asarray(s),kind="stable")[:k];v=np.asarray(y)[idx]
    return {"k":k,"true_mean":float(v.mean()),"true_median":float(np.median(v)),
            "true_min":float(v.min()),"high420":float((v>=4.2).mean()),"high435":float((v>=4.35).mean())}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--run",default="glom_seed20261013");ap.add_argument("--batch-size",type=int,default=64)
    a=ap.parse_args();root=a.root;setup_project(root);torch.set_num_threads(3)
    from pipeline.tgito.terrain_memory import TerrainMemoryBank
    from pipeline.tgito.terrain_potential import TerrainPotential,TerrainPotentialConfig
    base=root/"artifacts/models/random"/a.run;feat=base/"features";bankdir=base/"terrain_train_only"
    assert (bankdir/"DONE.json").exists()
    out=base/"risk_screening";out.mkdir(exist_ok=True)
    bank=TerrainMemoryBank.from_npz(bankdir/"terrain.npz",device="cuda")
    cfg=TerrainPotentialConfig(terrain_gate_mode="risk",topk_sources=5)
    # Default v13 risk policy is frozen. No selection against this new test.
    quality=pd.read_csv(bankdir/"source_quality.csv").sort_values("source_quality",ascending=False)
    best_source=str(quality.iloc[0].source)
    models={"risk":TerrainPotential(bank,cfg),
            "rule":TerrainPotential(bank,TerrainPotentialConfig(terrain_gate_mode="rule",topk_sources=5)),
            "single":TerrainPotential(bank,TerrainPotentialConfig(terrain_gate_mode="risk",topk_sources=1),sources=[best_source])}
    ix=pd.read_csv(feat/"row_index.csv");pred=np.load(feat/"prediction.npy")
    reps={s:np.load(feat/(s+".npy"),mmap_mode="r") for s in bank.available_sources}
    from calibrate_risk_policy import calibrate
    selected_cfg=calibrate(root,base,bank,ix,pred,reps,a.batch_size)
    models["calibrated"]=TerrainPotential(bank,selected_cfg)
    for role in ["calibration","test"]:
        rows=np.flatnonzero(ix.partition.eq(role));scores={}
        for mode,model in models.items():
            records=[]
            with torch.inference_mode():
                for start in range(0,len(rows),a.batch_size):
                    jj=rows[start:start+a.batch_size]
                    live={s:torch.tensor(np.asarray(x[jj]),device="cuda") for s,x in reps.items()}
                    live["pred_brightness_live"]=torch.tensor(pred[jj],device="cuda")
                    result=model(live)
                    record={"row_id":ix.row_id.iloc[jj].to_numpy(),"pred":pred[jj]}
                    for key,val in result.items():
                        if val.ndim==1 and len(val)==len(jj):record[key]=val.cpu().numpy()
                    records.append(pd.DataFrame(record))
            df=confidence(pd.concat(records,ignore_index=True))
            # Write frozen scores before opening the corresponding fluorescence labels.
            df.to_csv(out/f"{role}_{mode}_scores.csv",index=False);scores[mode]=df
        labels=pd.read_csv(root/f"data/{role}.csv").set_index("row_id")
        main=scores["risk"];y=labels.loc[main.row_id].brightness.to_numpy();p=main.pred.to_numpy()
        assert np.isfinite(y).all()
        rankings={"prediction":p,"confidence_full":main.confidence_score.to_numpy(),
                  "confidence_without_final_risk":main.confidence_without_final_risk.to_numpy(),
                  "low_risk":main.conf_lowrisk.to_numpy(),
                  "rule_gate_full_confidence":scores["rule"].confidence_score.to_numpy(),
                  "single_view_full_confidence":scores["single"].confidence_score.to_numpy(),
                  "calibrated_risk_full_confidence":scores["calibrated"].confidence_score.to_numpy()}
        for seed in [20261013,20261014,20261015]:
            rankings[f"random_seed{seed}"]=np.random.default_rng(seed).random(len(y))
        records=[]
        for name,rank in rankings.items():
            for k in [6,30,100]:records.append({"method":name,**selection(y,rank,k)})
        pd.DataFrame(records).to_csv(out/f"{role}_selection_metrics.csv",index=False)
        risk=main.terrain_selected_risk_wmean.to_numpy()+.5*main.terrain_selected_pred_gap_wmean.to_numpy()
        err=np.abs(y-p);bad=((p>=4.2)&(y<4.2)).astype(int)
        det={"risk_error_spearman":float(spearmanr(risk,err).statistic),"n":len(y),
             "overprediction_failure_definition":"pred >= 4.2 and true < 4.2",
             "failure_prevalence":float(bad.mean()),
             "failure_auroc":float(roc_auc_score(bad,risk)) if len(np.unique(bad))==2 else None,
             "failure_auprc":float(average_precision_score(bad,risk)) if bad.sum() else None}
        # Adjust for predicted brightness by reporting within-stratum risk/error association.
        bins=pd.qcut(pd.Series(p).rank(method="first"),10,labels=False);strata=[]
        for b in range(10):
            ii=np.flatnonzero(bins.eq(b))
            strata.append({"prediction_decile":b,"n":len(ii),"risk_error_spearman":float(spearmanr(risk[ii],err[ii]).statistic)})
        pd.DataFrame(strata).to_csv(out/f"{role}_risk_by_prediction_decile.csv",index=False)
        retention=[];order=np.argsort(risk,kind="stable")
        for fraction in np.arange(.1,1.01,.1):
            jj=order[:max(1,int(len(y)*fraction))]
            retention.append({"retained_fraction":fraction,"n":len(jj),"rmse":float(np.sqrt(np.mean((y[jj]-p[jj])**2))),
                              "mae":float(err[jj].mean())})
        pd.DataFrame(retention).to_csv(out/f"{role}_risk_retention.csv",index=False)
        write_json(out/f"{role}_risk_detection.json",det)
        # Paired resampling of smaller candidate libraries; no duplicate protein inside a library.
        draws=[];rng=np.random.default_rng(20261013)
        for repetition in range(200):
            jj=rng.choice(len(y),min(512,len(y)),replace=False)
            local_rankings={name:rank[jj] for name,rank in rankings.items()}
            for mode,raw in scores.items():
                local=confidence(raw.iloc[jj])
                if mode=="risk":
                    local_rankings.update(confidence_full=local.confidence_score.to_numpy(),
                        confidence_without_final_risk=local.confidence_without_final_risk.to_numpy(),
                        low_risk=local.conf_lowrisk.to_numpy())
                else:
                    name={"rule":"rule_gate_full_confidence","single":"single_view_full_confidence",
                          "calibrated":"calibrated_risk_full_confidence"}[mode]
                    local_rankings[name]=local.confidence_score.to_numpy()
            for name,rank in local_rankings.items():
                for k in [6,30]:draws.append({"library":repetition,"method":name,**selection(y[jj],rank,k)})
        pd.DataFrame(draws).to_csv(out/f"{role}_paired_candidate_libraries.csv",index=False)
        print(json.dumps({"partition":role,"selections":records,"risk":det}),flush=True)
    write_json(out/"DONE.json",{"run":a.run,"best_source_selected_on_calibration":best_source,
        "terrain_memory":"train only","quality_selection":"calibration only","confidence":"original v13 weights",
        "candidate_normalization":"min-max on each unlabeled candidate pool","topk_sources":5,
        "calibration_grid":"9 risk/top-k configurations, selected on calibration Top-30 mean before test",
        "free_generation":False,"evaluation":"measured held-out candidate screening"})
if __name__=="__main__":main()

