"""Choose TGITO risk hyperparameters on a separate measured calibration library."""
from pathlib import Path
import json,hashlib,dataclasses
import numpy as np
import pandas as pd
import torch
from experiment_core import write_json

def score_features(model,reps,rows,pred,index,batch_size=64):
    records=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            jj=rows[start:start+batch_size]
            live={s:torch.as_tensor(np.array(x[jj]),device="cuda") for s,x in reps.items()}
            live["pred_brightness_live"]=torch.as_tensor(pred[jj],device="cuda")
            result=model(live);record={"row_id":index.row_id.iloc[jj].to_numpy(),"pred":pred[jj]}
            for key,val in result.items():
                if val.ndim==1 and len(val)==len(jj):record[key]=val.cpu().numpy()
            records.append(pd.DataFrame(record))
    return pd.concat(records,ignore_index=True)

def calibrate(root,base,bank,index,pred,reps,batch_size=64):
    from pipeline.tgito.terrain_potential import TerrainPotential,TerrainPotentialConfig
    from evaluate_risk_screening import confidence,selection
    out=base/"risk_calibration";out.mkdir(exist_ok=True);frozen=out/"FROZEN_CONFIG.json"
    if frozen.exists():return TerrainPotentialConfig(**json.loads(frozen.read_text())["terrain_config"])
    rows=np.flatnonzero(index.partition.eq("calibration"));summary=[];configs={}
    # Written before any grid scores or calibration labels are read.
    write_json(out/"grid_protocol.json",{"topk_sources":[3,5,8],"terrain_risk_weight":[.325,.65,1.3],
        "selection":"calibration Top-30 measured mean; tie-break Top-6 mean, then smaller config ID",
        "confidence_weights":"unchanged v13.5", "test_labels_read":False})
    for topk in [3,5,8]:
        for weight in [.325,.65,1.3]:
            key=f"top{topk}_risk{weight:g}"
            cfg=TerrainPotentialConfig(terrain_gate_mode="risk",topk_sources=topk,terrain_risk_weight=weight)
            configs[key]=cfg;model=TerrainPotential(bank,cfg)
            scores=confidence(score_features(model,reps,rows,pred,index,batch_size))
            scores.to_csv(out/(key+"_calibration_scores.csv"),index=False)
            labels=pd.read_csv(root/"data/calibration.csv").set_index("row_id").loc[scores.row_id].brightness.to_numpy()
            m30=selection(labels,scores.confidence_score,30);m6=selection(labels,scores.confidence_score,6)
            summary.append({"config_id":key,"topk_sources":topk,"terrain_risk_weight":weight,
                "top30_true_mean":m30["true_mean"],"top30_true_min":m30["true_min"],"top30_high435":m30["high435"],
                "top6_true_mean":m6["true_mean"],"top6_true_min":m6["true_min"],"top6_high435":m6["high435"]})
            print(json.dumps({"risk_calibration":summary[-1]}),flush=True)
    grid=pd.DataFrame(summary);grid.to_csv(out/"calibration_grid.csv",index=False)
    selected=grid.sort_values(["top30_true_mean","top6_true_mean","config_id"],ascending=[False,False,True]).iloc[0]
    cfg=configs[selected.config_id]
    write_json(frozen,{"selected_id":selected.config_id,"terrain_config":dataclasses.asdict(cfg),
        "criterion":"calibration Top-30 mean, then Top-6 mean","calibration_n":len(rows),
        "model_sha256":hashlib.sha256((base/"best.pt").read_bytes()).hexdigest(),"test_labels_read":False,
        "fixed_reference":"topk=5, risk_weight=0.65 remains reported separately"})
    write_json(out/"DONE.json",{"configs_evaluated":len(grid),"selected_id":selected.config_id})
    return cfg
