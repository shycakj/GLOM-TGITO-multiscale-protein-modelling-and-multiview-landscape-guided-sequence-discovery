"""Official ProteinTTT 2026 on the PPLU reference, compared with its unadapted ESM2-650M."""
from pathlib import Path
import sys,argparse,json,copy,time,hashlib,os
os.environ.setdefault("OMP_NUM_THREADS","3")
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr,pearsonr
from experiment_core import seed_all,regression_metrics,write_json
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--steps",type=int,default=30);ap.add_argument("--smoke",action="store_true");a=ap.parse_args()
    root=a.root;sys.path.insert(0,str(root/"third_party/ProteinTTT"))
    import esm
    from proteinttt.models.esm2 import ESM2TTT,DEFAULT_ESM2_650M_TTT_CFG
    torch.set_num_threads(3);seed_all(20261013)
    out=root/"artifacts/proteinttt_650m";out.mkdir(exist_ok=True)
    if a.smoke:out=out/"smoke";out.mkdir(exist_ok=True)
    path=root/"public_checkpoints/esm2_t33_650M_UR50D.pt"
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    model,alphabet=esm.pretrained.load_model_and_alphabet_core("esm2_t33_650M_UR50D",checkpoint,None)
    del checkpoint;model=model.cuda().eval()
    wt=(root/"repro_project/data/processed/reference_sequence.txt").read_text().strip()
    convert=alphabet.get_batch_converter()
    tokens=convert([("WT",wt)])[2].cuda()
    d=pd.read_csv(root/"data/unique_sequences.csv",usecols=["row_id","sequence","partition"])
    ids=np.asarray([[alphabet.get_idx(c) for c in s] for s in d.sequence],dtype="int64")
    wtids=np.asarray([alphabet.get_idx(c) for c in wt],dtype="int64")
    def matrices(m):
        m.eval()
        # fair-esm caches rotary tensors; no_grad keeps them usable by the subsequent TTT backward pass.
        with torch.no_grad():
            native=torch.log_softmax(m(tokens)["logits"].float(),-1)[0,1:222].cpu().numpy()
            masked=np.empty_like(native)
            for start in range(0,221,4):
                pos=torch.arange(start,min(start+4,221),device="cuda")
                batch=tokens.repeat(len(pos),1);batch[torch.arange(len(pos),device="cuda"),pos+1]=alphabet.mask_idx
                lp=torch.log_softmax(m(batch)["logits"].float(),-1)
                masked[start:start+len(pos)]=lp[torch.arange(len(pos),device="cuda"),pos+1].cpu().numpy()
        return {"wt_marginals":native,"masked_marginals":masked}
    metrics=[]
    def evaluate(tag,tables):
        for kind,mat in tables.items():
            score=(mat[np.arange(221)[None,:],ids]-mat[np.arange(221),wtids][None,:]).sum(1)
            pd.DataFrame({"row_id":d.row_id,"score":score,"partition":d.partition}).to_csv(out/f"{tag}_{kind}_raw_scores.csv",index=False)
            tr=pd.read_csv(root/"data/train.csv").set_index("row_id")
            train=d.partition.eq("train").to_numpy()
            fit=LinearRegression().fit(score[train,None],tr.loc[d.row_id[train]].brightness.to_numpy())
            for role in ["validation","calibration","test"]:
                use=d.partition.eq(role).to_numpy();dd=pd.read_csv(root/f"data/{role}.csv").set_index("row_id")
                y=dd.loc[d.row_id[use]].brightness.to_numpy();p=fit.predict(score[use,None])
                row={"method":tag,"scoring":kind,"partition":role,"raw_spearman":float(spearmanr(y,score[use]).statistic),
                     "raw_pearson":float(pearsonr(y,score[use]).statistic),"calibration_slope":float(fit.coef_[0]),
                     "calibration_intercept":float(fit.intercept_),**regression_metrics(y,p)}
                metrics.append(row)
                pd.DataFrame({"row_id":d.row_id[use],"true":y,"pred":p,"raw_score":score[use]}).to_csv(out/f"{tag}_{kind}_{role}_predictions.csv",index=False)
            np.save(out/f"{tag}_{kind}_logprobs.npy",mat)
        pd.DataFrame(metrics).to_csv(out/"metrics.csv",index=False)
    evaluate("unadapted",matrices(model))
    cfg=copy.deepcopy(DEFAULT_ESM2_650M_TTT_CFG)
    cfg.steps=1 if a.smoke else a.steps
    cfg.msa=False
    wrapper=ESM2TTT.ttt_from_pretrained(model,ttt_cfg=cfg)
    for seed in ([20261013] if a.smoke else [20261013,20261014,20261015]):
        seed_all(seed);wrapper.ttt_cfg.seed=seed;begin=time.monotonic()
        wrapper.ttt(wt)
        evaluate(f"proteinttt_seed{seed}",matrices(wrapper))
        write_json(out/f"seed{seed}_runtime.json",{"seconds":time.monotonic()-begin,"peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30})
        wrapper.ttt_reset()
    import subprocess,dataclasses
    write_json(out/"DONE.json",{"method":"ProteinTTT, official ESM2-650M configuration",
        "source_commit":(subprocess.check_output(["git","-C",str(root/"third_party/ProteinTTT"),"rev-parse","HEAD"],text=True).strip()
            if (root/"third_party/ProteinTTT/.git").exists() else "f14ae176422cf57a80215108b437fcb0d9985e8e"),
        "cfg":{k:str(v) if isinstance(v,Path) else v for k,v in dataclasses.asdict(cfg).items()},
        "adaptation_unit":"one reference sequence for the DMS assay, as official ProteinGym-ProteinTTT pipeline",
        "fluorescence_labels_in_TTT":False,"msa":False,
        "error_metric_scale":"affine mapping fitted exclusively on training fluorescence; raw correlations also reported",
        "precision":"float32","checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    print("PROTEINTTT_COMPLETE",flush=True)
if __name__=="__main__":main()

