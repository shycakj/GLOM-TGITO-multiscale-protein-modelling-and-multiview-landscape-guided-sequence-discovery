"""Paired sequence bootstrap for prespecified seed 20261013; no best-test seed selection."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr

def metrics(y,p):
    err=y-p
    return np.array([spearmanr(y,p).statistic,pearsonr(y,p).statistic,
        np.sqrt(np.mean(err**2)),np.mean(np.abs(err)),1-np.sum(err**2)/np.sum((y-y.mean())**2)])

def read_predictions(path):
    d=pd.read_csv(path)
    d=d.rename(columns={"brightness":"true","prediction":"pred"})
    assert d.row_id.is_unique and {"row_id","true","pred"}.issubset(d.columns)
    return d[["row_id","true","pred"]].sort_values("row_id").set_index("row_id")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--replicates",type=int,default=500);a=ap.parse_args();root=a.root
    output=root/"reports";output.mkdir(exist_ok=True);rows=[]
    for split in ["random","depth"]:
        base=root/"artifacts/models"/split
        reference=base/"glom_seed20261013/test_predictions.csv"
        if not reference.exists():continue
        truth=read_predictions(reference);y=truth.true.to_numpy();g=truth.pred.to_numpy()
        baselines={p.parent.name.rsplit("_seed",1)[0]:p for p in base.glob("*_seed20261013/test_predictions.csv")}
        controls=root/"artifacts/baselines"
        if split=="depth":controls=controls/"depth"
        for p in controls.glob("*_test_predictions.csv"):
            name=p.name.removesuffix("_test_predictions.csv")
            if name in ["onehot_ridge","sequence_knn_hamming","esm_mean_ridge"]:baselines[name]=p
        for name,path in baselines.items():
            if name=="glom":continue
            other=read_predictions(path)
            assert other.index.equals(truth.index),(name,split,"row mismatch")
            assert np.allclose(other.true,y,atol=1e-6,rtol=0),(name,split,"label mismatch")
            p=other.pred.to_numpy();observed=metrics(y,g)-metrics(y,p)
            rng=np.random.default_rng(20261013);delta=[]
            for _ in range(a.replicates):
                ix=rng.integers(0,len(y),len(y))
                delta.append(metrics(y[ix],g[ix])-metrics(y[ix],p[ix]))
            lo,hi=np.quantile(delta,[.025,.975],axis=0)
            for i,key in enumerate(["spearman","pearson","rmse","mae","r2"]):
                rows.append({"split":split,"reference":"glom","comparison":name,"seed":20261013,
                    "metric":key,"difference_glom_minus_comparison":observed[i],"ci95_low":lo[i],"ci95_high":hi[i],
                    "favors_glom":"negative" if key in ["rmse","mae"] else "positive","n_sequences":len(y),
                    "bootstrap_replicates":a.replicates,"interval":"paired sequence percentile bootstrap"})
    if rows:pd.DataFrame(rows).to_csv(output/"paired_glom_comparisons.csv",index=False)
    (output/"paired_statistics_protocol.json").write_text(json.dumps({"records":len(rows),
        "prespecified_seed":20261013,"bootstrap_seed":20261013,"replicates":a.replicates,
        "interpretation":"uncertainty over held-out sequences for fixed fitted models; seed SD reported separately",
        "multiple_comparisons":"descriptive intervals; no familywise significance claim",
        "top_k":"candidate libraries are evaluated separately without replacement"},indent=2))
    print(json.dumps({"paired_comparison_records":len(rows)}))
if __name__=="__main__":main()
