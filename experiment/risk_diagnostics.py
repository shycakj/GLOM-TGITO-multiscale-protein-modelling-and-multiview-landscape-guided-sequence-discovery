"""Diagnostics of frozen rankings, including unfavorable candidate-library results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

def write_diagnostics(root):
    root=Path(root);base=root/"artifacts/models/random/glom_seed20261013/risk_screening"
    if not (base/"DONE.json").exists():return
    if not all((base/f"{p}_{m}_scores.csv").exists() for p in ["calibration","test"] for m in ["risk","calibrated","rule","single"]):return
    records=[];paired=[]
    for partition in ["calibration","test"]:
        labels=pd.read_csv(root/f"data/{partition}.csv").set_index("row_id")
        for mode in ["risk","calibrated","rule","single"]:
            d=pd.read_csv(base/f"{partition}_{mode}_scores.csv")
            y=labels.loc[d.row_id].brightness.to_numpy();p=d.pred.to_numpy()
            risk=d.terrain_selected_risk_wmean.to_numpy()+.5*d.terrain_selected_pred_gap_wmean.to_numpy()
            for threshold in [4.2,4.35]:
                selected=p>=threshold;bad=(y[selected]<threshold).astype(int)
                records.append({"partition":partition,"policy":mode,"brightness_threshold":threshold,
                    "n_predicted_above_threshold":int(selected.sum()),"failures":int(bad.sum()),
                    "failure_prevalence":float(bad.mean()) if len(bad) else None,
                    "risk_auroc":float(roc_auc_score(bad,risk[selected])) if len(np.unique(bad))==2 else None,
                    "risk_auprc":float(average_precision_score(bad,risk[selected])) if bad.sum() else None})
        libraries=pd.read_csv(base/f"{partition}_paired_candidate_libraries.csv")
        for k in [6,30]:
            x=libraries[libraries.k.eq(k)].pivot(index="library",columns="method",values="true_mean")
            for method in x.columns:
                if method=="prediction":continue
                diff=x[method]-x.prediction
                paired.append({"partition":partition,"method":method,"k":k,"libraries":len(diff),
                    "mean_brightness_difference":float(diff.mean()),"median_brightness_difference":float(diff.median()),
                    "q25_difference":float(diff.quantile(.25)),"q75_difference":float(diff.quantile(.75)),
                    "fraction_improved":float((diff>0).mean()),"fraction_tied":float(np.isclose(diff,0,atol=1e-12).mean())})
    pd.DataFrame(records).to_csv(base/"predicted_bright_risk_diagnostics.csv",index=False)
    pd.DataFrame(paired).to_csv(base/"paired_library_robustness_summary.csv",index=False)
    (base/"diagnostic_protocol.json").write_text(json.dumps({
        "analysis":"diagnostic assessment of already frozen policies; no refitting or selection",
        "conditional_domain":"predicted brightness at least the fixed v13.5 thresholds 4.2 and 4.35",
        "failure":"measured brightness below the same threshold",
        "library_intervals":"interquartile spread across overlapping candidate subsets, not confidence intervals",
        "all_candidate_libraries_reported":True},indent=2))

if __name__=="__main__":write_diagnostics(Path(__file__).resolve().parent)
