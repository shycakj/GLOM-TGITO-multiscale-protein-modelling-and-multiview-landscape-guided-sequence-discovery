"""Measured, held-out two-substitution libraries around the prespecified search starts."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from run_search_comparison import make_plan
from evaluate_risk_screening import selection,confidence
from experiment_core import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--run",default="glom_seed20261013");a=ap.parse_args();root=a.root
    base=root/"artifacts/models/random"/a.run;out=base/"local_measured_libraries";out.mkdir(exist_ok=True)
    if (out/"DONE.json").exists():return
    assert (base/"risk_screening/DONE.json").exists()
    # Starts use only training labels and representations, identical to native search.
    starts=make_plan(root,a.run)
    candidates=pd.read_csv(root/"data/test.csv",usecols=["row_id","sequence"])
    scores=pd.read_csv(base/"risk_screening/test_risk_scores.csv").set_index("row_id").loc[candidates.row_id]
    rankings={"glom_prediction":scores.pred.to_numpy(),"glom_full_confidence":scores.confidence_score.to_numpy(),
        "glom_without_final_risk":scores.confidence_without_final_risk.to_numpy(),"glom_low_risk":scores.conf_lowrisk.to_numpy()}
    raw_terrain={"risk":scores}
    for mode in ["rule","single","calibrated"]:
        x=pd.read_csv(base/f"risk_screening/test_{mode}_scores.csv").set_index("row_id").loc[candidates.row_id]
        raw_terrain[mode]=x
        rankings["glom_"+mode+"_confidence"]=x.confidence_score.to_numpy()
    for method in ["esmc_600m_mlp","esm2_650m_mlp","esmc_6b_mlp","esm2_150m_lora"]:
        p=root/"artifacts/models/random"/(method+"_seed20261013")/"test_predictions.csv"
        if p.exists():
            x=pd.read_csv(p).set_index("row_id").loc[candidates.row_id]
            rankings[method]=x["prediction" if "prediction" in x else "pred"].to_numpy()
    for seed in [20261013,20261014,20261015]:rankings[f"random_seed{seed}"]=np.random.default_rng(seed).random(len(candidates))
    # Distance/proposal membership uses sequence only; freeze before reading held-out fluorescence.
    chars=np.asarray([list(s) for s in candidates.sequence],dtype="U1")
    memberships=[];library_indices=[]
    for start in starts.itertuples():
        dist=np.count_nonzero(chars!=np.asarray(list(start.sequence))[None,:],axis=1)
        indices=np.flatnonzero((dist>=1)&(dist<=2));library_indices.append((start,indices))
        for j in indices:memberships.append({"seed_row":int(start.row_id),"candidate_row":int(candidates.row_id.iloc[j]),"distance":int(dist[j])})
    pd.DataFrame(memberships,columns=["seed_row","candidate_row","distance"]).to_csv(out/"candidate_membership.csv",index=False)
    # All rankings and candidate membership are now frozen.
    truth=pd.read_csv(root/"data/test.csv").set_index("row_id").loc[candidates.row_id].brightness.to_numpy()
    rows=[];coverage=[];chosen=[]
    for start,idx in library_indices:
        coverage.append({"seed_row":int(start.row_id),"cluster":int(start.cluster),"role":start.role,
            "seed_brightness":float(start.brightness),"reachable_test_candidates":len(idx),
            "eligible_top6":len(idx)>=6,"eligible_top30":len(idx)>=30})
        if not len(idx):continue
        local_rankings={name:ranking[idx] for name,ranking in rankings.items()}
        for mode,raw in raw_terrain.items():
            rescored=confidence(raw.iloc[idx])
            if mode=="risk":
                local_rankings.update(glom_full_confidence=rescored.confidence_score.to_numpy(),
                    glom_without_final_risk=rescored.confidence_without_final_risk.to_numpy(),
                    glom_low_risk=rescored.conf_lowrisk.to_numpy())
            else:local_rankings["glom_"+mode+"_confidence"]=rescored.confidence_score.to_numpy()
        for method,ranking in local_rankings.items():
            for k in [6,30]:
                if len(idx)<k:continue
                met=selection(truth[idx],ranking,k)
                rows.append({"seed_row":int(start.row_id),"cluster":int(start.cluster),"role":start.role,
                    "method":method,"reachable_test_candidates":len(idx),"seed_brightness":float(start.brightness),
                    "true_mean_gain_vs_seed":met["true_mean"]-float(start.brightness),**met})
                order=idx[np.argsort(-ranking,kind="stable")[:k]]
                for rank,j in enumerate(order,1):chosen.append({"seed_row":int(start.row_id),"method":method,"k":k,
                    "rank":rank,"candidate_row":int(candidates.row_id.iloc[j]),"true_brightness":float(truth[j])})
    pd.DataFrame(coverage).to_csv(out/"start_coverage.csv",index=False)
    if rows:
        d=pd.DataFrame(rows);d.to_csv(out/"selection_by_start.csv",index=False)
        d.groupby(["method","k"])[["true_mean","true_min","high420","high435","true_mean_gain_vs_seed"]].agg(["mean","std","count"]).to_csv(out/"selection_summary.csv")
        pd.DataFrame(chosen).to_csv(out/"selected_candidates.csv",index=False)
    write_json(out/"DONE.json",{"evaluation":"measured hidden-library screening within two substitutions of training starts",
        "starts":len(starts),"candidate_partition":"test","rankings":list(rankings),"max_hamming_from_start":2,
        "candidate_memberships":len(memberships),"selection_records":len(rows),"labels_used_for_start_selection":"train only",
        "confidence_normalization":"each unlabeled local library, before selection",
        "reporting":"per-start results and eligible-start counts; start libraries may overlap; not free-generation output"})
    print(json.dumps({"local_library_records":len(rows),"start_coverage":coverage}))
if __name__=="__main__":main()
