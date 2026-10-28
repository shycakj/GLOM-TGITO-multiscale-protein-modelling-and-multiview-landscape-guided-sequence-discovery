"""Collect completed, non-smoke experiments into one report without selecting winners."""
from pathlib import Path
import json,time
import pandas as pd
ROOT=Path(__file__).resolve().parent
out=ROOT/"reports";out.mkdir(exist_ok=True)
from evaluate_submission_policy import evaluate as evaluate_submission_policy
evaluate_submission_policy(ROOT)
rows=[]
for p in sorted((ROOT/"artifacts/models").glob("*/*")):
    if not p.is_dir() or "smoke" in p.name or not (p/"DONE.json").exists():continue
    done=json.loads((p/"DONE.json").read_text())
    if (p/"metrics.csv").exists():r=pd.read_csv(p/"metrics.csv").to_dict("records")
    elif (p/"evaluation.json").exists():r=json.loads((p/"evaluation.json").read_text())
    else:continue
    for x in r:
        if x["partition"]!="test":continue
        method=p.name.rsplit("_seed",1)[0]
        rows.append({"run":p.name,"method":method,"split":p.parent.name,"source":str(p.relative_to(ROOT)),
                     "seed":int(p.name.rsplit("_seed",1)[1]),**x})
for split in ["random","depth"]:
    directory=ROOT/"artifacts/baselines"
    if split=="depth":directory=directory/split
    for base in directory.glob("*_v2_metrics.csv"):
        for x in pd.read_csv(base).to_dict("records"):
            if x.get("partition")!="test":continue
            rows.append({"run":x["method"],"method":x["method"],"split":split,"seed":20261013,
                         "source":str(base.relative_to(ROOT)),**x})
# The official TTT comparison is a separate learning setting: sequence-only adaptation,
# followed by a train-label affine scale map for regression errors.
base=ROOT/"artifacts/proteinttt_650m/metrics.csv"
if base.exists():
    for x in pd.read_csv(base).to_dict("records"):
        if x.get("partition")!="test":continue
        tag=x.pop("method");kind=x.pop("scoring")
        seed=int(tag.rsplit("_seed",1)[1]) if "_seed" in tag else 20261013
        method=("proteinttt_650m" if "_seed" in tag else "esm2_650m_unadapted")+"_"+kind
        rows.append({"run":tag+"_"+kind,"method":method,"split":"random","seed":seed,
                     "setting":"sequence-only TTT; train-label affine score mapping",
                     "source":str(base.relative_to(ROOT)),**x})
if rows:
    d=pd.DataFrame(rows);d.to_csv(out/"completed_regression_runs.csv",index=False)
    metrics=["spearman","pearson","rmse","mae","r2","top6_true_mean","top30_true_mean"]
    summary=d.groupby(["split","method"])[metrics].agg(["mean","std","count"])
    summary.to_csv(out/"regression_mean_sd.csv")
    print(d[["split","method","seed",*metrics]].to_string(index=False))
required=[]
for seed in [20261013,20261014,20261015]:
    for method in ["glom","global_only","no_message","hp_transformer"]:
        required.append(f"artifacts/models/random/{method}_seed{seed}/evaluation.json")
    required.extend([f"artifacts/models/depth/glom_seed{seed}/evaluation.json",
                     f"artifacts/models/random/esm2_150m_lora_seed{seed}/DONE.json"])
    for split in ["random","depth"]:
        for method in ["esmc_600m_mlp","esm2_650m_mlp","esmc_6b_mlp"]:
            required.append(f"artifacts/models/{split}/{method}_seed{seed}/DONE.json")
required.append("artifacts/proteinttt_650m/DONE.json")
base="artifacts/models/random/glom_seed20261013/"
required.extend(base+name+"/DONE.json" for name in ["representation_analysis","risk_screening","risk_calibration","local_measured_libraries","search"])
required.append(base+"submission_policy/search_selection_DONE.json")
required.append(base+"search_recent/DONE.json")
required.append(base+"discovery_evaluation/DONE.json")
if (ROOT/base/'wide_search/PLAN_FROZEN.json').exists() and not (ROOT/'audit/DISCOVERY_REVIEW_HOLD.json').exists():
    required.extend([base+'wide_search/DONE.json',base+'wide_evaluation/prefix_900/DONE.json',
                     base+'wide_evaluation/prefix_180_ablations/DONE.json'])
if (ROOT/base/'full_discovery_v2/PROTOCOL.json').exists():
    required.extend([base+'full_discovery_v2/DONE.json',base+'full_discovery_v2/prefix_900/CROSS_SCORE_DONE.json',
                     base+'full_discovery_v2/prefix_900/STRUCTURE_DONE.json','audit/structure_validation/ASSAY_VALIDATION_DONE.json'])
pending=[p for p in required if not (ROOT/p).exists()]
status={"collected_unix":time.time(),"completed_test_runs":len(rows),"records":rows,
        "all_campaigns_finished":not pending,"pending_artifacts":pending,
        "wetlab_status":"six submitted candidates; measurements pending per user clarification 2026-10-13"}
status['discovery_progress']={
    'original_search_runs_completed':len(list((ROOT/base/'search').glob('start*/DONE.json'))),
    'recent_baseline_runs_completed':len(list((ROOT/base/'search_recent').glob('start*/DONE.json'))),
    'planned_total_search_runs':252,
    'independent_evaluation_complete':(ROOT/base/'discovery_evaluation/DONE.json').exists()}
status['discovery_progress']['interpretation']='12-start pilot; external scores auxiliary only'
if (ROOT/base/'wide_search/PLAN_FROZEN.json').exists():
    status['wide_discovery_plan']=json.loads((ROOT/base/'wide_search/PLAN_FROZEN.json').read_text())
    p=ROOT/base/'wide_search/PROGRESS.json'
    status['wide_discovery_progress']=json.loads(p.read_text()) if p.exists() else {'state':'plan frozen; launch audit pending'}
    hold=ROOT/'audit/DISCOVERY_REVIEW_HOLD.json'
    if hold.exists():
        status['wide_discovery_progress']['state']='held_for_design_review'
        status['wide_discovery_progress']['hold']=json.loads(hold.read_text())
    status['wide_discovery_progress']['interpretation']='Search-engine comparison conditional on shared multi-view starts and final scoring. Exact matches are ancillary rediscovery; unknown generated fitness is missing, never a failure.'
new=ROOT/base/'full_discovery_v2'
if (new/'PROTOCOL.json').exists():
    status['full_discovery_v2_protocol']=json.loads((new/'PROTOCOL.json').read_text())
    status['full_discovery_v2_progress']=json.loads((new/'PROGRESS.json').read_text()) if (new/'PROGRESS.json').exists() else {'state':'prepared'}
    status['full_discovery_v2_readouts']={}
    for folder in sorted(new.glob('prefix_*')):
        if (folder/'FROZEN.json').exists():
            status['full_discovery_v2_readouts'][folder.name]={
                'frozen':json.loads((folder/'FROZEN.json').read_text()),
                'cross_score_complete':(folder/'CROSS_SCORE_DONE.json').exists(),
                'structure_score_complete':(folder/'STRUCTURE_DONE.json').exists()}
(out/"rolling_results.json").write_text(json.dumps(status,indent=2))

