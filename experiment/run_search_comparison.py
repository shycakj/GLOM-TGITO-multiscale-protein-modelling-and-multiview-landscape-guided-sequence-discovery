"""Current TGITO search and controlled proposal baselines on paired starting sequences."""
from pathlib import Path
import argparse,json,os,time,sys,types,importlib,dataclasses,copy,random
os.environ.setdefault("OMP_NUM_THREADS","3")
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from experiment_core import setup_project,seed_all,write_json
from evaluate_risk_screening import confidence
def load_adalead(root):
    # Import the original FLEXS algorithm without optional TF/RNA landscape packages.
    code=root/"third_party/FLEXS/flexs"
    pkg=types.ModuleType("flexs");pkg.__path__=[str(code)];sys.modules["flexs"]=pkg
    pkg.Landscape=importlib.import_module("flexs.landscape").Landscape
    pkg.Model=importlib.import_module("flexs.model").Model
    pkg.Explorer=importlib.import_module("flexs.explorer").Explorer
    for mod,path in [("flexs.utils",code/"utils"),("flexs.baselines",code/"baselines"),
                     ("flexs.baselines.explorers",code/"baselines/explorers")]:
        p=types.ModuleType(mod);p.__path__=[str(path)];sys.modules[mod]=p
    util=importlib.import_module("flexs.utils.sequence_utils")
    return importlib.import_module("flexs.baselines.explorers.adalead").Adalead,util
def make_plan(root,run):
    base=root/"artifacts/models/random"/run
    f=base/"features";ix=pd.read_csv(f/"row_index.csv")
    tr=pd.read_csv(root/"data/train.csv").set_index("row_id")
    ti=np.flatnonzero(ix.partition.eq("train"));frame=tr.loc[ix.row_id.iloc[ti]].reset_index()
    h=np.load(f/"h32.npy")[ti];high=frame.brightness.ge(4.2).to_numpy()
    frame=frame.loc[high].reset_index(drop=True);h=h[high]
    cluster=KMeans(n_clusters=4,n_init=10,random_state=20261013).fit_predict(h)
    rows=[]
    for c in range(4):
        group=frame.loc[cluster==c].sort_values(["brightness","row_id"])
        for role,q in [("lower",.1),("middle",.5),("upper",.9)]:
            row=group.iloc[min(len(group)-1,int(q*(len(group)-1)))].to_dict()
            rows.append({**row,"cluster":c,"role":role})
    out=base/"search";out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/"starting_sequences.csv",index=False)
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--run",default="glom_seed20261013");ap.add_argument("--starts",type=int,default=12)
    ap.add_argument("--steps",type=int,default=70);ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--wide",action="store_true");ap.add_argument("--shard",type=int,default=0)
    ap.add_argument("--shards",type=int,default=1);ap.add_argument("--max-contexts",type=int,default=0)
    ap.add_argument("--ablations-only",action="store_true")
    a=ap.parse_args();root=a.root.resolve();setup_project(root);torch.set_num_threads(3)
    os.environ["PPLU_ESM_MODEL_PATH"]=str(root/"repro_project/assets/esm2_t30_150M_UR50D")
    from pipeline.tgito.esm_bridge import load_bridge
    from encoder_contract import TrainingCompatibleEncoder
    from pipeline.tgito.revised_slim_internal import load_revised_slim
    from pipeline.tgito.terrain_memory import TerrainMemoryBank
    from pipeline.tgito.terrain_potential import TerrainPotential,TerrainPotentialConfig
    from pipeline.tgito.optimizer import TGITOGenerator,GeneratorConfig,hamming
    from pipeline.tgito.aa import AA20,seq_to_ids0,ids0_to_seq
    base=root/"artifacts/models/random"/a.run;out=base/("wide_search" if a.wide else "search_pilot" if a.pilot else "search")
    out.mkdir(exist_ok=True);assert (base/"terrain_train_only/DONE.json").exists()
    plan=pd.read_csv(out/'starting_sequences.csv') if a.wide else make_plan(root,a.run).head(1 if a.pilot else a.starts)
    if a.wide:
        from build_wide_peak_plan import sha
        frozen=json.loads((out/'PLAN_FROZEN.json').read_text())
        assert sha(out/'starting_sequences.csv')==frozen['plan_sha256']
        if a.ablations_only:plan=plan[plan.ablation_subset]
        plan=plan[plan.start_id % a.shards == a.shard]
        if a.max_contexts:plan=plan.head(a.max_contexts)
    train=pd.read_csv(root/"data/train.csv");known=set(train.sequence)
    if "smoke" not in a.run:
        for partition in ["validation","calibration"]:
            known.update(pd.read_csv(root/f"data/{partition}.csv",usecols=["sequence"]).sequence)
    wt=(root/"repro_project/data/processed/reference_sequence.txt").read_text().strip()
    model,mean,std=load_revised_slim(base/"best.pt",device="cuda")
    bridge=load_bridge(root/"repro_project/assets/warm_best_model.pt",device="cuda")
    enc=TrainingCompatibleEncoder(bridge,device="cuda",freeze_esm=True,target_h_dim=1280).eval()
    bank=TerrainMemoryBank.from_npz(base/"terrain_train_only/terrain.npz",device="cuda")
    risk_cfg=TerrainPotentialConfig(terrain_gate_mode="risk",topk_sources=5)
    if "smoke" not in a.run:
        frozen=base/"risk_calibration/FROZEN_CONFIG.json"
        while not frozen.exists():time.sleep(15)
        risk_cfg=TerrainPotentialConfig(**json.loads(frozen.read_text())["terrain_config"])
    risk=TerrainPotential(bank,risk_cfg)
    rule=TerrainPotential(bank,dataclasses.replace(risk_cfg,terrain_gate_mode="rule"))
    quality=pd.read_csv(base/"terrain_train_only/source_quality.csv").sort_values("source_quality",ascending=False)
    single=TerrainPotential(bank,dataclasses.replace(risk_cfg,topk_sources=1),sources=[str(quality.iloc[0].source)])
    from fast_proposals import FastProposalMixin
    assert json.loads((root/"audit/fast_proposal_equivalence.json").read_text())["passed"]
    class Recorder(FastProposalMixin,TGITOGenerator):
        def reset(self):
            self.seen=set();self.forward_sequences=0;self.gradient_forward_sequences=0
        def _forward_from_probs(self,probs,*args,**kw):
            ret=super()._forward_from_probs(probs,*args,**kw)
            self.forward_sequences+=len(probs)
            if probs.requires_grad:self.gradient_forward_sequences+=len(probs)
            else:
                hard=bool(((probs==0)|(probs==1)).all())
                if hard:
                    ids=probs.detach().argmax(-1).cpu()
                    self.seen.update(ids0_to_seq(row) for row in ids)
            return ret
    gen=Recorder(enc,model,risk,y_mean=mean,y_std=std,device="cuda");gen.reset()
    from position_features import canonical_bits
    gen.aa_bits=canonical_bits('cuda')
    config=GeneratorConfig(length=221,steps=8 if a.pilot else a.steps,lr=.04,terrain_weight=.03,
        mutation_budget=2,budget_relative="seed",proposal_candidates=64,proposal_order=2,
        seed_explore_weight=.20,seed_explore_min_change=1.,amp_dtype="off",
        accept_by_objective=True,select_by_objective=True,reject_known_output=True,novelty_min_hamming=1)
    Adalead,utils=load_adalead(root);original_mutant=utils.generate_random_mutant
    ref=seq_to_ids0(wt,221,device="cuda");mask=torch.ones(221,dtype=torch.bool,device="cuda")
    def predict_batch(seqs):
        result=[]
        for start in range(0,len(seqs),32):
            ids=torch.stack([seq_to_ids0(s,221,device="cuda") for s in seqs[start:start+32]])
            result.extend(gen._predict_hard_ids_batch(ids,ref,mask,config).detach().float().cpu().numpy())
        return np.asarray(result)
    run_records=[];candidate_records=[]
    seeds=[20261013] if a.pilot else [20261013,20261014,20261015]
    for index,startrow in plan.iterrows():
        anchor=startrow.sequence
        for seed in ([int(startrow.random_seed)] if a.wide else seeds):
            native_budget=None
            methods=(["tgito_no_risk_gate","tgito_no_terrain","tgito_single_view"] if a.ablations_only else
                     ["tgito_risk","random","adalead"]) if a.wide else ["tgito_risk","tgito_no_risk_gate","tgito_no_terrain","tgito_single_view","random","adalead"]
            for method in methods:
                prefix=f"context{int(startrow.start_id):04d}" if a.wide else f"start{int(startrow.row_id)}"
                name=f"{prefix}_seed{seed}_{method}";folder=out/name
                if (folder/"DONE.json").exists():
                    record=json.loads((folder/"DONE.json").read_text());run_records.append(record)
                    if method=="tgito_risk":native_budget=record["forward_sequences"]
                    if not a.wide:
                        cand=pd.read_csv(folder/"candidate_pool.csv");candidate_records.append(cand)
                    continue
                folder.mkdir(exist_ok=True);seed_all(seed);random.seed(seed);rng=np.random.default_rng(seed)
                cfg=dataclasses.replace(config,seed=seed,terrain_weight=0. if method=="tgito_no_terrain" else config.terrain_weight)
                gen.terrain=rule if method=="tgito_no_risk_gate" else single if method=="tgito_single_view" else risk
                method_terrain_config=dataclasses.asdict(gen.terrain.cfg)
                gen.reset();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
                if method.startswith("tgito"):
                    final=gen.run(anchor,"all",folder,cfg,known_sequences=known,reference_seq=wt)
                    pd.DataFrame([final]).to_csv(folder/"native_final.csv",index=False)
                    if method=="tgito_risk":native_budget=gen.forward_sequences
                elif method=="random":
                    seqs=set()
                    while len(seqs)<native_budget:
                        child=list(anchor);n=int(rng.integers(1,3))
                        for pos in rng.choice(221,n,replace=False):
                            child[pos]=rng.choice([c for c in AA20 if c!=anchor[pos]])
                        child="".join(child)
                        if child not in known:seqs.add(child)
                    predict_batch(sorted(seqs))
                else:
                    def bounded_mutant(parent,mu,alphabet):
                        while True:
                            child=original_mutant(parent,mu,alphabet)
                            if 1<=hamming(anchor,child)<=2 and child not in known:return child
                    utils.generate_random_mutant=bounded_mutant
                    class Oracle:
                        cost=0;name="Frozen_GLOM"
                        def get_fitness(self,seqs):
                            n=min(len(seqs),max(0,native_budget-self.cost))
                            values=np.full(len(seqs),-np.inf)
                            if n:values[:n]=predict_batch(list(seqs[:n]));self.cost+=n
                            return values
                        def train(self,*args,**kwargs):pass
                    oracle=Oracle()
                    explorer=Adalead(oracle,rounds=1,sequences_batch_size=32,model_queries_per_batch=native_budget,
                                     starting_sequence=anchor,alphabet=AA20,mu=1,recomb_rate=0,rho=0,threshold=.05,eval_batch_size=16)
                    measured=train[["sequence","brightness"]].rename(columns={"brightness":"true_score"}).copy()
                    measured["true_score"]=-np.inf;measured.loc[measured.sequence.eq(anchor),"true_score"]=float(startrow.brightness)
                    explorer.propose_sequences(measured);utils.generate_random_mutant=original_mutant
                elapsed=time.monotonic()-begin;count=gen.forward_sequences;grad=gen.gradient_forward_sequences
                peak_allocated=torch.cuda.max_memory_allocated()/2**30
                peak_reserved=torch.cuda.max_memory_reserved()/2**30
                candidates=sorted(s for s in gen.seen if s not in known and 1<=hamming(anchor,s)<=2)
                # Common risk scoring is a separately logged post-processing pass for every method.
                gen.terrain=risk;gen.reset();records=[]
                for st in range(0,len(candidates),32):
                    seqs=candidates[st:st+32]
                    ids=torch.stack([seq_to_ids0(s,221,device="cuda") for s in seqs])
                    with torch.inference_mode():
                        p,terr,_,_=gen._forward_from_probs(torch.nn.functional.one_hot(ids,20).float(),seq_to_ids0(anchor,221,"cuda"),ref,mask,config)
                    cols={key:value.detach().float().cpu().numpy() for key,value in terr.items()
                          if value.ndim==1 and len(value)==len(seqs)}
                    for j,s in enumerate(seqs):
                        records.append({"sequence":s,"pred":float(p[j]),"seed_row":int(startrow.row_id),"random_seed":seed,
                            "method":method,"hamming_from_seed":hamming(anchor,s),**{k:float(v[j]) for k,v in cols.items()}})
                cand=pd.DataFrame(records)
                if cand.empty:
                    cand=pd.DataFrame(columns=["sequence","pred","seed_row","random_seed","method","hamming_from_seed"])
                if a.wide:
                    from freeze_discovery_evaluation import COLS
                    cand=cand.reindex(columns=COLS)
                    cand['start_id']=int(startrow.start_id);cand['mountain_id']=startrow.mountain_id;cand['role']=startrow.role
                cand.to_csv(folder/"candidate_pool.csv",index=False)
                if not a.wide:candidate_records.append(cand)
                record={"method":method,"seed_row":int(startrow.row_id),"cluster":int(startrow.cluster),"role":startrow.role,
                        "random_seed":seed,"forward_sequences":count,"gradient_forward_sequences":grad,
                        "common_rescoring_sequences":gen.forward_sequences,"seconds":elapsed,"valid_unique_candidates":len(candidates),
                        "peak_gpu_allocated_gib":peak_allocated,"peak_gpu_reserved_gib":peak_reserved,
                        "matched_random_adalead_forward_budget":native_budget,"config":dataclasses.asdict(cfg),
                        "terrain_config":method_terrain_config}
                if a.wide:
                    record.update(start_id=int(startrow.start_id),mountain_id=startrow.mountain_id,
                                  representative_source=startrow.representative_source)
                    from build_wide_peak_plan import sha
                    record['candidate_pool_sha256']=sha(folder/'candidate_pool.csv')
                    if (folder/'native_final.csv').exists():record['native_final_sha256']=sha(folder/'native_final.csv')
                write_json(folder/"DONE.json",record);run_records.append(record)
                runtime_name=f"runtime_shard{a.shard}_{'ablations' if a.ablations_only else 'core'}.csv" if a.wide else 'runtime_and_queries.csv'
                pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in run_records]).to_csv(out/runtime_name,index=False)
                print(json.dumps({k:v for k,v in record.items() if k not in ["config","terrain_config"]}),flush=True)
    if a.wide:
        write_json(out/f"SHARD_{a.shard}_{'ablations' if a.ablations_only else 'core'}_DONE.json",
                   dict(contexts=len(plan),methods=methods,shards=a.shards,partial=bool(a.max_contexts),
                        truth_during_search=False,per_context_outputs_frozen=True))
        return
    pools=pd.concat(candidate_records,ignore_index=True)
    if not pools.empty:
        # A shared unlabeled union fixes the normalization for between-method ranking.
        pools=confidence(pools);pools.to_csv(out/"all_candidates_common_confidence.csv",index=False)
        pd.DataFrame(run_records).to_csv(out/"runtime_and_queries.csv",index=False)
        truth=pd.read_csv(root/"data/unique_sequences.csv").set_index("sequence")
        pools["measured_brightness"]=pools.sequence.map(truth.brightness)
        pools["measurement_partition"]=pools.sequence.map(truth.partition)
        if "smoke" not in a.run:
            assert pools.loc[pools.measured_brightness.notna(),"measurement_partition"].eq("test").all()
        pools.to_csv(out/"all_candidates_exact_measured_matches.csv",index=False)
        tops=[]
        for method,group in pools.groupby("method"):
            dedup=group.sort_values("confidence_score",ascending=False).drop_duplicates("sequence")
            for rank_by in ["pred","confidence_score"]:
                chosen=dedup.sort_values(rank_by,ascending=False).head(30).copy()
                chosen["ranking"]=rank_by;chosen["rank"]=np.arange(1,len(chosen)+1);tops.append(chosen)
        pd.concat(tops,ignore_index=True).to_csv(out/"top30_by_method_and_ranking.csv",index=False)
    write_json(out/"DONE.json",{"starts":len(plan),"seeds":seeds,"methods":6,"pilot":a.pilot,
        "truth_during_search":False,"budget":"all encoder forward sequences; gradient calls and elapsed time separately reported",
        "known_sequence_partitions":["train","validation","calibration"] if "smoke" not in a.run else ["train"],
        "adalead":"official FLEXS with proposal rejection to enforce same hard two-mutation neighborhood",
        "proposal_implementation":"CPU enumeration after batched GPU top-k; ordered-candidate equivalence audited against native code",
        "confidence_normalization":"shared unlabeled union of all generated pools"})
if __name__=="__main__":main()

