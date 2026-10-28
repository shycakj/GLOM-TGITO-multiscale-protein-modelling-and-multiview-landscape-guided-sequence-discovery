"""Frozen full-pipeline experiment with explicit historical-HPO transfer.

Preflight is a configuration/implementation check, never a test-set HPO contest.
Candidates are frozen before cross-predictor evaluation; unknown assays stay missing.
"""
from pathlib import Path
import argparse,copy,dataclasses,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
from experiment_core import setup_project,seed_all,write_json
from hpo_config_contract import sha,load_confirmed_winner,resolve,validate_effective,verify_input_hashes

def paths(root):return root/'artifacts/models/random/glom_seed20261013',root/'artifacts/discovery_v3'
def load_protocol(root):return json.loads((paths(root)[1]/'PROTOCOL.json').read_text())

def prepare(root):
    from build_wide_peak_plan import build,SOURCES
    base,out=paths(root);out.mkdir(parents=True,exist_ok=True)
    if (out/'PROTOCOL.json').exists():
        verify_input_hashes(root,load_protocol(root)['input_hashes']);return
    historical=root/'repro_project/artifacts/tgito_hpo';hp=load_confirmed_winner(historical)
    ref=json.loads((root/'audit/hpo_configuration_20261014/HPO_TRIAL12_CORRECTED_REFERENCE.json').read_text())
    validate_effective(hp,ref)
    confirmation=json.loads((historical/'confirmation.json').read_text())
    c30=next(x for x in confirmation if x['trial']==30)
    hp30=copy.deepcopy(hp);hp30.update(c30['params']);hp30['hpo_trial']=30
    hp30.update(confirmation_mean=c30['confirmation_mean'],confirmation_std=c30['confirmation_std'])
    r30=resolve(hp30,ref['generator_config'],ref['terrain_config'],overrides=ref['explicit_overrides']);validate_effective(hp30,r30)
    v2=dict(generator_config=json.loads((base/'full_discovery_v2/PROTOCOL.json').read_text())['config'],
            terrain_config=json.loads((base/'risk_calibration/FROZEN_CONFIG.json').read_text())['terrain_config'],execution={'num_restarts':1})
    profiles={'trial12':dict(parameters=hp,resolved=ref),'trial30':dict(parameters=hp30,resolved=r30),'v2_diagnostic':dict(parameters=None,resolved=v2)}
    build(root,900,output=out/'neutral_peak_plan',source_prior=dict.fromkeys(SOURCES,1.0))
    formal=pd.read_csv(out/'neutral_peak_plan/starting_sequences.csv');assert len(formal)==2700
    formal.to_csv(out/'starting_sequences.csv',index=False)
    # Same historical v2 starting contexts isolate parameter changes in the pilot.
    old=pd.read_csv(base/'full_discovery_v2/starting_sequences.csv')
    mids=old.sort_values('mountain_order').drop_duplicates('representative_source').mountain_id
    pilot=old[old.mountain_id.isin(mids)].sort_values('start_id').reset_index(drop=True)
    assert len(pilot)==36 and pilot.representative_source.nunique()==12 and set(pilot.role)=={'summit','ridge','edge'}
    pilot.to_csv(out/'pilot_starting_sequences.csv',index=False)
    # Baselines choose strong observed training anchors, without our terrain initializer.
    train=pd.read_csv(root/'data/train.csv').sort_values(['brightness','row_id'],ascending=[False,True]).head(2700)
    train=train.sample(frac=1,random_state=20261014).reset_index(drop=True)
    train['start_id']=np.arange(2700);train['random_seed']=20261013+train.start_id*1009
    train['role']='observed_high';train.to_csv(out/'baseline_starting_sequences.csv',index=False)
    required=[base/'best.pt',base/'terrain_train_only/terrain.npz',root/'data/train.csv',root/'data/validation.csv',root/'data/calibration.csv',
              root/'data/split_manifest.json',root/'repro_project/assets/warm_best_model.pt',
              root/'repro_project/data/processed/reference_sequence.txt',base/'terrain_train_only/source_quality.csv',
              historical/'FROZEN_TGITO_PARAMS.json',historical/'confirmation.json']
    required += list(out.glob('*starting_sequences.csv'))
    for name in ['glom_seed20261014','glom_seed20261015','esm2_650m_mlp_seed20261013']:
        required.append(root/'artifacts/models/random'/name/'best.pt')
    for n in ['discovery_v3.py','discovery_v3_baselines.py','hpo_config_contract.py','encoder_contract.py','optimizer_consistency.py','fast_proposals.py','position_features.py',
              'build_wide_peak_plan.py','evaluate_risk_screening.py','evaluate_submission_policy.py',
              'run_search_comparison.py','run_recent_search_baseline.py','score_discovery_independent.py']:
        required.append(root/n)
    for folder in ['pipeline/tgito','pipeline/glom_model']:
        required+=list((root/'repro_project'/folder).glob('*.py'))
    required+=list((root/'third_party/FLEXS/flexs').rglob('*.py'))
    for modelname in ['esm2_t30_150M_UR50D','esm2_650m']:
        required+=list((root/'repro_project/assets'/modelname).glob('*.safetensors'))
        for ext in ['*.json','*.txt']:required+=list((root/'repro_project/assets'/modelname).glob(ext))
    hashes={p.relative_to(root).as_posix():sha(p) for p in required}
    protocol=dict(version='discovery_v3_historical_hpo_transfer',profiles=profiles,primary_profile='trial12',input_hashes=hashes,
        primary_selection='Historical confirmation winner transferred to corrected implementation; not selected by this pilot or any test results',
        precision='float32 GLOM and differentiable encoder; historical bf16 override explicit',
        initialization='900 source-balanced multiterrain peaks; uniform source prior; summit/ridge/edge per peak',
        pilot=dict(contexts=36,repetitions=2,profiles=list(profiles),selection_feedback=False),
        methods=['glom_tgito','glom_adalead','glom_random','glom_prothmso','esm650_prothmso'],
        baseline_initialization='2700 distinct highest measured training sequences, fixed shuffled order',
        baseline_ranking='Native fitness predictor, without GLOM-TGITO terrain confidence',
        budget='For each context index, baselines receive the TGITO forward-sequence count; backward and MLM calls/time are reported separately',
        repeats='One full 2700-context campaign; pilot repeats do not count as whole-campaign repeats',
        candidate_pool='One native endpoint per context for each method; also retain query histories for budget curves',
        selection=dict(batch_sizes=[6,30,100],minimum_pred=4.31,minimum_pairwise_hamming=2),
        measurement_policy='Unmatched sequences have missing fluorescence, never zero or a failure',
        evaluation=['Frozen generated batches: cross-seed prediction stability, novelty, diversity, support/risk, auxiliary structure',
                    'Separately: assay-measured hidden-library ranking, RMSE/R2/Spearman and top-batch true brightness; not generated fluorescence'],
        main_metric_excludes_sparse_catalogue_hit_rate=True,external_model_is_not_ground_truth=True,
        historical_hpo_test_independence='Retrospective transfer: historical HPO used some labels in current splits; do not call this historical policy a strictly unseen-test selection',
        test_labels_read=False)
    write_json(out/'PROTOCOL.json',protocol)
    print(json.dumps(dict(prepared=str(out),pilot_runs=216,formal_contexts=2700)),flush=True)

def load_generator(root,profile):
    import torch
    setup_project(root);torch.set_num_threads(2)
    os.environ['PPLU_ESM_MODEL_PATH']=str(root/'repro_project/assets/esm2_t30_150M_UR50D')
    from pipeline.tgito.optimizer import TGITOGenerator,GeneratorConfig
    from pipeline.tgito.aa import ids0_to_seq
    from pipeline.tgito.esm_bridge import load_bridge
    from pipeline.tgito.revised_slim_internal import load_revised_slim
    from pipeline.tgito.terrain_memory import TerrainMemoryBank
    from pipeline.tgito.terrain_potential import TerrainPotential,TerrainPotentialConfig
    from encoder_contract import TrainingCompatibleEncoder
    from position_features import canonical_bits
    from fast_proposals import FastProposalMixin
    from optimizer_consistency import ConsistentObjectiveMixin
    protocol=load_protocol(root);verify_input_hashes(root,protocol['input_hashes'])
    spec=protocol['profiles'][profile];effective=spec['resolved']
    if spec['parameters'] is not None:validate_effective(spec['parameters'],effective)
    base,out=paths(root);model,mean,std=load_revised_slim(base/'best.pt',device='cuda')
    enc=TrainingCompatibleEncoder(load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda'),device='cuda',freeze_esm=True,target_h_dim=1280).eval()
    bank=TerrainMemoryBank.from_npz(base/'terrain_train_only/terrain.npz',device='cuda');assert len(bank.available_sources)==12
    risk=TerrainPotential(bank,TerrainPotentialConfig(**effective['terrain_config']))
    class Generator(ConsistentObjectiveMixin,FastProposalMixin,TGITOGenerator):
        def reset(self):self.forward_count=0;self.gradient_count=0;self.seen=set();self.gate_checks=0
        def _forward_from_probs(self,probs,*args,**kwargs):
            result=super()._forward_from_probs(probs,*args,**kwargs);y,terrain,_,_=result
            self.forward_count+=len(probs)
            # Count forwards participating in a loss graph; actual backward steps separately recorded.
            if probs.requires_grad and torch.is_grad_enabled():self.gradient_count+=len(probs)
            if not torch.isfinite(y).all() or not torch.isfinite(terrain['terrain']).all():raise RuntimeError('Nonfinite live objective')
            weights=torch.stack([terrain['selected_weight_'+s] for s in bank.available_sources],1)
            if not torch.all((weights>0).sum(1)==effective['terrain_config']['topk_sources']):raise RuntimeError('Live gate count differs from HPO')
            if not torch.allclose(weights.sum(1),torch.ones_like(y),atol=1e-5):raise RuntimeError('Live gate weights do not sum to one')
            self.gate_checks+=len(probs)
            if not probs.requires_grad and bool(((probs==0)|(probs==1)).all()):self.seen.update(ids0_to_seq(row) for row in probs.argmax(-1).detach().cpu())
            return result
    g=Generator(enc,model,risk,y_mean=mean,y_std=std,device='cuda');g.aa_bits=canonical_bits('cuda');g.reset()
    cfg=GeneratorConfig(**effective['generator_config'])
    # Record the actual constructed objects, not just the intended JSON.
    actual=dict(effective,generator_config=dataclasses.asdict(cfg),terrain_config=dataclasses.asdict(risk.cfg))
    if spec['parameters'] is not None:validate_effective(spec['parameters'],actual)
    return g,cfg,actual

def run(root,profile,phase,shard,shards,limit=0):
    import torch
    base,out=paths(root);protocol=load_protocol(root)
    if phase=='formal':
        gate=json.loads((out/'PREFLIGHT_PASSED.json').read_text());assert gate['protocol_sha256']==sha(out/'PROTOCOL.json')
        assert profile=='trial12'
    plan=pd.read_csv(out/('pilot_starting_sequences.csv' if phase=='pilot' else 'starting_sequences.csv'))
    reps=2 if phase=='pilot' else 1
    tasks=[(i*reps+r,start,r) for i,start in enumerate(plan.itertuples()) for r in range(reps)]
    tasks=[t for t in tasks if t[0]%shards==shard]
    if limit:tasks=tasks[:limit]
    g,cfg,effective=load_generator(root,profile)
    from pipeline.tgito.aa import seq_to_ids0
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip();ref=seq_to_ids0(wt,221,'cuda');mask=torch.ones(221,dtype=torch.bool,device='cuda')
    known=set()
    for name in ['train','validation','calibration']:known.update(pd.read_csv(root/f'data/{name}.csv',usecols=['sequence']).sequence)
    target=out/phase/('glom_tgito' if phase=='formal' else profile);target.mkdir(parents=True,exist_ok=True)
    for taskid,start,rep in tasks:
        folder=target/f'context{int(start.start_id):04d}_rep{rep}';folder.mkdir(exist_ok=True)
        if (folder/'DONE.json').exists():
            done=json.loads((folder/'DONE.json').read_text());assert done['protocol_sha256']==sha(out/'PROTOCOL.json')
            for n,h in done['hashes'].items():assert sha(folder/n)==h
            continue
        seed=int(start.random_seed)+rep*10000019;seed_all(seed);g.reset();torch.cuda.reset_peak_memory_stats()
        config=dataclasses.replace(cfg,seed=seed);begin=time.monotonic()
        final=g.run(start.sequence,'all',folder,config,known_sequences=known,reference_seq=wt)
        pd.DataFrame([final]).to_csv(folder/'native_final.csv',index=False)
        count=g.forward_count;graph=g.gradient_count;seconds=time.monotonic()-begin
        history=pd.read_csv(folder/'history_restart0.csv')
        assert np.isfinite(history[['objective','hard_argmax_pred_brightness','terrain']].to_numpy()).all()
        assert history.step.max()==config.steps
        if final['valid_output']:
            sequence=final['final_seq'];distance=sum(a!=b for a,b in zip(sequence,start.sequence))
            assert sequence not in known and len(sequence)==221 and 1<=distance<=2
            ids=seq_to_ids0(sequence,221,'cuda')[None]
            with torch.inference_mode():pred,terr,_,_=g._forward_from_probs(torch.nn.functional.one_hot(ids,20).float(),seq_to_ids0(start.sequence,221,'cuda'),ref,mask,config)
            assert abs(float(pred[0])-float(final['final_pred_brightness']))<1e-5
            row=dict(sequence=sequence,pred=float(pred[0]),start_id=int(start.start_id),seed_row=int(start.row_id),
                     mountain_id=start.mountain_id,mountain_order=int(start.mountain_order),role=start.role,
                     representative_source=start.representative_source,random_seed=seed,rep=rep,hamming_from_seed=distance,method='glom_tgito')
            row.update({k:float(v[0]) for k,v in terr.items() if v.ndim==1 and len(v)==1})
            pd.DataFrame([row]).to_csv(folder/'hard_endpoint.csv',index=False)
        weights=history.filter(regex='^selected_weight_')
        weight_range=float(np.ptp(weights.to_numpy(),axis=0).max()) if len(weights.columns) else 0.
        pd.DataFrame({'sequence':sorted(g.seen)}).to_csv(folder/'queried_sequences.csv',index=False)
        write_json(folder/'EFFECTIVE_CONFIG.json',dict(effective,generator_config=dataclasses.asdict(config),seed_override='Per-context predeclared stochastic seed'))
        record=dict(start_id=int(start.start_id),rep=rep,profile=profile,phase=phase,valid=bool(final['valid_output']),
                    seconds=seconds,forward_sequences=count,gradient_graph_forwards=graph,backward_steps=config.steps,
                    unique_hard_queries=len(g.seen),postprocess_forwards=g.forward_count-count,live_gate_checks=g.gate_checks,
                    gate_topk=effective['terrain_config']['topk_sources'],dynamic_weight_range=weight_range,
                    peak_gpu_gib=torch.cuda.max_memory_reserved()/2**30,protocol_sha256=sha(out/'PROTOCOL.json'),
                    hashes={p.name:sha(p) for p in folder.iterdir() if p.suffix in ['.csv','.json'] and p.name!='DONE.json'},test_labels_read=False)
        write_json(folder/'DONE.json',record);print(json.dumps(record),flush=True)

def check_pilot(root):
    base,out=paths(root);protocol=load_protocol(root);verify_input_hashes(root,protocol['input_hashes'])
    plan=pd.read_csv(out/'pilot_starting_sequences.csv');rows=[]
    for profile in protocol['pilot']['profiles']:
        for sid in plan.start_id:
            for rep in range(2):
                folder=out/'pilot'/profile/f'context{sid:04d}_rep{rep}'
                done=json.loads((folder/'DONE.json').read_text());assert done['protocol_sha256']==sha(out/'PROTOCOL.json')
                for n,h in done['hashes'].items():assert sha(folder/n)==h
                assert done['live_gate_checks']>0 and done['backward_steps']==70
                if profile!='v2_diagnostic':assert done['gate_topk']==6
                rows.append(done)
    table=pd.DataFrame([{k:v for k,v in r.items() if k!='hashes'} for r in rows]);table.to_csv(out/'pilot_runtime_summary.csv',index=False)
    main=table[table.profile.eq('trial12')]
    baseline_rows=[]
    for method in protocol['methods'][1:]:
        files=sorted((out/'baseline_pilot'/method).glob('context*/DONE.json'))
        assert len(files)==6,(method,len(files))
        for p in files:
            done=json.loads(p.read_text());assert done['protocol_sha256']==sha(out/'PROTOCOL.json')
            assert done['forward_sequences']<=550 and done['forward_sequences']>0 and done['valid']
            for n,h in done['hashes'].items():assert sha(p.parent/n)==h
            baseline_rows.append(done)
    pd.DataFrame([{k:v for k,v in r.items() if k!='hashes'} for r in baseline_rows]).to_csv(out/'baseline_pilot_runtime.csv',index=False)
    # No relative brightness or baseline-win criterion determines whether the experiment runs.
    assert main.valid.mean()>=.70,'Too few feasible native endpoints; investigate before scaling'
    assert (main.dynamic_weight_range>1e-7).mean()>=.5,'Dynamic terrain weights not changing as expected'
    report=dict(passed=True,records=len(rows),baseline_records=len(baseline_rows),primary_profile='trial12',primary_valid_fraction=float(main.valid.mean()),
                no_test_or_evaluator_feedback=True,protocol_sha256=sha(out/'PROTOCOL.json'),
                max_reserved_gib=float(table.peak_gpu_gib.max()),median_context_seconds=float(main.seconds.median()),
                criterion='Implementation and feasibility only; historical winner is not reselected by predicted or measured test brightness')
    write_json(out/'PREFLIGHT_PASSED.json',report);print(json.dumps(report),flush=True)

def freeze(root,method,mountains=900):
    from evaluate_risk_screening import confidence
    from evaluate_submission_policy import choose
    base,out=paths(root);protocol=load_protocol(root);target=out/'formal'/method;dest=target/f'prefix_{mountains:03d}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'FROZEN.json').exists():return
    rows=[];runtimes=[];sources={}
    for sid in range(mountains*3):
        folder=target/f'context{sid:04d}_rep0';done=json.loads((folder/'DONE.json').read_text())
        assert done['protocol_sha256']==sha(out/'PROTOCOL.json')
        for n,h in done['hashes'].items():assert sha(folder/n)==h
        if done['valid']:rows.append(pd.read_csv(folder/'hard_endpoint.csv'))
        runtimes.append({k:v for k,v in done.items() if k!='hashes'});sources[str(sid)]=sha(folder/'DONE.json')
    end=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['sequence','pred'])
    if method=='glom_tgito' and len(end):end=confidence(end);ranking='confidence_score'
    else:ranking='pred'
    ranked=end.sort_values([ranking,'sequence'],ascending=[False,True],kind='stable').drop_duplicates('sequence').reset_index(drop=True)
    ranked.to_csv(dest/'all_unique_ranked.csv',index=False);selected=[]
    for k in protocol['selection']['batch_sizes']:
        ix=choose(ranked.sequence.to_numpy(),ranked.pred.to_numpy(),ranked[ranking].to_numpy(),k=k,
                  min_pred=protocol['selection']['minimum_pred'],min_distance=protocol['selection']['minimum_pairwise_hamming'])
        d=ranked.iloc[ix].copy();d['batch_size']=k;d['rank']=np.arange(1,len(d)+1);selected.append(d)
        d.to_csv(dest/f'top{k}.csv',index=False)
        (dest/f'top{k}.fasta').write_text(''.join(f'>{method}_rank{r.rank}_context{r.start_id}\n{r.sequence}\n' for r in d.itertuples()))
    pd.concat(selected,ignore_index=True).to_csv(dest/'selected.csv',index=False)
    pd.DataFrame(runtimes).to_csv(dest/'context_runtime.csv',index=False)
    write_json(dest/'FROZEN.json',dict(method=method,contexts=mountains*3,valid_contexts=len(end),unique_candidates=len(ranked),
                sources=sources,files={p.name:sha(p) for p in dest.iterdir() if p.suffix in ['.csv','.fasta']},
                protocol_sha256=sha(out/'PROTOCOL.json'),test_labels_read=False,evaluator_used_before_freeze=False))
    print(json.dumps(dict(frozen=str(dest),unique=len(ranked))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--action',required=True,choices=['prepare','run','check-pilot','freeze']);p.add_argument('--phase',choices=['pilot','formal'],default='pilot')
    p.add_argument('--profile',default='trial12');p.add_argument('--method',default='glom_tgito');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--mountains',type=int,default=900);a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    elif a.action=='run':run(a.root,a.profile,a.phase,a.shard,a.shards,a.limit)
    elif a.action=='check-pilot':check_pilot(a.root)
    else:freeze(a.root,a.method,a.mountains)
