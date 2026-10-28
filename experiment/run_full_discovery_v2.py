"""Versioned full GLOM--TGITO generation; frozen candidates precede cross-scoring.

Uses the existing 900-peak plan and all 12 terrains. Only the audited numerical
contract corrections are enabled. No hidden assay labels enter this script.
"""
from pathlib import Path
import argparse, dataclasses, json, os, shutil, time
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np
import pandas as pd
from experiment_core import setup_project, write_json, seed_all
from build_wide_peak_plan import sha


def paths(root):
    base=root/'artifacts/models/random/glom_seed20261013'
    return base,base/'full_discovery_v2'


def prepare(root):
    base,out=paths(root);out.mkdir(exist_ok=True)
    if (out/'PROTOCOL.json').exists():return
    replay=root/'audit/optimizer_consistency_replay'
    done=json.loads((replay/'DONE.json').read_text())
    assert done['records']==24 and json.loads((replay/'NUMERICAL_CONTRACT.json').read_text())['passed']
    runs=pd.read_csv(replay/'replay_metrics.csv')
    assert runs.query("variant == 'original_replay'").original_replay_exact.all()
    old=base/'wide_search';frozen=json.loads((old/'PLAN_FROZEN.json').read_text())
    for name,key in [('starting_sequences.csv','plan_sha256')]:assert sha(old/name)==frozen[key]
    assert sha(base/'best.pt')==frozen['model_sha256']
    assert sha(base/'terrain_train_only/terrain.npz')==frozen['terrain_sha256']
    assert sha(base/'risk_calibration/FROZEN_CONFIG.json')==frozen['risk_config_sha256']
    for name in ['starting_sequences.csv','mountains_selected.csv','PLAN_FROZEN.json']:shutil.copyfile(old/name,out/name)
    protocol=dict(version='full_discovery_v2',parent_protocol=frozen,
        corrections=['Baseline-corrected smooth mutation count','Hard objective for hard proposal acceptance and return'],
        pipeline=['12-terrain peak discovery','900 peaks with summit/ridge/edge starts','Dynamic multi-terrain TGITO','Native confidence and diversity selection'],
        config=json.loads((replay/'PROTOCOL.json').read_text())['config'],
        checkpoints_mountains=[12,60,180,900],output_batch_sizes=[6,30,100],
        prediction_floor=4.31,pairwise_hamming_min=2,
        candidate_pool='One native returned valid sequence per context; no promotion from the full query pool',
        evaluator_seeds=[20261014,20261015],evaluator_role='Post-freeze robustness, not measured fluorescence',
        evaluator_feedback_to_search=False,test_labels_read=False,
        source_sha256={n:sha(root/n) for n in ['run_full_discovery_v2.py','optimizer_consistency.py','fast_proposals.py','encoder_contract.py']},
        numerical_replay_sha256=sha(replay/'replay_metrics.csv'))
    write_json(out/'PROTOCOL.json',protocol)
    print(json.dumps(dict(prepared=str(out),contexts=2700)),flush=True)


def load_generator(root):
    import torch
    setup_project(root);torch.set_num_threads(3)
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
    base,out=paths(root);protocol=json.loads((out/'PROTOCOL.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(root/name)==digest
    model,mean,std=load_revised_slim(base/'best.pt',device='cuda')
    enc=TrainingCompatibleEncoder(load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda'),device='cuda',freeze_esm=True,target_h_dim=1280).eval()
    bank=TerrainMemoryBank.from_npz(base/'terrain_train_only/terrain.npz',device='cuda')
    assert len(bank.available_sources)==12
    risk=TerrainPotential(bank,TerrainPotentialConfig(**json.loads((base/'risk_calibration/FROZEN_CONFIG.json').read_text())['terrain_config']))
    class Generator(ConsistentObjectiveMixin,FastProposalMixin,TGITOGenerator):
        def reset(self):self.seen=set();self.forward_count=0;self.gradient_count=0
        def _forward_from_probs(self,probs,*args,**kw):
            ret=super()._forward_from_probs(probs,*args,**kw)
            self.forward_count+=len(probs)
            if probs.requires_grad:self.gradient_count+=len(probs)
            elif bool(((probs==0)|(probs==1)).all()):
                self.seen.update(ids0_to_seq(row) for row in probs.argmax(-1).detach().cpu())
            return ret
    g=Generator(enc,model,risk,y_mean=mean,y_std=std,device='cuda');g.aa_bits=canonical_bits('cuda');g.reset()
    return g,GeneratorConfig(**protocol['config'])


def run(root,shard,shards,max_mountains):
    import torch
    base,out=paths(root);protocol=json.loads((out/'PROTOCOL.json').read_text())
    assert sha(out/'starting_sequences.csv')==protocol['parent_protocol']['plan_sha256']
    plan=pd.read_csv(out/'starting_sequences.csv');plan=plan[(plan.start_id%shards==shard)&(plan.mountain_order<max_mountains)]
    g,config=load_generator(root)
    from pipeline.tgito.aa import seq_to_ids0
    known=set()
    for part in ['train','validation','calibration']:known.update(pd.read_csv(root/f'data/{part}.csv',usecols=['sequence']).sequence)
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    reference=seq_to_ids0(wt,221,'cuda');mask=torch.ones(221,dtype=torch.bool,device='cuda')
    for start in plan.itertuples():
        folder=out/f'context{int(start.start_id):04d}'
        if (folder/'DONE.json').exists():continue
        folder.mkdir(exist_ok=True);seed_all(int(start.random_seed));g.reset();torch.cuda.reset_peak_memory_stats()
        cfg=dataclasses.replace(config,seed=int(start.random_seed));begin=time.monotonic()
        final=g.run(start.sequence,'all',folder,cfg,known_sequences=known,reference_seq=wt)
        pd.DataFrame([final]).to_csv(folder/'native_final.csv',index=False)
        nforward=g.forward_count;ngrad=g.gradient_count;seconds=time.monotonic()-begin
        catalog=pd.DataFrame({'sequence':sorted(g.seen)});catalog['known']=catalog.sequence.isin(known)
        catalog.to_csv(folder/'queried_sequences.csv',index=False)
        if bool(final['valid_output']):
            sequence=final['final_seq'];assert sequence not in known
            distance=sum(a!=b for a,b in zip(sequence,start.sequence));assert 1<=distance<=2
            ids=seq_to_ids0(sequence,221,'cuda')[None]
            with torch.inference_mode():
                pred,terrain,_,_=g._forward_from_probs(torch.nn.functional.one_hot(ids,20).float(),seq_to_ids0(start.sequence,221,'cuda'),reference,mask,cfg)
            assert abs(float(pred[0])-float(final['final_pred_brightness']))<1e-5
            row=dict(sequence=sequence,pred=float(pred[0]),seed_row=int(start.row_id),start_id=int(start.start_id),
                mountain_id=start.mountain_id,mountain_order=int(start.mountain_order),role=start.role,
                representative_source=start.representative_source,random_seed=int(start.random_seed),
                hamming_from_seed=distance,mutation_depth=sum(a!=b for a,b in zip(sequence,wt)),
                method='full_glom_tgito_v2',mutations_vs_reference=final['final_mutations_vs_reference'])
            row.update({k:float(v[0]) for k,v in terrain.items() if v.ndim==1 and len(v)==1})
            pd.DataFrame([row]).to_csv(folder/'hard_endpoint.csv',index=False)
        record=dict(start_id=int(start.start_id),valid=bool(final['valid_output']),pred=float(final['final_pred_brightness']),
            seconds=seconds,forward_sequences=nforward,gradient_sequences=ngrad,unique_hard_queried=len(catalog),
            postprocess_forwards=g.forward_count-nforward,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
            hashes={f.name:sha(f) for f in folder.glob('*.csv')},test_labels_read=False,version='full_discovery_v2')
        write_json(folder/'DONE.json',record);print(json.dumps(record),flush=True)
    write_json(out/f'SHARD_{shard}_DONE.json',dict(shards=shards,contexts=len(plan),max_mountains=max_mountains))


def freeze(root,mountains):
    from evaluate_risk_screening import confidence
    from evaluate_submission_policy import choose
    base,out=paths(root);dest=out/f'prefix_{mountains:03d}';dest.mkdir(exist_ok=True)
    if (dest/'FROZEN.json').exists():return dest
    plan=pd.read_csv(out/'starting_sequences.csv');plan=plan[plan.mountain_order<mountains]
    assert len(plan)==3*mountains
    rows=[];sources=[];counts=[]
    for start in plan.itertuples():
        folder=out/f'context{start.start_id:04d}';done=json.loads((folder/'DONE.json').read_text())
        for name,digest in done['hashes'].items():assert sha(folder/name)==digest
        if done['valid']:rows.append(pd.read_csv(folder/'hard_endpoint.csv'))
        counts.append({k:v for k,v in done.items() if k!='hashes'})
        sources.append(dict(start_id=int(start.start_id),done_sha256=sha(folder/'DONE.json')))
    end=confidence(pd.concat(rows,ignore_index=True).sort_values('start_id'))
    end.to_csv(dest/'all_native_endpoints.csv',index=False)
    ranked=end.sort_values(['confidence_score','sequence'],ascending=[False,True],kind='stable').drop_duplicates('sequence').reset_index(drop=True)
    ranked.to_csv(dest/'all_unique_ranked.csv',index=False)
    chosen=[]
    for k in [6,30,100]:
        idx=choose(ranked.sequence.to_numpy(),ranked.pred.to_numpy(),ranked.confidence_score.to_numpy(),k=k)
        d=ranked.iloc[idx].copy();d['batch_size']=k;d['rank']=np.arange(1,len(d)+1);chosen.append(d)
        d.to_csv(dest/f'top{k}.csv',index=False)
        (dest/f'top{k}.fasta').write_text(''.join(f'>GLOMTGITO_v2_P{mountains}_rank{r.rank}_context{r.start_id}\n{r.sequence}\n' for r in d.itertuples()))
    pd.concat(chosen,ignore_index=True).to_csv(dest/'selected.csv',index=False)
    pd.DataFrame(counts).to_csv(dest/'context_runtime.csv',index=False)
    write_json(dest/'FROZEN.json',dict(mountains=mountains,contexts=len(plan),valid_contexts=len(end),unique_candidates=len(ranked),
        sources=sources,files={f.name:sha(f) for f in dest.iterdir() if f.suffix in ['.csv','.fasta']},
        protocol_sha256=sha(out/'PROTOCOL.json'),test_labels_read=False,evaluator_used_before_freeze=False))
    print(json.dumps(dict(frozen=str(dest),valid=len(end),unique=len(ranked))),flush=True)
    return dest


def score(root,mountains):
    import torch
    setup_project(root);torch.set_num_threads(3)
    os.environ['PPLU_ESM_MODEL_PATH']=str(root/'repro_project/assets/esm2_t30_150M_UR50D')
    from encoder_contract import TrainingCompatibleEncoder
    from pipeline.tgito.esm_bridge import load_bridge
    from pipeline.tgito.revised_slim_internal import load_revised_slim
    from pipeline.tgito.aa import seq_to_ids0,soft_pos_feats
    from position_features import canonical_bits
    from evaluate_discovery import min_hamming_to_reference,diversity
    base,out=paths(root);dest=out/f'prefix_{mountains:03d}'
    if (dest/'CROSS_SCORE_DONE.json').exists():return
    frozen=json.loads((dest/'FROZEN.json').read_text())
    for name,digest in frozen['files'].items():assert sha(dest/name)==digest
    chosen=pd.read_csv(dest/'selected.csv');starts=pd.read_csv(out/'starting_sequences.csv').set_index('start_id')
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    sequences=sorted(set(chosen.sequence)|set(starts.loc[chosen.start_id].sequence)|{wt})
    models={seed:load_revised_slim(root/f'artifacts/models/random/glom_seed{seed}/best.pt',device='cuda') for seed in [20261013,20261014,20261015]}
    enc=TrainingCompatibleEncoder(load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda'),device='cuda',freeze_esm=True,target_h_dim=1280).eval()
    reference=seq_to_ids0(wt,221,'cuda');bits=canonical_bits('cuda');records=[]
    with torch.inference_mode():
        for st in range(0,len(sequences),16):
            seqs=sequences[st:st+16];ids=torch.stack([seq_to_ids0(s,221,'cuda') for s in seqs]);probs=torch.nn.functional.one_hot(ids,20).float()
            H=enc(probs)['H'];P=soft_pos_feats(probs,reference,bits)
            values={seed:(m.forward_all(H,P)['y_norm']*std+mean).cpu().numpy() for seed,(m,mean,std) in models.items()}
            records.extend(dict(sequence=s,**{f'glom_{seed}':float(v[i]) for seed,v in values.items()}) for i,s in enumerate(seqs))
    predictions=pd.DataFrame(records).set_index('sequence');predictions['cross_mean']=predictions[['glom_20261014','glom_20261015']].mean(1)
    predictions['cross_sd']=predictions[['glom_20261014','glom_20261015']].std(1,ddof=0)
    predictions.to_csv(dest/'cross_predictions.csv')
    known=pd.concat([pd.read_csv(root/f'data/{p}.csv',usecols=['sequence']) for p in ['train','validation','calibration']])
    unique=sorted(set(chosen.sequence));distance=min_hamming_to_reference(unique,known.sequence.tolist(),wt)
    nearest=dict(zip(unique,distance));chosen['nearest_visible_hamming']=chosen.sequence.map(nearest)
    for col in predictions:chosen[col]=chosen.sequence.map(predictions[col])
    assert np.max(np.abs(chosen.pred-chosen.glom_20261013))<1e-4
    chosen['cross_gain_vs_start']=chosen.cross_mean-chosen.start_id.map(starts.sequence).map(predictions.cross_mean)
    chosen['cross_gain_vs_wt']=chosen.cross_mean-float(predictions.loc[wt,'cross_mean'])
    chosen['search_optimism_gap']=chosen.pred-chosen.cross_mean
    chosen.to_csv(dest/'selected_cross_evaluated.csv',index=False)
    summary=[]
    for k,g in chosen.groupby('batch_size'):
        summary.append(dict(batch_size=int(k),selected_n=len(g),search_pred_mean=float(g.pred.mean()),
            cross_pred_mean=float(g.cross_mean.mean()),cross_pred_min=float(g.cross_mean.min()),
            cross_gain_vs_start=float(g.cross_gain_vs_start.mean()),cross_gain_vs_wt=float(g.cross_gain_vs_wt.mean()),
            cross_predicted_ge420=int(g.cross_mean.ge(4.2).sum()),cross_seed_sd_mean=float(g.cross_sd.mean()),
            pairwise_hamming_mean=diversity(g.sequence.tolist()),nearest_visible_hamming_mean=float(g.nearest_visible_hamming.mean())))
    pd.DataFrame(summary).to_csv(dest/'cross_evaluation_summary.csv',index=False)
    write_json(dest/'CROSS_SCORE_DONE.json',dict(mountains=mountains,models={str(s):sha(root/f'artifacts/models/random/glom_seed{s}/best.pt') for s in models},
        evaluator_not_used_for_selection=True,shared_training_data_and_architecture=True,measured_fitness=False,
        status='Prediction robustness with validated task-specific heads; see audit/design_reliability for assay validation',
        source_sha256=sha(root/'run_full_discovery_v2.py')))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--action',choices=['prepare','run','freeze','score'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);p.add_argument('--mountains',type=int,default=900)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    elif a.action=='run':run(a.root,a.shard,a.shards,a.mountains)
    elif a.action=='freeze':freeze(a.root,a.mountains)
    else:score(a.root,a.mountains)
