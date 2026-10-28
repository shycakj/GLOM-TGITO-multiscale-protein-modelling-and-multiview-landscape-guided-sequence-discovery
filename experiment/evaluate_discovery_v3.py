"""Evaluate frozen batches; predicted support is separate from measured fitness."""
from pathlib import Path
import gc,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import torch
from experiment_core import setup_project,write_json,regression_metrics
from hpo_config_contract import sha
from discovery_v3 import paths,load_protocol

def quantile_higher(values,alpha=.10):
    values=np.asarray(values);rank=min(len(values),int(np.ceil((len(values)+1)*(1-alpha))))
    return float(np.sort(values)[rank-1])

def main():
    root=Path(__file__).resolve().parent;setup_project(root);torch.set_num_threads(2)
    base,out=paths(root);dest=out/'evaluation';dest.mkdir(exist_ok=True);protocol=load_protocol(root)
    evaluation_plan=json.loads((out/'EVALUATION_PLAN_FROZEN.json').read_text())
    assert sha(root/'evaluate_discovery_v3.py')==evaluation_plan['evaluation_source_sha256']
    for name,h in evaluation_plan['input_hashes'].items():assert sha(root/name)==h
    selected=[];frozen_files={};known=pd.concat([pd.read_csv(root/f'data/{p}.csv',usecols=['sequence']) for p in ['train','validation','calibration']]).sequence.tolist()
    for method in protocol['methods']:
        folder=out/'formal'/method/'prefix_900';frozen=json.loads((folder/'FROZEN.json').read_text())
        for name,h in frozen['files'].items():assert sha(folder/name)==h
        frozen_files[method]=sha(folder/'FROZEN.json')
        chosen=pd.read_csv(folder/'selected.csv');starts=pd.read_csv(out/('starting_sequences.csv' if method=='glom_tgito' else 'baseline_starting_sequences.csv')).set_index('start_id')
        chosen['method']=method;chosen['start_sequence']=chosen.start_id.map(starts.sequence);selected.append(chosen)
    chosen=pd.concat(selected,ignore_index=True)
    # Assay identities are added only after every native output set is frozen.
    cal=pd.read_csv(root/'data/calibration.csv');test=pd.read_csv(root/'data/test.csv')
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    sequences=sorted(set(chosen.sequence)|set(chosen.start_sequence)|set(cal.sequence)|set(test.sequence)|{wt})
    pd.DataFrame({'sequence':sequences}).to_csv(dest/'evaluator_sequences.csv',index=False)
    write_json(dest/'EVALUATION_INPUTS_FROZEN.json',dict(candidate_freezes=frozen_files,sequences_sha256=sha(dest/'evaluator_sequences.csv'),
                predictor_feedback_to_search=False,post_selection_analysis=True))
    from pipeline.tgito.revised_slim_internal import load_revised_slim
    from pipeline.tgito.esm_bridge import load_bridge
    from pipeline.tgito.aa import seq_to_ids0,soft_pos_feats
    from encoder_contract import TrainingCompatibleEncoder
    from position_features import canonical_bits
    os.environ['PPLU_ESM_MODEL_PATH']=str(root/'repro_project/assets/esm2_t30_150M_UR50D')
    predfile=dest/'glom_live_predictions.csv'
    if not predfile.exists():
        from pipeline.tgito.terrain_memory import TerrainMemoryBank
        from pipeline.tgito.terrain_potential import TerrainPotential,TerrainPotentialConfig
        bank=TerrainMemoryBank.from_npz(base/'terrain_train_only/terrain.npz',device='cuda')
        terrain=TerrainPotential(bank,TerrainPotentialConfig(**protocol['profiles']['trial12']['resolved']['terrain_config']))
        models={s:load_revised_slim(root/f'artifacts/models/random/glom_seed{s}/best.pt',device='cuda') for s in [20261013,20261014,20261015]}
        enc=TrainingCompatibleEncoder(load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda'),device='cuda',freeze_esm=True,target_h_dim=1280).eval()
        ref=seq_to_ids0(wt,221,'cuda');bits=canonical_bits('cuda');records=[];risk_records=[]
        with torch.inference_mode():
            for st in range(0,len(sequences),24):
                batch=sequences[st:st+24];ids=torch.stack([seq_to_ids0(s,221,'cuda') for s in batch]);probs=torch.nn.functional.one_hot(ids,20).float()
                H=enc(probs)['H'];P=soft_pos_feats(probs,ref,bits)
                ys={}
                for seed,(m,mu,sd) in models.items():
                    outputs=m.forward_all(H,P);y=outputs['y_norm']*sd+mu;ys[seed]=y.cpu().numpy()
                    if seed==20261013:
                        live=dict(outputs,pred_brightness_live=y,sequence=probs.reshape(len(batch),-1)@bank.sequence_projection)
                        values=terrain(live);columns={k:v.cpu().numpy() for k,v in values.items() if v.ndim==1 and len(v)==len(batch)}
                        risk_records.extend(dict(sequence=s,pred=float(y[i]),**{k:float(v[i]) for k,v in columns.items()}) for i,s in enumerate(batch))
                records.extend(dict(sequence=s,**{f'glom_{seed}':float(v[i]) for seed,v in ys.items()}) for i,s in enumerate(batch))
                if st%480==0:print(json.dumps(dict(stage='cross_glom',rows=st,total=len(sequences))),flush=True)
        pd.DataFrame(risk_records).to_csv(dest/'live_risk_scores.csv',index=False)
        pd.DataFrame(records).to_csv(predfile,index=False);del models,enc,H,P,bank,terrain;gc.collect();torch.cuda.empty_cache()
    prediction=pd.read_csv(predfile).set_index('sequence');assert set(prediction.index)==set(sequences)
    prediction['glom_cross_mean']=prediction[['glom_20261014','glom_20261015']].mean(1)
    prediction['glom_cross_sd']=prediction[['glom_20261014','glom_20261015']].std(1,ddof=0)
    # A second architecture is calibrated on measured sequences and reported alongside,
    # never treated as an infallible referee or averaged into a truth label.
    esmfile=dest/'esm650_live_predictions.csv'
    if not esmfile.exists():
        from transformers import AutoTokenizer,AutoModel
        from score_discovery_independent import head_model,apply_head
        path=root/'repro_project/assets/esm2_650m';tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        backbone=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval()
        heads={s:head_model(root/f'artifacts/models/random/esm2_650m_mlp_seed{s}/best.pt') for s in [20261013,20261014,20261015]};records=[]
        with torch.inference_mode():
            for st in range(0,len(sequences),16):
                batch=sequences[st:st+16];inp=tok(batch,return_tensors='pt',padding=True,return_special_tokens_mask=True)
                mask=(inp.pop('special_tokens_mask')==0)&inp['attention_mask'].bool();mask=mask.cuda();inp={k:v.cuda() for k,v in inp.items()}
                h=backbone(**inp).last_hidden_state.float();x=((h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy()
                ys={s:apply_head(c,m,x) for s,(c,m) in heads.items()}
                records.extend(dict(sequence=s,**{f'esm650_{seed}':float(v[i]) for seed,v in ys.items()}) for i,s in enumerate(batch))
                if st%480==0:print(json.dumps(dict(stage='cross_esm650',rows=st,total=len(sequences))),flush=True)
        pd.DataFrame(records).to_csv(esmfile,index=False);del backbone,heads;gc.collect();torch.cuda.empty_cache()
    prediction=prediction.join(pd.read_csv(esmfile).set_index('sequence'),validate='one_to_one')
    prediction['esm650_cross_mean']=prediction[['esm650_20261014','esm650_20261015']].mean(1)
    from evaluate_discovery import min_hamming_to_reference,diversity
    training=pd.read_csv(root/'data/train.csv',usecols=['sequence']).sequence.tolist()
    cal_distance=min_hamming_to_reference(cal.sequence.tolist(),training,wt)
    test_distance=min_hamming_to_reference(test.sequence.tolist(),training,wt)
    metrics=[];bounds=[]
    judges=['glom_cross_mean','esm650_cross_mean']
    for judge in judges:
        pc=cal.sequence.map(prediction[judge]).to_numpy();pt=test.sequence.map(prediction[judge]).to_numpy()
        for scope,mask in [('all',np.ones(len(test),bool)),('true_high420',test.brightness.ge(4.2).to_numpy()),('predicted_high420',pt>=4.2),
                           ('nearest_train_1',test_distance==1),('nearest_train_2plus',test_distance>=2)]:
            if mask.sum()>2:metrics.append(dict(judge=judge,scope=scope,**regression_metrics(test.brightness.to_numpy()[mask],pt[mask])))
        for scope,mc,mt in [('all',np.ones(len(cal),bool),np.ones(len(test),bool)),('predicted_high420',pc>=4.2,pt>=4.2),
                            ('high420_nearest_train_1',(pc>=4.2)&(cal_distance==1),(pt>=4.2)&(test_distance==1)),
                            ('high420_nearest_train_2plus',(pc>=4.2)&(cal_distance>=2),(pt>=4.2)&(test_distance>=2))]:
            if mc.sum()<30:continue
            q=quantile_higher(pc[mc]-cal.brightness.to_numpy()[mc])
            bounds.append(dict(judge=judge,scope=scope,calibration_n=int(mc.sum()),overprediction_q90=q,test_n=int(mt.sum()),
                test_coverage=float(np.mean(test.brightness.to_numpy()[mt]>=pt[mt]-q)),
                meaning='Empirical lower error envelope; optimizer-selected distribution has no automatic coverage guarantee'))
    pd.DataFrame(metrics).to_csv(dest/'judge_measured_test_metrics.csv',index=False);pd.DataFrame(bounds).to_csv(dest/'empirical_error_envelopes.csv',index=False)
    for c in prediction.columns:chosen[c]=chosen.sequence.map(prediction[c])
    unique=sorted(set(chosen.sequence));nearest=min_hamming_to_reference(unique,known,wt)
    chosen['nearest_visible_hamming']=chosen.sequence.map(dict(zip(unique,nearest)))
    nearest_training=min_hamming_to_reference(unique,training,wt)
    chosen['nearest_training_hamming']=chosen.sequence.map(dict(zip(unique,nearest_training)))
    assay=pd.concat([pd.read_csv(root/f'data/{p}.csv',usecols=['sequence','brightness']) for p in ['train','validation','calibration','test']]).set_index('sequence')
    chosen['measured_brightness_if_available']=chosen.sequence.map(assay.brightness)
    chosen['novel_to_measured_catalogue']=~chosen.sequence.isin(assay.index)
    for judge in judges:
        chosen[judge+'_gain_vs_start']=chosen[judge]-chosen.start_sequence.map(prediction[judge])
        q=next(b['overprediction_q90'] for b in bounds if b['judge']==judge and b['scope']=='predicted_high420')
        chosen[judge+'_empirical_lower']=chosen[judge]-q
    from validate_structure_evidence import Scorer
    structure=Scorer(root).score(sorted(set(chosen.sequence)|{wt})).set_index('sequence')
    structure.to_csv(dest/'structure_compatibility.csv')
    for c in structure.columns:chosen[c]=chosen.sequence.map(structure[c])
    chosen.to_csv(dest/'selected_candidates_evaluated.csv',index=False)
    summary=[]
    for (method,k),d in chosen.groupby(['method','batch_size']):
        r=dict(method=method,batch_size=int(k),selected_n=len(d),novel_catalogue_fraction=float(d.novel_to_measured_catalogue.mean()),
               mean_pairwise_hamming=diversity(d.sequence.tolist()),nearest_visible_hamming_mean=float(d.nearest_visible_hamming.mean()),
               measured_n=int(d.measured_brightness_if_available.notna().sum()),
               mpnn_log_likelihood_mean=float(d.mpnn_log_likelihood.mean()),
               canonical_chromophore_preserved_fraction=float(d.canonical_chromophore_preserved.mean()),
               mutations_outside_resolved_backbone=int(d.unscored_mutations.sum()))
        for judge in judges:r.update({judge+'_mean':float(d[judge].mean()),judge+'_gain_vs_start':float(d[judge+'_gain_vs_start'].mean()),
                                     judge+'_positive_gain_fraction':float(d[judge+'_gain_vs_start'].gt(0).mean()),
                                     judge+'_empirical_lower_mean':float(d[judge+'_empirical_lower'].mean())})
        summary.append(r)
    pd.DataFrame(summary).to_csv(dest/'full_pipeline_comparison.csv',index=False)
    # Matched measured-library ranking is a separate assay-supported screening task.
    from evaluate_submission_policy import choose
    measured=[]
    for judge in ['glom_20261013','esm650_20261013']:
        p=test.sequence.map(prediction[judge]).to_numpy()
        for k in [6,30,100]:
            ix=choose(test.sequence.to_numpy(),p,p,k=k)
            values=test.brightness.iloc[ix]
            measured.append(dict(method=judge,task='heldout_measured_library_screening',batch_size=k,selected_n=len(ix),true_mean=float(values.mean()),
                                 true_min=float(values.min()),true_high420_fraction=float(values.ge(4.2).mean())))
    pd.DataFrame(measured).to_csv(dest/'measured_library_predictor_ranking.csv',index=False)
    from evaluate_risk_screening import confidence
    risk=pd.read_csv(dest/'live_risk_scores.csv').set_index('sequence').loc[test.sequence].reset_index()
    risk=confidence(risk);ranking_records=[]
    rng=np.random.default_rng(20261014)
    libraries=[(-1,np.arange(len(test)))]+[(i,rng.choice(len(test),512,replace=False)) for i in range(200)]
    for library,idx in libraries:
        # Native pool normalization is recomputed within each unlabelled library.
        local=confidence(risk.iloc[idx]);scores={'glom_native_confidence':local.confidence_score.to_numpy(),'glom_prediction':local.pred.to_numpy()}
        for method,score in scores.items():
            for k in ([6,30,100] if library==-1 else [6,30]):
                jj=choose(test.sequence.to_numpy()[idx],local.pred.to_numpy(),score,k=k)
                truth=test.brightness.iloc[idx[jj]].to_numpy()
                ranking_records.append(dict(task='screening_only_from_fixed_hidden_library',library=library,method=method,batch_size=k,selected_n=len(jj),
                    true_mean=float(truth.mean()) if len(truth) else None,true_min=float(truth.min()) if len(truth) else None,
                    true_high420_fraction=float((truth>=4.2).mean()) if len(truth) else None))
    pd.DataFrame(ranking_records).to_csv(dest/'measured_library_full_policy.csv',index=False)
    write_json(dest/'DONE.json',dict(candidate_freezes=frozen_files,generated_fitness_is_measured=False,search_feedback=False,
                unmatched_assays_missing=True,methods=len(protocol['methods']),result_rows=len(chosen),unix_time=time.time(),
                repeated_library_scope='Overlapping retrospective screening tasks; not independent wet-lab or search repetitions'))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
