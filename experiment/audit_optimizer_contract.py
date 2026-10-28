"""Numerical contract tests and a fixed 12-context replay; no test labels.

Diagnostic replay only. It does not replace completed search outputs or authorize
resumption of the large campaign. Compares corrected and historical objectives.
"""
from pathlib import Path
import argparse,dataclasses,json,os,time,types,hashlib
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np
import pandas as pd
import torch
from experiment_core import setup_project,write_json,seed_all
from optimizer_consistency import ConsistentObjectiveMixin


def main(root):
    setup_project(root);torch.set_num_threads(3)
    os.environ['PPLU_ESM_MODEL_PATH']=str(root/'repro_project/assets/esm2_t30_150M_UR50D')
    from pipeline.tgito.optimizer import TGITOGenerator,GeneratorConfig
    from pipeline.tgito.aa import seq_to_ids0,ids0_to_seq,make_relaxed_logits_from_ids
    from pipeline.tgito.esm_bridge import load_bridge
    from pipeline.tgito.revised_slim_internal import load_revised_slim
    from pipeline.tgito.terrain_memory import TerrainMemoryBank
    from pipeline.tgito.terrain_potential import TerrainPotential,TerrainPotentialConfig
    from encoder_contract import TrainingCompatibleEncoder
    from position_features import canonical_bits
    from fast_proposals import FastProposalMixin
    base=root/'artifacts/models/random/glom_seed20261013'
    out=root/'audit/optimizer_consistency_replay';out.mkdir(exist_ok=True)
    starts=pd.read_csv(base/'wide_search/starting_sequences.csv').head(12)
    first=base/'wide_search'/f'context0000_seed{int(starts.iloc[0].random_seed)}_tgito_risk/DONE.json'
    cfg=GeneratorConfig(**json.loads(first.read_text())['config'])
    # Actual old run configuration; no tuning to observed performance.
    write_json(out/'PROTOCOL.json',dict(start_ids=starts.start_id.tolist(),selection='First twelve pre-existing source-balanced contexts',
        config=dataclasses.asdict(cfg),corrections=['subtract smooth-count initialization offset and normalize hard endpoint',
        'compare discrete proposals and returned candidates using hard objective'],
        unchanged=['GLOM checkpoint','12-source terrain bank','calibration-selected gate','learning rate','steps','proposal policy','radius'],
        test_labels_read=False,main_campaign_restart_authorized_by_this_test=False))
    model,mean,std=load_revised_slim(base/'best.pt',device='cuda')
    enc=TrainingCompatibleEncoder(load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda'),device='cuda',freeze_esm=True,target_h_dim=1280).eval()
    bank=TerrainMemoryBank.from_npz(base/'terrain_train_only/terrain.npz',device='cuda')
    riskcfg=TerrainPotentialConfig(**json.loads((base/'risk_calibration/FROZEN_CONFIG.json').read_text())['terrain_config'])
    terrain=TerrainPotential(bank,riskcfg)
    class Counted(FastProposalMixin,TGITOGenerator):
        def reset(self):self.forward_calls=0;self.grad_calls=0;self.proposal_overlap=[]
        def _forward_from_probs(self,probs,*args,**kwargs):
            self.forward_calls+=len(probs)
            if probs.requires_grad:self.grad_calls+=len(probs)
            return super()._forward_from_probs(probs,*args,**kwargs)
        def _try_discrete_proposals(self,logits,grad,*args,**kwargs):
            gcfg=args[3];cur=logits.argmax(-1)
            # Does the infinitesimal gradient step materially change the proposal ordering?
            old=logits.clone().scatter(1,cur[:,None],-1e9).flatten()
            new=(logits-gcfg.lr*grad).scatter(1,cur[:,None],-1e9).flatten()
            k=min(64,len(old));a=set(old.topk(k).indices.cpu().tolist());b=set(new.topk(k).indices.cpu().tolist())
            self.proposal_overlap.append(len(a&b)/k)
            return super()._try_discrete_proposals(logits,grad,*args,**kwargs)
    class Fixed(ConsistentObjectiveMixin,Counted):pass
    # Check zero/one/two hard changes and the relaxed baseline without running optimization.
    probe=Fixed(enc,model,terrain,y_mean=mean,y_std=std,device='cuda');probe.aa_bits=canonical_bits('cuda');probe.reset()
    ids=seq_to_ids0(starts.iloc[0].sequence,221,'cuda');mask=torch.ones(221,dtype=torch.bool,device='cuda')
    checks=[]
    for count in [0,1,2]:
        changed=ids.clone();changed[:count]=(changed[:count]+1)%20
        probs=torch.nn.functional.one_hot(changed,20).float()
        new=float(probe._confident_change_mass(probs,ids,mask,cfg,topk=0))
        old=float(TGITOGenerator._confident_change_mass(probe,probs,ids,mask,cfg,topk=0))
        assert abs(new-count)<1e-5
        checks.append(dict(kind='hard',actual=count,original=old,corrected=new))
        logits=make_relaxed_logits_from_ids(changed,221,mask,cfg.mutable_anchor_logit,cfg.mutable_alt_logit,cfg.lock_scale,'cuda')
        probs=probe._probs_from_logits(logits,ids,mask,cfg)
        new=float(probe._confident_change_mass(probs,ids,mask,cfg,topk=0))
        old=float(TGITOGenerator._confident_change_mass(probe,probs,ids,mask,cfg,topk=0))
        assert abs(new-count)<.01
        checks.append(dict(kind='relaxed',actual=count,original=old,corrected=new))
    write_json(out/'NUMERICAL_CONTRACT.json',dict(passed=True,cases=checks))
    known=set()
    for partition in ['train','validation','calibration']:known.update(pd.read_csv(root/f'data/{partition}.csv',usecols=['sequence']).sequence)
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    rows=[]
    for start in starts.itertuples():
        folder=out/f'context{start.start_id:04d}';folder.mkdir(exist_ok=True)
        historical=base/'wide_search'/f'context{start.start_id:04d}_seed{start.random_seed}_tgito_risk'
        for variant,klass in [('original_replay',Counted),('consistent_objective',Fixed)]:
            target=folder/variant;target.mkdir(exist_ok=True)
            if (target/'DONE.json').exists():rows.append(json.loads((target/'DONE.json').read_text()));continue
            seed_all(int(start.random_seed));g=klass(enc,model,terrain,y_mean=mean,y_std=std,device='cuda');g.aa_bits=canonical_bits('cuda');g.reset()
            runconfig=dataclasses.replace(cfg,seed=int(start.random_seed));t0=time.monotonic()
            final=g.run(start.sequence,'all',target,runconfig,known_sequences=known,reference_seq=wt)
            pd.DataFrame([final]).to_csv(target/'native_final.csv',index=False)
            h=pd.read_csv(target/'history_restart0.csv')
            old=pd.read_csv(historical/'native_final.csv').iloc[0]
            exact=final['final_seq']==old.final_seq
            if variant=='original_replay':assert exact,'Original replay differs; stop before interpreting corrected output'
            r=dict(start_id=int(start.start_id),variant=variant,valid=bool(final['valid_output']),
                pred=float(final['final_pred_brightness']),historical_pred=float(old.final_pred_brightness),
                objective=float(final['final_objective']),actual_changes=int(final['changed_vs_seed_count']),
                unique_hard_states=int(h.seq.nunique()),elapsed=time.monotonic()-t0,
                forward_sequences=g.forward_calls,gradient_sequences=g.grad_calls,
                original_replay_exact=exact,
                mean_gradient_vs_no_gradient_top64_overlap=float(np.mean(g.proposal_overlap)),
                test_labels_read=False)
            write_json(target/'DONE.json',r);rows.append(r);pd.DataFrame(rows).to_csv(out/'replay_metrics.csv',index=False)
            print(json.dumps(r),flush=True)
    write_json(out/'DONE.json',dict(contexts=len(starts),variants=2,records=len(rows),test_labels_read=False,
        note='Numerical/implementation diagnosis, not measured generated fitness or an equal-compute algorithm benchmark'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    a=p.parse_args();main(a.root)
