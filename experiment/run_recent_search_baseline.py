"""ProtHMSO (2026), paper Algorithm 1 adaptation to the fixed GFP search task.

Jointly mask one/two sites, retain top three joint AA combinations, score with the
same frozen GLOM. This is a GFP/budget adapter, not an unchanged AMP reproduction.
"""
from pathlib import Path
import argparse, dataclasses, hashlib, itertools, json, os, time
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import torch
from experiment_core import setup_project, seed_all, write_json

def joint_top3(logits, positions, aa_ids, alphabet):
    choices=[]
    for pos in positions:
        values=logits[pos+1,aa_ids].float().log_softmax(-1)
        v,j=values.topk(3)
        choices.append([(alphabet[int(ix)],float(score)) for ix,score in zip(j.cpu(),v.cpu())])
    combos=sorted(itertools.product(*choices),key=lambda c:sum(v for _,v in c),reverse=True)[:3]
    return [tuple(aa for aa,_ in combo) for combo in combos]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--max-runs',type=int,default=0);p.add_argument('--wide',action='store_true')
    a=p.parse_args();root=a.root
    setup_project(root);torch.set_num_threads(2)
    from transformers import AutoTokenizer,AutoModelForMaskedLM
    from pipeline.tgito.esm_bridge import load_bridge
    from encoder_contract import TrainingCompatibleEncoder
    from pipeline.tgito.revised_slim_internal import load_revised_slim
    from pipeline.tgito.terrain_memory import TerrainMemoryBank
    from pipeline.tgito.terrain_potential import TerrainPotential,TerrainPotentialConfig
    from pipeline.tgito.optimizer import TGITOGenerator,GeneratorConfig,hamming
    from pipeline.tgito.aa import AA20,seq_to_ids0
    from position_features import canonical_bits
    base=root/'artifacts/models/random/glom_seed20261013';out=base/('wide_search_recent' if a.wide else 'search_recent');out.mkdir(exist_ok=True)
    method='prothmso_top3_2026'
    parent=base/('wide_search' if a.wide else 'search')
    starts=pd.read_csv(parent/'starting_sequences.csv')
    if a.wide:
        from build_wide_peak_plan import sha
        assert sha(parent/'starting_sequences.csv')==json.loads((parent/'PLAN_FROZEN.json').read_text())['plan_sha256']
    known=set()
    for role in ['train','validation','calibration']:
        known.update(pd.read_csv(root/f'data/{role}.csv',usecols=['sequence']).sequence)
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    os.environ['PPLU_ESM_MODEL_PATH']=str(root/'repro_project/assets/esm2_t30_150M_UR50D')
    model,mean,std=load_revised_slim(base/'best.pt',device='cuda')
    bridge=load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda')
    enc=TrainingCompatibleEncoder(bridge,device='cuda',freeze_esm=True,target_h_dim=1280).eval()
    bank=TerrainMemoryBank.from_npz(base/'terrain_train_only/terrain.npz',device='cuda')
    cfg=TerrainPotentialConfig(**json.loads((base/'risk_calibration/FROZEN_CONFIG.json').read_text())['terrain_config'])
    terrain=TerrainPotential(bank,cfg)
    gen=TGITOGenerator(enc,model,terrain,y_mean=mean,y_std=std,device='cuda');gen.aa_bits=canonical_bits('cuda')
    config=GeneratorConfig(length=221,amp_dtype='off',mutation_budget=2,budget_relative='seed')
    ref=seq_to_ids0(wt,221,'cuda');mask=torch.ones(221,dtype=torch.bool,device='cuda')
    path=root/'repro_project/assets/esm2_650m'
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    mlm=AutoModelForMaskedLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval()
    aa_ids=torch.tensor([tok.convert_tokens_to_ids(aa) for aa in AA20],device='cuda')
    source=root/'third_party/ProtHMSO/OPT_Code/Random/randomVSESM2_1.py'
    write_json(out/'PROTOCOL.json',dict(method=method,paper='https://doi.org/10.1371/journal.pcbi.1014365',
        implementation='Paper Algorithm 1: simultaneous MLM masking and top-k fitness screening; GFP task adapter',
        official_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        differences_from_official_AMP_example=['GFP fitness replaces AMP/MIC', 'one/two mutable sites to match TGITO radius',
            'paper top-k=3 joint combinations replaces example argmax', 'same fitness forward budget',
            'AA20 vocabulary restriction replaces non-AA fallback K', 'bf16 MLM; float32 GLOM'],
        ranking='Same downstream frozen risk and final submission policies; external judges never guide generation',
        prior_model='ESM2-650M',additional_prior_forwards_reported=True,
        primary_readout='Frozen exact experimental test matches; external predictors are auxiliary only'))
    completed=0;records=[]
    for start in starts.itertuples():
        for seed in ([int(start.random_seed)] if a.wide else [20261013,20261014,20261015]):
            prefix=f'context{int(start.start_id):04d}' if a.wide else f'start{int(start.row_id)}'
            folder=out/f'{prefix}_seed{seed}_{method}'
            if (folder/'DONE.json').exists():
                records.append(json.loads((folder/'DONE.json').read_text()));continue
            budgetfile=parent/f'{prefix}_seed{seed}_tgito_risk/DONE.json'
            while not budgetfile.exists():time.sleep(30)
            budget=json.loads(budgetfile.read_text())['forward_sequences']
            folder.mkdir(exist_ok=True);seed_all(seed);rng=np.random.default_rng(seed)
            anchor=start.sequence;anchor_ids=seq_to_ids0(anchor,221,'cuda')
            tokens=tok(anchor,return_tensors='pt')['input_ids'].to('cuda')
            assert tokens.shape==(1,223)
            candidates=set();prior_forwards=0;attempts=0;t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
            while len(candidates)<budget and attempts<50*budget:
                positions=[sorted(rng.choice(221,int(rng.integers(1,3)),replace=False).tolist()) for _ in range(8)]
                ids=tokens.repeat(8,1)
                for j,pos in enumerate(positions):ids[j,[v+1 for v in pos]]=tok.mask_token_id
                with torch.inference_mode():logits=mlm(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
                prior_forwards+=len(ids);attempts+=len(ids)
                for j,pos in enumerate(positions):
                    for replacements in joint_top3(logits[j],pos,aa_ids,AA20):
                        seq=list(anchor)
                        for site,aa in zip(pos,replacements):seq[site]=aa
                        seq=''.join(seq)
                        if seq not in known and 1<=hamming(seq,anchor)<=2:candidates.add(seq)
                        if len(candidates)>=budget:break
                    if len(candidates)>=budget:break
                del logits,ids
            candidates=sorted(candidates);rows=[]
            for st in range(0,len(candidates),16):
                batch=candidates[st:st+16]
                ids=torch.stack([seq_to_ids0(s,221,'cuda') for s in batch])
                with torch.inference_mode():
                    pred=gen._predict_hard_ids_batch(ids,ref,mask,config).detach().float().cpu().numpy()
                rows.extend({'sequence':s,'pred':float(y),'seed_row':int(start.row_id),'random_seed':seed,
                             'method':method,'hamming_from_seed':hamming(s,anchor)} for s,y in zip(batch,pred))
            elapsed=time.monotonic()-t0
            alloc=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
            # Match other search methods: common risk rescoring is separately counted.
            for st in range(0,len(candidates),16):
                batch=candidates[st:st+16];ids=torch.stack([seq_to_ids0(s,221,'cuda') for s in batch])
                with torch.inference_mode():
                    pred,terr,_,_=gen._forward_from_probs(torch.nn.functional.one_hot(ids,20).float(),anchor_ids,ref,mask,config)
                cols={k:v.detach().float().cpu().numpy() for k,v in terr.items() if v.ndim==1 and len(v)==len(batch)}
                for j in range(len(batch)):
                    assert abs(rows[st+j]['pred']-float(pred[j]))<2e-5
                    rows[st+j].update({k:float(v[j]) for k,v in cols.items()})
            frame=pd.DataFrame(rows)
            if a.wide:
                from freeze_discovery_evaluation import COLS
                frame=frame.reindex(columns=COLS)
                frame['start_id']=int(start.start_id);frame['mountain_id']=start.mountain_id;frame['role']=start.role
            frame.to_csv(folder/'candidate_pool.csv',index=False)
            record=dict(method=method,seed_row=int(start.row_id),cluster=int(start.cluster),role=start.role,
                random_seed=seed,forward_sequences=len(candidates),gradient_forward_sequences=0,
                common_rescoring_sequences=len(candidates),mlm_proposal_sequences=prior_forwards,seconds=elapsed,
                valid_unique_candidates=len(candidates),matched_random_adalead_forward_budget=budget,
                peak_gpu_allocated_gib=alloc,peak_gpu_reserved_gib=reserved,budget_filled=len(candidates)==budget)
            if a.wide:
                record.update(start_id=int(start.start_id),mountain_id=start.mountain_id,
                    representative_source=start.representative_source,candidate_pool_sha256=sha(folder/'candidate_pool.csv'))
            write_json(folder/'DONE.json',record);records.append(record);completed+=1
            pd.DataFrame(records).to_csv(out/'runtime_and_queries.csv',index=False)
            print(json.dumps(record),flush=True)
            if a.max_runs and completed>=a.max_runs:return
    write_json(out/'DONE.json',dict(method=method,runs=len(records),starts=len(starts),seeds=1 if a.wide else 3))

if __name__=='__main__':main()
