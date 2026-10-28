"""Native full discovery baselines with their own anchors and fitness ranking.

Official FLEXS AdaLead and the ProtHMSO (2026) joint masked top-3 proposal rule,
adapted to the same GFP two-substitution task. No terrain/confidence is attached
to these methods. The two ProtHMSO predictors share label-free proposals only.
"""
from pathlib import Path
import argparse,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import torch
from experiment_core import setup_project,seed_all,write_json
from hpo_config_contract import sha,verify_input_hashes
from discovery_v3 import paths,load_protocol

class Fitness:
    def __init__(self,root,need_mlm=False,need_glom=True):
        setup_project(root);torch.set_num_threads(2)
        from pipeline.tgito.aa import AA20,seq_to_ids0,soft_pos_feats
        from position_features import canonical_bits
        self.aa=AA20;self.ids=seq_to_ids0;self.pos=soft_pos_feats;self.bits=canonical_bits('cuda');self.root=root
        wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip();self.ref=seq_to_ids0(wt,221,'cuda')
        if need_glom:
            from encoder_contract import TrainingCompatibleEncoder
            from pipeline.tgito.esm_bridge import load_bridge
            from pipeline.tgito.revised_slim_internal import load_revised_slim
            os.environ['PPLU_ESM_MODEL_PATH']=str(root/'repro_project/assets/esm2_t30_150M_UR50D')
            self.model,self.mean,self.std=load_revised_slim(paths(root)[0]/'best.pt',device='cuda')
            self.enc=TrainingCompatibleEncoder(load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda'),device='cuda',freeze_esm=True,target_h_dim=1280).eval()
        if need_mlm:
            from transformers import AutoTokenizer,AutoModelForMaskedLM
            from score_discovery_independent import head_model
            path=root/'repro_project/assets/esm2_650m'
            self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
            self.mlm=AutoModelForMaskedLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval()
            self.aa_ids=torch.tensor([self.tok.convert_tokens_to_ids(c) for c in self.aa],device='cuda')
            self.head_checkpoint,self.head=head_model(root/'artifacts/models/random/esm2_650m_mlp_seed20261013/best.pt')
    def predict(self,sequences,predictor='glom'):
        from score_discovery_independent import apply_head
        values=[]
        with torch.inference_mode():
            for start in range(0,len(sequences),16):
                batch=sequences[start:start+16]
                if predictor=='glom':
                    ids=torch.stack([self.ids(s,221,'cuda') for s in batch]);probs=torch.nn.functional.one_hot(ids,20).float()
                    out=self.model.forward_all(self.enc(probs)['H'],self.pos(probs,self.ref,self.bits))
                    y=(out['y_norm']*self.std+self.mean).cpu().numpy()
                else:
                    tokens=self.tok(batch,return_tensors='pt',padding=True,return_special_tokens_mask=True)
                    mask=(tokens.pop('special_tokens_mask')==0)&tokens['attention_mask'].bool()
                    tokens={k:v.cuda() for k,v in tokens.items()};mask=mask.cuda()
                    h=self.mlm.esm(**tokens).last_hidden_state.float()
                    x=((h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy()
                    y=apply_head(self.head_checkpoint,self.head,x)
                assert np.isfinite(y).all();values.extend(map(float,y))
        return np.asarray(values)

def distance(a,b):return sum(x!=y for x,y in zip(a,b))

def random_sequences(anchor,budget,known,rng,aa):
    candidates=set();attempts=0
    while len(candidates)<budget:
        child=list(anchor)
        for p in rng.choice(221,int(rng.integers(1,3)),replace=False):child[p]=rng.choice([c for c in aa if c!=anchor[p]])
        child=''.join(child);attempts+=1
        if child not in known:candidates.add(child)
        if attempts>budget*1000:raise RuntimeError('Random feasible-proposal exhaustion')
    return sorted(candidates)

def masked_sequences(fitness,anchor,budget,known,rng):
    from run_recent_search_baseline import joint_top3
    candidates=set();prior_forwards=0;tokens=fitness.tok(anchor,return_tensors='pt')['input_ids'].cuda()
    assert tokens.shape==(1,223)
    while len(candidates)<budget and prior_forwards<50*budget:
        positions=[sorted(rng.choice(221,int(rng.integers(1,3)),replace=False).tolist()) for _ in range(16)]
        ids=tokens.repeat(16,1)
        for j,pos in enumerate(positions):ids[j,[p+1 for p in pos]]=fitness.tok.mask_token_id
        with torch.inference_mode():logits=fitness.mlm(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
        prior_forwards+=len(ids)
        for j,pos in enumerate(positions):
            for replacements in joint_top3(logits[j],pos,fitness.aa_ids,fitness.aa):
                child=list(anchor)
                for p,c in zip(pos,replacements):child[p]=c
                child=''.join(child)
                if child not in known and 1<=distance(child,anchor)<=2:candidates.add(child)
                if len(candidates)==budget:break
            if len(candidates)==budget:break
    return sorted(candidates),prior_forwards

def run_adalead(root,fitness,anchor,brightness,budget,known):
    from run_search_comparison import load_adalead
    Adalead,utils=load_adalead(root);original=utils.generate_random_mutant;rows=[]
    def bounded(parent,mu,alphabet):
        for _ in range(100000):
            child=original(parent,mu,alphabet)
            if child not in known and 1<=distance(anchor,child)<=2:return child
        raise RuntimeError('AdaLead constrained proposal exhaustion')
    class Oracle:
        cost=0;name='Frozen_GLOM'
        def get_fitness(self,seqs):
            n=min(len(seqs),max(0,budget-self.cost));result=np.full(len(seqs),-np.inf)
            if n:
                s=list(seqs[:n]);result[:n]=fitness.predict(s);self.cost+=n
                rows.extend(dict(sequence=x,pred=float(y),query_index=self.cost-n+i+1) for i,(x,y) in enumerate(zip(s,result[:n])))
            return result
        def train(self,*a,**kw):pass
    oracle=Oracle();utils.generate_random_mutant=bounded
    try:
        explorer=Adalead(oracle,rounds=1,sequences_batch_size=32,model_queries_per_batch=budget,
                        starting_sequence=anchor,alphabet=fitness.aa,mu=1,recomb_rate=0,rho=0,threshold=.05,eval_batch_size=16)
        explorer.propose_sequences(pd.DataFrame([dict(sequence=anchor,true_score=brightness)]))
    finally:utils.generate_random_mutant=original
    return pd.DataFrame(rows),oracle.cost

def run(root,method,shard,shards,limit=0,pilot=False):
    base,out=paths(root);protocol=load_protocol(root);verify_input_hashes(root,protocol['input_hashes'])
    if not pilot:
        gate=json.loads((out/'PREFLIGHT_PASSED.json').read_text());assert gate['protocol_sha256']==sha(out/'PROTOCOL.json')
    plan=pd.read_csv(out/'baseline_starting_sequences.csv');plan=plan[plan.start_id%shards==shard]
    if limit:plan=plan.head(limit)
    fitness=Fitness(root,need_mlm=method=='prothmso_pair');known=set()
    for name in ['train','validation','calibration']:known.update(pd.read_csv(root/f'data/{name}.csv',usecols=['sequence']).sequence)
    phase='baseline_pilot' if pilot else 'formal'
    methods=['glom_prothmso','esm650_prothmso'] if method=='prothmso_pair' else [method]
    for start in plan.itertuples():
        folders={m:out/phase/m/f'context{start.start_id:04d}_rep0' for m in methods}
        pending=[m for m in methods if not (folders[m]/'DONE.json').exists()]
        if not pending:continue
        if pilot:budget=550
        else:
            budget_file=out/'formal/glom_tgito'/f'context{start.start_id:04d}_rep0/DONE.json'
            # The queue launches after the corresponding TGITO campaign has completed.
            budget=int(json.loads(budget_file.read_text())['forward_sequences'])
        for folder in folders.values():folder.mkdir(parents=True,exist_ok=True)
        seed_all(int(start.random_seed));rng=np.random.default_rng(int(start.random_seed));torch.cuda.reset_peak_memory_stats()
        begin=time.monotonic();prior_count=0;proposal_seconds=0.
        if method=='prothmso_pair':
            seqs,prior_count=masked_sequences(fitness,start.sequence,budget,known,rng);proposal_seconds=time.monotonic()-begin
        elif method=='glom_random':seqs=random_sequences(start.sequence,budget,known,rng,fitness.aa)
        for m in pending:
            begin_score=time.monotonic()
            if m=='glom_adalead':frame,count=run_adalead(root,fitness,start.sequence,start.brightness,budget,known)
            else:
                values=fitness.predict(seqs,'esm650' if m=='esm650_prothmso' else 'glom');count=len(values)
                frame=pd.DataFrame(dict(sequence=seqs,pred=values,query_index=np.arange(1,len(values)+1)))
            assert count<=budget and len(frame)>0
            folder=folders[m];frame.to_csv(folder/'query_history.csv',index=False)
            valid=frame[~frame.sequence.isin(known)].copy();valid=valid[valid.sequence.map(lambda s:1<=distance(s,start.sequence)<=2)]
            valid=valid.sort_values(['pred','sequence'],ascending=[False,True],kind='stable').drop_duplicates('sequence')
            if len(valid):
                chosen=valid.head(1).copy();chosen['start_id']=int(start.start_id);chosen['seed_row']=int(start.row_id)
                chosen['method']=m;chosen['random_seed']=int(start.random_seed);chosen['hamming_from_seed']=distance(chosen.iloc[0].sequence,start.sequence)
                chosen.to_csv(folder/'hard_endpoint.csv',index=False)
            record=dict(start_id=int(start.start_id),method=m,valid=bool(len(valid)),forward_sequences=count,
                        matched_budget=budget,budget_filled=count==budget,gradient_graph_forwards=0,backward_steps=0,
                        mlm_proposal_sequences=prior_count,unique_hard_queries=int(frame.sequence.nunique()),
                        seconds=proposal_seconds+time.monotonic()-begin_score,proposal_seconds=proposal_seconds,
                        proposal_cost_shared_with_other_predictor=(method=='prothmso_pair'),peak_gpu_gib=torch.cuda.max_memory_reserved()/2**30,
                        protocol_sha256=sha(out/'PROTOCOL.json'),test_labels_read=False,
                        hashes={p.name:sha(p) for p in folder.glob('*.csv')})
            write_json(folder/'DONE.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--method',required=True,choices=['glom_random','glom_adalead','prothmso_pair']);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    run(a.root,a.method,a.shard,a.shards,a.limit,a.pilot)
