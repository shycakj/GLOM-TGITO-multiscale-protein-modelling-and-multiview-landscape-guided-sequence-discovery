"""Scientific invariants for candidate selection, sparse novelty, and joint MLM proposals."""
from pathlib import Path
import argparse, itertools, json
import numpy as np
import pandas as pd
import torch
from evaluate_discovery import min_hamming_to_reference,diversity,chars
from run_recent_search_baseline import joint_top3
from score_discovery_independent import head_model,apply_head
from experiment_core import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--cohort',default='pilot_paired');a=ap.parse_args();root=a.root
    rng=np.random.default_rng(1907);alphabet='ACDEFGHIKLMNPQRSTVWY';reference='A'*17
    def make():
        s=list(reference)
        for j in rng.choice(17,int(rng.integers(0,12)),replace=False):s[j]=rng.choice(list(alphabet))
        return ''.join(s)
    seq=[make() for _ in range(60)];known=[make() for _ in range(40)]
    fast=min_hamming_to_reference(seq,known,reference)
    brute=np.asarray([min(sum(x!=y for x,y in zip(s,k)) for k in known) for s in seq])
    assert np.array_equal(fast,brute)
    expected=np.mean([sum(x!=y for x,y in zip(s,t)) for i,s in enumerate(seq) for t in seq[i+1:]])
    assert abs(diversity(seq)-expected)<1e-12
    for _ in range(20):
        logits=torch.tensor(rng.normal(size=(19,20)),dtype=torch.float32)
        pos=[2,11];aa_ids=torch.arange(20)
        got=joint_top3(logits,pos,aa_ids,alphabet)
        raw=logits[[p+1 for p in pos]].log_softmax(-1).numpy()
        combinations=sorted(itertools.product(range(20),repeat=2),key=lambda ij:raw[0,ij[0]]+raw[1,ij[1]],reverse=True)[:3]
        assert got==[tuple(alphabet[i] for i in ij) for ij in combinations]
    base=root/'artifacts/models/random/glom_seed20261013';out=base/'discovery_evaluation'/a.cohort
    s=pd.read_csv(out/'frozen_selected.csv')
    for _,g in s[s.policy.str.startswith('final6')].groupby(['scope','replicate','method','policy']):
        assert len(g)<=6 and g.pred.ge(4.31).all()
        c=chars(g.sequence)
        for i in range(len(c)-1):assert np.count_nonzero(c[i+1:]!=c[i],axis=1).min()>=2
    starts=pd.read_csv(base/'search/starting_sequences.csv');anchors=[]
    q=pd.read_csv(out/'evaluator_sequences.csv');index=pd.Series(np.arange(len(q)),index=q.sequence)
    for feature in ['esm2_650m','esmc_6b']:
        f=out/(feature+'_embedding_state.json')
        if not f.exists():continue
        state=json.loads(f.read_text())
        if state['completed_rows']!=state['total_rows']:continue
        new=np.load(out/(feature+'_generated_embeddings.npy'),mmap_mode='r')
        cached=np.load(root/'artifacts/embeddings'/(feature+'_mean.npy'),mmap_mode='r')
        newx=np.asarray(new[index.loc[starts.sequence].to_numpy()]);oldx=np.asarray(cached[starts.row_id.to_numpy()])
        for seed in [20261013,20261014,20261015]:
            c,m=head_model(root/'artifacts/models/random'/f'{feature}_mlp_seed{seed}/best.pt')
            error=float(np.max(np.abs(apply_head(c,m,newx)-apply_head(c,m,oldx))))
            assert error<.01, f'New-sequence encoder contract mismatch: {feature} {error}'
            anchors.append(dict(feature=feature,seed=seed,max_anchor_prediction_difference=error,anchors=len(starts)))
    result=dict(passed=True,sparse_hamming_vs_bruteforce=60,diversity_vs_bruteforce=True,
                joint_top3_vs_full_400_combinations=20,final_selection_constraints=True,encoder_anchor_checks=anchors)
    write_json(root/'audit'/f'discovery_evaluation_{a.cohort}.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
