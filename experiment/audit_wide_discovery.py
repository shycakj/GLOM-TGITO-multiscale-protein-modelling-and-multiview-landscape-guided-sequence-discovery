"""Meaningful contract checks before the 900-mountain launch."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from build_wide_peak_plan import find_peaks,sha
from evaluate_wide_discovery import distance_matrix,folder_for,CORE
from experiment_core import write_json
ROOT=Path(__file__).resolve().parent;BASE=ROOT/'artifacts/models/random/glom_seed20261013'
plan=pd.read_csv(BASE/'wide_search/starting_sequences.csv')
assert len(plan)==2700 and plan.mountain_id.nunique()==900
assert plan.groupby('mountain_id').role.apply(lambda x:set(x)=={'summit','ridge','edge'}).all()
assert plan.start_id.is_unique and plan.random_seed.is_unique
known={role:set(pd.read_csv(ROOT/f'data/{role}.csv',usecols=['sequence']).sequence) for role in ['train','validation','calibration','test']}
assert set(plan.sequence)<=known['train']
for role in ['validation','calibration','test']:assert not set(plan.sequence)&known[role]
# Synthetic peaks: plateaus are not strict maxima; isolated cells are retained.
xy=np.array([[0,0],[.5,.5],[1,1]],float)
assert len(find_peaks(xy,np.array([1.,2.,3.]),50))==3
xy=np.array([[0,0],[.001,.001],[1,1]],float)
assert set(i for i,_ in find_peaks(xy,np.array([1.,2.,3.]),50))=={1,2}
# Exact sparse Hamming agrees with direct full-length comparison.
rng=np.random.default_rng(999);reference='A'*221;seqs=[]
for _ in range(35):
    c=list(reference)
    for j in rng.choice(221,int(rng.integers(0,12)),replace=False):c[j]=rng.choice(list('ACDEFGHIKLMNPQRSTVWY'))
    seqs.append(''.join(c))
d=distance_matrix(seqs,seqs,reference)
brute=np.array([[sum(x!=y for x,y in zip(a,b)) for b in seqs] for a in seqs])
assert np.array_equal(d,brute)
start=next(plan.itertuples());meta={}
for method in CORE:
    folder=folder_for(BASE,start,method);done=json.loads((folder/'DONE.json').read_text())
    assert sha(folder/'candidate_pool.csv')==done['candidate_pool_sha256']
    p=pd.read_csv(folder/'candidate_pool.csv');assert set(p.start_id)=={int(start.start_id)}
    assert not set(p.sequence)&set.union(known['train'],known['validation'],known['calibration'])
    assert p.sequence.map(lambda s:sum(a!=b for a,b in zip(s,start.sequence))).between(1,2).all()
    meta[method]=done
    if method.startswith('tgito'):
        f=pd.read_csv(folder/'native_final.csv').iloc[0]
        assert f.valid_output and f.final_seq in set(p.sequence)
for m in CORE[1:]:assert meta[m]['forward_sequences']==meta['tgito_risk']['forward_sequences']
write_json(ROOT/'audit/wide_search_contract.json',dict(passed=True,mountains=900,contexts=2700,
    unique_start_sequences=int(plan.sequence.nunique()),plan_sha256=sha(BASE/'wide_search/starting_sequences.csv'),
    exact_hamming_pairs=35*35,paired_methods=CORE,
    first_context_fitness_budget=meta['tgito_risk']['forward_sequences'],test_labels_loaded=False,
    first_context_runtimes={m:r['seconds'] for m,r in meta.items()},
    first_context_peak_gpu_gib={m:r['peak_gpu_allocated_gib'] for m,r in meta.items()}))
print((ROOT/'audit/wide_search_contract.json').read_text())
