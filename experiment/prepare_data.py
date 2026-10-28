"""Freeze label-independent sequence splits and supervised training inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]/'handoff_20261001/repro_project')
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'data')
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    raw_path=a.project/'data/raw/pplu_31480_sequences.csv'
    raw=pd.read_csv(raw_path); raw.insert(0,'row_id',np.arange(len(raw)))
    raw=raw.rename(columns={'medianBrightness':'brightness'})
    d=raw.drop_duplicates('sequence',keep='first').copy()
    wt=(a.project/'data/processed/reference_sequence.txt').read_text().strip()
    d['mutation_depth']=[sum(x!=y for x,y in zip(s,wt)) for s in d.sequence]
    assert d.sequence.str.len().eq(221).all()
    d['sequence_sha256']=[hashlib.sha256(s.encode()).hexdigest() for s in d.sequence]
    def key(s): return hashlib.sha256(('glom-independent-20261013:'+s).encode()).hexdigest()
    order=sorted(range(len(d)),key=lambda i:key(d.iloc[i].sequence))
    n=len(order); limits=[0,int(.7*n),int(.8*n),int(.9*n),n]
    roles=['train','validation','calibration','test']
    role=np.empty(n,object)
    for r,lo,hi in zip(roles,limits[:-1],limits[1:]): role[order[lo:hi]]=r
    d['partition']=role
    cols=['row_id','sequence','sequence_sha256','brightness','mutation_depth','partition']
    d=d[cols].sort_values('row_id').reset_index(drop=True)
    for r in roles: d[d.partition.eq(r)].to_csv(a.output/(r+'.csv'),index=False)
    d.to_csv(a.output/'unique_sequences.csv',index=False)
    # A separate, fixed extrapolation challenge; not mixed into the random-split headline.
    ex=d.copy(); low=ex.index[ex.mutation_depth.le(2)].tolist()
    lo_order=sorted(low,key=lambda i:key(ex.at[i,'sequence']))
    nlow=len(lo_order); ex['partition']='test'
    for r,lo,hi in zip(roles[:3],[0,int(.8*nlow),int(.9*nlow)],[int(.8*nlow),int(.9*nlow),nlow]):
        ex.loc[lo_order[lo:hi],'partition']=r
    ex.to_csv(a.output/'depth_extrapolation.csv',index=False)
    summary={
      'protocol':'pplu-independent-retraining-20261013', 'split_seed':20261013,
      'label_independent_random_split':True,'raw_rows':len(raw),'unique_rows':len(d),
      'duplicate_policy':'retain first original row; no duplicate sequence crosses partitions',
      'partition_counts':d.partition.value_counts().to_dict(),
      'extrapolation_counts':ex.partition.value_counts().to_dict(),
      'extrapolation_rule':'train/validation/calibration depth<=2; test depth>=3',
      'train_seed_list':[20261013,20261014,20261015],
      'raw_sha256':digest(raw_path),
      'files':{p.name:digest(p) for p in sorted(a.output.glob('*.csv'))},
      'historical_supervised_checkpoint_allowed':False,
      'heldout_use':'test labels are read only by final scoring; calibration and validation are separate',
    }
    (a.output/'split_manifest.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['protocol','raw_rows','unique_rows','partition_counts','extrapolation_counts']},indent=2))

if __name__=='__main__':main()
