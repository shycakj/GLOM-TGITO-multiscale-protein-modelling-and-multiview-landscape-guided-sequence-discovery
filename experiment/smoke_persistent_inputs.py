from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
from discovery_v3_baselines import Fitness,random_sequences,masked_sequences,run_adalead
from experiment_core import write_json,seed_all
root=Path(__file__).resolve().parent
fit=Fitness(root,need_mlm=True);seed_all(20261014)
cal=pd.read_csv(root/'data/calibration.csv',usecols=['sequence','row_id']).head(32)
records={}
for predictor,run in [('glom','glom_seed20261013'),('esm650','esm2_650m_mlp_seed20261013')]:
    file=root/'artifacts/models/random'/run/'calibration_predictions.csv'
    header=pd.read_csv(file,nrows=0).columns
    field='prediction' if 'prediction' in header else 'pred'
    old=pd.read_csv(file,usecols=['row_id',field]).set_index('row_id')
    p=fit.predict(cal.sequence.tolist(),predictor);q=old.loc[cal.row_id,field].to_numpy();err=float(np.sqrt(np.mean((p-q)**2)))
    assert err<.02,(predictor,err)
    records[predictor]=dict(n=len(p),prediction_rmse_vs_original_cache=err,maximum_absolute_difference=float(abs(p-q).max()))
train=pd.read_csv(root/'data/train.csv');row=train.sort_values('brightness',ascending=False).iloc[0]
known=set(train.sequence)
for part in ['validation','calibration']:known.update(pd.read_csv(root/f'data/{part}.csv',usecols=['sequence']).sequence)
begin=time.monotonic();seqs,n=masked_sequences(fit,row.sequence,32,known,np.random.default_rng(20261014))
assert len(seqs)==32;records['mlm_proposals']=dict(n=32,mlm_forwards=n,seconds=time.monotonic()-begin)
frame,cost=run_adalead(root,fit,row.sequence,float(row.brightness),64,known)
assert cost==64 and np.isfinite(frame.pred).all()
records['adalead']=dict(cost=cost,records=len(frame))
records.update(passed=True,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,test_labels_read=False)
write_json(root/'audit/persistent_runtime_smoke.json',records);print(json.dumps(records),flush=True)
