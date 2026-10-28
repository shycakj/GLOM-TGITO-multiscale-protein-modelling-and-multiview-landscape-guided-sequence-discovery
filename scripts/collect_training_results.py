"""Collect completed training results without invoking historical search queues."""
from pathlib import Path
import json
import pandas as pd

R=Path(__file__).resolve().parent
out=R/'reports';out.mkdir(exist_ok=True)
rows=[]
for p in sorted((R/'artifacts/models').glob('*/*')):
    if not p.is_dir() or 'smoke' in p.name or not (p/'DONE.json').exists():continue
    if (p/'metrics.csv').exists(): records=pd.read_csv(p/'metrics.csv').to_dict('records')
    elif (p/'evaluation.json').exists():records=json.loads((p/'evaluation.json').read_text())
    else:continue
    method,seed=p.name.rsplit('_seed',1)
    for x in records:
        if x.get('partition')=='test':rows.append(dict(method=method,split=p.parent.name,seed=int(seed),**x))
for split,folder in [('random',R/'artifacts/baselines'),('depth',R/'artifacts/baselines/depth')]:
    for p in folder.glob('*_v2_metrics.csv'):
        for x in pd.read_csv(p).to_dict('records'):
            if x.get('partition')=='test':rows.append(dict(split=split,seed=20261013,**x))
p=R/'artifacts/proteinttt_650m/metrics.csv'
if p.exists():
    for x in pd.read_csv(p).to_dict('records'):
        if x.get('partition')!='test':continue
        tag=x.pop('method');kind=x.pop('scoring')
        seed=int(tag.rsplit('_seed',1)[1]) if '_seed' in tag else 20261013
        method=('proteinttt_650m' if '_seed' in tag else 'esm2_650m_unadapted')+'_'+kind
        rows.append(dict(method=method,split='random',seed=seed,**x))
d=pd.DataFrame(rows);d.to_csv(out/'completed_regression_runs.csv',index=False)
d.groupby(['split','method'])[['rmse','mae','r2','spearman','pearson']].agg(['mean','std','count']).to_csv(out/'regression_mean_sd.csv')
print('Collected',len(d),'test records:',out)
