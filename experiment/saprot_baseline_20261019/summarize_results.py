"""Compare unchanged GLOM predictions with every completed native SaProt run."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from experiment_core import regression_metrics, write_json
METHODS=['glom','saprot650_pred3di_mlp','saprot650_pred3di_lora']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args();spec=json.loads((HERE/'PROTOCOL.json').read_text())
    out=ROOT/'reports/saprot_baseline_20261019';out.mkdir(exist_ok=True)
    rows=[];predictions={};missing=[];sources={}
    for split in spec['splits']:
        for method in METHODS:
            for seed in spec['seeds']:
                folder=ROOT/'artifacts/models'/split/f'{method}_seed{seed}'
                archive=HERE/'reference_glom_results'/split/f'{method}_seed{seed}'
                if method=='glom' and archive.exists():folder=archive
                p=folder/'test_predictions.csv'
                if not (folder/'DONE.json').exists() or not p.exists():
                    missing.append(dict(split=split,method=method,seed=seed));continue
                d=pd.read_csv(p).rename(columns={'brightness':'true','prediction':'pred'}).sort_values('row_id')
                assert not d.row_id.duplicated().any()
                rows.append(dict(split=split,method=method,seed=seed,**regression_metrics(d.true,d.pred)))
                predictions[split,method,seed]=d[['row_id','true','pred']]
                sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'per_seed_metrics.csv',index=False)
    metrics=[c for c in frame.columns if c not in ['split','method','seed','n']]
    summary=frame.groupby(['split','method'])[metrics].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary['completed_seeds']=frame.groupby(['split','method']).size()
    summary.reset_index().to_csv(out/'mean_sd_metrics.csv',index=False)
    effects=[]
    for split in spec['splits']:
        for method in METHODS[1:]:
            keys=[(split,m,s) for m in ['glom',method] for s in spec['seeds']]
            if any(k not in predictions for k in keys):continue
            base=predictions[split,'glom',spec['seeds'][0]]
            for k in keys:
                d=predictions[k]
                assert np.array_equal(base.row_id,d.row_id),k
                assert np.allclose(base.true,d.true,atol=5e-7,rtol=0),k
            truth=base.true.to_numpy()
            g=np.stack([predictions[split,'glom',s].pred.to_numpy() for s in spec['seeds']])
            b=np.stack([predictions[split,method,s].pred.to_numpy() for s in spec['seeds']])
            def differences(ix):
                ge=g[:,ix]-truth[ix];be=b[:,ix]-truth[ix]
                return np.array([np.mean(np.sqrt(np.mean(ge**2,axis=1))-np.sqrt(np.mean(be**2,axis=1))),
                                 np.mean(np.abs(ge)-np.abs(be))])
            point=differences(np.arange(len(truth)));rng=np.random.default_rng(20261019)
            boot=np.asarray([differences(rng.integers(0,len(truth),len(truth))) for _ in range(args.bootstrap)])
            for j,metric in enumerate(['rmse','mae']):
                low,high=np.quantile(boot[:,j],[.025,.975])
                effects.append(dict(split=split,baseline=method,metric=metric,glom_minus_baseline=point[j],
                                    ci_low=low,ci_high=high,test_rows=len(truth),seeds=3,
                                    bootstrap_unit='paired test rows; average fixed-seed metrics'))
    pd.DataFrame(effects).to_csv(out/'paired_error_differences.csv',index=False)
    write_json(out/'STATUS.json',{'status':'PARTIAL' if missing else 'COMPLETE',
               'completed_model_runs':len(rows),'missing':missing,'source_sha256':sources,
               'native_saprot_inputs':'predicted 3Di paired with amino acids',
               'not_new_wetlab_or_discovery_results':True,
               'bootstrap_replicates':args.bootstrap,
               'uncertainty':'Row bootstrap conditional on fitted models; seed SD is reported separately'})
    print(summary[['completed_seeds','rmse_mean','mae_mean','r2_mean','spearman_mean']].to_string())
    print(json.dumps({'missing_model_runs':len(missing),'paired_comparisons':len(effects)}))

if __name__=='__main__':main()
