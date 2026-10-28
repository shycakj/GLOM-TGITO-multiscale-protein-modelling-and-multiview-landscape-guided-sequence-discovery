"""Validate design-region predictor reliability against existing real assays.

Fixed subsets, no model fitting, no selection of a winning evaluator. Calibrated
residual references are empirical: optimization shift does not inherit coverage.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from audit_discovery_design import distances

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'audit/design_reliability';OUT.mkdir(parents=True,exist_ok=True)


def wilson(success,n):
    if not n:return [None,None]
    z=1.959963984540054;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]


def frame(split,method,partition,seeds):
    frames=[]
    for seed in seeds:
        path=ROOT/f'artifacts/models/{split}/{method}_seed{seed}/{partition}_predictions.csv'
        d=pd.read_csv(path).set_index('row_id')
        column='prediction' if 'prediction' in d else 'pred'
        frames.append(d[column].rename(str(seed)))
    preds=pd.concat(frames,axis=1);assert preds.notna().all().all()
    if 'brightness' not in d:
        data=pd.read_csv(ROOT/'data'/('unique_sequences.csv' if split=='random' else 'depth_extrapolation.csv')).set_index('row_id')
        d=data.loc[preds.index]
    result=d.loc[preds.index,['sequence','brightness','mutation_depth']].copy()
    result['pred']=preds.mean(1);result['seed_sd']=preds.std(1,ddof=0)
    return result


def main():
    protocol=dict(subsets=['all','pred_ge4.2','pred_ge4.31','top6','top30','top100',
        'nearest_train_1','nearest_train_2plus','depth_3plus','depth_3plus_pred_ge4.31'],
        methods=['glom','esmc_6b_mlp','esm2_650m_mlp'],
        seed_groups={'search_seed':[20261013],'unused_search_seeds':[20261014,20261015],'ensemble3':[20261013,20261014,20261015]},
        same_training_data=True,independent_ground_truth='Experimental held-out labels',
        no_test_tuning=True,calibration_quantile=.9,
        interval_status='Empirical calibration-residual reference; no guaranteed coverage for optimized candidates; depth split deliberately tests shift',
        audit_after_prior_test_reports=True)
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    wt=pd.read_csv(ROOT/'data/train.csv').query('mutation_depth == 0').iloc[0].sequence
    rows=[];calibrations=[];hashes={}
    for split in ['random','depth']:
        if split=='random':train=pd.read_csv(ROOT/'data/train.csv')
        else:train=pd.read_csv(ROOT/'data/depth_extrapolation.csv').query("partition == 'train'")
        first=frame(split,'glom','test',[20261013])
        distance=distances(first.sequence.tolist(),train.sequence.tolist(),wt).min(1)
        nearest=pd.Series(distance,index=first.index)
        for method in protocol['methods']:
            groups=protocol['seed_groups'] if method=='glom' else {'ensemble3':[20261013,20261014,20261015]}
            for group,seeds in groups.items():
                cal=frame(split,method,'calibration',seeds);d=frame(split,method,'test',seeds)
                d['nearest_train']=nearest.loc[d.index]
                residual=(cal.pred-cal.brightness).sort_values().to_numpy()
                q=float(residual[min(len(residual)-1,int(np.ceil((len(residual)+1)*.9))-1)])
                d['lower_reference']=d.pred-q
                masks={'all':np.ones(len(d),bool),'pred_ge4.2':d.pred.ge(4.2),
                       'pred_ge4.31':d.pred.ge(4.31), 'nearest_train_1':d.nearest_train.eq(1),
                       'nearest_train_2plus':d.nearest_train.ge(2),'depth_3plus':d.mutation_depth.ge(3),
                       'depth_3plus_pred_ge4.31':d.mutation_depth.ge(3)&d.pred.ge(4.31),
                       'lower_reference_ge4.2':d.lower_reference.ge(4.2)}
                for k in [6,30,100]:
                    top=d.sort_values(['pred','sequence'],ascending=[False,True]).head(k).index
                    masks[f'top{k}']=d.index.isin(top)
                for subset,mask in masks.items():
                    x=d.loc[mask];n=len(x)
                    if not n:continue
                    y=x.brightness.to_numpy();p=x.pred.to_numpy();error=p-y
                    k=int((y>=4.2).sum());ci=wilson(k,n)
                    denom=float(np.square(y-y.mean()).sum())
                    rows.append(dict(split=split,method=method,seed_group=group,subset=subset,n=n,
                        true_mean=float(y.mean()),pred_mean=float(p.mean()),
                        high420_fraction=k/n,high420_wilson_low=ci[0],high420_wilson_high=ci[1],
                        high435_fraction=float((y>=4.35).mean()),rmse=float(np.sqrt(np.square(error).mean())),
                        mae=float(np.abs(error).mean()),bias=float(error.mean()),
                        r2=float(1-np.square(error).sum()/denom) if denom else None,
                        spearman=float(spearmanr(y,p).statistic) if n>2 and np.std(y)>0 and np.std(p)>0 else None,
                        lower_reference_coverage=float((y>=x.lower_reference).mean()),
                        calibration_overprediction_q90=q,nearest_train_mean=float(x.nearest_train.mean())))
                calibrations.append(dict(split=split,method=method,seed_group=group,
                    calibration_n=len(cal),overprediction_q90=q,test_n=len(d)))
                if method=='glom' and group=='unused_search_seeds':d.to_csv(OUT/f'{split}_heldout_reliability_rows.csv')
                for seed in seeds:
                    for partition in ['calibration','test']:
                        path=ROOT/f'artifacts/models/{split}/{method}_seed{seed}/{partition}_predictions.csv'
                        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    data=pd.DataFrame(rows);data.to_csv(OUT/'heldout_design_region_metrics.csv',index=False)
    (OUT/'DONE.json').write_text(json.dumps(dict(protocol=protocol,input_sha256=hashes,records=len(data)),indent=2))
    print(data[(data.method=='glom')&(data.seed_group=='unused_search_seeds')&data.subset.isin(['all','pred_ge4.31','top30','nearest_train_2plus'])].to_json(orient='records',indent=2))


if __name__=='__main__':main()
