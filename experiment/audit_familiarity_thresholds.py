"""Audit the frozen familiarity cutoff; never reselect a favorable threshold."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
P=ROOT/'artifacts/discovery_v3/embedding_familiarity_v1'
OUT=P/'threshold_audit'

def wilson(k,n):
    z=1.959963984540054;p=k/n;a=1+z*z/n
    center=(p+z*z/(2*n))/a;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return [float(center-half),float(center+half)]

def err_metrics(d):
    if not len(d):return dict(n=0)
    e=d.prediction.to_numpy()-d.brightness.to_numpy()
    return dict(n=len(d),rmse=float(np.mean(e**2)**.5),mae=float(abs(e).mean()),
        abs_error_gt01=float((abs(e)>.1).mean()),overprediction_gt01=float((e>.1).mean()),
        mean_overprediction=float(e.mean()))

def main():
    OUT.mkdir(exist_ok=True)
    plan=dict(purpose='Cutoff audit; retain original q95/k32 and native selection',
        original_plan_sha256=hashlib.sha256((P/'PLAN.json').read_bytes()).hexdigest(),
        quantiles=[.90,.95,.975,.99],ks=[8,16,32,64],threshold_bootstraps=2000,
        brightness_reference_minima=[4.2,4.25,4.3,4.31],changes_search=False,
        mean_metric='arithmetic mean d32 / common calibration q95; rankings invariant to a shared positive denominator',
        bootstrap_scope='calibration threshold estimation only; frozen candidate batches not resampled as independent experiments')
    (OUT/'AUDIT_PLAN.json').write_text(json.dumps(plan,indent=2))
    cols=['feature_id','sequence','partition','prediction']+[f'z768_d{k}' for k in plan['ks']]+['esm_mean_d32']
    s=pd.read_csv(P/'sequence_scores.csv',usecols=cols)
    m=pd.read_csv(P/'membership.csv');cand=m.merge(s.drop(columns='sequence'),on='feature_id',validate='many_to_one')
    top=cand[cand.scope=='top100'].copy()
    cal=s[s.partition=='calibration'].copy()
    test=pd.read_csv(ROOT/'data/test.csv',usecols=['sequence','brightness']).merge(s,on='sequence',validate='one_to_one')
    rng=np.random.default_rng(20261015);grid=[];checks=[];boots=[];boundaries=[];bins=[];refcontrols=[]
    for refname,ref,tt in [('all',cal,test),('predicted_high420',cal[cal.prediction>=4.2],test[test.prediction>=4.2])]:
        for k in plan['ks']:
            col=f'z768_d{k}'
            for q in plan['quantiles']:
                cutoff=float(np.quantile(ref[col],q,method='higher'))
                for method,d in top.groupby('method',sort=False):
                    vals=d[col].to_numpy()
                    grid.append(dict(reference=refname,k=k,q=q,threshold=cutoff,method=method,n=len(d),
                        outside_n=int((vals>cutoff).sum()),outside_fraction=float((vals>cutoff).mean()),
                        mean_raw_distance=float(vals.mean()),mean_normalized_distance=float(vals.mean()/cutoff)))
                mask=tt[col]>cutoff;nout=int(mask.sum());ci=wilson(nout,len(tt))
                checks.append(dict(reference=refname,k=k,q=q,threshold=cutoff,calibration_n=len(ref),test_n=len(tt),
                    expected_tail=1-q,test_tail=nout/len(tt),test_tail_n=nout,test_tail_wilson_low=ci[0],test_tail_wilson_high=ci[1],
                    inside=err_metrics(tt[~mask]),outside=err_metrics(tt[mask])))
        vals=ref.z768_d32.to_numpy();cutoff=float(np.quantile(vals,.95,method='higher'))
        qs=np.concatenate([np.quantile(vals[rng.integers(len(vals),size=(100,len(vals)))],.95,axis=1,method='higher') for _ in range(20)])
        lo,hi=np.quantile(qs,[.025,.975])
        for method,d in top.groupby('method',sort=False):
            v=d.z768_d32.to_numpy();counts=(v[:,None]>qs[None,:]).sum(0)
            boots.append(dict(reference=refname,method=method,n=len(d),threshold=cutoff,threshold_ci_low=float(lo),threshold_ci_high=float(hi),
                outside_n_at_original=int((v>cutoff).sum()),outside_n_ci_low=float(np.quantile(counts,.025)),outside_n_ci_high=float(np.quantile(counts,.975)),
                outside_n_median=float(np.median(counts)),mean_raw_distance=float(v.mean()),
                mean_normalized_distance=float(v.mean()/cutoff),ratio_ci_low=float(np.quantile(v.mean()/qs,.025)),ratio_ci_high=float(np.quantile(v.mean()/qs,.975))))
            for row in d.itertuples():
                ratio=row.z768_d32/cutoff
                if .9<=ratio<=1.1 or ratio>1:
                    boundaries.append(dict(reference=refname,method=method,native_rank=row.native_rank,feature_id=row.feature_id,
                        prediction=row.prediction,distance=row.z768_d32,ratio=ratio,margin_percent=100*(ratio-1),
                        bootstrap_outside_fraction=float((row.z768_d32>qs).mean()),
                        interpretation='fraction across calibration bootstrap thresholds, not a sequence failure probability'))
        # Risk as a continuous function of distance, using bins frozen from calibration only.
        edges=np.quantile(vals,[0,.2,.4,.6,.8,.9,.95,.975,.99,1],method='higher')
        edges[0]=-np.inf;edges[-1]=np.inf
        idx=np.searchsorted(edges[1:-1],tt.z768_d32.to_numpy(),side='left')
        for j in range(len(edges)-1):
            d=tt.iloc[np.flatnonzero(idx==j)]
            bins.append(dict(reference=refname,bin=j,distance_low=float(edges[j]),distance_high=float(edges[j+1]),**err_metrics(d)))
    pd.DataFrame(grid).to_csv(OUT/'top100_threshold_sensitivity.csv',index=False)
    pd.json_normalize(checks).to_csv(OUT/'measured_threshold_validation.csv',index=False)
    pd.DataFrame(boots).to_csv(OUT/'threshold_bootstrap.csv',index=False)
    pd.DataFrame(boundaries).to_csv(OUT/'boundary_candidates.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'distance_error_bins.csv',index=False)
    for floor in plan['brightness_reference_minima']:
        ref=cal[cal.prediction>=floor];tt=test[test.prediction>=floor]
        cut=float(np.quantile(ref.z768_d32,.95,method='higher')) if len(ref)>=100 else None
        for method,d in top.groupby('method',sort=False):
            refcontrols.append(dict(reference_prediction_min=floor,calibration_n=len(ref),test_n=len(tt),threshold=cut,
                method=method,candidate_n=len(d),candidate_meets_reference_condition_n=int((d.prediction>=floor).sum()),
                candidate_prediction_min=float(d.prediction.min()),candidate_prediction_median=float(d.prediction.median()),
                outside_n=int((d.z768_d32>cut).sum()) if cut else None,
                test_tail=float((tt.z768_d32>cut).mean()) if cut and len(tt) else None,
                calibration_prediction_max=float(ref.prediction.max())))
    pd.DataFrame(refcontrols).to_csv(OUT/'brightness_reference_sensitivity.csv',index=False)
    # Frozen batch contrasts are descriptive; shared normalization cannot change their order.
    comparisons=[];ours=top[top.method=='glom_tgito'].z768_d32.mean()
    for method,d in top.groupby('method',sort=False):
        comparisons.append(dict(method=method,n=len(d),raw_mean=float(d.z768_d32.mean()),raw_median=float(d.z768_d32.median()),
            raw_p95=float(d.z768_d32.quantile(.95)),raw_max=float(d.z768_d32.max()),
            ours_relative_reduction=float(1-ours/d.z768_d32.mean()),
            esm_mean_distance=float(d.esm_mean_d32.mean())))
    pd.DataFrame(comparisons).to_csv(OUT/'continuous_distance_comparison.csv',index=False)
    result=dict(status='complete',original_cutoff_preserved=True,candidate_scores_unchanged=True,plan=plan,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob('*.csv')})
    (OUT/'DONE.json').write_text(json.dumps(result,indent=2))
    compact=dict(q95_checks=[x for x in checks if x['k']==32 and x['q']==.95],
        high_bootstrap=[x for x in boots if x['reference']=='predicted_high420'],
        high_k32_grid=[x for x in grid if x['reference']=='predicted_high420' and x['k']==32],
        high_q95_k_grid=[x for x in grid if x['reference']=='predicted_high420' and x['q']==.95],
        continuous=comparisons,brightness_reference=refcontrols)
    (OUT/'COMPACT.json').write_text(json.dumps(compact,indent=2))
    print(pd.DataFrame(grid).query("reference=='predicted_high420' and k==32").pivot(index='method',columns='q',values='outside_n').to_string())
    print(pd.DataFrame(boots).query("reference=='predicted_high420'")[['method','threshold_ci_low','threshold_ci_high','outside_n_ci_low','outside_n_ci_high']].to_string(index=False))
    print('AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
