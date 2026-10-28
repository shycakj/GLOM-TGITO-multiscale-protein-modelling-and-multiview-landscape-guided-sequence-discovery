"""Descriptive multi-threshold comparison on frozen embeddings and native batches."""
from pathlib import Path
import json, hashlib, zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
P=ROOT/'artifacts/discovery_v3/embedding_familiarity_v1'
OUT=P/'threshold_grid_v2'
QS=[.50,.60,.70,.75,.80,.85,.90,.95,.975,.99]
METHODS=['glom_tgito','glom_adalead','glom_random','glom_prothmso','esm650_prothmso']

def wilson(k,n):
    if not n:return (None,None)
    z=1.95996398454;p=k/n;a=1+z*z/n
    c=(p+z*z/(2*n))/a;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return float(c-h),float(c+h)

def error_stats(d):
    e=d.prediction.to_numpy()-d.brightness.to_numpy();n=len(d)
    large=int((abs(e)>.1).sum());over=int((e>.1).sum());lo,hi=wilson(large,n)
    return dict(n=n,rmse=float(np.sqrt(np.mean(e*e))),mae=float(abs(e).mean()),large_error_n=large,large_error_rate=large/n,large_error_ci_low=lo,large_error_ci_high=hi,overprediction_n=over,overprediction_rate=over/n)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(quantiles=QS,k=32,reference_populations=['all','predicted_high420'],primary_metric_unchanged=True,denominator_for_primary_mean='Original high-reference q95, never changed across the new cutoff grid',candidate_selection='Frozen native Top6/30/100 and complete candidate pools, no reranking',purpose='Compare central support and sparse tails over the complete declared grid',threshold_selection='Calibration quantiles only; no maximizing between-method differences',bootstrap_n=2000,bootstrap_seed=20261016,scope='Calibration threshold variation with candidate sets fixed; not independent campaign repeats',lower_quantile_interpretation='Exceeding Q50-Q85 indicates less central representation support; it is not a binary OOD or biological failure label',high_quantile_interpretation='Q90-Q99 describe the more distant tail; measured sample sizes and error rates accompany each threshold')
    (OUT/'PLAN.json').write_text(json.dumps(plan,indent=2))
    s=pd.read_csv(P/'sequence_scores.csv',usecols=['feature_id','sequence','partition','prediction','z768_d32'])
    m=pd.read_csv(P/'membership.csv')
    cand=m.merge(s.drop(columns='sequence'),on='feature_id',validate='many_to_one')
    cal=s[s.partition=='calibration']
    test=pd.read_csv(ROOT/'data/test.csv',usecols=['sequence','brightness']).merge(s,on='sequence',validate='one_to_one')
    rng=np.random.default_rng(20261016);grid=[];validation=[];uncertainty=[];compact=[]
    for refname,ref,tt in [('all',cal,test),('predicted_high420',cal[cal.prediction>=4.2],test[test.prediction>=4.2])]:
        vals=ref.z768_d32.to_numpy();cuts=np.quantile(vals,QS,method='higher')
        draws=vals[rng.integers(0,len(vals),size=(2000,len(vals)))]
        bootcuts=np.quantile(draws,QS,axis=1,method='higher');del draws
        for j,(q,cut) in enumerate(zip(QS,cuts)):
            mask=tt.z768_d32>cut;outside=error_stats(tt[mask]);inside=error_stats(tt[~mask]);lo,hi=wilson(int(mask.sum()),len(tt))
            validation.append(dict(reference=refname,q=q,threshold=float(cut),calibration_n=len(ref),test_n=len(tt),expected_exceedance=1-q,test_exceedance=float(mask.mean()),test_exceedance_ci_low=lo,test_exceedance_ci_high=hi,inside=inside,outside=outside,rmse_ratio=outside['rmse']/inside['rmse'],large_error_risk_ratio=outside['large_error_rate']/inside['large_error_rate'],min_group_n=min(inside['n'],outside['n'])))
            for (method,scope),d in cand.groupby(['method','scope'],sort=False):
                v=d.z768_d32.to_numpy();count=int((v>cut).sum())
                grid.append(dict(reference=refname,quantile=q,threshold=float(cut),method=method,scope=scope,n=len(v),exceed_n=count,exceed_fraction=count/len(v),mean_raw_distance=float(v.mean())))
                if scope=='top100':
                    counts=(v[:,None]>bootcuts[j][None,:]).sum(0)
                    cilo,cihi=np.quantile(counts,[.025,.975]);tlo,thi=np.quantile(bootcuts[j],[.025,.975])
                    uncertainty.append(dict(reference=refname,quantile=q,method=method,n=len(v),exceed_n=count,exceed_n_ci_low=float(cilo),exceed_n_ci_high=float(cihi),threshold=float(cut),threshold_ci_low=float(tlo),threshold_ci_high=float(thi)))
        # Download only compact calibration/test/Top100 rows to make subsequent statistics local.
    selected_ids=set(cand[cand.scope=='top100'].feature_id)
    small=s[s.partition.isin(['calibration','test'])|s.feature_id.isin(selected_ids)][['feature_id','partition','prediction','z768_d32']].copy()
    truth=test.set_index('feature_id').brightness
    small['brightness']=small.feature_id.map(truth)
    small.to_csv(OUT/'compact_sequence_scores.csv.gz',index=False,compression='gzip')
    cand[cand.scope=='top100'][['method','scope','native_rank','feature_id']].to_csv(OUT/'compact_top100_membership.csv',index=False)
    gd=pd.DataFrame(grid);vd=pd.json_normalize(validation);ud=pd.DataFrame(uncertainty)
    gd.to_csv(OUT/'threshold_comparison.csv',index=False);vd.to_csv(OUT/'measured_validation.csv',index=False);ud.to_csv(OUT/'calibration_bootstrap.csv',index=False)
    # Check earlier evaluated cutoffs exactly, preserving their meaning.
    old=pd.read_csv(P/'threshold_audit/top100_threshold_sensitivity.csv');old=old[old.k==32]
    new=gd[gd.scope=='top100'].merge(old,on=['reference','method'],suffixes=('_new','_old'))
    new=new[np.isclose(new['quantile'],new['q'])]
    assert len(new)==40 and (new.exceed_n==new.outside_n).all()
    assert np.allclose(new.threshold_new,new.threshold_old,atol=1e-12)
    done=dict(status='complete',old40cutoff_counts_reproduced=True,generated_labels_imputed=False,source_hashes={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['sequence_scores.csv','membership.csv','PLAN.json']},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'DONE.json').write_text(json.dumps(done,indent=2))
    with zipfile.ZipFile(OUT/'tables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['PLAN.json','DONE.json','threshold_comparison.csv','measured_validation.csv','calibration_bootstrap.csv','compact_top100_membership.csv']:z.write(OUT/name,name)
    print(gd[(gd.reference=='predicted_high420')&(gd.scope=='top100')].pivot(index='method',columns='quantile',values='exceed_n').loc[METHODS].to_string())
    print(vd[vd.reference=='predicted_high420'][['q','inside.n','outside.n','inside.rmse','outside.rmse','large_error_risk_ratio']].to_string(index=False))
    print('TABLES_BYTES', (OUT/'tables.zip').stat().st_size,'COMPACT_BYTES',(OUT/'compact_sequence_scores.csv.gz').stat().st_size,flush=True)

if __name__=='__main__':main()
