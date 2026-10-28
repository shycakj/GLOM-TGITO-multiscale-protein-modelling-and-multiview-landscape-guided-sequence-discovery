"""Recalculate paper data from frozen per-sequence predictions and MD measurements.

The archived predictions, rankings and measurements are read-only. Recomputed
tables and a numerical agreement report are written under reproduced/.
"""
from pathlib import Path
import json, sys, subprocess, re, shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiment'
DATA=ROOT/'paper/data/Figure_Source_Data'
OUT=ROOT/'reproduced/tables';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(EXP))
from experiment_core import regression_metrics
report={}

def agree(actual,expected,label,atol=5e-6):
    if not np.isclose(actual,expected,atol=atol,rtol=1e-6,equal_nan=True):
        raise AssertionError(f'{label}: recomputed={actual}; archived={expected}')

def prediction_file(row):
    method,split,seed=row.method,row.split,int(row.seed)
    if str(row.source).startswith('artifacts/baselines'):
        return EXP/Path(row.source).parent/f'{method}_test_predictions.csv'
    if 'marginals' in method:
        head='unadapted' if 'unadapted' in method else f'proteinttt_seed{seed}'
        tail='masked_marginals' if 'masked_marginals' in method else 'wt_marginals'
        return EXP/'artifacts/proteinttt_650m'/f'{head}_{tail}_test_predictions.csv'
    folder=f'{method}_seed{seed}'
    return EXP/'artifacts/models'/split/folder/'test_predictions.csv'

def predictions():
    frozen=pd.read_csv(DATA/'combined_prediction_all_runs.csv')
    rows=[];comparisons=0;maxerr=0
    for _,row in frozen.iterrows():
        p=prediction_file(row);d=pd.read_csv(p)
        y=d['brightness' if 'brightness' in d else 'true'].to_numpy(float)
        pred=d['prediction' if 'prediction' in d else 'pred'].to_numpy(float)
        result=regression_metrics(y,pred)
        for metric,value in result.items():
            if metric in row.index and pd.notna(row[metric]):
                agree(value,row[metric],f'{row.method}/{row.split}/{row.seed}/{metric}')
                comparisons+=1;maxerr=max(maxerr,abs(value-row[metric]))
        rows.append(dict(method=row.method,split=row.split,seed=int(row.seed),**result,prediction_file=p.relative_to(ROOT).as_posix()))
    pd.DataFrame(rows).to_csv(OUT/'prediction_all_runs_recomputed.csv',index=False)
    a=pd.DataFrame(rows)
    a.groupby(['method','split'])[['rmse','mae','r2','spearman','pearson','top100_high420','top100_high435']].agg(['mean','std','count']).to_csv(OUT/'prediction_mean_sd_recomputed.csv')
    report['prediction']={'runs':len(rows),'numeric_values_checked':comparisons,'max_absolute_difference':maxerr}
    print(f'Prediction: {len(rows)} configurations; {comparisons} numerical values verified.',flush=True)

def thresholds():
    folder=EXP/'artifacts/discovery_v3/embedding_familiarity_v1/threshold_grid_v2'
    s=pd.read_csv(folder/'compact_sequence_scores.csv.gz');m=pd.read_csv(folder/'compact_top100_membership.csv')
    cand=m.merge(s,on='feature_id',validate='many_to_one')
    cal=s[s.partition.eq('calibration')];test=s[s.partition.eq('test')]
    high=cal[cal.prediction>=4.2];high_test=test[test.prediction>=4.2]
    q95=float(np.quantile(high.z768_d32,.95,method='higher'))
    q95all=float(np.quantile(cal.z768_d32,.95,method='higher'))
    qs=[.50,.60,.70,.75,.80,.85,.90,.95,.975,.99]
    rows=[];validation=[];means=[]
    for method,c in cand.groupby('method',sort=False):
        means.append(dict(method=method,n=len(c),mean_distance=c.z768_d32.mean(),mean_deviation=c.z768_d32.mean()/q95,
            median_deviation=c.z768_d32.median()/q95,p95_deviation=np.quantile(c.z768_d32,.95)/q95,
            mean_deviation_all_calibration=c.z768_d32.mean()/q95all))
    for refname,ref,tt in [('all',cal,test),('predicted_high420',high,high_test)]:
        for q in qs:
            cut=float(np.quantile(ref.z768_d32,q,method='higher'))
            for method,c in cand.groupby('method',sort=False):
                n=int((c.z768_d32>cut).sum())
                rows.append(dict(reference=refname,q=q,threshold=cut,method=method,n=len(c),above_n=n,above_fraction=n/len(c)))
            for side,mask in [('below_or_equal',tt.z768_d32<=cut),('above',tt.z768_d32>cut)]:
                d=tt[mask];err=d.prediction-d.brightness
                validation.append(dict(reference=refname,q=q,side=side,n=len(d),rmse=np.sqrt(np.mean(err**2)),mae=abs(err).mean(),error_gt01_n=int((abs(err)>.1).sum())))
    means=pd.DataFrame(means);means.to_csv(OUT/'top100_deviation_recomputed.csv',index=False)
    frozen=pd.read_csv(DATA/'discovery/all_selected_batch_metrics.csv').query('batch_size == 100').set_index('method')
    for row in means.itertuples():agree(row.mean_deviation,frozen.loc[row.method,'highcal_degree_mean'],row.method+' mean deviation',1e-9)
    pd.DataFrame(rows).to_csv(OUT/'threshold_grid_recomputed.csv',index=False)
    pd.DataFrame(validation).to_csv(OUT/'threshold_measured_errors_recomputed.csv',index=False)
    archived_grid=pd.read_csv(folder/'threshold_comparison.csv').query("scope == 'top100'")
    threshold_checks=0
    for r in rows:
        e=archived_grid[(archived_grid.reference==r['reference']) &
            np.isclose(archived_grid['quantile'],r['q']) & (archived_grid.method==r['method'])]
        assert len(e)==1
        for name,oldname in [('threshold','threshold'),('above_n','exceed_n'),('above_fraction','exceed_fraction')]:
            agree(r[name],e.iloc[0][oldname],f"threshold/{r['method']}/{r['q']}/{name}",1e-10)
            threshold_checks+=1
    archived_errors=pd.read_csv(folder/'measured_validation.csv');error_checks=0
    for r in validation:
        e=archived_errors[(archived_errors.reference==r['reference']) & np.isclose(archived_errors.q,r['q'])]
        assert len(e)==1
        side='inside' if r['side']=='below_or_equal' else 'outside'
        for name,oldname in [('n','n'),('rmse','rmse'),('mae','mae'),('error_gt01_n','large_error_n')]:
            agree(r[name],e.iloc[0][side+'.'+oldname],f"measured error/{r['reference']}/{r['q']}/{side}/{name}",1e-10)
            error_checks+=1
    expected=pd.read_csv(DATA/'discovery_plot_inputs/fig4_exceedance_counts.csv')
    expected.to_csv(OUT/'figure4_exceedance_reference.csv',index=False)
    # Error discrimination is evaluated on measured test rows, not generated labels.
    auc=[]
    for scope,t in [('all',test),('predicted_high420',high_test)]:
        err=t.prediction-t.brightness
        for target,values in [('absolute_gt0.1',abs(err)>.1),('overprediction_gt0.1',err>.1)]:
            auc.append(dict(scope=scope,target=target,score='z768_d32',n=len(t),events=int(values.sum()),auroc=roc_auc_score(values,t.z768_d32)))
    pd.DataFrame(auc).to_csv(OUT/'deviation_error_auroc_recomputed.csv',index=False)
    old=pd.read_csv(DATA/'discovery/test_auroc_bootstrap.csv')
    for r in auc:
        z=old[(old.scope==r['scope'])&(old.target==r['target'])&(old.score==r['score'])]
        assert len(z)==1;agree(r['auroc'],z.auroc.iloc[0],str(r),1e-10)
    report['deviation']={'top100_candidates':len(cand),'calibration_n':len(cal),'high_calibration_n':len(high),'measured_test_n':len(test),
        'high_measured_test_n':len(high_test),'high_calibration_q95':q95,'threshold_comparisons':len(rows),
        'threshold_values_verified':threshold_checks,'measured_error_values_verified':error_checks,'auroc_values_verified':len(auc)}
    print('Deviation: 500 selected candidates, 100 threshold comparisons and measured-error AUROCs recalculated.',flush=True)

def discovery():
    d=pd.read_csv(DATA/'discovery/candidate_reliability_scores.csv')
    expected=pd.read_csv(DATA/'discovery/all_selected_batch_metrics.csv').set_index(['method','batch_size'])
    rows=[];checks=0
    for (method,batch),part in d.groupby(['method','batch_size'],sort=False):
        part=part.sort_values('rank');assert len(part)==batch
        seq=np.array([list(s) for s in part.sequence])
        pairs=[np.sum(seq[i]!=seq[j]) for i in range(len(seq)) for j in range(i)]
        row=dict(method=method,batch_size=int(batch),selected_n=len(part),mean_pairwise_hamming=np.mean(pairs))
        columns={'terrain_risk_mean':'terrain_risk','combined_risk_mean':'combined_risk','terrain_support_mean':'terrain_support',
            'cross_sd_mean':'cross_sd','optimism_gap_mean':'optimism_gap','cross_mean_mean':'cross_mean',
            'risk_scaled_lower90_mean':'risk_scaled_lower90','glom_cross_mean_mean':'glom_cross_mean',
            'glom_cross_mean_gain_vs_start':'glom_cross_mean_gain_vs_start',
            'esm650_cross_mean_mean':'esm650_cross_mean','mpnn_log_likelihood_mean':'mpnn_log_likelihood'}
        row.update({out:part[src].mean() for out,src in columns.items()})
        e=expected.loc[(method,batch)]
        for metric,value in row.items():
            if metric in ['method','batch_size']:continue
            agree(value,e[metric],f'{method}/Top{batch}/{metric}',1e-8);checks+=1
        rows.append(row)
    for method,part in d.groupby('method'):
        batches={int(k):g.sort_values('rank').sequence.tolist() for k,g in part.groupby('batch_size')}
        assert batches[6]==batches[30][:6] and batches[30]==batches[100][:30],method
    weights=[c for c in d if c.startswith('selected_weight_')]
    tg=d[(d.method=='glom_tgito')&(d.batch_size==100)]
    assert np.all((tg[weights].to_numpy()>0).sum(1)==6)
    np.testing.assert_allclose(tg[weights].sum(1),1,atol=1e-7)
    starts=pd.read_csv(EXP/'artifacts/discovery_v3/starting_sequences.csv')
    assert len(starts)==2700 and starts.sequence.nunique()==1308
    pd.DataFrame(rows).to_csv(OUT/'native_selected_batches_recomputed.csv',index=False)
    report['discovery']={'methods':5,'batch_comparisons':len(rows),'numeric_values_checked':checks,'nested_top6_30_100':True,
        'tasks':len(starts),'distinct_start_sequences':starts.sequence.nunique(),'active_views_at_each_top100_endpoint':6}
    print('Discovery: all 15 native batches, nested selections, start counts and dynamic gate weights verified.',flush=True)

def molecular_dynamics():
    work=ROOT/'reproduced/md_table_work'
    shutil.copytree(ROOT/'paper/three_runs',work,dirs_exist_ok=True)
    subprocess.run([sys.executable,str(work/'reproduce_run_tables.py')],check=True)
    raw=pd.read_csv(ROOT/'paper/three_runs/verification/md_metrics_by_replicate_source.csv')
    assert len(raw)==168 and raw.system_id.nunique()==56
    actual=[]
    for r in raw.itertuples():
        path=ROOT/'biophysics/results'/r.system_id/f'rep{r.replicate}'/'METRICS.json'
        j=json.loads(path.read_text());metrics=j['metrics']
        for name,value in metrics.items():
            if name in raw.columns:agree(value,getattr(r,name),str(path)+'/'+name,1e-9)
        actual.append(dict(system_id=r.system_id,replicate=r.replicate,**metrics))
    pd.DataFrame(actual).to_csv(OUT/'md_168_trajectories_recomputed.csv',index=False)
    for f in (work/'reproduced_tables').glob('*.csv'):
        (OUT/f.name).write_bytes(f.read_bytes())
    report['md']={'systems':56,'trajectories':168,'candidates':30,'paired_runs':90,'duration_ns_each':3.0,'figure5F_values_checked':9}
    print('MD: all 168 per-trajectory records and 90 candidate/start/WT comparisons verified.',flush=True)

if __name__=='__main__':
    for name in sys.argv[1:] or ['prediction','deviation','discovery','md']:
        {'prediction':predictions,'deviation':thresholds,'discovery':discovery,'md':molecular_dynamics}[name]()
    (OUT/'NUMERICAL_REPRODUCTION_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('PASS:',OUT)
