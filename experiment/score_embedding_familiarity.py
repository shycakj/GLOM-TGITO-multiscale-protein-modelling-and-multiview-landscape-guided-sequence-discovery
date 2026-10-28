"""Common, label-free embedding support scores for every frozen method output."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score,r2_score
from experiment_core import write_json
from hpo_config_contract import sha
from audit_embedding_features import REL,prepare

def load_features(root):
    out=root/REL;plan=prepare(root);index=pd.read_csv(out/'feature_index.csv');n=len(index)
    filled=np.zeros(n,bool);z=np.empty((n,768),np.float32);pred=np.empty(n,np.float32);esm=None
    for shard in [0,1]:
        done=json.loads((out/f'shard_{shard}/DONE.json').read_text());assert done['plan_sha']==sha(out/'PLAN.json')
        for file in sorted((out/f'shard_{shard}').glob('chunk_*.npz')):
            with np.load(file) as data:
                assert str(data['plan_sha'])==done['plan_sha'];ids=data['feature_id']
                assert not filled[ids].any();filled[ids]=True
                z[ids]=data['z768'];pred[ids]=data['prediction']
                if esm is None:esm=np.empty((n,data['esm_mean'].shape[1]),np.float32)
                esm[ids]=data['esm_mean']
    assert filled.all() and np.isfinite(z).all() and np.isfinite(esm).all()
    return index,z,esm,pred

def distances(features,reference_ids,device='cuda',ks=(8,16,32,64),batch=192):
    # Double precision prevents cancellation for almost identical normalized proteins.
    x=torch.as_tensor(features,dtype=torch.float64,device=device)
    norms=x.norm(dim=1,keepdim=True);assert (norms>1e-12).all()
    x=x/norms;ref=x[reference_ids];result=np.empty((len(x),len(ks)),np.float64)
    self_column=np.full(len(x),-1,dtype=np.int64);self_column[reference_ids]=np.arange(len(reference_ids))
    for st in range(0,len(x),batch):
        q=x[st:st+batch];d2=(q.square().sum(1)[:,None]+ref.square().sum(1)[None,:]-2*(q@ref.T)).clamp_min_(0)
        cols=self_column[st:st+len(q)];rows=np.flatnonzero(cols>=0)
        if len(rows):d2[torch.as_tensor(rows,device=device),torch.as_tensor(cols[rows],device=device)]=torch.inf
        values=torch.topk(d2,max(ks),largest=False,sorted=True).values.sqrt()
        result[st:st+len(q)]=values[:,[k-1 for k in ks]].cpu().numpy()
    # Independent direct subtraction check, including identity exclusion.
    for i in np.linspace(0,len(x)-1,7,dtype=int):
        direct=(ref-x[i]).square().sum(1).sqrt()
        if self_column[i]>=0:direct[self_column[i]]=torch.inf
        expected=direct.topk(max(ks),largest=False,sorted=True).values[[k-1 for k in ks]].cpu().numpy()
        np.testing.assert_allclose(result[i],expected,rtol=1e-6,atol=2e-8)
    return result

def validation_table(frame,score_names):
    rows=[]
    for scope,part in [('all',frame),('predicted_high420',frame[frame.prediction>=4.2])]:
        if len(part)<3:continue
        e=part.prediction.to_numpy()-part.brightness.to_numpy()
        for score in score_names:
            s=part[score].to_numpy();rec=dict(scope=scope,score=score,n=len(part),
                absolute_error_spearman=float(spearmanr(s,abs(e)).statistic),
                overprediction_spearman=float(spearmanr(s,e).statistic))
            for errorcut in [.1,.2]:
                for what,target in [('absolute',abs(e)>errorcut),('overprediction',e>errorcut)]:
                    stem=f'{what}_gt{errorcut}'
                    rec[stem+'_prevalence']=float(target.mean())
                    rec[stem+'_auroc']=float(roc_auc_score(target,s)) if len(np.unique(target))==2 else None
                    rec[stem+'_auprc']=float(average_precision_score(target,s)) if target.any() else None
            rows.append(rec)
    return pd.DataFrame(rows)

def metrics(d):
    e=d.prediction.to_numpy()-d.brightness.to_numpy()
    return dict(n=len(d),rmse=float(np.mean(e**2)**.5),mae=float(abs(e).mean()),
        r2=float(r2_score(d.brightness,d.prediction)) if len(d)>1 else None,
        spearman=float(spearmanr(d.brightness,d.prediction).statistic) if len(d)>2 else None,
        mean_overprediction=float(e.mean()),false_high420=float((d.brightness<4.2).mean()) if (d.prediction>=4.2).all() else None)

def main(a):
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False
    root=a.root;out=root/REL;index,z,esm,pred=load_features(root)
    ref=index.index[index.partition=='train'].to_numpy();cal=index.partition.eq('calibration').to_numpy()
    highcal=cal&(pred>=4.2);assert highcal.sum()>50
    scored=index.copy();scored['prediction']=pred;thresholds={};started=time.monotonic()
    for name,features in [('z768',z),('G',z[:,:256]),('Rf',z[:,256:512]),('Kf',z[:,512:]),('esm_mean',esm)]:
        target=out/f'distances_{name}.npz'
        if target.exists():
            with np.load(target) as old:
                assert str(old['plan_sha'])==sha(out/'PLAN.json');dist=old['distances']
        else:
            dist=distances(features,ref,device=a.device)
            np.savez(target,distances=dist,plan_sha=sha(out/'PLAN.json'))
        for j,k in enumerate([8,16,32,64]):
            col=f'{name}_d{k}';v=dist[:,j];q=float(np.quantile(v[cal],.95,method='higher'));qh=float(np.quantile(v[highcal],.95,method='higher'))
            assert q>0 and qh>0
            scored[col]=v;scored[col+'_q95_ratio']=v/q;scored[col+'_highcal_q95_ratio']=v/qh
            thresholds[col]=dict(q95=q,highcal_q95=qh,cal_n=int(cal.sum()),highcal_n=int(highcal.sum()))
        print(json.dumps(dict(stage='distance',representation=name,rows=len(index),seconds=time.monotonic()-started)),flush=True)
    primary=scored.z768_d32.to_numpy();cal_sorted=np.sort(primary[cal])
    scored['familiarity_percentile']=np.searchsorted(cal_sorted,primary,side='right')/len(cal_sorted)
    scored['beyond_calibration_max']=primary>cal_sorted[-1]
    scored['outside_q95']=scored.z768_d32_q95_ratio>1
    scored['outside_highcal_q95']=scored.z768_d32_highcal_q95_ratio>1
    scored.to_csv(out/'sequence_scores.csv',index=False)
    membership=pd.read_csv(out/'membership.csv');candidates=membership.merge(scored.drop(columns='sequence'),on='feature_id',validate='many_to_one')
    candidates.to_csv(out/'candidate_scores.csv',index=False)
    rows=[]
    for (method,scope),d in candidates.groupby(['method','scope'],sort=False):
        rows.append(dict(method=method,scope=scope,n=len(d),outside_q95_n=int(d.outside_q95.sum()),outside_q95_fraction=float(d.outside_q95.mean()),
            outside_highcal_q95_n=int(d.outside_highcal_q95.sum()),outside_highcal_q95_fraction=float(d.outside_highcal_q95.mean()),
            distance_median=float(d.z768_d32.median()),degree_mean=float(d.z768_d32_q95_ratio.mean()),
            degree_median=float(d.z768_d32_q95_ratio.median()),degree_p95=float(d.z768_d32_q95_ratio.quantile(.95)),
            highcal_degree_mean=float(d.z768_d32_highcal_q95_ratio.mean()),beyond_cal_max_fraction=float(d.beyond_calibration_max.mean()),
            **{f'{v}_outside_q95_fraction':float((d[f'{v}_d32_q95_ratio']>1).mean()) for v in ['G','Rf','Kf','esm_mean']},
            **{f'k{k}_outside_q95_fraction':float((d[f'z768_d{k}_q95_ratio']>1).mean()) for k in [8,16,64]}))
    pd.DataFrame(rows).to_csv(out/'method_summary.csv',index=False)
    test=pd.read_csv(root/'data/test.csv',usecols=['sequence','brightness']).merge(scored,on='sequence',validate='one_to_one')
    test.to_csv(out/'test_scores.csv',index=False)
    validation_table(test,[f'{v}_d32' for v in ['z768','G','Rf','Kf','esm_mean']]).to_csv(out/'test_score_validation.csv',index=False)
    groups=[]
    for scope,d in [('all',test),('predicted_high420',test[test.prediction>=4.2])]:
        for threshold in ['outside_q95','outside_highcal_q95']:
            for flag,part in d.groupby(threshold):
                groups.append(dict(scope=scope,threshold=threshold,outside=bool(flag),**metrics(part)))
        for fraction in [.2,.4,.6,.8,1.]:
            part=d.sort_values('z768_d32',kind='stable').head(max(1,int(len(d)*fraction)))
            groups.append(dict(scope=scope,threshold='risk_coverage',retained_fraction=fraction,**metrics(part)))
    pd.DataFrame(groups).to_csv(out/'test_group_metrics.csv',index=False)
    old=pd.read_csv(root/'artifacts/discovery_v3/evaluation/glom_live_predictions.csv',usecols=['sequence','glom_20261013'])
    check=old.merge(scored[['sequence','prediction']],on='sequence');delta=check.prediction-check.glom_20261013
    replay=dict(n=len(check),rmse=float(np.sqrt(np.mean(delta**2))),max_abs=float(abs(delta).max()))
    assert replay['max_abs']<1e-4, replay
    done=dict(status='complete',rows=len(index),reference_n=len(ref),method_groups=len(rows),thresholds=thresholds,
        source_script_sha=sha(Path(__file__)),plan_sha=sha(out/'PLAN.json'),live_prediction_replay=replay,
        sequence_identity_metric=False,fluorescence_imputed=False,seconds=time.monotonic()-started,
        note='Outside q95 is operational representation support, not error probability or pretraining membership.')
    write_json(out/'SCORES_DONE.json',done);print(json.dumps(done),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--device',default='cuda');main(p.parse_args())
