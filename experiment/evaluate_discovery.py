"""Evaluate frozen end-to-end outputs, with explicit measured and proxy columns."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from experiment_core import write_json
from freeze_discovery_evaluation import sha

KEY=['scope','replicate','method','policy']

def chars(sequences):
    return np.asarray([list(s.encode('ascii')) for s in sequences],dtype=np.uint8)

def min_hamming_to_reference(sequences, known, reference):
    # Exact sparse-mutation identity avoids a sequence x train x position tensor.
    k=chars(known);wt=chars([reference])[0];depth=np.count_nonzero(k!=wt,axis=1)
    result=[]
    for seq in sequences:
        c=chars([seq])[0];pos=np.flatnonzero(c!=wt)
        distances=depth+len(pos)
        if len(pos):
            distances=distances-np.count_nonzero(k[:,pos]!=wt[pos],axis=1)-np.count_nonzero(k[:,pos]==c[pos],axis=1)
        result.append(int(distances.min()))
    return np.asarray(result)

def diversity(sequences):
    if len(sequences)<2:return np.nan
    c=chars(sequences)
    total=sum(np.count_nonzero(c[i+1:]!=c[i],axis=1).sum() for i in range(len(c)-1))
    return float(total/(len(c)*(len(c)-1)/2))

def evaluate(root,cohort,partial=False):
    base=root/'artifacts/models/random/glom_seed20261013'
    out=base/'discovery_evaluation'/cohort
    frozen=json.loads((out/'FROZEN.json').read_text())
    for name,digest in frozen['frozen_files'].items():assert sha(out/name)==digest
    s=pd.read_csv(out/'frozen_selected.csv');counts=pd.read_csv(out/'frozen_selection_counts.csv')
    starts=pd.read_csv(base/'search/starting_sequences.csv').set_index('row_id')
    judges=[f for f in ['esm2_650m','esmc_6b'] if (out/(f+'_DONE.json')).exists()]
    if not partial:assert len(judges)==2
    # Only this post-freeze stage opens test fluorescence labels.
    truth=pd.read_csv(root/'data/test.csv').set_index('sequence')
    s['measured_brightness']=s.sequence.map(truth.brightness)
    train=pd.read_csv(root/'data/train.csv',usecols=['sequence'])
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    seqs=sorted(set(s.sequence));dist=min_hamming_to_reference(seqs,train.sequence.tolist(),wt)
    s['nearest_train_hamming']=s.sequence.map(dict(zip(seqs,dist)))
    assert s.nearest_train_hamming.ge(1).all()
    for feature in judges:
        d=pd.read_csv(out/(feature+'_predictions.csv')).set_index('sequence')
        j=json.loads((out/(feature+'_DONE.json')).read_text())
        assert sha(out/(feature+'_predictions.csv'))==j['predictions_sha256']
        s[feature+'_mean']=s.sequence.map(d['mean'])
        s[feature+'_seed_sd']=s.sequence.map(d.seed_sd)
        s[feature+'_lower_reference']=s[feature+'_mean']-j['calibration_overprediction_q90']
        startmap=starts.sequence.map(d['mean']).to_dict()
        s[feature+'_gain_vs_origin']=s[feature+'_mean']-s.seed_row.map(startmap)
        for seed in [20261013,20261014,20261015]:s[feature+f'_seed{seed}']=s.sequence.map(d[f'pred_seed{seed}'])
    s.to_csv(out/'selected_candidates_evaluated.csv',index=False)
    rows=[]
    for keys,g in s.groupby(KEY,sort=True):
        measured=g.measured_brightness.dropna()
        r=dict(zip(KEY,keys));r.update(glom_pred_mean=float(g.pred.mean()),glom_pred_min=float(g.pred.min()),
            pairwise_hamming_mean=diversity(g.sequence.tolist()),nearest_train_hamming_mean=float(g.nearest_train_hamming.mean()),
            hamming_from_origin_mean=float(g.hamming_from_seed.mean()),exact_measured_n=len(measured),
            exact_measured_fraction=len(measured)/len(g),
            measured_mean=float(measured.mean()) if len(measured) else None,
            measured_high420=float(measured.ge(4.2).mean()) if len(measured) else None,
            measured_high435=float(measured.ge(4.35).mean()) if len(measured) else None)
        for feature in judges:
            p=g[feature+'_mean']
            r.update({feature+'_mean':float(p.mean()),feature+'_minimum':float(p.min()),
                feature+'_predicted_high420':float(p.ge(4.2).mean()),feature+'_predicted_high435':float(p.ge(4.35).mean()),
                feature+'_mean_gain_vs_origin':float(g[feature+'_gain_vs_origin'].mean()),
                feature+'_glom_optimism_gap':float((g.pred-p).mean()),
                feature+'_head_sd_mean':float(g[feature+'_seed_sd'].mean())})
            for seed in [20261013,20261014,20261015]:r[feature+f'_seed{seed}_mean']=float(g[feature+f'_seed{seed}'].mean())
        rows.append(r)
    summary=counts.merge(pd.DataFrame(rows),on=KEY,how='left',validate='one_to_one')
    summary.to_csv(out/'end_to_end_by_replicate.csv',index=False)
    numeric=['selected_n','full_batch','glom_pred_mean','pairwise_hamming_mean','nearest_train_hamming_mean',
             'exact_measured_n','measured_mean']+[f+'_mean' for f in judges]+[f+'_mean_gain_vs_origin' for f in judges]
    agg=summary.groupby(['scope','method','policy'])[numeric].agg(['mean','std','count'])
    agg.columns=['_'.join(c) for c in agg.columns];agg.reset_index().to_csv(out/'end_to_end_summary.csv',index=False)
    # All evaluated unique proposals are also checked against true test labels, not only top-ranked matches.
    pools=pd.read_csv(out/'frozen_candidate_pools.csv')
    pools['measured_brightness']=pools.sequence.map(truth.brightness)
    exact=[];matchrows=[]
    for method,g in pools.groupby('method'):
        g=g.sort_values('pred',ascending=False).drop_duplicates('sequence');m=g[g.measured_brightness.notna()].copy()
        exact.append(dict(method=method,unique_generated=len(g),exact_test_matches=len(m),
            coverage=len(m)/len(g) if len(g) else 0,measured_mean=float(m.measured_brightness.mean()) if len(m) else None,
            measured_high420=float(m.measured_brightness.ge(4.2).mean()) if len(m) else None))
        matchrows.append(m)
    pd.DataFrame(exact).to_csv(out/'generated_exact_match_summary.csv',index=False)
    pd.concat(matchrows,ignore_index=True).to_csv(out/'generated_exact_matches.csv',index=False)
    # Cluster resampling: average seeds within each start before resampling starts.
    paired=[];rng=np.random.default_rng(20261013)
    block=summary[summary.scope.eq('start_seed')].copy()
    block['start']=block.replicate.str.split('_').str[0]
    for policy,g in block.groupby('policy'):
        for metric in ['selected_n','glom_pred_mean','pairwise_hamming_mean']+[f+'_mean' for f in judges]:
            ref=g[g.method.eq('tgito_risk')][['replicate','start',metric]].rename(columns={metric:'reference'})
            for method,h in g.groupby('method'):
                if method=='tgito_risk':continue
                z=ref.merge(h[['replicate',metric]],on='replicate').dropna(subset=['reference',metric])
                z['delta']=z.reference-z[metric];values=z.groupby('start').delta.mean().to_numpy()
                if not len(values):continue
                boot=rng.choice(values,(2000,len(values)),replace=True).mean(1)
                paired.append(dict(policy=policy,metric=metric,comparator=method,paired_blocks=len(z),
                    independent_starts=len(values),tgito_minus_comparator=float(values.mean()),
                    ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),
                    conditional_note='Quality uses nonempty matched batches; selected_n includes empty batches'))
    pd.DataFrame(paired).to_csv(out/'paired_start_cluster_bootstrap.csv',index=False)
    # Main operational comparison includes all seeds, including zero-yield batches.
    q=summary[summary.scope.eq('campaign_seed') & summary.policy.eq('final6_confidence')]
    if q.empty:q=summary[summary.scope.eq('start_seed') & summary.policy.eq('screen30_prediction')]
    if not q.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
        metrics=[f+'_mean' for f in judges]+['selected_n','pairwise_hamming_mean']
        labels=[f.replace('esm2_650m','ESM-2 650M').replace('esmc_6b','ESMC-6B')+' prediction' for f in judges]+['Selected candidates','Pairwise Hamming distance']
        names={'tgito_risk':'Full TGITO','tgito_no_risk_gate':'No risk gate','tgito_no_terrain':'No terrain',
               'tgito_single_view':'Single view','random':'Random','adalead':'AdaLead','prothmso_top3_2026':'ProtHMSO (2026)'}
        fig,axes=plt.subplots(1,len(metrics),figsize=(3.3*len(metrics)+1.5,4.5),squeeze=False,sharey=True)
        for panel,(ax,metric,label) in enumerate(zip(axes[0],metrics,labels)):
            m=q.groupby('method')[metric].agg(['mean','std']).sort_index()
            for i,(method,row) in enumerate(m.iterrows()):
                ax.errorbar(row['mean'],i,xerr=0 if pd.isna(row['std']) else row['std'],fmt='o',
                            color='#147d92' if method=='tgito_risk' else '#738797',capsize=3,markersize=6)
            ax.set_yticks(np.arange(len(m)));ax.set_yticklabels([names.get(x,x) for x in m.index])
            ax.set_xlabel(label);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.margins(y=.12,x=.18)
        fig.suptitle('End-to-end discovery: mean and SD across three search seeds' if frozen['complete_campaign']
                     else 'Workflow pilot: one paired start and seed, Top-30 screening')
        fig.tight_layout();fig.savefig(out/'end_to_end_comparison.png',dpi=180);fig.savefig(out/'end_to_end_comparison.svg');plt.close(fig)
        table=q.groupby('method')[['selected_n','glom_pred_mean']+[f+'_mean' for f in judges]+['pairwise_hamming_mean']].mean()
        (out/'table_end_to_end.tex').write_text(table.to_latex(float_format='%.4f',escape=True),encoding='utf-8')
    write_json(out/'EVALUATION_DONE.json',dict(cohort=cohort,complete_campaign=frozen['complete_campaign'],
        complete_judges=len(judges)==2,judges=judges,selected_rows=len(s),exact_selected_matches=int(s.measured_brightness.notna().sum()),
        new_wetlab_results=False,selection_and_judges_frozen_before_readout=True,
        biological_readout='Exact hidden-test matches only; other sequence scores remain computational predictions',
        primary='Small 12-start pilot diagnostics only; external predictions are auxiliary, not experimental ground truth',
        secondary='Start-cluster paired bootstrap and whole-pool exact-measurement coverage'))
    print(json.dumps({'evaluated':cohort,'judges':judges,'rows':len(s),'complete':frozen['complete_campaign']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--cohort',default='full');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    evaluate(a.root,a.cohort,a.allow_partial)
