"""Evaluate the actual v13.5 final selection rule, preserving raw-ranking diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate_risk_screening import confidence

def choose(sequences,pred,ranking,k=6,min_pred=4.31,min_distance=2):
    chars=np.asarray([list(s) for s in sequences],dtype='U1');selected=[]
    for j in np.argsort(-np.asarray(ranking),kind='stable'):
        if pred[j]<min_pred:continue
        if all(np.count_nonzero(chars[j]!=chars[i])>=min_distance for i in selected):selected.append(int(j))
        if len(selected)>=k:break
    return np.asarray(selected,dtype=int)

def evaluate(root):
    root=Path(root);base=root/'artifacts/models/random/glom_seed20261013';risk=base/'risk_screening'
    if not (risk/'DONE.json').exists():return
    out=base/'submission_policy';out.mkdir(exist_ok=True)
    if not (out/'measured_libraries_DONE.json').exists():
        # Source constants are verified against the actual v13.5 launch script and selection code.
        protocol={'prediction_floor':4.31,'minimum_pairwise_hamming':2,'requested_candidates':6,
            'source':'LL_GF/v13_working_code/scripts_v13_5/04_confidence_and_select_candidates.ps1 calls modal_v13_2_candidate_confidence.py with its defaults',
            'normalization':'all candidates in the unlabeled pool, before filtering, matching original code',
            'novelty':'all evaluated sequences belong to hidden test/calibration libraries; the current library is not in training memory',
            'comparison':'all rankings use the same GLOM eligibility rule and pairwise-distance constraint',
            'earlier_unfiltered_analysis':'retained as score-component diagnostics; not replaced',
            'parameter_selection':'no new tuning; both fixed and already frozen calibration-selected terrain policies reported'}
        (out/'selection_protocol.json').write_text(json.dumps(protocol,indent=2))
        records=[];selected_rows=[]
        for partition in ['calibration','test']:
            data=pd.read_csv(root/f'data/{partition}.csv').set_index('row_id')
            raw={mode:pd.read_csv(risk/f'{partition}_{mode}_scores.csv') for mode in ['risk','calibrated','rule','single']}
            ids=raw['risk'].row_id.to_numpy()
            for d in raw.values():assert np.array_equal(d.row_id,ids)
            seq=data.loc[ids].sequence.to_numpy();y=data.loc[ids].brightness.to_numpy()
            rng=np.random.default_rng(20261013)
            pools=[('whole_library',-1,np.arange(len(ids)))]
            pools += [('subset_512',i,rng.choice(len(ids),min(512,len(ids)),replace=False)) for i in range(200)]
            for scope,library,idx in pools:
                scores={mode:confidence(d.iloc[idx]) for mode,d in raw.items()};p=scores['risk'].pred.to_numpy()
                rankings={'prediction':p,'confidence_full':scores['risk'].confidence_score.to_numpy(),
                    'confidence_without_final_risk':scores['risk'].confidence_without_final_risk.to_numpy(),
                    'calibrated_risk_full_confidence':scores['calibrated'].confidence_score.to_numpy(),
                    'rule_gate_full_confidence':scores['rule'].confidence_score.to_numpy(),
                    'single_view_full_confidence':scores['single'].confidence_score.to_numpy()}
                for method,r in rankings.items():
                    chosen=choose(seq[idx],p,r);values=y[idx[chosen]]
                    records.append({'partition':partition,'scope':scope,'library':library,'method':method,
                        'pool_n':len(idx),'eligible_n':int((p>=4.31).sum()),'selected_n':len(chosen),'full_six':len(chosen)==6,
                        'true_mean':float(values.mean()) if len(values) else None,
                        'true_min':float(values.min()) if len(values) else None,
                        'high420':float((values>=4.2).mean()) if len(values) else None,
                        'high435':float((values>=4.35).mean()) if len(values) else None})
                    if scope=='whole_library':
                        for rank,j in enumerate(chosen,1):selected_rows.append({'partition':partition,'method':method,
                            'rank':rank,'row_id':int(ids[idx[j]]),'sequence':seq[idx[j]],'pred':float(p[j]),'true':float(y[idx[j]])})
        d=pd.DataFrame(records);d.to_csv(out/'measured_policy_by_library.csv',index=False)
        pd.DataFrame(selected_rows).to_csv(out/'whole_library_selected_top6.csv',index=False)
        summary=[]
        for partition in ['calibration','test']:
            q=d[d.partition.eq(partition)&d.scope.eq('subset_512')]
            values=q.pivot(index='library',columns='method',values='true_mean')
            counts=q.pivot(index='library',columns='method',values='selected_n')
            for method in values:
                common=counts[method].eq(6)&counts.prediction.eq(6)
                diff=values.loc[common,method]-values.loc[common,'prediction']
                summary.append({'partition':partition,'method':method,'libraries':len(values),
                    'full_six_libraries':int(counts[method].eq(6).sum()),'jointly_full_with_prediction':int(common.sum()),
                    'mean_gain_vs_prediction_common_full':float(diff.mean()) if len(diff) else None,
                    'fraction_improved_common_full':float((diff>0).mean()) if len(diff) else None,
                    'median_selected_n':float(counts[method].median())})
        pd.DataFrame(summary).to_csv(out/'submission_policy_robustness.csv',index=False)
        (out/'measured_libraries_DONE.json').write_text(json.dumps({'records':len(d),'label_use':'final evaluation only'}))
    search=base/'search'
    if (search/'DONE.json').exists() and not (out/'search_selection_DONE.json').exists():
        pools=pd.read_csv(search/'all_candidates_exact_measured_matches.csv');selected=[];summaries=[]
        for method,group in pools.groupby('method'):
            g=group.sort_values('confidence_score',ascending=False,kind='stable').drop_duplicates('sequence').reset_index(drop=True)
            for ranking in ['pred','confidence_score']:
                jj=choose(g.sequence.to_numpy(),g.pred.to_numpy(),g[ranking].to_numpy());s=g.iloc[jj].copy()
                s['ranking']=ranking;s['rank']=np.arange(1,len(s)+1);selected.append(s)
                measured=s.measured_brightness.dropna()
                summaries.append({'method':method,'ranking':ranking,'pool_n':len(g),'eligible_n':int(g.pred.ge(4.31).sum()),
                    'selected_n':len(s),'predicted_mean':float(s.pred.mean()),'exact_measured_matches':len(measured),
                    'measured_mean_if_matched':float(measured.mean()) if len(measured) else None})
        pd.concat(selected,ignore_index=True).to_csv(out/'search_selected_top6.csv',index=False)
        pd.DataFrame(summaries).to_csv(out/'search_submission_summary.csv',index=False)
        (out/'search_selection_DONE.json').write_text(json.dumps({'methods':len(summaries)//2,'rule':'same original v13.5 final selection'}))
    (out/'DONE.json').write_text(json.dumps({'measured_policy_complete':True,
        'search_selection_complete':(out/'search_selection_DONE.json').exists()}))

if __name__=='__main__':evaluate(Path(__file__).resolve().parent)
