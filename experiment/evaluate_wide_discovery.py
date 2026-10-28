"""Measured rediscovery and navigation after freezing a source-balanced campaign.

The primary operational pool has ONE returned proposal per start, as in v13.5.
All queried proposals are evaluated separately for exact experimental rediscovery.
Hamming-neighborhood coverage is navigation evidence, never candidate fluorescence.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from build_wide_peak_plan import sha
from evaluate_risk_screening import confidence
from evaluate_submission_policy import choose
from experiment_core import write_json

CORE=['tgito_risk','random','adalead','prothmso_top3_2026']
ABLATIONS=['tgito_no_risk_gate','tgito_no_terrain','tgito_single_view']

def folder_for(base,start,method):
    return base/('wide_search_recent' if method=='prothmso_top3_2026' else 'wide_search')/(
        f'context{int(start.start_id):04d}_seed{int(start.random_seed)}_{method}')

def distance_matrix(sequences,targets,reference):
    """Exact full-length Hamming via sparse differences to a fixed reference."""
    k=np.array([list(s.encode('ascii')) for s in targets],np.uint8)
    wt=np.frombuffer(reference.encode('ascii'),np.uint8)
    depth=np.count_nonzero(k!=wt,axis=1)
    result=np.empty((len(sequences),len(targets)),np.int16)
    for i,s in enumerate(sequences):
        c=np.frombuffer(s.encode('ascii'),np.uint8);pos=np.flatnonzero(c!=wt)
        d=depth+len(pos)
        if len(pos):d=d-np.count_nonzero(k[:,pos]!=wt[pos],axis=1)-np.count_nonzero(k[:,pos]==c[pos],axis=1)
        result[i]=d
    return result

def freeze(root,mountains,ablations=False):
    base=root/'artifacts/models/random/glom_seed20261013'
    tag=f'prefix_{mountains:03d}'+('_ablations' if ablations else '')
    out=base/'wide_evaluation'/tag;out.mkdir(parents=True,exist_ok=True)
    if (out/'FROZEN.json').exists():
        f=json.loads((out/'FROZEN.json').read_text())
        for name,h in f['files'].items():assert sha(out/name)==h
        return out
    plan=pd.read_csv(base/'wide_search/starting_sequences.csv')
    plan=plan[plan.mountain_order<mountains]
    assert len(plan)==mountains*3
    methods=CORE+(ABLATIONS if ablations else [])
    sources=[];endpoints=[];counts=[];budgets=[]
    known=set()
    for p in ['train','validation','calibration']:
        known.update(pd.read_csv(root/f'data/{p}.csv',usecols=['sequence']).sequence)
    for start in plan.itertuples():
        for method in methods:
            folder=folder_for(base,start,method)
            if not (folder/'DONE.json').exists():raise RuntimeError(f'Not ready: {folder.name}')
            meta=json.loads((folder/'DONE.json').read_text());budgets.append(meta)
            digest=sha(folder/'candidate_pool.csv');assert digest==meta['candidate_pool_sha256']
            pool=pd.read_csv(folder/'candidate_pool.csv')
            assert not set(pool.sequence)&known
            assert set(pool.start_id)<={int(start.start_id)} and set(pool.method)<={method}
            assert pool.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{221}').all()
            d=pool.sequence.map(lambda s:sum(a!=b for a,b in zip(s,start.sequence)))
            assert d.between(1,2).all() and np.array_equal(d.to_numpy(),pool.hamming_from_seed.to_numpy())
            source=dict(folder=str(folder.relative_to(root)),pool_sha256=digest)
            valid=False
            if method.startswith('tgito'):
                source['native_final_sha256']=sha(folder/'native_final.csv')
                assert source['native_final_sha256']==meta['native_final_sha256']
                native=pd.read_csv(folder/'native_final.csv').iloc[0]
                final_seq=native.final_seq
                picked=pool[pool.sequence.eq(final_seq)]
                # Invalid/unchanged/known outputs remain zero-yield contexts.
                valid=bool(native.valid_output) and len(picked)==1
            else:
                picked=pool.sort_values(['pred','sequence'],ascending=[False,True]).head(1)
                valid=len(picked)==1
            if valid:
                row=picked.iloc[0].to_dict();row['representative_source']=start.representative_source
                row['mountain_order']=int(start.mountain_order);endpoints.append(row)
            counts.append(dict(start_id=int(start.start_id),mountain_id=start.mountain_id,
                seed_row=int(start.row_id),role=start.role,method=method,valid_endpoint=int(valid),pool_n=len(pool)))
            sources.append(source)
    end=pd.DataFrame(endpoints);assert len(end)>0
    end=confidence(end)
    selected=[];selection_counts=[]
    for method in methods:
        d=end[end.method.eq(method)].sort_values(['confidence_score','sequence'],ascending=[False,True]).drop_duplicates('sequence').reset_index(drop=True)
        for k in [6,30]:
            for policy,ranking in [('confidence','confidence_score'),('prediction','pred'),('no_final_risk','confidence_without_final_risk')]:
                jj=choose(d.sequence.to_numpy(),d.pred.to_numpy(),d[ranking].to_numpy(),k=k)
                x=d.iloc[jj].copy();x['policy']=policy;x['k']=k;x['rank']=np.arange(1,len(x)+1)
                selected.append(x);selection_counts.append(dict(method=method,k=k,policy=policy,
                    selected_n=len(x),full_batch=int(len(x)==k),unique_endpoint_pool=len(d)))
    end.to_csv(out/'frozen_endpoints.csv',index=False)
    pd.concat(selected,ignore_index=True).to_csv(out/'frozen_selected.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'endpoint_counts.csv',index=False)
    pd.DataFrame(selection_counts).to_csv(out/'selection_counts.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in budgets]).to_csv(out/'query_budgets.csv',index=False)
    plan.to_csv(out/'frozen_starts.csv',index=False)
    files=['frozen_endpoints.csv','frozen_selected.csv','endpoint_counts.csv','selection_counts.csv','query_budgets.csv','frozen_starts.csv']
    write_json(out/'FROZEN.json',dict(mountains=mountains,contexts=len(plan),methods=methods,
        files={n:sha(out/n) for n in files},sources=sources,
        plan_sha256=sha(base/'wide_search/starting_sequences.csv'),
        model_sha256=json.loads((base/'wide_search/PLAN_FROZEN.json').read_text())['model_sha256'],
        test_labels_read_before_freeze=False,external_predictors_used=False,
        primary_pool='One native returned TGITO proposal per context; highest-fitness proposal per baseline context',
        exact_match_scope='Returned proposals primary; full evaluated pools secondary',
        main_brightness_threshold=4.2,secondary_brightness_threshold=4.35,
        neighborhood_radii=[1,2],endpoint_hamming_constraints=[1,2],selection_floor=4.31,
        selected_pairwise_hamming_min=2,all_thresholds_fixed_before_wide_readout=True))
    return out

def evaluate(root,mountains,ablations=False):
    out=freeze(root,mountains,ablations);f=json.loads((out/'FROZEN.json').read_text())
    if (out/'DONE.json').exists():print(json.dumps(dict(already_done=str(out))));return
    # All choices above are immutable before accessing experimental test fluorescence.
    test=pd.read_csv(root/'data/test.csv');truth=test.set_index('sequence').brightness
    starts=pd.read_csv(out/'frozen_starts.csv');end=pd.read_csv(out/'frozen_endpoints.csv')
    selected=pd.read_csv(out/'frozen_selected.csv');counts=pd.read_csv(out/'endpoint_counts.csv')
    budgets=pd.read_csv(out/'query_budgets.csv');ref=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    sequences=sorted(set(starts.sequence)|set(end.sequence));lookup={s:i for i,s in enumerate(sequences)}
    distances=distance_matrix(sequences,test.sequence.tolist(),ref)
    # Deterministic independent direct comparison protects the exact-distance evaluation.
    rng=np.random.default_rng(20261013)
    for _ in range(80):
        i=int(rng.integers(len(sequences)));j=int(rng.integers(len(test)))
        assert distances[i,j]==sum(a!=b for a,b in zip(sequences[i],test.sequence.iloc[j]))
    initial=distances[[lookup[s] for s in starts.sequence]].min(0)
    reachable=initial<=2
    end['measured_brightness']=end.sequence.map(truth)
    selected['measured_brightness']=selected.sequence.map(truth)
    end.to_csv(out/'endpoints_evaluated.csv',index=False);selected.to_csv(out/'selected_evaluated.csv',index=False)
    rows=[];percontext=[];curve=[]
    bright=test.brightness.ge(4.2).to_numpy();strong=test.brightness.ge(4.35).to_numpy()
    for method in f['methods']:
        g=end[end.method.eq(method)].drop_duplicates('sequence');ii=[lookup[s] for s in g.sequence]
        near=distances[ii].min(0) if ii else np.full(len(test),999)
        measured=g.measured_brightness.dropna();cost=budgets[budgets.method.eq(method)]
        r=dict(method=method,contexts=len(starts),unique_returned=len(g),valid_contexts=int(counts[counts.method.eq(method)].valid_endpoint.sum()),
            exact_test_hits=len(measured),exact_bright420_hits=int((near[bright]==0).sum()),
            exact_bright435_hits=int((near[strong]==0).sum()),
            exact_bright420_recall=float((near[bright]==0).mean()),
            reachable_bright420_n=int((reachable&bright).sum()),
            exact_bright420_reachable_recall=float((near[reachable&bright]==0).mean()) if (reachable&bright).any() else None,
            measured_match_mean=float(measured.mean()) if len(measured) else None,
            measured_match_high420=float(measured.ge(4.2).mean()) if len(measured) else None,
            fitness_forward_sequences=int(cost.forward_sequences.sum()),
            search_seconds=float(cost.seconds.sum()),
            common_rescoring_sequences=int(cost.common_rescoring_sequences.sum()))
        for radius in [1,2]:
            b=near<=radius;b0=initial<=radius
            br=float(b[bright].mean());dr=float(b[~bright].mean())
            r.update({f'r{radius}_bright_covered':int((b&bright).sum()),
                f'r{radius}_initial_bright_covered':int((b0&bright).sum()),
                f'r{radius}_new_bright_covered':int((b&~b0&bright).sum()),
                f'r{radius}_lost_bright_covered':int((~b&b0&bright).sum()),
                f'r{radius}_bright_recall':br,f'r{radius}_low_brightness_recall':dr,
                f'r{radius}_bright_to_low_recall_ratio':br/dr if dr else None})
        rows.append(r)
        cg=end[end.method.eq(method)].set_index('start_id')
        for st in starts.itertuples():
            d0=distances[lookup[st.sequence]]
            d1=distances[lookup[cg.loc[st.start_id,'sequence']]] if st.start_id in cg.index else np.full(len(test),999)
            q=dict(method=method,start_id=int(st.start_id),mountain_id=st.mountain_id,seed_row=int(st.row_id),role=st.role,
                valid_endpoint=int(st.start_id in cg.index),exact_bright420_hits=int((d1[bright]==0).sum()))
            for radius in [1,2]:
                before=d0<=radius;after=d1<=radius
                q.update({f'r{radius}_new_bright':int((after&~before&bright).sum()),
                    f'r{radius}_lost_bright':int((~after&before&bright).sum()),
                    f'r{radius}_bright_minus_low_recall':float(after[bright].mean()-after[~bright].mean())})
            percontext.append(q)
        for n in sorted(set([x for x in [12,30,60,90,180,300,600,mountains] if x<=mountains])):
            part=end[end.method.eq(method)&end.mountain_order.lt(n)]
            qd=distances[[lookup[s] for s in set(part.sequence)]].min(0) if len(part) else np.full(len(test),999)
            curve.append(dict(method=method,mountains=n,contexts=n*3,exact_bright420_hits=int((qd[bright]==0).sum()),
                bright_radius1_covered=int((qd[bright]<=1).sum()),bright_radius2_covered=int((qd[bright]<=2).sum())))
    pd.DataFrame(rows).to_csv(out/'measured_discovery_summary.csv',index=False)
    pc=pd.DataFrame(percontext);pc.to_csv(out/'paired_context_metrics.csv',index=False)
    pd.DataFrame(curve).to_csv(out/'coverage_by_mountain_budget.csv',index=False)
    # Two-way cluster resampling handles shared mountain contexts AND repeated anchor sequences.
    paired=[];metrics=['valid_endpoint','exact_bright420_hits','r1_new_bright','r2_new_bright','r1_bright_minus_low_recall']
    for method in f['methods']:
        if method=='tgito_risk':continue
        a=pc[pc.method.eq('tgito_risk')].set_index('start_id');b=pc[pc.method.eq(method)].set_index('start_id').loc[a.index]
        mi=pd.factorize(a.mountain_id)[0];si=pd.factorize(a.seed_row)[0];nm=mi.max()+1;ns=si.max()+1
        delta=a[metrics].to_numpy()-b[metrics].to_numpy();res=[]
        for _ in range(1000):
            wm=rng.multinomial(nm,np.full(nm,1/nm));ws=rng.multinomial(ns,np.full(ns,1/ns));w=wm[mi]*ws[si]
            if w.sum():res.append(np.average(delta,axis=0,weights=w))
        res=np.asarray(res)
        for j,metric in enumerate(metrics):paired.append(dict(comparator=method,metric=metric,
            tgito_minus_comparator=float(delta[:,j].mean()),ci95_low=float(np.quantile(res[:,j],.025)),
            ci95_high=float(np.quantile(res[:,j],.975)),mountains=int(nm),unique_anchors=int(ns),contexts=len(a)))
    pd.DataFrame(paired).to_csv(out/'paired_two_way_cluster_bootstrap.csv',index=False)
    # Exact rediscovery among ALL evaluated hard candidates: no nearest-neighbor label imputation.
    pool_sets={m:set() for m in f['methods']};pool_hits={m:set() for m in f['methods']}
    truth_set=set(truth.index)
    for src in f['sources']:
        folder=root/src['folder'];assert sha(folder/'candidate_pool.csv')==src['pool_sha256']
        p=pd.read_csv(folder/'candidate_pool.csv',usecols=['sequence','method'])
        if p.empty:continue
        m=str(p.method.iloc[0]);seqset=set(p.sequence);pool_sets[m].update(seqset);pool_hits[m].update(seqset&truth_set)
    pooled=[];hit_rows=[]
    for method in f['methods']:
        v=truth.reindex(sorted(pool_hits[method])).dropna()
        pooled.append(dict(method=method,unique_queried_candidates=len(pool_sets[method]),exact_test_hits=len(v),
            exact_bright420_hits=int(v.ge(4.2).sum()),exact_bright435_hits=int(v.ge(4.35).sum()),
            measured_match_mean=float(v.mean()) if len(v) else None,
            measured_match_high420=float(v.ge(4.2).mean()) if len(v) else None))
        hit_rows.extend(dict(method=method,sequence=s,measured_brightness=float(y)) for s,y in v.items())
    pd.DataFrame(pooled).to_csv(out/'all_queried_exact_rediscovery.csv',index=False)
    pd.DataFrame(hit_rows,columns=['method','sequence','measured_brightness']).to_csv(out/'all_queried_measured_hits.csv',index=False)
    chosen=pd.read_csv(out/'selection_counts.csv');sel=[]
    for r in chosen.to_dict('records'):
        g=selected[selected.method.eq(r['method'])&selected.policy.eq(r['policy'])&selected.k.eq(r['k'])]
        y=g.measured_brightness.dropna();r.update(exact_measured_n=len(y),
            measured_mean=float(y.mean()) if len(y) else None,
            measured_high420=float(y.ge(4.2).mean()) if len(y) else None)
        sel.append(r)
    pd.DataFrame(sel).to_csv(out/'final_selection_measured_metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    tab=pd.DataFrame(rows).set_index('method');fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    names={'tgito_risk':'TGITO','random':'Random','adalead':'AdaLead','prothmso_top3_2026':'ProtHMSO (2026)',
        'tgito_no_risk_gate':'No risk gate','tgito_no_terrain':'No terrain','tgito_single_view':'Single view'}
    for ax,col,title in zip(axes,['exact_bright420_hits','r1_new_bright_covered','r1_bright_to_low_recall_ratio'],
                            ['Exact measured bright rediscovery','New bright-region coverage (Hamming <=1)','Bright / low-bright coverage rate']):
        values=pd.to_numeric(tab[col],errors='coerce')
        ax.barh([names.get(m,m) for m in tab.index],values.fillna(0),color=['#147d92' if m=='tgito_risk' else '#99acb8' for m in tab.index])
        xmax=max(1.,float(values.max()) if values.notna().any() else 1.)
        ax.set_xlim(0,xmax*1.2)
        for j,value in enumerate(values):
            label='n/a' if pd.isna(value) else f'{value:.2f}' if 'ratio' in col else str(int(value))
            ax.text((0 if pd.isna(value) else value)+xmax*.025,j,label,va='center',fontsize=8,color='#425766')
        ax.set_title(title,fontsize=9);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f'{mountains} mountain'+('s' if mountains!=1 else '')+f' / {mountains*3} start contexts; one returned proposal per context')
    fig.tight_layout();fig.savefig(out/'measured_discovery.png',dpi=180);fig.savefig(out/'measured_discovery.svg');plt.close(fig)
    (out/'measured_discovery_table.tex').write_text(tab[['unique_returned','exact_bright420_hits','r1_new_bright_covered','r2_new_bright_covered']].to_latex(escape=True),encoding='utf-8')
    write_json(out/'DONE.json',dict(mountains=mountains,contexts=len(starts),methods=f['methods'],
        exact_hamming_audit_cases=80,test_n=len(test),test_bright420_n=int(bright.sum()),
        interpretation='Exact sequence matches provide measured brightness; radius coverage describes navigation only',
        no_external_model_arbitration=True,new_wetlab_data=False,full_wide_campaign=mountains==900))
    print(json.dumps(dict(completed=str(out),mountains=mountains,methods=len(f['methods']))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--mountains',type=int,required=True);p.add_argument('--ablations',action='store_true')
    a=p.parse_args();evaluate(a.root,a.mountains,a.ablations)
