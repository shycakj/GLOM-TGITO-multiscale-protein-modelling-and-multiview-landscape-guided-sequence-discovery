"""Build publication tables/figures from frozen discovery-v3 evidence.

No search, candidate ranking, evaluator fitting, or threshold tuning is performed.
Run with the project's scientific Python; render_paper_pdf.py uses bundled Python.
"""
from pathlib import Path
import hashlib
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DEFAULT = HERE if (HERE / 'manuscript_content.json').exists() and HERE.name == 'full_discovery_paper_20261014' else HERE / 'reports/full_discovery_paper_20261014'
METHODS = ['glom_tgito', 'glom_adalead', 'glom_random', 'glom_prothmso', 'esm650_prothmso']
LABELS = ['GLOM-TGITO', 'GLOM-AdaLead', 'GLOM-random', 'GLOM-ProtHMSO', 'ESM650-ProtHMSO']
SHORT = ['TGITO', 'AdaLead', 'Random', 'ProtHMSO / GLOM', 'ProtHMSO / ESM650']
COLORS = ['#087F8C', '#6E7E91', '#B29C7D', '#8069A7', '#D27756']
SOURCE_PATHS = {
    'familiarity': 'artifacts/discovery_v3/embedding_familiarity_v1/method_summary.csv',
    'continuous': 'artifacts/discovery_v3/embedding_familiarity_v1/threshold_audit/continuous_distance_comparison.csv',
    'sensitivity': 'artifacts/discovery_v3/embedding_familiarity_v1/threshold_audit/top100_threshold_sensitivity.csv',
    'threshold_validation': 'artifacts/discovery_v3/embedding_familiarity_v1/threshold_audit/measured_threshold_validation.csv',
    'bootstrap': 'artifacts/discovery_v3/embedding_familiarity_v1/threshold_audit/threshold_bootstrap.csv',
    'bins': 'artifacts/discovery_v3/embedding_familiarity_v1/threshold_audit/distance_error_bins.csv',
    'reference_sensitivity': 'artifacts/discovery_v3/embedding_familiarity_v1/threshold_audit/brightness_reference_sensitivity.csv',
    'auroc': 'artifacts/discovery_v3/embedding_familiarity_v1/test_auroc_bootstrap.csv',
    'error_validation': 'artifacts/discovery_v3/embedding_familiarity_v1/test_score_validation.csv',
    'reliability': 'reports/discovery_v3_reliability_comparison/main_comparison.csv',
    'candidates': 'reports/discovery_v3_reliability_comparison/candidate_reliability_scores.csv',
    'discovery': 'artifacts/discovery_v3/evaluation/full_pipeline_comparison.csv',
    'matched': 'reports/discovery_v3_reliability_comparison/matched_measured_validation.csv',
    'matched_effect': 'reports/discovery_v3_reliability_comparison/matched_measured_effect_sizes.csv',
    'joint': 'reports/discovery_v3_reliability_comparison/joint_novelty_reliability_primary.csv',
    'joint_grid': 'reports/discovery_v3_reliability_comparison/joint_novelty_reliability_full_grid.csv',
    'joint_test': 'reports/discovery_v3_reliability_comparison/joint_rule_measured_validation.csv',
    'risk_validity': 'reports/discovery_v3_reliability_comparison/measured_score_validity.csv',
    'judge_accuracy': 'artifacts/discovery_v3/evaluation/judge_measured_test_metrics.csv',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def ordered(df):
    return df.set_index('method').loc[METHODS].reset_index()

def tex_escape(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('{',r'\{').replace('}',r'\}').replace('≥', r'$\geq$').replace('−', '-')

def tabular(df, headers, cols, fmts, caption, label):
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c, fmt in zip(cols, fmts):
            v = r[c]
            cells.append(tex_escape(v) if fmt is None else format(v, fmt))
        rows.append(' & '.join(cells) + r' \\')
    return ('\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n'
            + '\\caption{' + caption + '}\n\\label{' + label + '}\n'
            + '\\begin{tabular}{l' + 'r'*(len(cols)-1) + '}\n\\toprule\n'
            + ' & '.join(headers) + r' \\' + '\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n')

def panel(ax, letter, title):
    ax.set_title(title, loc='left', fontsize=9, pad=11)
    ax.text(-.02, 1.15, letter, transform=ax.transAxes, fontsize=11, fontweight='bold')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#CBD2D7')
    ax.spines['bottom'].set_color('#CBD2D7')
    ax.tick_params(labelsize=8, length=3, color='#9CA6AE')

def dotplot(ax, vals, title, letter, fmt='.3f', show_y=True):
    panel(ax, letter, title)
    vals = np.asarray(vals)
    span = max(np.ptp(vals), abs(vals.mean())*.03)
    for i, (v, c) in enumerate(zip(vals, COLORS)):
        ax.scatter(v, i, s=45 if i == 0 else 28, c=c, zorder=3, edgecolors='white', linewidths=.5)
        ax.annotate(format(v, fmt), (v, i), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8, color=c if i == 0 else '#39444E', fontweight='bold' if i == 0 else 'normal')
    ax.set_yticks(range(5), LABELS if show_y else ['']*5)
    ax.set_ylim(4.6, -.6)
    ax.set_xlim(vals.min()-span*.15, vals.max()+span*.6)
    ax.grid(axis='x', color='#E9EDF0', linewidth=.6)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.locator_params(axis='x', nbins=3)

def savefig(fig, out, name):
    fig.savefig(out/'figures'/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out/'figures'/f'{name}.svg', bbox_inches='tight')
    fig.savefig(out/'figures'/f'{name}.png', dpi=450, bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=DEFAULT)
    ap.add_argument('--snapshot', action='store_true', help='Rebuild from packaged source_data snapshots')
    args=ap.parse_args(); out=args.output.resolve()
    if out.name != 'full_discovery_paper_20261014':
        raise ValueError('Use a dedicated full_discovery_paper_20261014 output folder; never the experiment root.')
    for d in ['figures','tables','latex','source_data','qa']:
        (out/d).mkdir(parents=True, exist_ok=True)
    if not args.snapshot:
        manifest=[]
        for key, rel in SOURCE_PATHS.items():
            src=HERE/rel; dest=out/'source_data'/f'{key}.csv'
            shutil.copy2(src,dest)
            manifest.append(dict(key=key, original_path=rel, snapshot=dest.relative_to(out).as_posix(), sha256=sha(src)))
        for key, rel in {
            'embedding_plan':'artifacts/discovery_v3/embedding_familiarity_v1/PLAN.json',
            'embedding_done':'artifacts/discovery_v3/embedding_familiarity_v1/SCORES_DONE.json',
            'protocol':'artifacts/discovery_v3/PROTOCOL.json',
            'campaign_audit':'reports/discovery_v3_review/AUDIT.json',
            'reliability_rules':'reports/discovery_v3_reliability_comparison/ANALYSIS_RULES.json',
            'joint_rules':'reports/discovery_v3_reliability_comparison/JOINT_ANALYSIS_RULES.json',
        }.items():
            src=HERE/rel; dest=out/'source_data'/f'{key}.json';shutil.copy2(src,dest)
            manifest.append(dict(key=key,original_path=rel,snapshot=dest.relative_to(out).as_posix(),sha256=sha(src)))
        write_json(out/'source_manifest.json',manifest)
    for record in json.loads((out/'source_manifest.json').read_text(encoding='utf-8')):
        assert sha(out/record['snapshot'])==record['sha256'], record['snapshot']
    frames={k:pd.read_csv(out/'source_data'/f'{k}.csv') for k in SOURCE_PATHS}
    f=frames['familiarity']; c=frames['candidates']; r=frames['reliability']; e=frames['discovery']
    plan=json.loads((out/'source_data/embedding_plan.json').read_text(encoding='utf-8'))
    checks=[]
    if not args.snapshot:
        for m in METHODS:
            for b in [6,30,100]:
                rel=f'artifacts/discovery_v3/formal/{m}/prefix_900/top{b}.csv'
                assert sha(HERE/rel)==plan['input_hashes'][rel],rel
                original=pd.read_csv(HERE/rel)
                evaluated=c[(c.method==m)&(c.batch_size==b)]
                assert set(original.sequence)==set(evaluated.sequence),(m,b)
                assert len(evaluated)==b
        checks.append('All 15 native frozen batches match pre-familiarity SHA256 and evaluated sequence membership')
    assert c.measured_brightness_if_available.isna().all()
    assert np.allclose(c.cross_mean,(c.glom_20261014+c.glom_20261015)/2)
    assert np.allclose(c.cross_sd,abs(c.glom_20261014-c.glom_20261015)/2)
    for (m,b), g in c.groupby(['method','batch_size']):
        ref=r[(r.method==m)&(r.batch_size==b)].iloc[0]
        for col,summary in [('terrain_risk','terrain_risk_mean'),('terrain_support','terrain_support_mean'),('cross_mean','cross_mean_mean'),('cross_sd','cross_sd_mean'),('risk_scaled_lower90','risk_scaled_lower90_mean')]:
            assert np.isclose(g[col].mean(),ref[summary],atol=1e-9), (m,b,col)
        sequences=np.array([list(s) for s in g.sequence],dtype='U1')
        dd=(sequences[:,None,:]!=sequences[None,:,:]).sum(-1)
        pairwise=dd[np.triu_indices(b,1)].mean()
        ref_e=e[(e.method==m)&(e.batch_size==b)].iloc[0]
        assert np.isclose(pairwise,ref_e.mean_pairwise_hamming)
        assert g.novel_to_measured_catalogue.all()
    checks+=['Every generated experimental fluorescence remains missing', 'Candidate-level means, seed SD (ddof=0), pairwise diversity and catalogue novelty independently reconciled']
    q=frames['threshold_validation']
    qhigh=q[(q.reference=='predicted_high420')&(q.k==32)&(q.q==.95)].iloc[0]
    qall=q[(q.reference=='all')&(q.k==32)&(q.q==.95)].iloc[0]
    fsel=f[f.scope.isin(['top6','top30','top100'])].copy()
    fsel['batch_size']=fsel.scope.str.replace('top','').astype(int)
    merged=fsel.merge(r,on=['method','batch_size','n'],validate='one_to_one').merge(e,on=['method','batch_size'],validate='one_to_one')
    merged=merged.merge(frames['joint'][['method','batch_size','novel_and_reliability_pass_n']],on=['method','batch_size'],validate='one_to_one')
    merged['method_label']=merged.method.map(dict(zip(METHODS,LABELS)))
    merged['mean_raw_embedding_distance']=merged.highcal_degree_mean*qhigh.threshold
    means=c.groupby(['method','batch_size']).glom_20261013.mean().rename('search_glom_mean').reset_index()
    merged=merged.merge(means,on=['method','batch_size'],validate='one_to_one')
    top=ordered(merged[merged.batch_size==100])
    assert top.highcal_degree_mean.idxmin()==0
    assert np.allclose(top.mean_raw_embedding_distance,ordered(frames['continuous']).raw_mean)
    merged.to_csv(out/'tables/all_selected_batch_metrics.csv',index=False)
    top.to_csv(out/'tables/top100_complete_metrics.csv',index=False)
    f.to_csv(out/'tables/all_pool_and_selected_embedding_metrics.csv',index=False)
    frames['sensitivity'].to_csv(out/'tables/neighborhood_and_threshold_sensitivity.csv',index=False)
    frames['bootstrap'].to_csv(out/'tables/tail_calibration_uncertainty.csv',index=False)
    frames['auroc'].to_csv(out/'tables/measured_error_discrimination.csv',index=False)
    frames['joint_grid'].to_csv(out/'tables/exploratory_joint_rule_full_grid.csv',index=False)
    frames['judge_accuracy'][['judge','scope','n','rmse','mae','r2','spearman','pearson']].to_csv(out/'tables/common_evaluator_measured_accuracy.csv',index=False)
    candidate_cols=['method','batch_size','rank','sequence','glom_20261013','cross_mean','cross_sd','terrain_risk','terrain_support','combined_risk','risk_scaled_lower90','nearest_visible_hamming','novel_to_measured_catalogue','measured_brightness_if_available']
    c[candidate_cols].to_csv(out/'tables/frozen_selected_candidates.csv',index=False)
    effect=[]
    for i,row in top.iloc[1:].iterrows():
        effect.append(dict(baseline=row.method,mean_deviation_reduction_pct=100*(1-top.iloc[0].highcal_degree_mean/row.highcal_degree_mean),terrain_risk_reduction_pct=100*(1-top.iloc[0].terrain_risk_mean/row.terrain_risk_mean),terrain_support_increase_pct=100*(top.iloc[0].terrain_support_mean/row.terrain_support_mean-1),pairwise_diversity_increase_pct=100*(top.iloc[0].mean_pairwise_hamming/row.mean_pairwise_hamming-1),cross_seed_predicted_brightness_difference=top.iloc[0].cross_mean_mean-row.cross_mean_mean))
    pd.DataFrame(effect).to_csv(out/'tables/effects_vs_baselines.csv',index=False)
    campaign=json.loads((out/'source_data/campaign_audit.json').read_text(encoding='utf-8'))
    pd.DataFrame(campaign['generation']).to_csv(out/'tables/search_budgets.csv',index=False)
    write_json(out/'ANALYSIS_SPECIFICATION.json',dict(primary='Mean GLOM z768 kth-training-neighbor distance, L2-normalized representations, k=32, divided by high-prediction calibration q95',primary_choice='Requested after prior results; retrospective descriptive primary endpoint, not preregistered',calibration_high_prediction_min=4.2,calibration_high_n=int(qhigh.calibration_n),calibration_high_q95=float(qhigh.threshold),calibration_all_n=int(qall.calibration_n),calibration_all_q95=float(qall.threshold),reference='22003 model-training sequences only',candidate_selection='Original native rankings; no selection or reranking on embedding familiarity',selected_sizes=[6,30,100],tail='Secondary; calibration-bootstrap variation only; do not rank 1 vs 2 exceedances',unit='One completed search campaign per method; candidates are not independent campaign replications',brightness='Predicted dataset log-fluorescence, original target scale',hamming_role='Sequence novelty/diversity only; never familiarity',terrain_baselines='Post-selection diagnostic scoring only; native baselines did not receive multiterrain guidance',history='Historical trial12 transfer with explicit float32 override; retrospective HPO/test label overlap',representations='GLOM task-specific z768 primary; raw ESM150 token mean retained as representation control'))
    # Scientific plots: true numerical scales, no radar/composite scores or false replicate intervals.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.labelcolor':'#26313B','text.color':'#26313B','axes.titleweight':'normal','figure.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(9.0,5.0))
    fig.subplots_adjust(left=.19,right=.98,bottom=.10,top=.88,wspace=.35,hspace=.58)
    specs=[('highcal_degree_mean','Mean embedding deviation ↓','.4f'),('cross_mean_mean','Predicted log-fluorescence ↑','.3f'),('terrain_risk_mean','Terrain risk ↓','.3f'),('terrain_support_mean','Terrain support ↑','.3f'),('mean_pairwise_hamming','Sequence diversity ↑','.2f'),('cross_sd_mean','Cross-seed SD ↓','.4f')]
    for j,(col,title,fmt) in enumerate(specs):
        dotplot(axes.flat[j],top[col],title,chr(65+j),fmt,j%3==0)
    fig.text(.19,.015,'Frozen native Top 100 per method  |  Means from one campaign  |  Fluorescence is predicted',fontsize=8,color='#64727D')
    savefig(fig,out,'fig1_complete_discovery')
    # Error-based validation of the scores, including the high-brightness target regime.
    fig,axes=plt.subplots(2,2,figsize=(8.0,6.3))
    fig.subplots_adjust(left=.10,right=.98,bottom=.11,top=.91,wspace=.40,hspace=.68)
    ax=axes[0,0];panel(ax,'A','Embedding distance tracks measured error')
    bins=frames['bins'].query("reference == 'all'")
    ax.plot(range(len(bins)),bins.rmse,'o-',color=COLORS[0],lw=1.6,ms=4)
    ax.set_xticks(range(9),['0–20','20–40','40–60','60–80','80–90','90–95','95–97.5','97.5–99','99–100'],rotation=50,ha='right',fontsize=7)
    ax.set_xlabel('Calibration distance percentile bin',fontsize=8);ax.set_ylabel('Test RMSE (log-fluorescence)',fontsize=8)
    ax.set_ylim(0,.59)
    for j,rr in enumerate(bins.itertuples()):ax.annotate(str(rr.n),(j,rr.rmse),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6.7)
    ax.text(.03,.96,'Numbers show test sequences per bin',transform=ax.transAxes,va='top',fontsize=7,color='#64727D')
    ax=axes[0,1];panel(ax,'B','Large-error discrimination (|error| > 0.1)')
    scores=['z768_d32','esm_mean_d32','maxwent_sd'];scorelabels=['GLOM embedding','ESM150 embedding','MaxWEnt head']
    for group,scope in enumerate(['all','predicted_high420']):
        aa=frames['auroc'].query('scope == @scope and target == "absolute_gt0.1"').set_index('score').loc[scores]
        yy=np.arange(3)+group*4
        for j,(_,row) in enumerate(aa.iterrows()):
            ax.errorbar(row.auroc,yy[j],xerr=[[row.auroc-row.ci_low],[row.ci_high-row.auroc]],fmt='o',color=[COLORS[0],COLORS[1],COLORS[3]][j],capsize=2,ms=4,lw=1.2)
        ax.text(.49,group*4-.65,'All test (n=3,144)' if group==0 else 'Predicted ≥ 4.2 (n=973)',fontsize=7,fontweight='bold')
    ax.set_yticks([0,1,2,4,5,6],scorelabels*2,fontsize=7);ax.set_ylim(6.7,-1.15);ax.set_xlim(.48,.85)
    ax.axvline(.5,ls=':',color='#B6BFC7');ax.set_xlabel('AUROC with 95% bootstrap interval',fontsize=8)
    ax=axes[1,0];panel(ax,'C','Secondary boundary separates error levels')
    for j,rr in enumerate([qall,qhigh]):
        for k,prefix in enumerate(['inside','outside']):
            x=j*3+k;v=rr[f'{prefix}.rmse']
            ax.bar(x,v,color=COLORS[0] if k==0 else COLORS[4],width=.72)
            ax.text(x,v+.012,f'{v:.3f}\nn={int(rr[f"{prefix}.n"]):,}',ha='center',fontsize=7)
    ax.set_xticks([.5,3.5],['All test','Predicted ≥ 4.2']);ax.set_ylabel('Test RMSE (log-fluorescence)',fontsize=8);ax.set_ylim(0,.52)
    ax.legend(handles=[Line2D([0],[0],marker='s',ls='',color=COLORS[0],label='Within q95'),Line2D([0],[0],marker='s',ls='',color=COLORS[4],label='Beyond q95')],frameon=False,fontsize=7,loc='upper right')
    ax=axes[1,1];panel(ax,'D','Terrain-risk validation within matched strata')
    mm=frames['matched'].set_index('group').loc[['low','high']]
    ax.bar([0,1],mm.rmse,color=[COLORS[0],COLORS[4]],width=.55)
    ax.errorbar([0,1],mm.rmse,yerr=[mm.rmse-mm.rmse_ci_low,mm.rmse_ci_high-mm.rmse],fmt='none',color='#3B4853',capsize=3,lw=1)
    for j,(_,rr) in enumerate(mm.iterrows()):ax.text(j,rr.rmse_ci_high+.006,f'{rr.rmse:.4f}\nn={int(rr.n)}',ha='center',fontsize=8)
    ax.set_xticks([0,1],['Low risk','High risk']);ax.set_ylim(0,.19);ax.set_ylabel('Stratum-standardized test RMSE',fontsize=8)
    ax.text(.03,.96,'Similar predicted means: 4.242 / 4.240',transform=ax.transAxes,va='top',fontsize=7,color='#64727D')
    savefig(fig,out,'fig2_measured_validation')
    # Robustness plus an explicit representation control and complete preselection pool.
    fig,axes=plt.subplots(2,2,figsize=(8.3,6.5))
    fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.90,wspace=.35,hspace=.58)
    ss=frames['sensitivity'];handles=[]
    ax=axes[0,0];panel(ax,'A','Neighborhood-size sensitivity')
    for m,c0,lab in zip(METHODS,COLORS,LABELS):
        x=ss[(ss.method==m)&(ss.reference=='predicted_high420')&(ss.q==.95)].sort_values('k')
        line,=ax.plot(x.k,x.mean_normalized_distance,'o-',color=c0,label=lab,lw=2 if m==METHODS[0] else 1.2,ms=4)
        handles.append(line)
    ax.set_xscale('log',base=2);ax.set_xticks([8,16,32,64],['8','16','32','64']);ax.set_xlabel('Training-neighbor rank k',fontsize=8);ax.set_ylabel('Mean embedding deviation',fontsize=8)
    ax=axes[0,1];panel(ax,'B','Boundary sensitivity (secondary)')
    for m,c0 in zip(METHODS,COLORS):
        x=ss[(ss.method==m)&(ss.reference=='predicted_high420')&(ss.k==32)].sort_values('q')
        ax.plot(range(4),100*x.outside_fraction,'o-',color=c0,lw=2 if m==METHODS[0] else 1.2,ms=4)
    ax.set_xticks(range(4),['90%','95%','97.5%','99%']);ax.set_xlabel('High-reference calibration quantile',fontsize=8);ax.set_ylabel('Top 100 beyond boundary (%)',fontsize=8);ax.set_ylim(-1,36)
    ax=axes[1,0];panel(ax,'C','Full output pool and native selection')
    for m,c0 in zip(METHODS,COLORS):
        ff=f[f.method==m].set_index('scope').loc[['all_unique','top100']]
        ax.plot([0,1],ff.highcal_degree_mean,'o-',color=c0,lw=2 if m==METHODS[0] else 1.2,ms=4)
    ax.set_xticks([0,1],['All unique outputs','Native Top 100']);ax.set_xlim(-.1,1.1);ax.set_ylabel('Mean embedding deviation',fontsize=8)
    ax=axes[1,1];panel(ax,'D','Representation control: raw ESM150')
    cont=ordered(frames['continuous'])
    for j,(v,c0) in enumerate(zip(cont.esm_mean_distance,COLORS)):
        ax.scatter(v,j,c=c0,s=30);ax.annotate(f'{v:.4f}',(v,j),xytext=(5,0),textcoords='offset points',va='center',fontsize=7)
    ax.set_yticks(range(5),SHORT,fontsize=7);ax.set_ylim(4.5,-.5);ax.set_xlim(.0225,.0325);ax.set_xlabel('Mean normalized-embedding distance (raw units)',fontsize=7.5)
    fig.legend(handles=handles,labels=LABELS,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,fontsize=8)
    savefig(fig,out,'figS1_robustness_and_controls')
    # Manuscript tables.
    display=top.copy();display['tail_percent']=100*display.outside_highcal_q95_fraction
    maintex=tabular(display,['Method',r'$\bar D\downarrow$',r'Pred. $\bar y\uparrow$',r'Risk$\downarrow$',r'Support$\uparrow$',r'Diversity$\uparrow$',r'Tail (\%)'],['method_label','highcal_degree_mean','cross_mean_mean','terrain_risk_mean','terrain_support_mean','mean_pairwise_hamming','tail_percent'],[None,'.4f','.3f','.4f','.4f','.2f','.0f'],'Complete discovery pipelines evaluated on their frozen native Top 100 candidates. Mean normalized embedding deviation is the primary descriptive endpoint. Predictions use two GLOM seeds not used for search; diversity is mean pairwise Hamming distance. Tail exceedance is relative to the high-prediction calibration 95th percentile and is secondary.','tab:discovery-main')
    (out/'latex/table_main.tex').write_text(maintex,encoding='utf-8')
    sup=top.copy()
    suptex=tabular(sup,['Method','Search pred.',r'Cross SD$\downarrow$','Emp. lower','Nearest seq.','Joint count'],['method_label','search_glom_mean','cross_sd_mean','risk_scaled_lower90_mean','nearest_visible_hamming_mean','novel_and_reliability_pass_n'],[None,'.3f','.4f','.3f','.2f','.0f'],r'Supporting Top 100 metrics. Search pred. is the seed-20261013 GLOM prediction, also used as a common evaluator for ESM650 outputs. Emp. lower is the mean terrain-scaled empirical 90\% lower reference. Nearest seq. is minimum Hamming distance to train, validation and calibration sequences. Joint count applies the documented exploratory novelty and reliability rule; it is not a measured success count.','tab:discovery-support')
    (out/'latex/table_supporting.tex').write_text(suptex,encoding='utf-8')
    batchtex=tabular(merged.sort_values(['batch_size','method']),['Method','Batch',r'$\bar D\downarrow$','Pred. mean','Risk','Support'],['method_label','batch_size','highcal_degree_mean','cross_mean_mean','terrain_risk_mean','terrain_support_mean'],[None,'.0f','.4f','.3f','.4f','.4f'],'Sensitivity to native batch size. All batches were frozen before the embedding evaluation.','tab:discovery-batches')
    (out/'latex/table_batch_sizes.tex').write_text(batchtex,encoding='utf-8')
    judge=frames['judge_accuracy'].query("scope == 'all'")
    judgetex=tabular(judge,['Evaluator','n','RMSE','MAE',r'$R^2$',r'Spearman $\rho$'],['judge','n','rmse','mae','r2','spearman'],[None,'.0f','.4f','.4f','.4f','.4f'],'Accuracy of common evaluators on the same measured test sequences. Cross-mean predictions use two seeds unused during search.','tab:evaluator-accuracy')
    (out/'latex/table_evaluator_accuracy.tex').write_text(judgetex,encoding='utf-8')
    sections = build_text(top, qhigh, qall)
    write_json(out/'manuscript_content.json',sections)
    for name in ['results','methods','captions']:
        content='\n\n'.join(title+'\n'+body for title,body in sections[name])
        (out/f'{name}_EN.txt').write_text(content+'\n',encoding='utf-8')
        tex='\n\n'.join('\\paragraph{'+tex_escape(title)+'}\n'+tex_escape(body) for title,body in sections[name])
        (out/'latex'/f'{name}.tex').write_text(tex+'\n',encoding='utf-8')
    (out/'latex/metric_equation.tex').write_text(r'''\[
\widehat z(x)=\frac{[G(x);R_f(x);K_f(x)]}{\|[G(x);R_f(x);K_f(x)]\|_2},\quad
d_k(x)=\operatorname{kth}_{u\in\mathcal T}\|\widehat z(x)-\widehat z(u)\|_2,
\]
\[
D_k(x)=\frac{d_k(x)}{Q_{0.95}\{d_k(c):c\in\mathcal C,\widehat y_{13}(c)\geq4.2\}},
\qquad \overline D_k(S)=\frac{1}{|S|}\sum_{x\in S}D_k(x),\qquad k=32.
\]
''',encoding='utf-8')
    bibliography=r'''@inproceedings{sun2022knn,
 title={Out-of-Distribution Detection with Deep Nearest Neighbors},
 author={Sun, Yiyou and Ming, Yifei and Zhu, Xiaojin and Li, Yixuan},
 booktitle={Proceedings of the 39th International Conference on Machine Learning},
 volume={162}, pages={20827--20840}, year={2022}, url={https://proceedings.mlr.press/v162/sun22d.html}}
@article{greenman2025uq,
 title={Benchmarking uncertainty quantification for protein engineering},
 author={Greenman, Kevin P. and Amini, Ava P. and Yang, Kevin K.},
 journal={PLOS Computational Biology}, volume={21}, number={1}, pages={e1012639}, year={2025}, doi={10.1371/journal.pcbi.1012639}}
@article{demathelin2025maxwent,
 title={Deep Out-of-Distribution Uncertainty Quantification via Weight Entropy Maximization},
 author={de Mathelin, Antoine and Deheeger, Fran\c{c}ois and Mougeot, Mathilde and Vayatis, Nicolas},
 journal={Journal of Machine Learning Research}, volume={26}, number={4}, pages={1--68}, year={2025}, url={https://www.jmlr.org/papers/v26/23-1359.html}}
@article{wang2026prothmso,
 title={Heuristic multi-site optimization for protein sequence design using Masked Protein Language Models},
 author={Wang, L. and Wang, Y. and Qiu, C. and Xiao, L. and Liu, X. and Chen, J.},
 journal={PLOS Computational Biology}, volume={22}, number={6}, pages={e1014365}, year={2026}, doi={10.1371/journal.pcbi.1014365}}
@misc{sinai2020adalead,
 title={AdaLead: A simple and robust adaptive greedy search algorithm for sequence design},
 author={Sinai, Sam and Wang, Richard and Whatley, Alexander and Slocum, Stewart and Locane, Elina and Kelsic, Eric D.},
 year={2020}, eprint={2010.02141}, archivePrefix={arXiv}, url={https://arxiv.org/abs/2010.02141}}
'''
    (out/'latex/references.bib').write_text(bibliography,encoding='utf-8')
    (out/'latex/main.tex').write_text(r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,booktabs,graphicx,hyperref,natbib}
\graphicspath{{../figures/}}
\title{Computational evaluation of complete GLOM--TGITO protein discovery}
\author{}\date{}
\begin{document}\maketitle
\section*{Results}\input{results}
\input{table_main}\input{table_supporting}
\clearpage\section*{Methods}\input{methods}\input{metric_equation}
The embedding-distance evaluation follows the methodological rationale of
\citet{sun2022knn}; protein uncertainty is validated against measured errors as
recommended by \citet{greenman2025uq}. The MaxWEnt head control adapts
\citet{demathelin2025maxwent}; the recent masked-language-model search baseline
is adapted from \citet{wang2026prothmso}.
The native AdaLead baseline follows \citet{sinai2020adalead}.
\clearpage\section*{Figures}
\begin{figure}[h]\centering\includegraphics[width=\textwidth]{fig1_complete_discovery.pdf}\caption{Complete native Top 100 comparison. See detailed captions below.}\end{figure}
\clearpage\begin{figure}[h]\centering\includegraphics[width=\textwidth]{fig2_measured_validation.pdf}\caption{Validation against measured fluorescence.}\end{figure}
\clearpage\begin{figure}[h]\centering\includegraphics[width=\textwidth]{figS1_robustness_and_controls.pdf}\caption{Sensitivity and representation controls.}\end{figure}
\clearpage\input{captions}\input{table_batch_sizes}\input{table_evaluator_accuracy}
\bibliographystyle{plainnat}\bibliography{references}
\end{document}
''',encoding='utf-8')
    # The manuscript module is kept separate from previously frozen paper drafts.
    readme=f'''本轮完整蛋白发现干实验论文模块（2026-10-14）

主要指标：Top 100 平均标准化 GLOM embedding 距离，越小表征支撑越强。
完整方法 {top.iloc[0].highcal_degree_mean:.4f}，对比四个基线为 {', '.join(f'{v:.4f}' for v in top.highcal_degree_mean.iloc[1:])}。
相对降低 {', '.join(f'{a["mean_deviation_reduction_pct"]:.2f}%' for a in effect)}。
分母统一为预测亮度≥4.2的950个校准样本距离95%分位数，{qhigh.threshold:.9f}。
全校准集归一化的主方法均值为{top.iloc[0].degree_mean:.4f}；两种归一化排序相同，不能当作两份独立证据。

优先阅读 full_discovery_results.pdf；results_EN.txt 是可直接修改使用的英文结果段落。
figures 中三个图组均有矢量 PDF、可编辑 SVG 和450 dpi PNG。
latex 中有正文、方法、公式、图注、三张表和参考文献。把所需片段接入论文；独立示例 main.tex 使用相邻 figures 文件夹。
tables 保留五种方法、Top6/30/100、完整候选池、k/阈值敏感性、原始候选与预算。
source_data 是冻结证据副本，source_manifest.json 记录原路径与哈希。

论文叙述重点：相对于其他完整发现流程，最终候选在任务相关表征空间中偏离较小、地势风险较低、支撑较强，同时序列新颖且多样。
所有Top100均未在完整测量库出现；实际荧光仍留空。序列差异只统计新颖性和多样性，不作为embedding可信度。
平均偏离是本轮按用户要求强调的回顾性主要指标，没有据此重排或挑选候选。
原生AdaLead没有接多地势图。地势风险和支撑只在其输出冻结后统一诊断。
完整候选池中本方法平均偏离不最低；最终优势对应整个搜索加原生置信度/多样性筛选流程。
ESM150原始embedding对照中本方法也不是最近；本文明确结论属于GLOM任务相关表征。
实测测试集证明距离与误差有关：整体AUROC0.773，高预测亮度区0.684；高亮度越界样本RMSE0.246，对应界内0.088。
Top100高亮度参考越界数2/1/4/4/21，仅为辅助。校准重采样显示前两者为1–3和1–6，不据2对1判优。
额外有效指标：跨种子预测亮度、地势风险和支撑、序列多样性、跨种子SD，以及补充材料中的探索性新颖/可靠联合通过数。
不把多个相关指标相加成一个新总分，不把预测值写成实测值，不把100个候选当成100次独立实验。

复现图表：python build_full_discovery_paper.py --snapshot --output .
生成PDF：使用装有reportlab与pypdf的Python运行 render_paper_pdf.py --output .（同目录）。
本次图表与正文不改动历史实验、HPO、原生输出或原稿。
'''
    (out/'阅读说明与中文结论.txt').write_text(readme,encoding='utf-8')
    write_json(out/'qa/numerical_validation.json',dict(passed=True,checks=checks,top100_primary_values=top.highcal_degree_mean.tolist(),unknown_fluorescence_rows=int(c.measured_brightness_if_available.isna().sum()),top100_rows=500,all_selected_memberships=len(c),campaign_replicates_per_method=1))
    print(json.dumps(dict(output=str(out),checks=checks,main=top[['method','highcal_degree_mean','cross_mean_mean','terrain_risk_mean','terrain_support_mean','mean_pairwise_hamming']].to_dict('records')),ensure_ascii=False))

def build_text(top,qhigh,qall):
    results=[
      ('The complete discovery pipeline produces supported and diverse candidate batches',
       'We evaluated the complete GLOM-TGITO pipeline, including multiterrain peak selection, dynamically gated climbing, and native confidence/diversity selection, against GLOM-AdaLead, GLOM-random, GLOM-ProtHMSO and ESM650-ProtHMSO. The corrected historical-HPO campaign covered 900 peak regions with three starting roles per region (2,700 contexts). Each method used 1,476,063 forward sequence evaluations; GLOM-TGITO additionally used 189,000 gradient steps. Candidate batches were frozen before the representation-support analysis. We used mean normalized GLOM embedding deviation as the primary descriptive endpoint for the final Top 100, supported by predicted brightness, terrain risk/support, novelty and diversity (Figure 1; Table 1).'),
      ('Lower mean embedding deviation in the final selected batch',
       'GLOM-TGITO achieved the lowest mean embedding deviation, 0.8048, compared with 0.8290 for GLOM-AdaLead, 0.8323 for GLOM-random, 0.8310 for GLOM-ProtHMSO and 0.8763 for ESM650-ProtHMSO. These correspond to reductions of 2.91%, 3.30%, 3.14% and 8.16%, respectively. The corresponding unscaled mean distances were 0.4698, 0.4839, 0.4858, 0.4851 and 0.5116. The advantage persisted for neighbor ranks 8, 16, 32 and 64, and at native batch sizes 6 and 30 (Figure S1; supplementary tables). Using the full calibration population for normalization gave the same ordering. Across all unique GLOM-TGITO outputs, mean deviation was 0.9034; native selection reduced it to 0.8048, a 10.91% reduction. Other methods had smaller deviations in their full output pools, locating the observed advantage at the level of the complete final-batch pipeline.'),
      ('Representation support is associated with actual predictive error',
       'On 3,144 measured test sequences, the common cross-seed GLOM evaluator had RMSE 0.1649, R-squared 0.8981 and Spearman correlation 0.8962; the corresponding ESM650 values were 0.2426, 0.7794 and 0.7945. For the search GLOM checkpoint, embedding distance discriminated absolute prediction errors greater than 0.1 with AUROC 0.773 (95% bootstrap interval 0.752-0.792). Within the 973 sequences predicted to have log-fluorescence at least 4.2, AUROC was 0.684 (0.624-0.742), and AUROC for overprediction greater than 0.1 was 0.709 (0.632-0.780). At the calibration-derived 95th-percentile boundary, 49/973 high-prediction test sequences exceeded the boundary (5.04%; Wilson 95% interval 3.83-6.60%). Their RMSE was 0.246, compared with 0.088 for the 924 sequences within the boundary. Thus, the distance endpoint has an empirical connection to prediction reliability on measured proteins (Figure 2). The raw ESM150 representation and a MaxWEnt regression-head pilot were retained as controls: neither was uniformly best across the full test set and the high-prediction subset.'),
      ('Terrain reliability and sequence diversity complement the primary endpoint',
       'The GLOM-TGITO Top 100 had mean terrain risk 0.1182 and support 0.3168. Relative to GLOM-AdaLead, risk was 12.8% lower and support was 32.2% higher; relative to GLOM-ProtHMSO, the differences were 6.2% and 16.8%. A separate matched analysis of measured test sequences compared low- and high-risk groups at similar predicted fluorescence (4.242 and 4.240), yielding stratum-standardized RMSEs of 0.0508 and 0.1090. All five Top 100 batches consisted of sequences absent from the measured catalogue. GLOM-TGITO also had the largest mean pairwise sequence difference, 7.62 residues versus 6.66, 6.72, 6.09 and 6.38, respectively. Its mean nearest-visible-sequence distance was 1.90 residues, compared with 1.44, 1.42, 1.20 and 1.27. Sequence novelty and diversity therefore accompanied closer support in the task-related embedding space.'),
      ('Brightness and stability identify the performance trade-off',
       'The common cross-seed GLOM evaluator gave mean predicted log-fluorescence of 4.391 for GLOM-TGITO, compared with 4.400, 4.401, 4.414 and 4.317 for the four baselines. Mean cross-seed SD was 0.0192, compared with 0.0167, 0.0181, 0.0163 and 0.0207. GLOM-TGITO therefore combined stronger representation and terrain support with competitive predicted brightness, while GLOM-ProtHMSO had the highest cross-seed predicted brightness and lowest dispersion. The exploratory joint novelty/reliability criterion retained 38 GLOM-TGITO candidates versus 16, 14, 14 and 11; its complete threshold grid is provided in the supplementary data. These are computationally accepted candidates, with experimental fluorescence awaiting measurement.'),
      ('Tail exceedance provides secondary context',
       'At the high-prediction calibration 95th percentile, the Top 100 exceedance counts were 2, 1, 4, 4 and 21. Calibration resampling gave count ranges of 1-3, 1-6, 1-5, 4-6 and 18-24, respectively (central 95% ranges with candidate sets fixed). The difference between one and two exceedances does not establish a reliability ranking. Under the full-calibration boundary, the respective counts were 0, 0, 0, 0 and 9. The full sensitivity analysis, including quantiles 90%, 95%, 97.5% and 99%, accompanies the continuous primary endpoint (Figure S1).'),
    ]
    methods=[
      ('Campaign and comparison scope',
       'The primary experiment used the corrected discovery_v3 pipeline with historical HPO trial 12 explicitly transferred to the effective runtime configuration. The terrain coefficient was 0.0464303621, learning rate 0.0388846264, gate temperature 0.2037978423 and seed-exploration coefficient 0.2224716472. Twelve terrain sources contributed to peak selection with a uniform source prior; dynamic gating selected six sources using support, risk and prediction-gap weights 0.5249775580, 0.7454479755 and 0.1654493334, and source-quality power 0.8665330602. Each context used 70 steps, a two-residue mutation radius from its seed and a discrete proposal every six steps. Float32 execution was an explicit numerical override of historical bfloat16. The final native score weighted prediction, support, terrain potential, low risk and terrain-weight entropy by 0.35, 0.25, 0.20, 0.15 and 0.05, followed by a prediction floor of 4.31 and pairwise diversity constraint. Baselines retained their native proposals, starts and selection; terrain scoring of their frozen outputs was evaluation only. AdaLead used no terrain guidance. ProtHMSO was adapted to the GFP task using masked-language-model proposals, rather than reproducing its original assay system. Equal forward-query budgets did not imply equal gradient cost. The comparison evaluates whole pipelines, including their different starting policies, rather than an isolated optimizer ablation.'),
      ('Learned representation deviation',
       'The sequence representation was the concatenation of the learned global, regional and key-site GLOM features, z = [G; Rf; Kf], with 768 dimensions. All representations were computed consistently in float32 using the search GLOM checkpoint (seed 20261013). After L2 normalization, d_k(x) was the Euclidean distance to the kth nearest member of the 22,003-sequence training reference, with k=32. This distance is derived from learned protein features; it does not count amino-acid substitutions. We defined D_k(x) as d_k(x) divided by the 95th-percentile d_k among calibration sequences with seed-20261013 predicted log-fluorescence at least 4.2. This reference contained 950 sequences and yielded a denominator of 0.5837557863. The primary batch statistic was the arithmetic mean of D_k over all 100 frozen candidates. The full calibration set of 3,143 sequences provided a complementary denominator of 0.6986382698. As both denominators are common across methods, changing between them preserves method ordering. The embedding endpoint was selected as the main reporting measure retrospectively; no candidate was selected, filtered or reranked using this distance.'),
      ('Calibration, error validation and controls',
       'Calibration quantiles used the higher order-statistic convention. Sensitivity analyses covered k in {8,16,32,64} and quantiles {0.90,0.95,0.975,0.99}. Calibration-bootstrap intervals used 2,000 resamples with candidate batches fixed; these quantify reference-threshold uncertainty. Error discrimination was evaluated against measured test fluorescence using absolute and signed overprediction events above 0.1, with 1,000 bootstrap resamples for AUROC intervals. High-prediction validation used the same seed-20261013 prediction threshold of 4.2 as the reference. All four GLOM-driven Top 100 batches met this threshold; 92/100 ESM650-ProtHMSO candidates met it, so the latter batch distance is a common-reference descriptor, not a conditional high-prediction error probability. Controls used the raw ESM150 token-mean embedding and a fixed-mean MaxWEnt regression-head pilot trained on GLOM features, with noise scales selected on validation data and 64 weight draws. Thresholds and distances were not interpreted as calibrated probabilities of activity for optimized sequences.'),
      ('Supporting endpoints and experimental labels',
       'Common predicted brightness was the mean of GLOM seeds 20261014 and 20261015, neither used for search; dispersion was their population SD. These models shared architecture and training data. Terrain risk and support were recomputed with the same evaluator for all frozen candidate batches. The supplementary empirical lower brightness reference used calibration residuals scaled by terrain-derived risk. The exploratory joint rule required this lower reference to be at least 4.2, combined risk no greater than the high-prediction calibration median (0.1576107955), cross-seed SD no greater than its 75th percentile (0.0145756602), and minimum Hamming distance of two to visible train/validation/calibration sequences. Novelty was also checked against the full measured catalogue, including test sequences. Mean pairwise Hamming distance describes batch diversity only. It was not used as a measure of model familiarity. Experimental brightness for generated sequences without assays remained missing.'),
      ('Statistical interpretation',
       'Each method contributed one completed campaign and one frozen native batch at each requested size. Candidate means describe these batches; no candidate-level significance test was interpreted as replication of the entire search. Historical HPO used labels overlapping the current test partition, making the policy comparison a retrospective transfer study. Measured-sequence score validation supports the reliability interpretation of the computational endpoints; generated-sequence functional confirmation requires their assays. The primary result concerns task-specific GLOM representation support. Raw ESM150 distance and all-output-pool controls are reported to show how this result depends on representation and native selection.'),
    ]
    captions=[
      ('Figure 1. Complete discovery pipeline comparison',
       'Frozen native Top 100 batches from the complete GLOM-TGITO pipeline and four baselines. (A) Primary endpoint: mean GLOM embedding deviation normalized by the high-prediction calibration 95th percentile. (B) Common predicted log-fluorescence from two GLOM seeds unused during search. (C,D) Common post-selection terrain-risk and support diagnostics. (E) Mean pairwise Hamming distance, measuring sequence diversity. (F) Mean cross-seed prediction SD. Points denote batch means from one campaign; no independent-campaign confidence intervals are implied. All 100 candidates per method are absent from the measured catalogue and have no experimental fluorescence labels.'),
      ('Figure 2. Validation of computational reliability against measured fluorescence',
       '(A) Test RMSE across calibration-defined GLOM distance bins; annotations show the number of measured test sequences. (B) Discrimination of absolute prediction errors greater than 0.1 for GLOM embedding distance, raw ESM150 embedding distance and the MaxWEnt head pilot. Error bars are 95% bootstrap intervals from 1,000 resamples. (C) Test RMSE within and beyond each population-specific calibration 95th-percentile boundary. (D) Terrain-risk groups compared after standardizing over common predicted-brightness and nearest-training-sequence strata; error bars are conditional stratified bootstrap intervals. Panel D uses cross-seed prediction errors, whereas panels A-C evaluate the search GLOM checkpoint. The resulting sample counts and error values are therefore not interchangeable.'),
      ('Figure S1. Robustness, native selection and representation controls',
       '(A) Mean normalized embedding deviation for neighbor ranks 8, 16, 32 and 64, each normalized by its own common high-prediction calibration 95th percentile. (B) Top 100 tail fractions at four calibration quantiles. (C) Mean deviation in all unique outputs and the corresponding native Top 100, displaying selection as part of the complete discovery pipeline. The all-output pools contain 2,594, 2,700, 2,700, 2,694 and 2,697 sequences, respectively. (D) Mean raw distances in L2-normalized ESM150 token-mean embeddings. This control does not give GLOM-TGITO the lowest distance; the primary finding is specific to the task-related GLOM representation. Lines connect descriptive summaries, not independent optimization replicates.'),
    ]
    refs=[
      'Sun Y, Ming Y, Zhu X, Li Y. Out-of-Distribution Detection with Deep Nearest Neighbors. ICML 2022;162:20827-20840. https://proceedings.mlr.press/v162/sun22d.html',
      'Greenman KP, Amini AP, Yang KK. Benchmarking uncertainty quantification for protein engineering. PLOS Computational Biology. 2025;21(1):e1012639. https://doi.org/10.1371/journal.pcbi.1012639',
      'de Mathelin A, Deheeger F, Mougeot M, Vayatis N. Deep Out-of-Distribution Uncertainty Quantification via Weight Entropy Maximization. JMLR. 2025;26(4):1-68. https://www.jmlr.org/papers/v26/23-1359.html',
      'Wang L, Wang Y, Qiu C, Xiao L, Liu X, Chen J. Heuristic multi-site optimization for protein sequence design using Masked Protein Language Models. PLOS Computational Biology. 2026;22(6):e1014365. https://doi.org/10.1371/journal.pcbi.1014365',
        'Sinai S, Wang R, Whatley A, Slocum S, Locane E, Kelsic ED. AdaLead: A simple and robust adaptive greedy search algorithm for sequence design. arXiv:2010.02141 (2020). https://arxiv.org/abs/2010.02141',
    ]
    return dict(results=results,methods=methods,captions=captions,references=refs)

if __name__=='__main__':main()
