"""Compare reliability of all five frozen discovery pipelines.

Post-hoc analysis: rules are recorded before this script computes comparisons.
No search, native ranking, candidate identities or model weights are changed.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from audit_discovery_v3_reliability import nearest_distance

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'artifacts/discovery_v3'
OUT = ROOT / 'reports/discovery_v3_reliability_comparison'
METHODS = ['glom_tgito', 'glom_adalead', 'glom_random', 'glom_prothmso', 'esm650_prothmso']
NAMES = dict(zip(METHODS, ['GLOM-TGITO', 'GLOM + AdaLead', 'GLOM + random', 'GLOM + ProtHMSO', 'ESM650 + ProtHMSO']))
SEED = 20261014


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def qhigh(x, alpha):
    a = np.sort(np.asarray(x, float))
    rank = int(np.ceil((len(a)+1)*(1-alpha)))
    if rank > len(a):
        return float('inf')
    return float(a[rank-1])


def wilson(k, n):
    if not n:
        return np.nan, np.nan
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half


def freeze_analysis():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [RUN/'PROTOCOL.json', RUN/'evaluation/glom_live_predictions.csv',
             RUN/'evaluation/esm650_live_predictions.csv', RUN/'evaluation/live_risk_scores.csv',
             RUN/'evaluation/selected_candidates_evaluated.csv', ROOT/'data/train.csv',
             ROOT/'data/calibration.csv', ROOT/'data/test.csv']
    plan = dict(created_local=datetime.now().astimezone().isoformat(),
        status='Post-hoc follow-up; prior aggregate results were already inspected',
        methods=METHODS, batch_sizes=[6,30,100],
        scores=['terrain_risk', 'combined_risk', 'terrain_support', 'prediction_terrain_gap',
                'unused_GLOM_seed_sd', 'primary_minus_unused_mean', 'risk_scaled_empirical_lower',
                'unscaled_empirical_lower', 'ProteinMPNN_compatibility', 'ESM650_support'],
        risk_formula='terrain_selected_risk_wmean + 0.5 * terrain_selected_pred_gap_wmean',
        calibration='Same calibration set for every method; no method-specific fit or thresholds',
        bound_formula='cross_seed_mean - quantile((cross_seed_mean-y)/scale) * scale',
        bound_scales=['constant 1', 'combined_risk'], alphas=[0.1,0.05],
        strata='cross-seed mean >=4.2 versus <4.2; nearest training Hamming 1 versus >=2',
        eligibility='For summaries, empirical lower >=4.2 or >=4.35, risk <= calibration median among cross-predicted >=4.2, unused seed SD <= calibration 75th percentile in that group',
        coverage_validation='Measured test overall and per fixed stratum; report interval width and observed success among accepted measured cases',
        bootstrap='2000 deterministic resamples; measured validation stratified resampling of sequence rows; selected candidate intervals resample peak/anchor groups within this one campaign',
        bootstrap_limits='Conditional descriptive uncertainty; not repeated campaigns or a proof of independent sequences',
        random_seed=SEED, comparisons_not_used_for_reranking=True,
        unknown_measured_fluorescence_remains_missing=True,
        historical_HPO_label_overlap_retained=True,
        references=['https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012639',
                    'https://pmc.ncbi.nlm.nih.gov/articles/PMC9618043/'],
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths})
    target=OUT/'ANALYSIS_RULES.json'
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        assert old['input_hashes']==plan['input_hashes']
        comparable=lambda d:{k:v for k,v in d.items() if k!='created_local'}
        assert comparable(old)==comparable(plan), 'Do not silently change frozen evaluation rules'
    else:
        write_json(target,plan)
    # Verify selections against the frozen main evaluation and downloaded manifest.
    done=json.loads((RUN/'evaluation/DONE.json').read_text())
    for m in METHODS:
        folder=RUN/'formal'/m/'prefix_900'
        assert sha(folder/'FROZEN.json')==done['candidate_freezes'][m]
        frozen=json.loads((folder/'FROZEN.json').read_text())
        for filename,digest in frozen['files'].items():
            assert sha(folder/filename)==digest, (m,filename)
    manifest=json.loads((ROOT/'reports/discovery_v3_snapshot_manifest.json').read_text())
    for item in manifest['files']:
        if item['path'].startswith('artifacts/discovery_v3/evaluation/'):
            assert sha(ROOT/item['path'])==item['sha256']
    return plan


def load_data():
    p=pd.read_csv(RUN/'evaluation/glom_live_predictions.csv')
    r=pd.read_csv(RUN/'evaluation/live_risk_scores.csv')
    p=p.merge(r[['sequence','terrain_selected_risk_wmean','terrain_selected_support_wmean',
                  'terrain_selected_pred_gap_wmean','terrain_weight_entropy','terrain']],on='sequence',validate='one_to_one')
    p=p.rename(columns={'terrain_selected_risk_wmean':'terrain_risk','terrain_selected_support_wmean':'terrain_support',
                        'terrain_selected_pred_gap_wmean':'prediction_terrain_gap'})
    p['combined_risk']=p.terrain_risk+.5*p.prediction_terrain_gap
    assert p.combined_risk.gt(0).all()
    p['cross_mean']=p[['glom_20261014','glom_20261015']].mean(axis=1)
    p['cross_sd']=p[['glom_20261014','glom_20261015']].std(axis=1,ddof=0)
    p['optimism_gap']=p.glom_20261013-p.cross_mean
    p['abs_optimism_gap']=p.optimism_gap.abs()
    p['negative_support']=-p.terrain_support
    train=pd.read_csv(ROOT/'data/train.csv',usecols=['sequence']).sequence.tolist()
    ref=ROOT/'repro_project/data/processed/reference_sequence.txt'
    if not ref.exists(): ref=ROOT.parent/'handoff_20261001/repro_project/data/processed/reference_sequence.txt'
    wt=ref.read_text().strip()
    frames={}
    for split in ['calibration','test']:
        d=pd.read_csv(ROOT/f'data/{split}.csv').merge(p,on='sequence',validate='one_to_one')
        d['nearest_training_hamming']=nearest_distance(d.sequence,train,wt)
        d['error']=d.cross_mean-d.brightness
        d['stratum']=np.where(d.cross_mean.ge(4.2),'high','low')+'_'+np.where(d.nearest_training_hamming.eq(1),'d1','d2plus')
        frames[split]=d
    selected=pd.read_csv(RUN/'evaluation/selected_candidates_evaluated.csv')
    s=selected.drop(columns=[c for c in p.columns if c!='sequence' and c in selected.columns]).merge(p,on='sequence',validate='many_to_one')
    s['stratum']=np.where(s.cross_mean.ge(4.2),'high','low')+'_'+np.where(s.nearest_training_hamming.eq(1),'d1','d2plus')
    s['resample_group']=np.where(s.method.eq('glom_tgito'),'peak_'+s.mountain_id.fillna('missing').astype(str),'anchor_'+s.seed_row.astype(int).astype(str))
    assert not s.loc[s.method.eq('glom_tgito'),'resample_group'].eq('peak_missing').any()
    assert len(s)==680 and s.measured_brightness_if_available.isna().all()
    return frames['calibration'],frames['test'],s


def calibrated_bounds(cal,test,s):
    rows=[]
    for alpha in [.1,.05]:
        tag=str(int(round((1-alpha)*100)))
        for model in ['constant','risk_scaled']:
            for name,c in cal.groupby('stratum'):
                scale=c.combined_risk.to_numpy() if model=='risk_scaled' else np.ones(len(c))
                lower_q=qhigh(c.error.to_numpy()/scale,alpha)
                abs_q=qhigh(np.abs(c.error.to_numpy())/scale,alpha)
                for d in [test,s]:
                    mask=d.stratum.eq(name)
                    v=d.loc[mask,'combined_risk'].to_numpy() if model=='risk_scaled' else np.ones(mask.sum())
                    d.loc[mask,f'{model}_lower{tag}']=d.loc[mask,'cross_mean']-lower_q*v
                    d.loc[mask,f'{model}_width{tag}']=2*abs_q*v
                t=test[test.stratum.eq(name)]
                covered=t.brightness.ge(t[f'{model}_lower{tag}'])
                two=(t.error.abs()<=t[f'{model}_width{tag}']/2)
                low,high=wilson(int(covered.sum()),len(t))
                rows.append(dict(alpha=alpha,nominal=1-alpha,model=model,stratum=name,calibration_n=len(c),test_n=len(t),
                    lower_q=lower_q,absolute_q=abs_q,lower_coverage=covered.mean(),coverage_wilson_low=low,coverage_wilson_high=high,
                    two_sided_coverage=two.mean(),mean_two_sided_width=t[f'{model}_width{tag}'].mean(),
                    mean_lower_margin=(t.cross_mean-t[f'{model}_lower{tag}']).mean()))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'empirical_bound_test_validation.csv',index=False)
    return frame


def measured_validation(cal,test):
    associations=[];curves=[];screening=[]
    highcal=cal[cal.cross_mean.ge(4.2)]
    thresholds=dict(combined_risk_median=float(highcal.combined_risk.median()),cross_sd_q75=float(highcal.cross_sd.quantile(.75)))
    for name,mask in [('all',np.ones(len(test),bool)),('predicted_high420',test.cross_mean.ge(4.2)),('predicted_high431',test.cross_mean.ge(4.31))]:
        t=test[mask]
        c=cal if name=='all' else cal[cal.cross_mean.ge(4.2 if name=='predicted_high420' else 4.31)]
        for score in ['combined_risk','terrain_risk','prediction_terrain_gap','negative_support','cross_sd','abs_optimism_gap']:
            failure=t.brightness.lt(4.2)
            large_error=t.error.abs().gt(.1)
            associations.append(dict(scope=name,score=score,n=len(t),
                spearman_absolute_error=spearmanr(t[score],t.error.abs()).statistic,
                spearman_overprediction=spearmanr(t[score],t.error).statistic,
                measured_below420_n=int(failure.sum()),measured_below420_auc=roc_auc_score(failure,t[score]) if failure.nunique()==2 else np.nan,
                large_error_n=int(large_error.sum()),large_error_auc=roc_auc_score(large_error,t[score]) if large_error.nunique()==2 else np.nan,
                large_error_auprc=average_precision_score(large_error,t[score]) if large_error.nunique()==2 else np.nan))
            for fraction in [.1,.2,.25,.4,.5,.6,.75,.8,.9,1.]:
                cutoff=c[score].quantile(fraction) if fraction<1 else np.inf
                keep=t[t[score].le(cutoff)]
                curves.append(dict(scope=name,score=score,calibration_keep=fraction,cutoff=cutoff,
                    test_n=len(t),retained_n=len(keep),retained_fraction=len(keep)/len(t),
                    rmse=float(np.sqrt(np.mean(keep.error**2))) if len(keep) else np.nan,
                    mae=float(keep.error.abs().mean()) if len(keep) else np.nan,
                    false_bright_fraction=float(keep.brightness.lt(4.2).mean()) if len(keep) else np.nan))
    for target in [4.2,4.35]:
        for policy in ['prediction_only','constant_lower90','risk_scaled_lower90','risk_scaled_lower95','risk_scaled_lower90_lowrisk_stable']:
            if policy=='prediction_only': keep=test.cross_mean.ge(target)
            elif policy.endswith('lowrisk_stable'):
                keep=test.risk_scaled_lower90.ge(target)&test.combined_risk.le(thresholds['combined_risk_median'])&test.cross_sd.le(thresholds['cross_sd_q75'])
            else: keep=test[policy].ge(target)
            t=test[keep];success=t.brightness.ge(target);lo,hi=wilson(int(success.sum()),len(t))
            screening.append(dict(target=target,policy=policy,test_n=len(test),selected_n=len(t),
                true_success_n=int(success.sum()),true_success_fraction=success.mean() if len(t) else np.nan,
                wilson_low=lo,wilson_high=hi,rmse=np.sqrt(np.mean(t.error**2)) if len(t) else np.nan,
                actual_mean=t.brightness.mean() if len(t) else np.nan))
    pd.DataFrame(associations).to_csv(OUT/'measured_score_validity.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'risk_coverage_curves.csv',index=False)
    pd.DataFrame(screening).to_csv(OUT/'measured_acceptance_validation.csv',index=False)
    write_json(OUT/'CALIBRATION_THRESHOLDS.json',thresholds)
    return pd.DataFrame(curves),pd.DataFrame(screening),thresholds


def bootstrap_mean(group, columns, rng, reps=2000):
    blocks=group.groupby('resample_group')[columns].agg(['sum','count'])
    idx=rng.integers(0,len(blocks),size=(reps,len(blocks)))
    output={}
    for col in columns:
        sums=blocks[(col,'sum')].to_numpy()[idx].sum(1)
        ns=blocks[(col,'count')].to_numpy()[idx].sum(1)
        output[col]=sums/ns
    return output,len(blocks)


def candidate_comparison(cal,s,thresholds):
    for target,name in [(4.2,'420'),(4.35,'435')]:
        s[f'lower90_pass{name}']=s.risk_scaled_lower90.ge(target).astype(float)
        s[f'lower95_pass{name}']=s.risk_scaled_lower95.ge(target).astype(float)
        s[f'lowrisk_stable_pass{name}']=(s.risk_scaled_lower90.ge(target)&s.combined_risk.le(thresholds['combined_risk_median'])&s.cross_sd.le(thresholds['cross_sd_q75'])).astype(float)
        s[f'both_unused_seeds_pass{name}']=(s.glom_20261014.ge(target)&s.glom_20261015.ge(target)).astype(float)
    s['primary_prediction_exceeds_calibration_max']=s.glom_20261013.gt(cal.glom_20261013.max()).astype(float)
    s['cross_prediction_exceeds_calibration_max']=s.cross_mean.gt(cal.cross_mean.max()).astype(float)
    s['nearest_visible_ge2']=s.nearest_visible_hamming.ge(2).astype(float)
    columns=['terrain_risk','combined_risk','terrain_support','prediction_terrain_gap','cross_mean','cross_sd','optimism_gap',
             'abs_optimism_gap','constant_lower90','risk_scaled_lower90','risk_scaled_lower95','risk_scaled_width90',
             'lower90_pass420','lower90_pass435','lower95_pass420','lower95_pass435','lowrisk_stable_pass420','lowrisk_stable_pass435',
             'both_unused_seeds_pass420','both_unused_seeds_pass435','nearest_visible_ge2','primary_prediction_exceeds_calibration_max',
             'cross_prediction_exceeds_calibration_max','mpnn_log_likelihood','esm650_cross_mean']
    summaries=[];effects=[];draws={};rng=np.random.default_rng(SEED)
    for k in [6,30,100]:
        for m in METHODS:
            d=s[s.method.eq(m)&s.batch_size.eq(k)].copy()
            assert len(d)==k and d.sequence.nunique()==k
            boots,nblocks=bootstrap_mean(d,columns,rng)
            row=dict(method=m,batch_size=k,n=len(d),resample_groups=nblocks)
            for col in columns:
                a=d[col]
                row[col+'_mean']=float(a.mean())
                row[col+'_median']=float(a.median())
                row[col+'_ci_low']=float(np.quantile(boots[col],.025))
                row[col+'_ci_high']=float(np.quantile(boots[col],.975))
                draws[(m,k,col)]=boots[col]
            summaries.append(row)
        for m in METHODS[1:]:
            for col in columns:
                ours=s[s.method.eq(METHODS[0])&s.batch_size.eq(k)][col].mean()
                baseline=s[s.method.eq(m)&s.batch_size.eq(k)][col].mean()
                delta=draws[(METHODS[0],k,col)]-draws[(m,k,col)]
                effects.append(dict(batch_size=k,baseline=m,metric=col,ours=ours,baseline_value=baseline,
                    delta=ours-baseline,relative_change=(ours/baseline-1) if baseline!=0 else np.nan,
                    delta_ci_low=np.quantile(delta,.025),delta_ci_high=np.quantile(delta,.975),
                    interval_scope='Within frozen candidate pools; peak/anchor block resampling, not independent campaigns'))
    table=pd.DataFrame(summaries)
    table.to_csv(OUT/'all_methods_reliability.csv',index=False)
    pd.DataFrame(effects).to_csv(OUT/'baseline_effect_sizes.csv',index=False)
    s.to_csv(OUT/'candidate_reliability_scores.csv',index=False)
    compact=['method','batch_size','n']+[x+'_mean' for x in ['terrain_risk','combined_risk','terrain_support','cross_sd','optimism_gap','cross_mean','risk_scaled_lower90','lowrisk_stable_pass420','lowrisk_stable_pass435']]
    table[compact].to_csv(OUT/'main_comparison.csv',index=False)
    return table


def matched_measured_effect(cal,test):
    rng=np.random.default_rng(SEED)
    groups=[]
    for distance in [1,2]:
        for lo,hi in zip([4.2,4.25,4.3,4.35,4.4],[4.25,4.3,4.35,4.4,np.inf]):
            cm=(cal.nearest_training_hamming.eq(1) if distance==1 else cal.nearest_training_hamming.ge(2))&cal.cross_mean.ge(lo)&cal.cross_mean.lt(hi)
            tm=(test.nearest_training_hamming.eq(1) if distance==1 else test.nearest_training_hamming.ge(2))&test.cross_mean.ge(lo)&test.cross_mean.lt(hi)
            c=cal[cm];t=test[tm]
            if len(c)<20:continue
            q25,q75=c.combined_risk.quantile([.25,.75])
            low=t[t.combined_risk.le(q25)];high=t[t.combined_risk.ge(q75)]
            if min(len(low),len(high))>=5:groups.append((len(t),low,high))
    total=sum(g[0] for g in groups); rows=[]; all_draws={}
    for gname,j in [('low',1),('high',2)]:
        mse=0;failure=0;pred=0; mse_draw=np.zeros(2000); fail_draw=np.zeros(2000);n=0
        for size,*pair in groups:
            d=pair[j-1];w=size/total;n+=len(d)
            error=d.error.to_numpy();bad=d.brightness.lt(4.2).to_numpy()
            idx=rng.integers(0,len(d),size=(2000,len(d)))
            mse+=w*np.mean(error**2);failure+=w*bad.mean();pred+=w*d.cross_mean.mean()
            mse_draw+=w*np.mean(error[idx]**2,axis=1);fail_draw+=w*bad[idx].mean(axis=1)
        all_draws[gname]=(np.sqrt(mse_draw),fail_draw)
        rows.append(dict(group=gname,n=n,strata=len(groups),represented_test_n=total,mean_prediction=pred,rmse=np.sqrt(mse),false_bright_fraction=failure,
            rmse_ci_low=np.quantile(np.sqrt(mse_draw),.025),rmse_ci_high=np.quantile(np.sqrt(mse_draw),.975),
            false_bright_ci_low=np.quantile(fail_draw,.025),false_bright_ci_high=np.quantile(fail_draw,.975)))
    pd.DataFrame(rows).to_csv(OUT/'matched_measured_validation.csv',index=False)
    effects=[]
    for metric,j in [('rmse',0),('false_bright_fraction',1)]:
        delta=all_draws['low'][j]-all_draws['high'][j]
        effects.append(dict(metric=metric,delta_low_minus_high=rows[0][metric]-rows[1][metric],
            ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975)),
            interval_scope='Stratified sequence bootstrap; conditional on this split and fixed strata'))
    pd.DataFrame(effects).to_csv(OUT/'matched_measured_effect_sizes.csv',index=False)
    return pd.DataFrame(rows)


def export_report(table,coverage,screening,matched):
    # All selected outputs are reported; no best-metric or best-batch selection.
    compact=table[table.batch_size.eq(100)].set_index('method')
    rows=['GLOM-TGITO 与基线的可信度统计（2026-10-14）','',
          '五种方法各完成 2700 个上下文；比较已冻结 Top6/30/100。以下为结果分析，不改变搜索或候选顺序。',
          '统计依据：风险评分先与实测误差验证，再对各方法候选使用统一评分。跨种子只使用未参与搜索的两个 GLOM 种子。',
          '风险越低、支撑越高表示模型内证据较强；跨种子波动和 GLOM 主种子相对交叉种子的分数回落反映稳定性。',
          '对于 ESM650 搜索，GLOM 主种子差值只是共同评价模型的种子差异，不是其原生搜索分数的回落。',
          '经验亮度下界由校准层残差估计，按高预测亮度/训练距离分层，并在测试层检验覆盖。','',
          'Top100 结果：']
    for m in METHODS:
        d=compact.loc[m]
        rows.append(f'{NAMES[m]}：地势风险 {d.terrain_risk_mean:.4f}；综合风险 {d.combined_risk_mean:.4f}；支撑 {d.terrain_support_mean:.4f}；跨种子 SD {d.cross_sd_mean:.4f}；GLOM 主种子差值 {d.optimism_gap_mean:.4f}；经验90%下界均值 {d.risk_scaled_lower90_mean:.4f}；通过低风险/稳定性/下界>=4.2 的候选 {round(d.lowrisk_stable_pass420_mean*100)}/100。')
    rows.extend(['','分层实测验证（未参与搜索的 GLOM 种子平均预测）：'])
    for d in matched.itertuples():
        rows.append(f'{d.group}: n={d.n}; 平均预测={d.mean_prediction:.4f}; RMSE={d.rmse:.4f} [{d.rmse_ci_low:.4f}, {d.rmse_ci_high:.4f}]; 虚高比例={d.false_bright_fraction:.1%}。')
    rows.extend(['','论文解释范围：',
        '原生置信度的亮度、地势、风险权重保持原样。综合风险为原有 risk + 0.5*prediction_gap，不是看完基线后挑权重。',
        '生成候选的实测荧光仍缺失；可信度指标、经验下界和跨模型支持是计算证据。',
        '候选均值置信区间来自本次冻结候选按峰/起点组重采样，不代表多轮独立900峰实验。',
        '历史 HPO 与当前测试标签有交集，分析属于回顾性策略迁移；实测验证与完整策略独立盲测区分报告。',
        '优化候选存在分布偏移；覆盖率和接受规则在实测测试集上的表现单独给出，不把它当作新候选成功概率。',
        'AdaLead 没有使用多地势图。对其输出计算地势指标只发生在冻结候选后的共同评价环节。','',
        '方法依据：Greenman et al., PLOS Computational Biology (2025), doi:10.1371/journal.pcbi.1012639；Fannjiang et al., PNAS (2022), doi:10.1073/pnas.2204569119。'])
    (OUT/'可信度对照结果.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    latex=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Method & Terrain risk $\downarrow$ & Support $\uparrow$ & Seed SD $\downarrow$ & GLOM seed gap & Empirical lower $\uparrow$ \\',r'\midrule']
    for m in METHODS:
        d=compact.loc[m]
        latex.append(f'{NAMES[m]} & {d.terrain_risk_mean:.4f} & {d.terrain_support_mean:.4f} & {d.cross_sd_mean:.4f} & {d.optimism_gap_mean:.4f} & {d.risk_scaled_lower90_mean:.4f} '+r'\\')
    latex.extend([r'\bottomrule',r'\end{tabular}',r'% Top100 frozen candidates. Bounds are empirical, not fluorescence measurements.'])
    (OUT/'table_reliability_top100.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    payload={name:json.loads(frame.to_json(orient='records')) for name,frame in [('summary',table),('coverage',coverage),('screening',screening),('matched',matched)]}
    payload['method_names']=NAMES
    write_json(OUT/'workbook_data.json',payload)


def main():
    freeze_analysis();print('Inputs and rules verified',flush=True)
    cal,test,s=load_data();print('Loaded measured and generated candidates',flush=True)
    coverage=calibrated_bounds(cal,test,s)
    curves,screening,thresholds=measured_validation(cal,test)
    matched=matched_measured_effect(cal,test)
    table=candidate_comparison(cal,s,thresholds)
    export_report(table,coverage,screening,matched)
    cols=['sequence','brightness','cross_mean','cross_sd','combined_risk','terrain_support','nearest_training_hamming','stratum','error','risk_scaled_lower90','risk_scaled_lower95','risk_scaled_width90']
    test[cols].to_csv(OUT/'test_reliability_rows.csv',index=False)
    cal[[c for c in cols if c in cal]].to_csv(OUT/'calibration_reliability_rows.csv',index=False)
    write_json(OUT/'COMPUTATION_DONE.json',dict(completed_local=datetime.now().astimezone().isoformat(),
        script_sha256=sha(Path(__file__)),analysis_rules_sha256=sha(OUT/'ANALYSIS_RULES.json'),
        selected_rows=len(s),distinct_candidates=s.sequence.nunique(),methods=5,
        candidate_labels_missing=True,search_unchanged=True,
        files={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.csv','.txt','.tex','.json'] and p.name!='COMPUTATION_DONE.json'}))
    print(table[table.batch_size.eq(100)][['method','terrain_risk_mean','terrain_support_mean','cross_sd_mean','optimism_gap_mean','risk_scaled_lower90_mean','lowrisk_stable_pass420_mean']].to_string(index=False),flush=True)


if __name__=='__main__':main()
