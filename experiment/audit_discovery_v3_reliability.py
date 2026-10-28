"""Post-hoc reliability diagnostics; never changes frozen search or selection.

Risk is validated against measured test residuals. Calibration-only quantiles
define rejection thresholds. Matched summaries are descriptive, not a causal
ablation or an independent prospective experiment.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
EVAL = ROOT / 'artifacts/discovery_v3/evaluation'
OUT = ROOT / 'reports/discovery_v3_reliability_audit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_distance(sequences, known, reference):
    k = np.asarray([list(s.encode('ascii')) for s in known], dtype=np.uint8)
    wt = np.frombuffer(reference.encode('ascii'), dtype=np.uint8)
    depth = np.count_nonzero(k != wt, axis=1)
    result = []
    for seq in sequences:
        c = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        pos = np.flatnonzero(c != wt)
        d = depth + len(pos)
        if len(pos):
            d = d - np.count_nonzero(k[:, pos] != wt[pos], axis=1)
            d = d - np.count_nonzero(k[:, pos] == c[pos], axis=1)
        result.append(int(d.min()))
    return np.asarray(result)


def metrics(d):
    if not len(d):
        return {'n': 0}
    return dict(n=len(d), rmse=float(np.sqrt(np.mean(d.error ** 2))),
                mae=float(d.error.abs().mean()), mean_overprediction=float(d.error.mean()),
                measured_below420_n=int(d.brightness.lt(4.2).sum()),
                measured_below420_fraction=float(d.brightness.lt(4.2).mean()),
                mean_prediction=float(d.pred.mean()), mean_measured=float(d.brightness.mean()))


def mask(d, scope):
    if scope == 'all':
        return np.ones(len(d), dtype=bool)
    return d.pred.ge(float(scope.split('_')[-1])).to_numpy()


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    pred = pd.read_csv(EVAL / 'glom_live_predictions.csv')
    risk = pd.read_csv(EVAL / 'live_risk_scores.csv')
    risk['combined_risk'] = risk.terrain_selected_risk_wmean + .5 * risk.terrain_selected_pred_gap_wmean
    train = pd.read_csv(ROOT / 'data/train.csv', usecols=['sequence']).sequence.tolist()
    reference_path = ROOT / 'repro_project/data/processed/reference_sequence.txt'
    if not reference_path.exists():
        reference_path = ROOT.parent / 'handoff_20261001/repro_project/data/processed/reference_sequence.txt'
    wt = reference_path.read_text().strip()
    assert len(wt) == 221
    frames = {}
    for split in ['calibration', 'test']:
        d = pd.read_csv(ROOT / f'data/{split}.csv').merge(risk, on='sequence', validate='one_to_one')
        d = d.merge(pred, on='sequence', validate='one_to_one')
        assert np.allclose(d.pred, d.glom_20261013, rtol=0, atol=1e-7)
        d['nearest_train'] = nearest_distance(d.sequence, train, wt)
        assert d.nearest_train.ge(1).all()
        d['distance_group'] = np.where(d.nearest_train.eq(1), '1', '2plus')
        d['error'] = d.pred - d.brightness
        d['seed_sd'] = d[['glom_20261014', 'glom_20261015']].std(axis=1, ddof=0)
        frames[split] = d
        print(json.dumps({'loaded': split, 'n': len(d)}), flush=True)
    cal, test = frames['calibration'], frames['test']
    diagnostics, curves, quartiles = [], [], []
    for scope in ['all', 'pred_ge_4.2', 'pred_ge_4.31']:
        c, t = cal[mask(cal, scope)], test[mask(test, scope)]
        for name in ['combined_risk', 'terrain_selected_risk_wmean', 'terrain_selected_pred_gap_wmean', 'seed_sd']:
            diagnostics.append(dict(scope=scope, score=name, **metrics(t),
                spearman_absolute_error=float(spearmanr(t[name], t.error.abs()).statistic),
                spearman_overprediction=float(spearmanr(t[name], t.error).statistic),
                measured_below420_auc=float(roc_auc_score(t.brightness.lt(4.2), t[name]))
                    if t.brightness.lt(4.2).nunique() == 2 else None))
            for fraction in [.25, .5, .75, 1.]:
                cutoff = float(c[name].quantile(fraction)) if fraction < 1 else float('inf')
                keep = t[t[name].le(cutoff)]
                curves.append(dict(scope=scope, score=name, calibration_keep_fraction=fraction,
                    calibration_n=len(c), cutoff=cutoff, test_keep_fraction=len(keep)/len(t), **metrics(keep)))
        # Test quartiles illustrate association only; not calibration-selected deployment rules.
        for q, g in t.groupby(pd.qcut(t.combined_risk, 4, labels=False, duplicates='drop')):
            quartiles.append(dict(scope=scope, test_risk_quartile=int(q)+1, **metrics(g)))
    pd.DataFrame(diagnostics).to_csv(OUT/'measured_risk_association.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT/'calibration_threshold_risk_coverage.csv', index=False)
    pd.DataFrame(quartiles).to_csv(OUT/'descriptive_test_quartiles.csv', index=False)

    # Control approximately for brightness and sequence novelty using fixed bins.
    # Low/high risk cutoffs use calibration labels neither directly nor through tuning here.
    edges = [4.2, 4.25, 4.3, 4.35, 4.4, np.inf]
    matched = []
    for distance in ['1', '2plus']:
        for lo, hi in zip(edges[:-1], edges[1:]):
            c = cal[cal.distance_group.eq(distance) & cal.pred.ge(lo) & cal.pred.lt(hi)]
            t = test[test.distance_group.eq(distance) & test.pred.ge(lo) & test.pred.lt(hi)]
            if len(c) < 20:
                continue
            q25, q75 = c.combined_risk.quantile([.25, .75])
            for group, selected in [('low', t[t.combined_risk.le(q25)]), ('high', t[t.combined_risk.ge(q75)])]:
                matched.append(dict(distance=distance, pred_lo=lo, pred_hi=hi, calibration_n=len(c),
                    test_stratum_n=len(t), group=group, cutoff_low=q25, cutoff_high=q75, **metrics(selected)))
    matched = pd.DataFrame(matched)
    matched.to_csv(OUT/'brightness_distance_matched_risk.csv', index=False)
    standardized = []
    eligible = []
    for key, g in matched.groupby(['distance', 'pred_lo', 'pred_hi']):
        if len(g) == 2 and g.n.ge(5).all():
            eligible.append(g)
    if eligible:
        joined = pd.concat(eligible)
        for group, g in joined.groupby('group'):
            weights = g.test_stratum_n / g.test_stratum_n.sum()
            standardized.append(dict(group=group, strata=len(g), n=int(g.n.sum()),
                represented_test_n=int(g.test_stratum_n.sum()),
                rmse=float(np.sqrt(np.sum(weights*g.rmse**2))),
                measured_below420_fraction=float(np.sum(weights*g.measured_below420_fraction)),
                mean_prediction=float(np.sum(weights*g.mean_prediction))))
    pd.DataFrame(standardized).to_csv(OUT/'matched_standardized_summary.csv', index=False)

    selected = pd.read_csv(EVAL/'selected_candidates_evaluated.csv')
    profile = []
    for (method, size), d in selected.groupby(['method', 'batch_size']):
        s = d[['sequence']].merge(risk, on='sequence', validate='one_to_one')
        profile.append(dict(method=method, batch_size=int(size), n=len(s),
            mean_terrain_risk=float(s.terrain_selected_risk_wmean.mean()),
            mean_prediction_terrain_gap=float(s.terrain_selected_pred_gap_wmean.mean()),
            mean_combined_risk=float(s.combined_risk.mean()),
            mean_terrain_support=float(s.terrain_selected_support_wmean.mean()),
            primary_glom_prediction_above_calibration_max_fraction=float(s.pred.gt(cal.pred.max()).mean()),
            measured_candidate_n=int(d.measured_brightness_if_available.notna().sum())))
    profiles = pd.DataFrame(profile)
    profiles.to_csv(OUT/'frozen_candidate_risk_profiles.csv', index=False)
    sources = [ROOT/'data/calibration.csv', ROOT/'data/test.csv', ROOT/'data/train.csv',
               EVAL/'glom_live_predictions.csv', EVAL/'live_risk_scores.csv',
               EVAL/'selected_candidates_evaluated.csv', ROOT/'discovery_v3_baselines.py',
               ROOT/'artifacts/discovery_v3/PROTOCOL.json', Path(__file__)]
    audit = dict(created_local=datetime.now().astimezone().isoformat(),
        analysis_status='Post-hoc descriptive audit, not preregistered confirmatory evidence',
        search_or_selection_changed=False, generated_labels_imputed=False,
        adalead_uses_terrain=False, primary_predictor='glom_seed20261013',
        risk_formula='terrain_selected_risk_wmean + 0.5 * terrain_selected_pred_gap_wmean',
        historical_hpo_overlaps_current_test=True,
        scope='Predictor-level heldout residual validation; not a new independent policy blind test',
        native_confidence_is_calibrated_probability=False,
        matched_summary='Fixed brightness bins and training-distance strata; common test-stratum weights; descriptive, not causal',
        conformal_scope='Existing empirical coverage on measured test; no automatic guarantee on optimized sequences',
        input_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
        reference_sequence_sha256=sha(reference_path),
        standardized_comparison=standardized,
        calibration_primary_prediction_max=float(cal.pred.max()),
        calibration_predicted_ge435_n=int(cal.pred.ge(4.35).sum()),
        test_predicted_ge431_n=int(test.pred.ge(4.31).sum()))
    (OUT/'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    q = pd.DataFrame(quartiles)
    hq = q[q.scope.eq('pred_ge_4.2')].set_index('test_risk_quartile')
    report = [
        'GLOM-TGITO 可信度补充核查（冻结结果的事后诊断）',
        '',
        'AdaLead 未使用多地势、动态风险门控或原生综合置信度。它使用 GLOM 亮度作为适应度。',
        '本次只评价冻结候选与已有实测误差，没有修改搜索、候选排名或历史 HPO。',
        '',
        f'实测测试集共 {len(test)} 条；搜索 GLOM 预测 >=4.2 的 {len(test[test.pred.ge(4.2)])} 条。',
        f'其中最低风险四分位 RMSE={hq.loc[1,"rmse"]:.6f}，最高四分位 RMSE={hq.loc[4,"rmse"]:.6f}。',
        f'两组实测 <4.2 比例分别为 {hq.loc[1,"measured_below420_fraction"]:.2%}、{hq.loc[4,"measured_below420_fraction"]:.2%}。',
        '以上分位比较存在预测亮度/距离差异；brightness_distance_matched_risk.csv 提供分层复核。',
        '这些数字检验风险对实际错误的识别能力，不能当作自由生成候选的实测成功率。',
        '',
        '可信度评价需要同时报告：实际误差关联、拒绝高风险后的误差/虚高率、校准覆盖与区间宽度、生成分布适用性。',
        '原生 confidence_score 是混合亮度、支撑、地势、低风险和熵的排序分，不是成功概率。',
        '生成候选的地势风险/支撑是模型内指标；需要先由实测误差验证。多样性或新颖性不等于可靠性。',
        f'高亮区域样本稀疏；校准集主模型预测 >=4.35 仅 {int(cal.pred.ge(4.35).sum())} 条。普通随机测试覆盖不自动适用于优化后的极端候选。',
        '历史 HPO 使用标签与当前 test 有重叠，本次保留为回顾性诊断，不称完全独立策略盲测。',
        '',
        '方法依据：',
        'Greenman et al. 2025, Benchmarking uncertainty quantification for protein engineering.',
        'https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012639',
        'Fannjiang et al. 2022, Conformal prediction under feedback covariate shift for biomolecular design.',
        'https://pmc.ncbi.nlm.nih.gov/articles/PMC9618043/',
    ]
    if standardized:
        comparison = pd.DataFrame(standardized).set_index('group')
        report.extend([
            '', '预测亮度与训练距离分层的共同权重比较：',
            f'纳入 {int(comparison.loc["low","strata"])} 个有足够样本的分层，代表 {int(comparison.loc["low","represented_test_n"])} 条测试序列。',
            '分层内低/高风险阈值只按校准集的 25%/75% 分位数确定，不按测试结果调阈值。',
            f'低风险 n={int(comparison.loc["low","n"])}，高风险 n={int(comparison.loc["high","n"])}；以下为按相同分层权重标准化的描述性统计：',
            f'低风险 RMSE={comparison.loc["low","rmse"]:.6f}，高风险 RMSE={comparison.loc["high","rmse"]:.6f}。',
            f'低风险预测>=4.2但实测<4.2比例={comparison.loc["low","measured_below420_fraction"]:.2%}，高风险={comparison.loc["high","measured_below420_fraction"]:.2%}。',
            f'低风险平均预测={comparison.loc["low","mean_prediction"]:.6f}，高风险={comparison.loc["high","mean_prediction"]:.6f}。',
            '本结果支持风险分数与真实误差有关；尚不是多地势导致改善的因果消融，也不是新序列的成功率。',
        ])
    report.extend(['', '冻结 Top100 候选的共同模型内评分（仅输出后评阅，没有把地势接到基线搜索）：'])
    for row in profiles[profiles.batch_size.eq(100)].itertuples():
        report.append(f'{row.method}: 地势风险均值 {row.mean_terrain_risk:.6f}；支撑度 {row.mean_terrain_support:.6f}；主模型预测超出校准最大预测的比例 {row.primary_glom_prediction_above_calibration_max_fraction:.0%}。')
    report.append('这些方法都经优化挑选；评分分布有明显变化，不能把模型内低风险直接当成更高真实成功率。')
    (OUT/'可信度核查.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(OUT), 'standardized':standardized}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
