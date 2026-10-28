"""Audit and summarize the completed frozen campaign without changing its results."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'artifacts/discovery_v3'
OUT = ROOT / 'reports/discovery_v3_review'
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protocol = json.loads((RUN / 'PROTOCOL.json').read_text())
    done = json.loads((RUN / 'evaluation/DONE.json').read_text())
    campaign = json.loads((RUN / 'CAMPAIGN_DONE.json').read_text())
    assert campaign['completed'] and done['methods'] == 5
    order = protocol['methods']
    names = {'glom_tgito': 'GLOM-TGITO', 'glom_adalead': 'GLOM + AdaLead',
             'glom_random': 'GLOM + random', 'glom_prothmso': 'GLOM + ProtHMSO',
             'esm650_prothmso': 'ESM650 + ProtHMSO'}
    records = []
    for method in order:
        folder = RUN / 'formal' / method / 'prefix_900'
        frozen = json.loads((folder / 'FROZEN.json').read_text())
        assert sha(folder / 'FROZEN.json') == done['candidate_freezes'][method]
        assert frozen['protocol_sha256'] == sha(RUN / 'PROTOCOL.json')
        for filename, digest in frozen['files'].items():
            assert sha(folder / filename) == digest, (method, filename)
        for sid, digest in frozen['sources'].items():
            p = folder.parent / f'context{int(sid):04d}_rep0/DONE.json'
            assert sha(p) == digest, (method, sid)
        runtime = pd.read_csv(folder / 'context_runtime.csv').sort_values('start_id')
        assert len(runtime) == 2700 and runtime.start_id.nunique() == 2700
        if method == 'glom_tgito':
            budgets = runtime.forward_sequences.to_numpy()
        else:
            assert np.array_equal(runtime.forward_sequences.to_numpy(), budgets)
        records.append({'method': method, 'contexts': 2700,
                        'valid_contexts': frozen['valid_contexts'],
                        'unique_candidates': frozen['unique_candidates'],
                        'forward_sequences': int(runtime.forward_sequences.sum()),
                        'backward_steps': int(runtime.backward_steps.sum())})
    # Download manifest records authenticate the evaluation files against the snapshot.
    manifest = json.loads((ROOT / 'reports/discovery_v3_snapshot_manifest.json').read_text())
    eval_files = [x for x in manifest['files'] if x['path'].startswith('artifacts/discovery_v3/evaluation/')]
    for item in eval_files:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    summary = pd.read_csv(RUN / 'evaluation/full_pipeline_comparison.csv')
    selected = pd.read_csv(RUN / 'evaluation/selected_candidates_evaluated.csv')
    assert len(summary) == 15 and len(selected) == 680
    assert selected.measured_brightness_if_available.isna().all()
    extra = []
    for (method, k), d in selected.groupby(['method', 'batch_size']):
        assert len(d) == k and d.sequence.nunique() == k
        extra.append({'method': method, 'batch_size': k,
                      'nearest_visible_at_least_2_fraction': float(d.nearest_visible_hamming.ge(2).mean()),
                      'glom_seed14_mean': float(d.glom_20261014.mean()),
                      'glom_seed15_mean': float(d.glom_20261015.mean()),
                      'native_predicted_mean': float(d.pred.mean())})
    table = summary.merge(pd.DataFrame(extra), on=['method', 'batch_size'], validate='one_to_one')
    table['method_order'] = table.method.map({m: i for i, m in enumerate(order)})
    table = table.sort_values(['batch_size', 'method_order']).drop(columns='method_order')
    table.to_csv(OUT / 'comparison_all_batch_sizes.csv', index=False)
    metrics = pd.read_csv(RUN / 'evaluation/judge_measured_test_metrics.csv')
    metrics[['judge', 'scope', 'n', 'spearman', 'pearson', 'rmse', 'mae', 'r2']].to_csv(OUT / 'judge_accuracy_by_scope.csv', index=False)
    measured = pd.read_csv(RUN / 'evaluation/measured_library_full_policy.csv')
    measured[measured.library.eq(-1)].to_csv(OUT / 'whole_measured_library_screening.csv', index=False)
    paired = []
    for k in [6, 30]:
        d = measured[measured.library.ge(0) & measured.batch_size.eq(k)]
        scores = d.pivot(index='library', columns='method', values='true_mean')
        counts = d.pivot(index='library', columns='method', values='selected_n')
        mask = counts.glom_native_confidence.eq(k) & counts.glom_prediction.eq(k)
        delta = scores.loc[mask, 'glom_native_confidence'] - scores.loc[mask, 'glom_prediction']
        paired.append({'requested_batch_size': k, 'sampled_libraries': len(scores),
                       'both_filled_requested_batch': int(mask.sum()),
                       'mean_true_brightness_difference': float(delta.mean()) if len(delta) else None,
                       'fraction_improved': float((delta > 1e-10).mean()) if len(delta) else None,
                       'scope': 'Overlapping hidden-library screening subsets; not independent search replicates'})
    pd.DataFrame(paired).to_csv(OUT / 'measured_screening_paired_summary.csv', index=False)
    report = {'reviewed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'campaign_completed_server_timestamp_cst': datetime.fromtimestamp(campaign['unix_time'], timezone(timedelta(hours=8))).isoformat(),
              'frozen_outputs_verified': True, 'context_records_verified': 13500,
              'evaluation_files_verified': len(eval_files), 'all_forward_budgets_match': True,
              'generation': records, 'generated_batch_rows': len(selected),
              'unique_selected_sequences_across_methods_and_batch_sizes': selected.sequence.nunique(),
              'all_selected_sequences_novel_to_assay_catalogue': bool(selected.novel_to_measured_catalogue.all()),
              'candidate_measurements_available': 0, 'paired_measured_screening': paired,
              'scope': 'One full campaign per method. Cross-GLOM seeds share architecture and training data. No search or ranking changes after evaluation.'}
    (OUT / 'AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    health = {'reviewed_at_cst': report['reviewed_at_cst'], 'campaign_complete': True,
              'evaluation_complete': True, 'formal_counts': {r['method']: r['contexts'] for r in records},
              'all_forward_budgets_match': True, 'frozen_outputs_verified': True,
              'context_records_verified': 13500, 'evaluation_files_verified': len(eval_files),
              'evidence': 'reports/discovery_v3_review/AUDIT.json'}
    (ROOT / 'reports/discovery_v3_health_check.json').write_text(json.dumps(health, indent=2), encoding='utf-8')
    lines = ['GLOM–TGITO 完整流程对照：本轮结果复核', '',
             '900 峰、四组基线和统一评价全部完成；每组 2700 个上下文，每组前向序列预算 1,476,063。',
             '已核对五组冻结文件、13,500 个完成记录及评价备份文件校验值。',
             '下列交叉评分使用未参与本轮搜索的 GLOM 种子 20261014/20261015；数值是模型预测亮度。', '']
    for k in [6, 30, 100]:
        lines.append(f'Top {k}：方法 | GLOM 交叉均分 | ESM650 交叉均分 | 至最近可见序列距离 | 批内平均两两距离')
        for row in table[table.batch_size.eq(k)].itertuples():
            lines.append(f'{names[row.method]} | {row.glom_cross_mean_mean:.6f} | {row.esm650_cross_mean_mean:.6f} | '
                         f'{row.nearest_visible_hamming_mean:.3f} | {row.mean_pairwise_hamming:.3f}')
        lines.append('')
    lines += ['结果判断：',
              '1. Top 6 的 GLOM 交叉亮度评分与 AdaLead 基本相同；Top 30/100 的亮度评分低于其余 GLOM 搜索基线。',
              '2. GLOM–TGITO 的 Top 30/100 在距可见序列距离和批内多样性上均高于四组对照。',
              '3. ESM650 对不同方法的排序与 GLOM 不同。两类评分分开保留；均不可当作生成候选的实测荧光。',
              '4. 五种方法的已选候选全部未出现在实测库中；目前这些新候选均无实测荧光值。',
              '5. 固定实测候选库的筛选实验单独报告。Top 100 因亮度下限和多样性约束只能选出 48 个 GLOM 候选，不能标成实测 Top 100。',
              '6. 隐藏子库的成对比较仅纳入两种策略均选满预定数量的子库；重叠子库不当成独立搜索重复。',
              '7. 本轮支持的具体表现是：在较高预测亮度下探索更远、更分散的新候选；亮度排序未全面优于强基线。', '',
              '预测器在 3144 条实测测试序列上的准确性：']
    for row in metrics[metrics.scope.eq('all')].itertuples():
        lines.append(f'{row.judge}: Spearman={row.spearman:.4f}, RMSE={row.rmse:.4f}, R²={row.r2:.4f}')
    lines += ['', '完整实测库筛选和子库汇总见同目录 CSV；历史 HPO 为回顾性参数迁移，其标签来源与当前测试划分有重叠。',
              '本轮冻结结果保持原样；后续排查或消融须另立目录与协议。']
    (OUT / '结果解读.txt').write_text('\n'.join(lines), encoding='utf-8')
    plot(table, order, names)
    print(json.dumps(report, ensure_ascii=False))
    print(table[table.batch_size.eq(100)][['method', 'glom_cross_mean_mean', 'nearest_visible_hamming_mean', 'mean_pairwise_hamming', 'nearest_visible_at_least_2_fraction']].to_string(index=False))

def plot(table, order, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ['#087E8B', '#59677B', '#BE8D3C', '#7D5BA6', '#C76860']
    panels = [('glom_cross_mean_mean', 'GLOM cross-seed score', 'Predicted brightness'),
              ('esm650_cross_mean_mean', 'ESM650 cross-seed score', 'Predicted brightness'),
              ('nearest_visible_hamming_mean', 'Distance from visible sequences', 'Mean nearest Hamming distance'),
              ('mean_pairwise_hamming', 'Candidate diversity', 'Mean pairwise Hamming distance')]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    for ax, (col, title, ylabel), letter in zip(axes.flat, panels, 'ABCD'):
        for method, color in zip(order, colors):
            d = table[table.method.eq(method)].sort_values('batch_size')
            ax.plot(range(3), d[col], marker='o' if method == 'glom_tgito' else 's', color=color,
                    linewidth=2.6 if method == 'glom_tgito' else 1.3, markersize=5,
                    label=names[method], zorder=3 if method == 'glom_tgito' else 2)
        ax.set_xticks([0, 1, 2], ['Top 6', 'Top 30', 'Top 100'])
        ax.set_title(f'{letter}   {title}', loc='left', fontsize=11, pad=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis='y', color='#E6E9EC', linewidth=.7)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .985), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.12, left=.09, right=.98, hspace=.48, wspace=.30)
    fig.text(.09, .035, 'Frozen batches; one campaign per method. Cross-seed scores are predictions, not measured fluorescence.', fontsize=9, color='#52606D')
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'full_pipeline_comparison.{suffix}', dpi=200, facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    main()
