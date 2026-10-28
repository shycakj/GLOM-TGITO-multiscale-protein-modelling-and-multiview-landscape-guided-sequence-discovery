"""Verify synchronized full-generation artifacts and write a concise status.

Reads measured labels only after verifying frozen final selections. Does not
rerank or modify the selected sequences. Missing measurements remain missing.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/models/random/glom_seed20261013/full_discovery_v2'
DEST=BASE/'prefix_900'
CST=timezone(timedelta(hours=8))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen=json.loads((DEST/'FROZEN.json').read_text())
    done=json.loads((BASE/'DONE.json').read_text())
    assert done['completed_contexts']==2700 and all(w['exit_code']==0 for w in done['workers'])
    assert len(list(BASE.glob('context*/DONE.json')))==2700
    verified=[]
    # The old sync excludes FASTA; reconstruct identical bytes from frozen CSVs.
    for k in [6,30,100]:
        d=pd.read_csv(DEST/f'top{k}.csv')
        body=''.join(f'>GLOMTGITO_v2_P900_rank{r.rank}_context{r.start_id}\n{r.sequence}\n' for r in d.itertuples()).encode()
        assert hashlib.sha256(body).hexdigest()==frozen['files'][f'top{k}.fasta']
        if not (DEST/f'top{k}.fasta').exists():(DEST/f'top{k}.fasta').write_bytes(body)
    for name,expected in frozen['files'].items():
        assert digest(DEST/name)==expected,name
        verified.append(name)
    for source in frozen['sources']:
        assert digest(BASE/f"context{source['start_id']:04d}/DONE.json")==source['done_sha256']
    crossdone=json.loads((DEST/'CROSS_SCORE_DONE.json').read_text())
    structdone=json.loads((DEST/'STRUCTURE_DONE.json').read_text())
    assert digest(DEST/'structure_compatibility_scores.csv')==structdone['scores_sha256']
    assert digest(ROOT/'run_full_discovery_v2.py')==crossdone['source_sha256']
    assert digest(ROOT/'validate_structure_evidence.py')==structdone['source_sha256']
    selected=pd.read_csv(DEST/'selected_cross_evaluated.csv')
    original=pd.read_csv(DEST/'selected.csv')
    assert selected[['batch_size','rank','sequence']].equals(original[['batch_size','rank','sequence']])
    assert selected.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{221}').all()
    reference=pd.read_csv(ROOT/'data/train.csv').query('mutation_depth == 0').iloc[0].sequence
    truth=pd.read_csv(ROOT/'data/unique_sequences.csv').set_index('sequence')
    selected['exact_in_measurement_library']=selected.sequence.isin(truth.index)
    selected['measured_brightness']=selected.sequence.map(truth.brightness)
    structure=pd.read_csv(DEST/'selected_structure_evaluated.csv')
    cols=['mpnn_log_likelihood','mpnn_gain_vs_start','unscored_mutations','canonical_chromophore_preserved']
    selected=selected.merge(structure[['batch_size','rank','sequence']+cols],on=['batch_size','rank','sequence'],validate='one_to_one')
    selected.to_csv(DEST/'selected_status_checked.csv',index=False)
    compact=['rank','mutations_vs_reference','sequence','pred','cross_mean','cross_gain_vs_start','cross_gain_vs_wt',
             'confidence_score','terrain_selected_risk_wmean','nearest_visible_hamming','mutation_depth',
             'exact_in_measurement_library',*cols]
    for k in [6,30,100]:
        selected[selected.batch_size.eq(k)][compact].to_csv(DEST/f'top{k}_compact.csv',index=False)
    allseq=pd.read_csv(DEST/'all_unique_ranked.csv');runtime=pd.read_csv(DEST/'context_runtime.csv')
    summaries=[]
    for k,d in selected.groupby('batch_size'):
        chars=np.asarray([list(s) for s in d.sequence]);pair=[]
        for i in range(len(chars)):pair.extend(np.count_nonzero(chars[i+1:]!=chars[i],axis=1).tolist())
        assert min(pair)>=2
        summaries.append(dict(batch_size=int(k),n=len(d),search_pred_mean=float(d.pred.mean()),
            cross_mean=float(d.cross_mean.mean()),cross_min=float(d.cross_mean.min()),
            gain_vs_start_mean=float(d.cross_gain_vs_start.mean()),positive_gain_n=int(d.cross_gain_vs_start.gt(0).sum()),
            negative_gain_n=int(d.cross_gain_vs_start.lt(0).sum()),cross_high420_n=int(d.cross_mean.ge(4.2).sum()),
            observed_in_full_measurement_library=int(d.exact_in_measurement_library.sum()),
            nearest_visible_counts={str(int(a)):int(b) for a,b in d.nearest_visible_hamming.value_counts().sort_index().items()},
            pairwise_hamming_mean=float(np.mean(pair)),mpnn_gain_vs_start_mean=float(d.mpnn_gain_vs_start.mean()),
            unscored_mutation_sequences=int(d.unscored_mutations.gt(0).sum())))
    metrics=pd.read_csv(ROOT/'audit/structure_validation/assay_validation_metrics.csv')
    rows=metrics[(metrics.partition=='test')&(metrics.metric=='mpnn_log_likelihood')&(metrics.threshold==4.2)]
    status=dict(checked_cst=datetime.now(CST).isoformat(),completed_cst=datetime.fromtimestamp(done['updated_unix'],CST).isoformat(),
        complete=True,contexts=2700,valid_contexts=frozen['valid_contexts'],unique_candidates=frozen['unique_candidates'],
        constraints_satisfied_fraction=frozen['valid_contexts']/2700,
        unique_candidates_observed_in_measurement_library=int(allseq.sequence.isin(truth.index).sum()),
        verified_frozen_files=verified,verified_context_manifests=len(frozen['sources']),
        computational_readout=summaries,structure_assay_validation=rows.to_dict('records'),
        forward_sequences=int(runtime.forward_sequences.sum()),gradient_sequences=int(runtime.gradient_sequences.sum()),
        summed_search_worker_hours=float(runtime.seconds.sum()/3600),
        full_pipeline_baseline_comparison='Not yet run under revised protocol; prior comparisons share multiterrain components',
        wetlab='Measurements of the earlier six submitted candidates are still pending per latest user information',
        generation_complete_does_not_mean_all_research_questions_complete=True)
    interpretation=BASE/'INTERPRETATION_20261014.json'
    if interpretation.exists():
        status['configuration_interpretation']=json.loads(interpretation.read_text())
        status['historical_hpo_applied']=False
    report=ROOT/'reports/discovery_status_20261014.json';report.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['GLOM–TGITO完整生成检查（2026-10-14）','',
          f"完成时间（北京时间）：{status['completed_cst']}",
          f"900峰×3起点=2700条轨迹全部完成；{frozen['valid_contexts']}次输出满足生成约束；去重{frozen['unique_candidates']}条。",
          '已核验10个冻结候选文件、2700份上下文完成清单、代码来源以及结构评分哈希。','',
          'Top批次 | 数量 | 交叉预测均值 | 相对各自起点均值变化 | 逐条提高数 | 已测库精确重复数']
    if interpretation.exists():
        text[2:2]=['【配置审计更正】该批为非历史HPO配置的完整流程诊断，不能标为HPO最优版本结果。',
                    'HPO十个搜索参数中九个不一致；参见 audit/hpo_configuration_20261014/AUDIT.json 和 HPO核对与重跑决策_20261014.txt。',
                    '轨迹、候选和评分原样保留；配置更正后的搜索尚未启动。','']
    text.extend(f"Top{x['batch_size']} | {x['n']} | {x['cross_mean']:.5f} | {x['gain_vs_start_mean']:+.5f} | {x['positive_gain_n']}/{x['n']} | {x['observed_in_full_measurement_library']}" for x in summaries)
    text.extend(['','亮度数值为数据集log-brightness标度上的计算预测。交叉评价来自未参与搜索的两个GLOM种子，架构和训练数据共享。',
        '结构指标已在3143校准+3144测试序列上完成检验。测试集总体AUROC约0.636，高预测区域内约0.620；它只能提供有限辅助信息。',
        '候选平均ProteinMPNN兼容性分数较起点略降；不将“完成结构评分”写成“结构验证全部通过”。',
        '尚待关键工作：修订协议下完整流程对完整流程的基线比较，以及完整多地势链条的归因实验。',
        '历史湿实验6条与本轮新候选不同；尚未收到历史6条的实测数值。',
        '', '候选目录：artifacts/models/random/glom_seed20261013/full_discovery_v2/prefix_900/',
        '优先查看top6.fasta、top6_compact.csv、top30_compact.csv和selected_status_checked.csv。'])
    (ROOT/'reports/完整发现实验状态_20261014.txt').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
