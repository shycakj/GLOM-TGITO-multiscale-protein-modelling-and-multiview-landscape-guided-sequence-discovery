"""Package verified reliability statistics and a manuscript-ready result fragment."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import zipfile
import pandas as pd
from compare_discovery_v3_reliability import ROOT, OUT, METHODS, NAMES, sha, write_json


def main():
    table=pd.read_csv(OUT/'all_methods_reliability.csv')
    joint=pd.read_csv(OUT/'joint_novelty_reliability_primary.csv')
    top=table[table.batch_size.eq(100)].set_index('method')
    j=joint[joint.batch_size.eq(100)].set_index('method')
    document=r'''\paragraph{Reliability assessment of the complete discovery pipeline.}
We compared frozen candidate batches from GLOM--TGITO and four native discovery
baselines using common post-selection reliability criteria. The original searches,
historical hyperparameters and native rankings were preserved. Terrain risk and
support were scored on every output after freezing. Predictions from two GLOM
training seeds that had not been used by the search provided a common stability
assessment. These predictors share architecture and training data.

\paragraph{Validation against measured fluorescence.}
Within fixed prediction and nearest-training-distance strata, low-risk test
sequences had a standardized RMSE of 0.0508, compared with 0.1090 for high-risk
sequences. Mean predicted fluorescence was similar (4.2419 versus 4.2401).
The corresponding fractions with measured fluorescence below 4.2 were 27.0\%
and 35.7\%. The low-minus-high RMSE difference was $-0.0582$ (conditional
stratified bootstrap 95\% interval: $[-0.0982,-0.0166]$); the below-threshold
fraction difference was $-8.70$ percentage points ($[-16.07,-0.98]$).
This retrospective analysis supports an association between the terrain-derived
risk score and actual prediction errors. The analysis follows the distinction
between predictive accuracy and uncertainty quality emphasized in protein UQ
benchmarking~\cite{greenman2025uq}.

\paragraph{Reliability and novelty of generated candidates.}
For the frozen Top100 batch, GLOM--TGITO had mean terrain risk 0.1182 and mean
terrain support 0.3168. Relative to GLOM--AdaLead, these corresponded to 12.8\%
lower risk and 32.2\% higher support; relative to GLOM--ProtHMSO, the differences
were 6.2\% and 16.8\%, respectively. A common computational acceptance rule
required a terrain-scaled empirical lower fluorescence bound of at least 4.2,
risk no greater than the calibration median, and cross-seed SD no greater than
the calibration 75th percentile. Both risk and SD thresholds used calibration
sequences with cross-seed predicted fluorescence at least 4.2.
Combining this rule with a minimum Hamming distance of two from every visible
sequence retained 38 GLOM--TGITO candidates, versus 16 for GLOM--AdaLead, 14 for
GLOM--random, 14 for GLOM--ProtHMSO and 11 for ESM650--ProtHMSO.
Varying the risk cutoff from the calibration 25th to 75th percentile retained
35--39 GLOM--TGITO candidates versus 15--20 GLOM--AdaLead candidates.
At a lower-bound threshold of 4.30 and the median risk cutoff, the counts were
26 and 10, respectively. GLOM--TGITO was not uniformly superior in cross-seed
dispersion, empirical lower brightness, or at the stricter threshold of 4.35.
The full threshold grid and all three batch sizes are reported.

\paragraph{Interpretation.}
The evidence supports a favorable combination of terrain support, lower modeled
risk and the yield of novel candidates satisfying a common computational rule.
The joint rule was added as a documented post-hoc analysis; it did not rerank
the generated batches. The empirical lower bounds are calibrated residual
references, not guarantees for adaptively optimized sequences under feedback
covariate shift~\cite{fannjiang2022feedback}. Terrain quality also uses calibration
data, so these estimates are reported as empirical rather than as a formal
split-conformal guarantee. Historical HPO label usage overlaps the present test
partition; the study is retrospective policy transfer. Generated candidates
without assays retain missing fluorescence values. Bootstrap intervals for
candidate scores describe this campaign's peak/anchor groups, rather than
independent repetitions of the complete discovery campaign.
'''
    (OUT/'reliability_results_fragment.tex').write_text(document,encoding='utf-8')
    bib=r'''@article{greenman2025uq,
  author={Greenman, Kevin P. and Amini, Ava P. and Yang, Kevin K.},
  title={Benchmarking uncertainty quantification for protein engineering},
  journal={PLOS Computational Biology}, year={2025}, volume={21}, number={1},
  pages={e1012639}, doi={10.1371/journal.pcbi.1012639}
}
@article{fannjiang2022feedback,
  author={Fannjiang, Clara and others},
  title={Conformal prediction under feedback covariate shift for biomolecular design},
  journal={Proceedings of the National Academy of Sciences}, year={2022},
  doi={10.1073/pnas.2204569119}
}
'''
    (OUT/'reliability_references.bib').write_text(bib,encoding='utf-8')
    lines=['可信度统计交付说明','',
        '主文件：reliability_comparison.xlsx（方法对照、实测验证、680行候选评分、规则定义）。',
        '论文图：reliability_measured_validation.pdf 与 reliability_frozen_candidate_comparison.pdf；同时提供 PNG/SVG。',
        'LaTeX：table_reliability_top100.tex、reliability_results_fragment.tex、reliability_references.bib。',
        '详细差值和区间：baseline_effect_sizes.csv；全部阈值：joint_novelty_reliability_full_grid.csv。','',
        'Top100 同一规则下的新颖候选数（距全部可见序列>=2、经验亮度下界>=4.2、低风险、跨种子稳定）：']
    for m in METHODS:lines.append(f'{NAMES[m]}：{int(j.loc[m,"novel_and_reliability_pass_n"])} / 100')
    lines.extend(['',
        '我们相对 AdaLead：地势风险降低12.8%，支撑度提高32.2%，通过联合规则的新颖候选38对16。',
        '主风险阈值为高预测校准集的中位数，稳定性阈值为该组的75%分位数。阈值数值在 CALIBRATION_THRESHOLDS.json。',
        '阈值敏感性：亮度下界4.2、风险阈值25%至75%分位时，我们35至39个，AdaLead15至20个。',
        '亮度下界提高至4.3、中位风险阈值时，我们26个，AdaLead10个；4.35时我们3个，GLOM-ProtHMSO5个。',
        '跨种子波动和保守亮度下界未全面领先，所有方法及批次均保留。',
        '',
        '实测验证：未参与搜索的 GLOM 种子平均预测经亮度/距离分层后，低风险组RMSE 0.0508，高风险组0.1090；虚高比例27.0%和35.7%。',
        '统一接受规则在实测测试集中接受80条，其中78条实测>=4.2（97.5%，Wilson参考区间91.3%至99.3%）。',
        '其中距训练序列>=2的接受样本仅14条、实测达标14条（Wilson下限78.5%）；该小样本结果不外推成新候选100%成功。',
        '',
        '分析状态：事后统计，生成/排名/HPO保持冻结。多地势风险为模型内指标，先以实测误差核验，再用于共同输出评价。',
        '经验下界不自动具有优化候选的覆盖保证；地势质量使用过校准数据。历史HPO与测试标签交集已保留记录。',
        '候选评分区间是本次候选池按峰/起点组重采样的描述性区间，不是多轮独立900峰重复。',
        '未知实测亮度保持缺失。AdaLead搜索没有使用多地势图。','',
        '复现：在原项目 experiments_20261013 目录，按顺序运行：',
        'python compare_discovery_v3_reliability.py',
        'python summarize_reliable_novel_discovery.py',
        'python plot_discovery_v3_reliability.py',
        'Excel builder 位于报告目录 workbook_builder/build.mjs，依赖已安装的 @oai/artifact-tool。',
        '所需输入及哈希见 ANALYSIS_RULES.json；原始大规模搜索结果不在本增量交付包内重复打包。',
    ])
    (OUT/'交付说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    handoff_path=ROOT/'reports/active_run_handoff.json'
    handoff=json.loads(handoff_path.read_text(encoding='utf-8'))
    handoff['updated_local']=datetime.now().astimezone().isoformat()
    handoff['reliability_statistics_complete']=True
    handoff['reliability_comparison_directory']='reports/discovery_v3_reliability_comparison/'
    handoff['reliability_statistics']={
        'frozen_methods':5,'batch_sizes':[6,30,100],
        'top100_joint_rule_novel_counts':{m:int(j.loc[m,'novel_and_reliability_pass_n']) for m in METHODS},
        'joint_rule':'empirical lower>=4.2, calibration-median risk, calibration-q75 seed SD, visible distance>=2',
        'interpretation':'Lower terrain risk/higher support and more novel candidates passing the computational rule; seed SD and lower brightness not uniformly superior',
        'scope':'Post-hoc retrospective statistics; no generated fluorescence imputed'}
    write_json(handoff_path,handoff)
    scripts=[ROOT/p for p in ['audit_discovery_v3_reliability.py','compare_discovery_v3_reliability.py',
                            'summarize_reliable_novel_discovery.py','plot_discovery_v3_reliability.py','finalize_reliability_delivery.py']]
    # Flat enumeration deliberately excludes the node_modules junction.
    files=[p for p in OUT.iterdir() if p.is_file() and p.suffix in {'.csv','.json','.txt','.tex','.bib','.pdf','.svg','.png','.xlsx'}
           and 'render' not in p.stem and 'preview' not in p.stem and p.name!='DELIVERY_MANIFEST.json']
    files += scripts+[OUT/'workbook_builder/build.mjs',handoff_path]
    manifest={str(p.relative_to(ROOT).as_posix()):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    write_json(OUT/'DELIVERY_MANIFEST.json',dict(created_local=datetime.now().astimezone().isoformat(),files=manifest))
    files.append(OUT/'DELIVERY_MANIFEST.json')
    archive=ROOT/'discovery_v3_reliability_delivery.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files),sha256=sha(archive)),ensure_ascii=False))


if __name__=='__main__':main()
