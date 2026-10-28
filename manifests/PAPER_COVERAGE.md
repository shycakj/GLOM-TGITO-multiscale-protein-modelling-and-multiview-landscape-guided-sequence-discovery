# 论文图表—数据—代码索引

下列路径均相对本包根目录。统一命令的输出在 `reproduced/`；`paper/submission/` 保留实际投稿文件。`paper/data/Figure_Source_Data/` 以下简写为 `FD/`，`experiment/artifacts/discovery_v3/` 简写为 `D3/`。

## 正文及补充图

| 图 | 原始依据 | 重建入口 |
|---|---|---|
| Fig. 1 | `experiment/repro_project/pipeline/tgito/`、`encoder_contract.py`、`position_features.py`、D3 协议；2G3O 结构 | `paper/figure_reproduction/source/reproduce_architecture.py`；`reproduce.py figures` |
| Fig. 2 | D3 正式 GLOM–TGITO context 0054 的记录；FD `refinement_20261025/` 中的投影轨迹和地势数据 | `render_final_figures.py terrain`；同一搜索的综合目标曲线 |
| Fig. 3 | 78 组固定测试预测；FD `combined_prediction_all_runs.csv` | `reproduce.py tables`；`render_final_figures.py performance` |
| Fig. 4 | D3 `embedding_familiarity_v1/`；FD `discovery/`、`discovery_plot_inputs/` | `reproduce.py tables`、`statistics`、`figures` |
| Fig. 5A–E | `biophysics/md/*/rep0/`、`results/`、`mechanism_analysis_20261027/` | `analyze_md.py` 与机制分析；`paper/figure_reproduction/source/reproduce_figures.py` |
| Fig. 5F | `biophysics/results/*/rep[0-2]/METRICS.json`、`paper/three_runs/` | `reproduce.py tables`；`paper/three_runs/reproduce_run_tables.py` |
| Fig. S1 | 2G3O、结构映射、窗口及关键位点 | `paper/support_plotting/source/render_structure_plate.py` |
| Fig. S2、S11、S12 | 固定测试预测、表征 CKA 和 ridge 探针结果 | `render_updated_figures.py`、`render_selected_figures.py` |
| Fig. S3、S10 | 与 Fig. 2 相同的训练记忆和轨迹；测量/预测地势、门控和硬序列记录 | `render_final_figures.py` |
| Fig. S4、S5 | FD `discovery/` 中的完整候选批次、联合筛选及实测误差表 | `render_supplement_evidence.py` |
| Fig. S6 | D3 原生端点地势门控 | `render_terrain_journal_supplement.py:endpoints` |
| Fig. S7 | 训练记忆—动态门控—交替更新—原生筛选的实现 | `render_tgito_mechanism.py` |
| Fig. S8 | D3 偏离度、多阈值、MaxWEnt 和实测误差验证 | `render_discovery.py` |
| Fig. S9 | 固定划分与数据角色 | `render_protocol.py` |
| Fig. S13、S14 | 6 个 Top 30 代表的模拟结构、全部 Top 30 和三次重复 | `paper/figure_reproduction/source/reproduce_figures.py` |

`reproduce.py figures` 按依赖顺序运行上述入口，并应用最终稿中已经确认的图内标注精简。`reproduced/figures/REPRODUCTION.json` 逐图记录实际使用的生成来源。

## 补充表

| 表 | 内容 | 数据及复算 |
|---|---|---|
| S1 | HPO trial 12 与有效运行参数 | `experiment/repro_project/artifacts/tgito_hpo/`、D3 `PROTOCOL.json`；`verify` |
| S2–S3 | 随机留出、深度外推预测 | `experiment/artifacts/models/`、`artifacts/baselines/`；`tables prediction` |
| S4 | ProteinTTT 与未适应模型 | `experiment/artifacts/proteinttt_650m/`；`tables prediction` |
| S5 | 全局、区域、关键位点和联合表征探针 | `experiment/artifacts/models/random/glom_seed20261013/representation_analysis/`；`probe_representations.py` |
| S6 | 高预测实测子集上的阈值内外误差 | D3 `embedding_familiarity_v1/threshold_grid_v2/measured_validation.csv`；`tables deviation` |
| S7 | 邻域大小敏感性 | D3 `embedding_familiarity_v1/threshold_audit/`；`statistics` |
| S8 | 全部唯一候选与原生 Top100 | D3 `embedding_familiarity_v1/method_summary.csv`；`score_embedding_familiarity.py` |
| S9 | 完整多阈值表 | D3 `embedding_familiarity_v1/threshold_grid_v2/`；`tables deviation`、`statistics` |
| S10 | 公共评价器的实测测试精度 | D3 `evaluation/`；`evaluate_discovery_v3.py` |
| S11 | 搜索模型、其他种子、离散度与 optimism gap | D3 `evaluation/`、`experiment/reports/discovery_v3_reliability_comparison/`；`tables discovery`、`statistics` |
| S12 | 六个代表 Top 30 的身份与完整序列 | `paper/data/Supplementary_Data_1_six_representatives.fasta`、`biophysics/inputs/candidate_start_pairs.csv` |
| S13 | 五个完整方法的 Top6/30/100 | FD `discovery/all_selected_batch_metrics.csv`；`tables discovery` |
| S14 | 已测蛋白的回顾性优先级筛选 | D3 `evaluation/`；`evaluate_discovery_v3.py`；FD `discovery/` |
| S15–S17 | 联合新颖性/可信度、实测验证、匹配分层误差 | `experiment/reports/discovery_v3_reliability_comparison/`；`statistics` |
| S18 | Top100 阈值和联合规则摘要 | 同 S9、S15；FD `discovery_plot_inputs/` |
| S19 | 四种 SaProt 配置 | `experiment/saprot_baseline_20261019/PROTOCOL.json`、`saprot_official_recipe_20261019/PROTOCOL.json`、各模型 `config.json` |
| S20 | 单种子 GLOM 与 SaProt | `experiment/artifacts/models/`；`tables prediction` |
| S21 | 配对误差差及 bootstrap 区间 | `experiment/reports/saprot_official_recipe_20261019/paired_error_differences.csv`；`statistics` |
| S22 | GLOM 和 SaProt 实测 Top-k | 各模型 `test_predictions.csv`；`tables prediction` |
| S23 | 全部 Top 30 首轮生物物理变化 | `biophysics/results/`、`mechanism_analysis_20261027/tables/`；MD 分析与第一轮机制汇总 |
| S24 | 六个代表的绝对生物物理数值 | 同 S23；`paper/figure_reproduction/data/` |
| S25 | 30 个候选/起点/WT 的三次配对 | `paper/three_runs/verification/md_metrics_by_replicate_source.csv`；`tables md` |

最终表格的 LaTeX 逐表副本位于 `paper/table_extracts/`，与最终正文的标题和编号同步；它们是排版后的表格文本，数值复算使用上表的数据文件。

## 数据角色与重现层次

- **精确数值复算**：固定逐序列预测、排名及模拟轨迹；CPU 表格/统计入口会比较归档数值并失败退出，而不是仅写出新表。
- **完整算法重跑**：使用归档检查点和 trial 12 重跑正式 900 峰与原生基线。新的输出目录和冻结记录独立保存。
- **重新训练**：从通用 PLM 和给定冻结 Bridge 开始训练 GLOM、对照及基线，使用固定训练/验证/测试划分和预定种子。不是重新预训练 ESM2 或 SaProt。
- **生物物理重算**：可直接从 168 条轨迹重算指标，也可从已参数化系统重新进行三次模拟。第一轮分析与三次重复汇总各有独立来源。

所有已报告的序列、候选排序、测试预测、主要学习权重及模拟输入/输出均在本包中。大型公开基础权重和 SPURS 检查点按固定身份下载；下载清单见 `PUBLIC_ASSETS.json`。
