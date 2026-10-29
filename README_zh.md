# GLOM–TGITO 精简复现包

**论文与复现材料归档：[Zenodo · DOI: 10.5281/zenodo.23725684](https://doi.org/10.5281/zenodo.23725684)**

本仓库保留模型代码、关键权重、固定数据划分、训练种子、正式搜索配置和复现入口。公共大模型与可选结果附件按需下载；分子动力学轨迹不包含在 Git 或 Release 中。

## 最快检查

在仓库根目录使用 Python 3.11/3.12：

```bash
python reproduce.py fetch data
python reproduce.py verify
```

这两步不需要 GPU。固定数据压缩包已经在仓库中，解压后会逐个核对 SHA256、划分人数、序列重复、训练种子和 HPO 参数路由。

## 复算论文数值

```bash
python -m pip install -r environment/requirements_cpu.txt
python reproduce.py fetch results
python reproduce.py tables
python reproduce.py statistics
python reproduce.py mechanisms
```

这里从保存的逐序列预测、候选排名和生物分析数值重新计算表格与检验。它不启动重新训练，也不重新运行 MD。输出位于 `reproduced/`。

## 使用模型预测

先按 [运行说明](docs/REPRODUCING.md) 配置训练环境，然后：

```bash
python reproduce.py fetch weights
python reproduce.py assets esm2_150m
python reproduce.py predict --device cuda --fasta your_sequences.fasta
```

可以改为 `--device cpu`。不指定 FASTA 时预测参考序列。序列必须是 221 个标准氨基酸，输出为预测 log10 荧光。模型包含冻结 ESM2 → 冻结 SaProt 蒸馏 Bridge → 每残基 1,280 维特征，以及 mutation 特征；不需要重新训练 Bridge。

## 重新训练与搜索

```bash
# 六个 GLOM 拟合：两种划分，各三个种子。
python reproduce.py prepare training --glom-only --destination runs/glom
python reproduce.py run runs/glom/RUN_PLAN.json --execute --gpus 0

# 正式多峰搜索，先运行两条轨迹的检查，再运行完整任务。
python reproduce.py fetch terrain
python reproduce.py assets esm2_650m
python reproduce.py verify --experiment
python reproduce.py prepare discovery --tgito-only --destination runs/discovery
python reproduce.py run runs/discovery/RUN_PLAN.json --execute --gpus 0,1
```

完整基线训练、完整搜索比较、图表重绘和生物分析的命令在 [docs/REPRODUCING.md](docs/REPRODUCING.md)。每张 GPU 最多启动一个任务，并预留显存。一个阶段成功后才进入后续依赖阶段；同一计划可以续跑。

固定训练种子为 `20261013、20261014、20261015`。主实验是 900 个峰区、2,700 个搜索任务、1,308 条独立起始序列。正式搜索使用保存的 trial-12 HPO 配置，保持多地势选峰、动态 TGITO 和综合置信度/多样性筛选完整衔接。

## 按需下载

```bash
python reproduce.py fetch --list
```

`weights` 为关键权重；`terrain` 为正式搜索的冻结地势库；`baseline_weights` 为其余已训练基线与消融权重；`results` 为数值复算材料；`figures` 为绘图与 LaTeX；`md_inputs` 为准备好的结构和模拟输入。分子动力学轨迹无需下载，也不会上传至本仓库。

数据来源、原始下载地址、标签定义和固定划分见 [docs/DATA.md](docs/DATA.md)。请使用归档的行号和划分，不要重新随机划分后与论文数值直接比较。

2026-10-29 版本将论文与 LaTeX 同步至投稿定稿，并为原创软件补充 [MIT 许可证](LICENSE)。实验数值、数据划分和模型权重沿用已核验的 2026-10-28 版本；下载脚本按清单获取各版本资产并校验 SHA256。第三方代码、数据、模型及论文文件的许可范围见 [NOTICE.md](NOTICE.md)。

Zenodo 归档包含代码、权重、固定划分、论文与补充数据。完整实验输出（`results`）和模拟输入（`md_inputs`）通过已固定版本的 GitHub Release 下载，使用 `reproduce.py fetch results md_inputs` 获取；完整分子动力学轨迹不包含在归档中。
