# GLOM–TGITO

**Multiscale protein modelling and multiview landscape-guided sequence discovery**

GLOM–TGITO combines a multiscale fluorescence predictor with sequence optimization across multiple learned landscapes. Frozen ESM2 features pass through a SaProt-distilled Bridge, then join mutation features in global, regional and key-site branches. TGITO uses these representations to select peak regions, guide sequence updates and rank diverse candidates by local evidence.

This repository provides the code, fitted weights, fixed data partitions, random seeds and configurations used in the paper. Large assets are downloaded separately; a normal clone does not include public language-model weights or molecular-dynamics trajectories.

[中文复现指南](README_zh.md) · [Experiment instructions](docs/REPRODUCING.md) · [Data sources](docs/DATA.md) · [Current release](../../releases/tag/repro-v2026.10.29)

The 2026-10-29 release synchronizes the manuscript and LaTeX with the journal submission and adds an explicit MIT license for the original software. Numerical experiment assets remain identical to the 2026-10-28 release. `reproduce.py fetch` uses the versioned URLs and SHA256 checksums in `manifests/ASSETS.json`.

The code, fitted weights, fixed data partitions, manuscript sources and supplementary source data are archived at [Zenodo, doi:10.5281/zenodo.23725684](https://doi.org/10.5281/zenodo.23725684). Full experiment outputs (`results`) and prepared simulation inputs (`md_inputs`) remain available from the versioned GitHub Releases through `reproduce.py fetch`. The Zenodo README gives the commands for combining these files. Molecular-dynamics trajectories are not included.

## Start here

Use Python 3.11 or 3.12. From the repository root:

```bash
# Extract the compact dataset included in the repository.
python reproduce.py fetch data

# Check exact data identities, partitions, seeds and HPO routing.
python reproduce.py verify
```

The data check uses the Python standard library. To reproduce numerical tables:

```bash
python -m pip install -r environment/requirements_cpu.txt
python reproduce.py fetch results
python reproduce.py tables
python reproduce.py statistics
```

These commands recompute prediction metrics, selected-batch comparisons, representation-deviation thresholds and recorded biophysical summaries. Outputs go to `reproduced/`; frozen inputs remain unchanged.

## Model inference

Use the training environment described in [the experiment guide](docs/REPRODUCING.md). Download the fitted models and the pinned ESM2-150M backbone:

```bash
python reproduce.py fetch weights
python reproduce.py assets esm2_150m
python reproduce.py predict --device cuda --fasta your_sequences.fasta
```

Input sequences must contain exactly 221 standard amino acids in the ppluGFP2 reference coordinate system. Omitting `--fasta` predicts the supplied reference sequence. `--device cpu` is supported. The output is predicted log10 fluorescence; `--split random|depth` and `--seed` select one of the six GLOM fits. This inference entry point uses float32, matching the search encoder convention.

## What is included

| Location | Contents |
| --- | --- |
| `experiment/repro_project/pipeline/` | GLOM, frozen ESM2/SaProt Bridge, TGITO and multiterrain components |
| `experiment/` | Training, baselines, feature extraction, discovery and evaluation code |
| `configs/SPLITS.json` | Fixed partition sizes, data checksums and training seeds |
| `configs/TRAINING_RUNS.csv` | All 78 fitted configurations in the prediction comparison |
| `configs/SEARCH_HPO_TRIAL12.json` | Historical winner and its effective runtime parameter mapping |
| `experiment/artifacts/discovery_v3/` | Frozen primary protocol and exact starting-context tables |
| `assets/` | Compact dataset archive, including fixed splits and mutation features |
| `manifests/` | Asset SHA256 values, upstream versions and source-data provenance |
| `environment/` | Dependency specifications and recorded package versions |
| `biophysics/` | Structure preparation, simulation and analysis code; no trajectories |

## Download only what you need

```bash
python reproduce.py fetch --list
python reproduce.py fetch weights terrain
```

| Asset group | Use |
| --- | --- |
| `data` | Fixed dataset, ordered row IDs, reference sequence, mutation channels and biological-system inputs; included in Git as a compressed archive |
| `weights` | Frozen Bridge, six GLOM fits and supporting predictor weights |
| `terrain` | Exact train-only memory for the published discovery campaign |
| `baseline_weights` | Optional fitted ablation and baseline models, including SaProt controls |
| `results` | Original per-sequence predictions, final candidate rankings, statistical inputs and numerical biological analyses |
| `figures` | Plotting inputs/code and LaTeX; requires `results` |
| `md_inputs` | Prepared structures and topology/coordinate files for fresh simulations |

Every asset is checked before extraction. Public ESM2, ESMC and SaProt models are obtained from pinned upstream revisions using `python reproduce.py assets NAME`; they are not duplicated in this repository. The SaProt baselines use general pretrained weights and are trained on the same fixed GFP partitions.

## Fixed experimental design

| Protocol | Train | Validation | Calibration | Test |
| --- | ---: | ---: | ---: | ---: |
| Sequence-hash random split | 22,003 | 3,144 | 3,143 | 3,144 |
| Mutation-depth split | 13,723 | 1,715 | 1,716 | 14,280 |

The dataset has 31,480 records and 31,434 unique sequences. The depth test contains variants with at least three substitutions; the other depth partitions contain at most two. The training seeds are **20261013, 20261014 and 20261015**.

The primary discovery campaign uses **900 peak regions, 2,700 search tasks and 1,308 distinct starting sequences**, with the confirmed trial-12 parameter mapping. It combines peak selection, dynamic multiterrain TGITO and native confidence/diversity selection. `discovery_v3.py` checks the identities of its 102 frozen runtime inputs before a run. Earlier diagnostic configurations are not substituted for this protocol.

[The experiment guide](docs/REPRODUCING.md) gives commands for retraining, the complete discovery comparison, and biological analyses. It also distinguishes regenerating experiments from recomputing the paper tables from archived outputs.

## Data and citation

The fluorescence data originate from Gonzalez Somermeyer et al., *Heterogeneity of the GFP fitness landscape and data-driven protein design*, eLife 11:e75842 (2022), [doi:10.7554/eLife.75842](https://doi.org/10.7554/eLife.75842). The pinned source download and the exact input-file hashes are documented in [Data sources](docs/DATA.md).

For this project, use [CITATION.cff](CITATION.cff). The original software is available under the [MIT License](LICENSE); third-party source, data and model terms are described in [NOTICE.md](NOTICE.md).
