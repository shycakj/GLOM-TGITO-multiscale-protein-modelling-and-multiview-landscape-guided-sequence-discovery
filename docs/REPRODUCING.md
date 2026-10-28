# Reproducing the experiments

Run commands from the repository root. Python 3.11/3.12 is recommended. The code uses the fixed 221-residue ppluGFP2 reference throughout.

## Environments

For numerical reconstruction on CPU:

```bash
python -m pip install -r environment/requirements_cpu.txt
```

The original training runtime was NVIDIA PyTorch 25.03, CUDA 12.8 and `torch 2.7.0a0+7c8ec84dab.nv25.03`. Retain the CUDA-enabled torch build in that runtime and install `environment/requirements_training.txt`. A container recipe is provided in `environment/Dockerfile.training`. The recorded package list is `environment/venv_pip_freeze.txt`.

For inference in a separate environment, install a torch build suitable for your CPU/GPU and then `environment/requirements_inference.txt`. Do not replace the recorded training environment when reproducing the published GPU fits.

ESMC models use the separate `requirements_recent.txt` environment, recorded in `venv_recent_pip_freeze.txt`. Pass its Python executable to `run --recent-python PATH`; it is not required for GLOM-only training or TGITO-only search.

## Fixed inputs

```bash
python reproduce.py fetch data weights
python reproduce.py assets esm2_150m
python reproduce.py verify --all-installed
```

`data` is a local compressed archive. The other groups come from the repository's Release. Each archive and extracted file is SHA256-verified. If files have already been downloaded manually, use `fetch GROUP --from-dir DIRECTORY`.

The six GLOM fits use random/depth partitions and seeds 20261013, 20261014 and 20261015. The exact run metadata are under `experiment/artifacts/models/{split}/{method}_seed{seed}/config.json`. Search-task seeds are recorded in the starting-context CSV files rather than inferred from task order.

## Retrain GLOM

```bash
python reproduce.py prepare training --glom-only --destination runs/glom
python reproduce.py run runs/glom/RUN_PLAN.json
python reproduce.py run runs/glom/RUN_PLAN.json --execute --gpus 0
```

The plan regenerates mutation features, caches the frozen Bridge representations, fits GLOM for all six split/seed combinations, evaluates them and computes the representation probes. GLOM fitting uses learning rate 3e-5, batch size 64, up to 120 epochs and patience 18. The Bridge and ESM2 weights remain fixed.

The residue-feature cache alone occupies approximately 17.8 GB. Fresh runs need more disk space than a clone of this repository. All new caches and outputs are written inside the chosen run workspace.

## Retrain the complete prediction benchmark

```bash
python reproduce.py assets esm2_650m esmc_600m esmc_6b saprot_650m_af2 esm2_native650
python reproduce.py prepare training --destination runs/all_prediction_models
python reproduce.py run runs/all_prediction_models/RUN_PLAN.json --execute --gpus 0,1 --recent-python /path/to/esmc-env/bin/python
```

The plan enumerates the original 78 fitted configurations, including GLOM, ablations, sequence controls, ESM2/ESMC models and SaProt controls. `configs/TRAINING_RUNS.csv` identifies the split and seed for each configuration. Matched LoRA training uses ESM2-150M batch size 32 with accumulation 2; SaProt uses batch size 16 with accumulation 4. The SaProt protocols and their official-recipe variants remain in their respective experiment subdirectories.

Optional: `python reproduce.py fetch baseline_weights` installs the additional archived fits when inference from existing models is preferred to retraining. This download is not required to start new baseline training.

## Reproduce the complete discovery pipeline

```bash
python reproduce.py fetch data weights terrain
python reproduce.py assets esm2_150m esm2_650m
python reproduce.py verify --experiment
python reproduce.py prepare discovery --tgito-only --destination runs/tgito
python reproduce.py run runs/tgito/RUN_PLAN.json --execute --gpus 0,1
```

This plan checks all 102 frozen inputs, runs two trial-12 preflight trajectories, then runs 2,700 formal search tasks and the native final selection. A single GPU can use `--gpus 0`; the two shards then run sequentially. The 900 peak regions contain 1,308 distinct starting sequences across summit/ridge/edge contexts. Their frozen list is supplied, and `build_wide_peak_plan.py` provides the construction.

For all four discovery comparators and post-search evaluation, omit `--tgito-only`:

```bash
python reproduce.py prepare discovery --destination runs/full_discovery
python reproduce.py run runs/full_discovery/RUN_PLAN.json --execute --gpus 0,1
```

The comparison uses GLOM–TGITO, GLOM–AdaLead, GLOM–random search, GLOM–ProtHMSO and ESM2-650M–ProtHMSO. Their own ranking rules are retained. TGITO forward-sequence counts determine the matched comparator budgets; forward, backward and proposal costs remain separately recorded. Top 6/30/100 selection uses the frozen brightness/diversity constraints and is nested.

The published search uses the archived random-split seed-20261013 predictor, its train-only terrain memory and the confirmed historical trial-12 parameters. It does not automatically replace them with a newly trained checkpoint. The runtime contract preserves every historical-to-effective parameter mapping and rejects changed inputs. To study a newly fitted predictor, create and document a separate protocol.

Representation deviation is computed from learned features. Missing experimental fluorescence remains missing. Final discovery rankings are frozen before their numerical comparisons are generated.

## Recompute the reported tables and statistics

```bash
python reproduce.py fetch data results
python reproduce.py tables
python reproduce.py statistics
python reproduce.py mechanisms
```

`tables` checks 78 prediction runs, deviation thresholds, all 15 native selected batches, and the 168 recorded per-simulation metric records. It does not require trajectories. `statistics` repeats the bootstrap, threshold and joint reliability analyses. `mechanisms` reconstructs the first-run biological summaries from stored analysis outputs.

These are numerical reconstructions from archived outputs. New training/search/simulation jobs above regenerate the experimental outputs. Floating-point kernels and GPU types can affect new-run values; the fixed archived predictions remain the reference for exact table comparisons.

## Figures and manuscript sources

```bash
python reproduce.py fetch results figures
python reproduce.py figures
python reproduce.py paper --tectonic /path/to/tectonic
```

The plotting workflow regenerates five main and fourteen supplementary figures in `reproduced/figures/`. The figure/table data mapping is `manifests/PAPER_COVERAGE.md`. Rendering additionally requires the dependencies in the CPU requirements; LaTeX compilation requires a TeX engine. Source document copies are downloaded with `figures` rather than stored in Git history.

## Biological analyses and fresh simulations

There are 56 systems: WT, 30 selected candidates and 25 distinct starts. The three velocity seeds are 20261027, 20262027 and 20263027; each run uses 0.5 ns equilibration and 3 ns production. The source implements mature CR2, ff14SB, OPC water, 300 K, 1 bar and 0.15 M added NaCl.

```bash
python reproduce.py fetch data md_inputs
conda create --prefix /path/to/md-env --file biophysics/tools/conda_explicit.txt
python reproduce.py prepare md --destination runs/md
python reproduce.py run runs/md/RUN_PLAN.json --execute --gpus 0,1 --md-python /path/to/md-env/bin/python
```

The plan begins with WT/C01 short preflights before the three production repetitions. Prepared topology/coordinate files are included in `md_inputs` so the initial systems match. `build_structures.py` and `parameterize.py` document the preparation; the latter expects an AmberTools environment at `biophysics/env`. No DCD/XTC/TRR trajectories are distributed. Trajectories created by a fresh run stay in its run workspace.

Static sequence/structure predictors and their recorded outputs are included as supporting analyses. The SPURS public checkpoint can be obtained using `python reproduce.py assets spurs`.

## Execution and resumption

`prepare` creates a new workspace and never overwrites a prior run. `run` only lists the jobs unless `--execute` is supplied. At most one process runs on each selected GPU, with a 6,000 MiB memory reserve and stage-specific checks. Successful jobs are recorded in the run workspace; rerunning the same plan skips those jobs. Each dependency group finishes before the next group starts.

Use `--plan-only` with `prepare` to inspect a plan without creating files. Logs, run plans and generated caches are excluded from Git by `.gitignore`.
