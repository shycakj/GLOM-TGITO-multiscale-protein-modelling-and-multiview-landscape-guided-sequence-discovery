#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
create_env() {
  local env_name="$1" py_version="$2"
  conda env list | awk '{print $1}' | grep -qx "$env_name" || conda create -n "$env_name" "python=$py_version" -y
}
case "${1:-all}" in
  glom|all)
    create_env "$TGITO_ENV" 3.10
    conda run -n "$TGITO_ENV" python -m pip install 'pip<26' setuptools wheel
    conda run -n "$TGITO_ENV" python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    conda run -n "$TGITO_ENV" python -m pip install -r "$ROOT/portable/requirements_gfp_repro_recorded.txt"
    ;;
  musearch) ;;
  *) echo 'Usage: bash portable/setup_environments.sh [all|glom|musearch]' >&2; exit 2 ;;
esac
if [[ "${1:-all}" == all || "${1:-all}" == musearch ]]; then
  create_env "$MUSEARCH_ENV" 3.8
  conda run -n "$MUSEARCH_ENV" python -m pip install 'pip<24' 'setuptools<70' wheel
  # Baseline driver runs on CPU; only the separate GLOM oracle uses the GPU.
  conda run -n "$MUSEARCH_ENV" python -m pip install torch==1.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
  conda run -n "$MUSEARCH_ENV" python -m pip install -r "$ROOT/portable/requirements_musearch_prescription.txt"
  conda run -n "$MUSEARCH_ENV" python -m pip install --no-deps -e "$ROOT/external/Mu-Protein/mu-search/src/flexs"
  conda run -n "$MUSEARCH_ENV" python -m pip install --no-deps -e "$ROOT/external/Mu-Protein/mu-search/src/flexs/flexs/baselines/explorers/stable_baselines3"
  conda run -n "$MUSEARCH_ENV" python "$ROOT/portable/smoke_baselines.py" --methods pure_random adalead evobo musearch --import-only
fi
# Scoring uses gfp_repro's pandas/numpy; separate core environment is retained as
# an original snapshot but need not be installed for this handoff's main search.
