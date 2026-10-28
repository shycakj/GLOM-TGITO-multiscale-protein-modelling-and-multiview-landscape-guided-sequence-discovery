#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
ACTION="${1:-check}"
shift || true
case "$ACTION" in
  check)
    conda run --no-capture-output -n "$TGITO_ENV" python "$ROOT/portable/check_handoff.py" --root "$ROOT" "$@" ;;
  check-baseline)
    CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n "$MUSEARCH_ENV" python "$ROOT/portable/check_handoff.py" --root "$ROOT" --mode baseline "$@" ;;
  oracle)
    conda run --no-capture-output -n "$TGITO_ENV" python "$ROOT/search/oracle_server.py" --root "$ROOT" --pipeline-root "$ROOT/pipeline" --bridge-ckpt "$GLOM_BRIDGE_CKPT" --revised-ckpt "$ROOT/artifacts/glom_current/best_model_pplu_current_split.pt" --port "$GLOM_ORACLE_PORT" "$@" ;;
  baseline-smoke)
    METHOD="${1:-pure_random}"
    CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n "$MUSEARCH_ENV" python "$ROOT/search/run_flexs_multipeak.py" --root "$ROOT" --method "$METHOD" --replicate-seed 20260902 --oracle-url "$GLOM_ORACLE_URL" --max-starts 12 ;;
  baselines)
    for METHOD in pure_random adalead evobo musearch; do
      for SEED in 20260902 20260903 20260904; do
        CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n "$MUSEARCH_ENV" python "$ROOT/search/run_flexs_multipeak.py" --root "$ROOT" --method "$METHOD" --replicate-seed "$SEED" --oracle-url "$GLOM_ORACLE_URL" "$@"
      done
    done ;;
  tgito)
    for SEED in 20260902 20260903 20260904; do
      conda run --no-capture-output -n "$TGITO_ENV" python "$ROOT/search/run_tgito_multipeak_frozen.py" --root "$ROOT" --pipeline-root "$ROOT/pipeline" --bridge-ckpt "$GLOM_BRIDGE_CKPT" --oracle-url "$GLOM_ORACLE_URL" --replicate-seed "$SEED" --resident "$@"
    done ;;
  score)
    conda run --no-capture-output -n "$TGITO_ENV" python "$ROOT/search/score_hidden400.py" --root "$ROOT" --include-saved "$@" ;;
  regenerate-h)
    conda run --no-capture-output -n "$TGITO_ENV" python "$ROOT/portable/regenerate_h.py" --root "$ROOT" "$@" ;;
  *) echo 'Actions: check, check-baseline, oracle, baseline-smoke [method], baselines, tgito, score, regenerate-h' >&2; exit 2 ;;
esac
