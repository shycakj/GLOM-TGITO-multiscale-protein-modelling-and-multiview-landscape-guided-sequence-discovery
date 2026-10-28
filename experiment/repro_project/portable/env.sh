#!/usr/bin/env bash
# Source in every terminal. No machine-specific absolute paths.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PPLU_PROJECT_ROOT="$ROOT"
export TGITO_ENV="${TGITO_ENV:-gfp_repro}"
export BASELINE_ENV="${BASELINE_ENV:-pplu_baseline_core}"
export MUSEARCH_ENV="${MUSEARCH_ENV:-pplu_musearch}"
export GLOM_BRIDGE_CKPT="$ROOT/assets/warm_best_model.pt"
export PPLU_ESM_MODEL_PATH="$ROOT/assets/esm2_t30_150M_UR50D"
export PPLU_H_NPY="$ROOT/assets/pplu_31k_H_embeddings.npy"
export PPLU_P4_NPY="$ROOT/assets/pplu_pmutation_p4.npy"
export PPLU_LEGACY_GLOM_CKPT="$ROOT/provenance/legacy_hpo_assets/best_model_pplu_31k.pt"
export PPLU_LEGACY_TERRAIN_12SOURCE="$ROOT/provenance/legacy_hpo_assets/terrain_12source.npz"
export PPLU_DATA_DIR="$ROOT/data/raw"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GLOM_ORACLE_URL="${GLOM_ORACLE_URL:-http://127.0.0.1:8765}"
export GLOM_ORACLE_PORT="${GLOM_ORACLE_PORT:-8765}"

cd "$ROOT"
