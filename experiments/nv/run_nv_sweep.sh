#!/usr/bin/env bash
# =============================================================================
# run_nv_sweep.sh — SODA sweep launcher for the NV experiment machine
#
# Usage (from repo root after sourcing env.sh):
#   source env.sh
#   bash experiments/nv/run_nv_sweep.sh [OPTIONS]
#
# Options:
#   --mode      prefill | decode         (default: prefill)
#   --models    comma-separated keys     (default: all)
#                 e.g. gpt2_small,gpt2_medium,llama_3_2_1b
#   --batch-sizes  comma-separated ints  (default: per-model in config.py)
#                 e.g. 1,2,4
#   --seq-lens  comma-separated ints     (default: per-model in config.py)
#                 e.g. 128,256,512
#   --warmup    int                      (default: 10)
#   --runs      int                      (default: 50)
#   -h | --help
#
# Examples:
#   # Full prefill sweep with all models at default shapes
#   bash experiments/nv/run_nv_sweep.sh
#
#   # Decode sweep, GPT-2 family only
#   bash experiments/nv/run_nv_sweep.sh --mode decode \
#       --models gpt2_small,gpt2_medium,gpt2_large,gpt2_xl
#
#   # Quick debug: two models, small shapes
#   bash experiments/nv/run_nv_sweep.sh \
#       --models gpt2_small,llama_3_2_1b \
#       --batch-sizes 1,2 --seq-lens 128,256 \
#       --warmup 2 --runs 5
#
#   # Decode sweep for Llama models only
#   bash experiments/nv/run_nv_sweep.sh --mode decode \
#       --models llama_3_2_1b,llama_3_2_3b,llama_3_1_8b
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Guard: must be run from repo root
# ---------------------------------------------------------------------------
if [ ! -f "pyproject.toml" ]; then
    echo "Error: run this script from the SODA repo root." >&2
    echo "  cd /path/to/SODA---System-Offload-Dynamics-Analyzer-NV" >&2
    echo "  source env.sh" >&2
    echo "  bash experiments/nv/run_nv_sweep.sh [OPTIONS]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Guard: env.sh must be sourced
# ---------------------------------------------------------------------------
if [ -z "${SODA_ENV_LOADED:-}" ]; then
    echo "Error: SODA environment not loaded." >&2
    echo "Please run: source env.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Parse arguments
# ---------------------------------------------------------------------------
MODE="prefill"
MODELS=""
BATCH_SIZES=""
SEQ_LENS=""
WARMUP=""
RUNS=""

usage() {
    sed -n '/^# Usage/,/^# ===/p' "$0" | grep -v "^# ===" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2";        shift 2 ;;
        --models)      MODELS="$2";      shift 2 ;;
        --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
        --seq-lens)    SEQ_LENS="$2";    shift 2 ;;
        --warmup)      WARMUP="$2";      shift 2 ;;
        --runs)        RUNS="$2";        shift 2 ;;
        -h|--help)     usage ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "prefill" && "$MODE" != "decode" ]]; then
    echo "Error: --mode must be 'prefill' or 'decode' (got: '$MODE')" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Activate Python environment
# ---------------------------------------------------------------------------
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    echo "[env] Using conda env: $CONDA_DEFAULT_ENV"
elif [ -d "${PYTHON_VENV:-}" ]; then
    echo "[env] Activating venv: $PYTHON_VENV"
    # shellcheck disable=SC1090
    source "$PYTHON_VENV/bin/activate"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "[env] Using conda base"
else
    echo "Warning: no virtual environment detected. Using system Python." >&2
fi

# ---------------------------------------------------------------------------
# 5. Build python command
# ---------------------------------------------------------------------------
PYTHON_CMD=(python experiments/nv/soda_nv_sweep.py --mode "$MODE")

[ -n "$MODELS" ]      && PYTHON_CMD+=(--models      "$MODELS")
[ -n "$BATCH_SIZES" ] && PYTHON_CMD+=(--batch-sizes "$BATCH_SIZES")
[ -n "$SEQ_LENS" ]    && PYTHON_CMD+=(--seq-lens    "$SEQ_LENS")
[ -n "$WARMUP" ]      && PYTHON_CMD+=(--warmup      "$WARMUP")
[ -n "$RUNS" ]        && PYTHON_CMD+=(--runs        "$RUNS")

# ---------------------------------------------------------------------------
# 6. Print summary and run
# ---------------------------------------------------------------------------
echo "============================================================"
echo " SODA NV Sweep"
echo "  mode        : $MODE"
echo "  models      : ${MODELS:-<all>}"
echo "  batch-sizes : ${BATCH_SIZES:-<per-model defaults>}"
echo "  seq-lens    : ${SEQ_LENS:-<per-model defaults>}"
echo "  warmup      : ${WARMUP:-<config default>}"
echo "  runs        : ${RUNS:-<config default>}"
echo "  output dir  : ${SODA_OUTPUT:-output}"
echo "============================================================"
echo ""

echo "[run] ${PYTHON_CMD[*]}"
echo ""

"${PYTHON_CMD[@]}"
