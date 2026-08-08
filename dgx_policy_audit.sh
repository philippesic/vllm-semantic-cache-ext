#!/usr/bin/env bash
# Unified, fail-closed DGX audit for semantic eviction versus LRU and ARC.
# Run from this repository after pulling both sibling repositories.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

VLLM_REPO=${VLLM_REPO:-/raid/ppesic/tmp/vllm-semantic-cache}
VENV_DIR=${VENV_DIR:-"$VLLM_REPO/.venv"}
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing vLLM virtual environment activation script: $VENV_DIR/bin/activate" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PYTHON_BIN=${PYTHON_BIN:-"$VENV_DIR/bin/python"}
VLLM_CLI=${VLLM_CLI:-"$(dirname "$PYTHON_BIN")/vllm"}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
# GPUs 0-3 are reserved by other workloads on the DGX. Never dispatch there
# unless the operator explicitly overrides GPUS after confirming availability.
GPUS=${GPUS:-4,5,6,7}
SEEDS=${SEEDS:-1,2,3}
CPU_BYTES=${CPU_BYTES:-91750400}
GPU_BLOCKS=${GPU_BLOCKS:-120}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
SCALE=${SCALE:-0.04}
SERVING_DURATION_S=${SERVING_DURATION_S:-180}
ABLATION_PROMPTS=${ABLATION_PROMPTS:-24}
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-7200}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=${OUTPUT_ROOT:-"$SCRIPT_DIR/dgx_logs/policy_audit_$TIMESTAMP"}
mkdir -p "$OUTPUT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable Python environment: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$VLLM_CLI" ]]; then
  echo "Missing vLLM CLI beside selected Python: $VLLM_CLI" >&2
  exit 2
fi
if [[ ! -d "$VLLM_REPO/vllm" ]]; then
  echo "VLLM_REPO does not point to a vLLM checkout: $VLLM_REPO" >&2
  exit 2
fi

export PYTHONPATH="$SCRIPT_DIR:$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_CLI
export PATH="$(dirname "$VLLM_CLI"):$PATH"
export SEMANTIC_OFFLOAD_TIMING=${SEMANTIC_OFFLOAD_TIMING:-1}
export SEMANTIC_OFFLOAD_TIMING_EVERY=${SEMANTIC_OFFLOAD_TIMING_EVERY:-500}
export NEEDLE_SHARED_CONTENT=1

{
  echo "semantic_repo=$(git rev-parse HEAD)"
  echo "vllm_repo=$(git -C "$VLLM_REPO" rev-parse HEAD)"
  echo "model=$MODEL"
  echo "python_bin=$PYTHON_BIN"
  echo "venv_dir=$VENV_DIR"
  echo "vllm_cli=$VLLM_CLI"
  echo "gpus=$GPUS"
  echo "seeds=$SEEDS"
  echo "cpu_bytes=$CPU_BYTES"
  echo "gpu_blocks=$GPU_BLOCKS"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "scale=$SCALE"
  echo "semantic_offload_timing=$SEMANTIC_OFFLOAD_TIMING"
  echo "semantic_offload_timing_every=$SEMANTIC_OFFLOAD_TIMING_EVERY"
  echo "needle_shared_content=$NEEDLE_SHARED_CONTENT"
  git status --short
  echo "tracked_diff_hash=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "worktree_content_hash=$(git ls-files -co --exclude-standard -z | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
  git -C "$VLLM_REPO" status --short
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} >"$OUTPUT_ROOT/manifest.txt"

echo "[1/4] Running complete CPU/unit suite"
"$PYTHON_BIN" -m pytest -q tests 2>&1 | tee "$OUTPUT_ROOT/unit_tests.log"

COMMON_ARGS=(
  --model "$MODEL"
  --seeds "$SEEDS"
  --gpus "$GPUS"
  --cpu-bytes-to-use "$CPU_BYTES"
  --num-gpu-blocks-override "$GPU_BLOCKS"
  --max-model-len "$MAX_MODEL_LEN"
  --scale "$SCALE"
  --cell-timeout-s "$CELL_TIMEOUT_S"
)

AUDIT_FAILED=0
run_grid() {
  local variant=$1
  shift
  echo "Running variant=$variant"
  if ! "$PYTHON_BIN" benchmarks/run_grid_sweep.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUTPUT_ROOT/$variant" \
    "$@" 2>&1 | tee "$OUTPUT_ROOT/${variant}.log"; then
    AUDIT_FAILED=1
  fi
}

echo "[2/4] Measuring eviction quality against LRU and ARC"
# One reference-count case per process gives every case a fresh server/cache.
for ref_count in 0 1 2; do
  run_grid "leaderboard_ref${ref_count}" \
    --policies lru,arc,semantic-minmax,semantic-mean,semantic-cuboid-mean \
    --workloads needle-v2 \
    --needle-reference-counts "$ref_count" \
    --num-prompts 12 \
    --extra-config '{"probe_layer":"middle","head_aggregation":"mean","prefetch_budget_fraction":0}'
done

echo "[3/4] Measuring serving cost and pressure behavior"
run_grid serving \
  --policies lru,arc,semantic-minmax,semantic-mean,semantic-cuboid-mean \
  --workloads chat,rag,mixed \
  --request-rates 2.0,8.0 \
  --target-duration-s "$SERVING_DURATION_S" \
  --extra-config '{"probe_layer":"middle","head_aggregation":"mean","prefetch_budget_fraction":0}'

echo "[4/4] Running controlled semantic-mean ablations"
declare -A ABLATIONS=(
  [signal_first_max]='{"probe_layer":"first","head_aggregation":"max","prefetch_budget_fraction":0}'
  [signal_middle_max]='{"probe_layer":"middle","head_aggregation":"max","prefetch_budget_fraction":0}'
  [signal_middle_mean]='{"probe_layer":"middle","head_aggregation":"mean","prefetch_budget_fraction":0}'
  [session_decay8]='{"probe_layer":"middle","head_aggregation":"mean","session_aware":true,"session_bonus_half_life":8,"prefetch_budget_fraction":0}'
  [prefetch_001]='{"probe_layer":"middle","head_aggregation":"mean","prefetch_budget_fraction":0.01}'
  [prefetch_005]='{"probe_layer":"middle","head_aggregation":"mean","prefetch_budget_fraction":0.05}'
  [capture_stride4]='{"probe_layer":"middle","head_aggregation":"mean","capture_stride":4,"prefetch_budget_fraction":0}'
)

for variant in \
  signal_first_max signal_middle_max signal_middle_mean session_decay8 \
  prefetch_001 prefetch_005 capture_stride4; do
  run_grid "$variant" \
    --policies semantic-mean \
    --workloads needle-v2,chat,rag \
    --request-rates 8.0 \
    --needle-reference-counts 1 \
    --num-prompts "$ABLATION_PROMPTS" \
    --extra-config "${ABLATIONS[$variant]}"
done

if ! "$PYTHON_BIN" benchmarks/summarize_policy_audit.py \
  "$OUTPUT_ROOT" --expected-seeds "$SEEDS"; then
  AUDIT_FAILED=1
  "$PYTHON_BIN" benchmarks/summarize_policy_audit.py \
    "$OUTPUT_ROOT" --expected-seeds "$SEEDS" --allow-errors || true
fi

tar -czf "$OUTPUT_ROOT.tar.gz" -C "$(dirname "$OUTPUT_ROOT")" \
  "$(basename "$OUTPUT_ROOT")"

echo "Audit output: $OUTPUT_ROOT"
echo "Summary: $OUTPUT_ROOT/audit_summary.md"
echo "Archive: $OUTPUT_ROOT.tar.gz"
exit "$AUDIT_FAILED"
