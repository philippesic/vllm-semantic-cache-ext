#!/usr/bin/env bash
# Unified, fail-closed DGX audit for semantic eviction versus LRU and ARC.
# Run from this repository after pulling both sibling repositories.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

repo_revision() {
  git -C "$1" rev-parse HEAD
}

repo_state_fingerprint() {
  (
    cd "$1"
    {
      git rev-parse HEAD
      git diff HEAD --binary
      git ls-files --others --exclude-standard -z | sort -z | xargs -0 sha256sum --
    } | sha256sum | cut -d' ' -f1
  )
}

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

# Claim the run directory atomically after preflight. Plain mkdir, rather than
# an existence check followed by mkdir -p, prevents concurrent launches from
# ever sharing one evidence root.
mkdir -p "$(dirname "$OUTPUT_ROOT")"
if ! mkdir "$OUTPUT_ROOT"; then
  echo "Refusing to reuse existing audit output root: $OUTPUT_ROOT" >&2
  echo "Choose a new OUTPUT_ROOT so manifests and results cannot mix across runs." >&2
  exit 2
fi

SEMANTIC_REVISION_START=$(repo_revision "$SCRIPT_DIR")
VLLM_REVISION_START=$(repo_revision "$VLLM_REPO")
SEMANTIC_STATE_START=$(repo_state_fingerprint "$SCRIPT_DIR")
VLLM_STATE_START=$(repo_state_fingerprint "$VLLM_REPO")
{
  echo "semantic_revision_start=$SEMANTIC_REVISION_START"
  echo "semantic_state_start=$SEMANTIC_STATE_START"
  echo "vllm_revision_start=$VLLM_REVISION_START"
  echo "vllm_state_start=$VLLM_STATE_START"
} >"$OUTPUT_ROOT/repository_state.txt"

export PYTHONPATH="$SCRIPT_DIR:$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_CLI
VLLM_CLI_DIR=$(dirname "$VLLM_CLI")
export PATH="$VLLM_CLI_DIR:$PATH"
export SEMANTIC_OFFLOAD_TIMING=${SEMANTIC_OFFLOAD_TIMING:-1}
export SEMANTIC_OFFLOAD_TIMING_EVERY=${SEMANTIC_OFFLOAD_TIMING_EVERY:-500}
export NEEDLE_SHARED_CONTENT=1

{
  echo "semantic_repo=$SEMANTIC_REVISION_START"
  echo "semantic_state=$SEMANTIC_STATE_START"
  echo "vllm_repo=$VLLM_REVISION_START"
  echo "vllm_state=$VLLM_STATE_START"
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
  local variant_dir="$OUTPUT_ROOT/$variant"
  local -a variant_args=("${COMMON_ARGS[@]}" --output-dir "$variant_dir" "$@")
  local semantic_revision_start vllm_revision_start
  local semantic_state_start vllm_state_start
  semantic_revision_start=$(repo_revision "$SCRIPT_DIR")
  vllm_revision_start=$(repo_revision "$VLLM_REPO")
  semantic_state_start=$(repo_state_fingerprint "$SCRIPT_DIR")
  vllm_state_start=$(repo_state_fingerprint "$VLLM_REPO")
  mkdir -p "$variant_dir"
  {
    printf 'variant=%s\n' "$variant"
    printf 'python_bin=%s\n' "$PYTHON_BIN"
    printf 'semantic_revision_start=%s\n' "$semantic_revision_start"
    printf 'semantic_state_start=%s\n' "$semantic_state_start"
    printf 'vllm_revision_start=%s\n' "$vllm_revision_start"
    printf 'vllm_state_start=%s\n' "$vllm_state_start"
    local option value
    for option in --policies --workloads --request-rates \
      --needle-reference-counts --num-prompts --target-duration-s \
      --seeds --gpus --extra-config; do
      value=""
      local i
      for ((i = 0; i < ${#variant_args[@]} - 1; i++)); do
        if [[ "${variant_args[$i]}" == "$option" ]]; then
          value=${variant_args[$((i + 1))]}
        fi
      done
      printf '%s=%s\n' "${option#--}" "$value"
    done
    printf 'command='
    printf '%q ' "$PYTHON_BIN" benchmarks/run_grid_sweep.py "${variant_args[@]}"
    printf '\n'
  } >"$variant_dir/variant_manifest.txt"
  echo "Running variant=$variant"
  if ! "$PYTHON_BIN" benchmarks/run_grid_sweep.py \
    "${variant_args[@]}" 2>&1 | tee "$OUTPUT_ROOT/${variant}.log"; then
    AUDIT_FAILED=1
  fi
  local semantic_revision_end vllm_revision_end
  local semantic_state_end vllm_state_end
  semantic_revision_end=$(repo_revision "$SCRIPT_DIR")
  vllm_revision_end=$(repo_revision "$VLLM_REPO")
  semantic_state_end=$(repo_state_fingerprint "$SCRIPT_DIR")
  vllm_state_end=$(repo_state_fingerprint "$VLLM_REPO")
  {
    printf 'semantic_revision_end=%s\n' "$semantic_revision_end"
    printf 'semantic_state_end=%s\n' "$semantic_state_end"
    printf 'vllm_revision_end=%s\n' "$vllm_revision_end"
    printf 'vllm_state_end=%s\n' "$vllm_state_end"
  } >>"$variant_dir/variant_manifest.txt"
  if [[ "$semantic_revision_start" != "$semantic_revision_end" ||
    "$semantic_state_start" != "$semantic_state_end" ||
    "$vllm_revision_start" != "$vllm_revision_end" ||
    "$vllm_state_start" != "$vllm_state_end" ]]; then
    echo "Repository state changed while variant=$variant was running" >&2
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
  [signal_middle_mean]='{"probe_layer":"middle","head_aggregation":"mean","alpha":0.5,"prefetch_budget_fraction":0}'
  [signal_middle_mean_alpha06]='{"probe_layer":"middle","head_aggregation":"mean","alpha":0.6,"prefetch_budget_fraction":0}'
  [session_decay8]='{"probe_layer":"middle","head_aggregation":"mean","session_aware":true,"session_bonus_half_life":8,"prefetch_budget_fraction":0}'
  [prefetch_001]='{"probe_layer":"middle","head_aggregation":"mean","prefetch_budget_fraction":0.01}'
  [prefetch_005]='{"probe_layer":"middle","head_aggregation":"mean","prefetch_budget_fraction":0.05}'
  [capture_stride4]='{"probe_layer":"middle","head_aggregation":"mean","capture_stride":4,"prefetch_budget_fraction":0}'
)

for variant in \
  signal_first_max signal_middle_max signal_middle_mean \
  signal_middle_mean_alpha06 session_decay8 \
  prefetch_001 prefetch_005 capture_stride4; do
  run_grid "$variant" \
    --policies semantic-mean \
    --workloads needle-v2,chat,rag \
    --request-rates 8.0 \
    --needle-reference-counts 1 \
    --num-prompts "$ABLATION_PROMPTS" \
    --extra-config "${ABLATIONS[$variant]}"
done

SEMANTIC_REVISION_END=$(repo_revision "$SCRIPT_DIR")
VLLM_REVISION_END=$(repo_revision "$VLLM_REPO")
SEMANTIC_STATE_END=$(repo_state_fingerprint "$SCRIPT_DIR")
VLLM_STATE_END=$(repo_state_fingerprint "$VLLM_REPO")
{
  echo "semantic_revision_end=$SEMANTIC_REVISION_END"
  echo "semantic_state_end=$SEMANTIC_STATE_END"
  echo "vllm_revision_end=$VLLM_REVISION_END"
  echo "vllm_state_end=$VLLM_STATE_END"
} >>"$OUTPUT_ROOT/repository_state.txt"
if [[ "$SEMANTIC_REVISION_START" != "$SEMANTIC_REVISION_END" ||
  "$SEMANTIC_STATE_START" != "$SEMANTIC_STATE_END" ||
  "$VLLM_REVISION_START" != "$VLLM_REVISION_END" ||
  "$VLLM_STATE_START" != "$VLLM_STATE_END" ]]; then
  echo "Repository state changed during the audit; results are not comparable" >&2
  AUDIT_FAILED=1
fi

if ! "$PYTHON_BIN" benchmarks/summarize_policy_audit.py \
  "$OUTPUT_ROOT" --expected-seeds "$SEEDS" --pre-summary; then
  AUDIT_FAILED=1
  "$PYTHON_BIN" benchmarks/summarize_policy_audit.py \
    "$OUTPUT_ROOT" --expected-seeds "$SEEDS" --pre-summary --allow-errors || true
fi

SEMANTIC_REVISION_SUMMARY_END=$(repo_revision "$SCRIPT_DIR")
VLLM_REVISION_SUMMARY_END=$(repo_revision "$VLLM_REPO")
SEMANTIC_STATE_SUMMARY_END=$(repo_state_fingerprint "$SCRIPT_DIR")
VLLM_STATE_SUMMARY_END=$(repo_state_fingerprint "$VLLM_REPO")
{
  echo "semantic_revision_summary_end=$SEMANTIC_REVISION_SUMMARY_END"
  echo "semantic_state_summary_end=$SEMANTIC_STATE_SUMMARY_END"
  echo "vllm_revision_summary_end=$VLLM_REVISION_SUMMARY_END"
  echo "vllm_state_summary_end=$VLLM_STATE_SUMMARY_END"
} >>"$OUTPUT_ROOT/repository_state.txt"
if [[ "$SEMANTIC_REVISION_START" != "$SEMANTIC_REVISION_SUMMARY_END" ||
  "$SEMANTIC_STATE_START" != "$SEMANTIC_STATE_SUMMARY_END" ||
  "$VLLM_REVISION_START" != "$VLLM_REVISION_SUMMARY_END" ||
  "$VLLM_STATE_START" != "$VLLM_STATE_SUMMARY_END" ]]; then
  echo "Repository state changed during summary generation" >&2
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
