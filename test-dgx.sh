#!/usr/bin/env bash
# Menu of every GPU-dependent test/benchmark for this project, meant to be
# cloned onto the B200 box and driven remotely: from the dev machine, say
# "run options 2 4" and paste this script's stdout back.
#
# Usage:
#   ./test-dgx.sh            # interactive menu
#   ./test-dgx.sh 2 4        # run options 2 and 4, in that order, no prompts
#   ./test-dgx.sh all        # run every option in menu order (slow -- see 6)
#
# Each run's full output is also teed to dgx_logs/<n>_<name>_<timestamp>.log
# in case terminal scrollback truncates it -- paste that file if so.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
GPUS="${GPUS:-}"
LOG_DIR="dgx_logs"
mkdir -p "$LOG_DIR"

if [ -z "$GPUS" ]; then
  n_gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  if [ -z "$n_gpus" ] || [ "$n_gpus" -eq 0 ]; then
    GPUS="0"
  else
    GPUS=$(seq -s, 0 $((n_gpus - 1)))
  fi
fi

_run() {
  local name="$1"; shift
  local ts logfile status
  ts=$(date +%Y%m%d_%H%M%S)
  logfile="$LOG_DIR/${name}_${ts}.log"
  echo "=============================================================="
  echo "=== $name -- $(date)"
  echo "=== log: $logfile"
  echo "=============================================================="
  ( "$@" ) 2>&1 | tee "$logfile"
  status=${PIPESTATUS[0]}
  echo "--------------------------------------------------------------"
  if [ "$status" -eq 0 ]; then
    echo "=== $name: OK (exit 0)"
  else
    echo "=== $name: FAILED (exit $status)"
  fi
  echo "=============================================================="
  echo
  return "$status"
}

_require_free_gpu() {
  # Fail fast with a clear diagnostic instead of launching vllm serve and
  # hitting a buried CUDA-OOM error N minutes into a run. Not automatic --
  # option 0 is the only thing that kills anything, and only when you run
  # it yourself.
  local min_free_mib="${1:-40000}"
  local best
  best=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
  if [ -z "$best" ] || [ "$best" -lt "$min_free_mib" ]; then
    echo "No GPU with >= ${min_free_mib} MiB free (best available: ${best:-none} MiB)."
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
    echo "Run option 0 to kill stray 'vllm serve' processes if that's safe" \
      "right now, or free a GPU yourself, then re-run."
    return 1
  fi
}

opt_0_kill_stale() {
  # Explicit only -- never part of 'all'. Kills every 'vllm serve' process
  # group on this box. Confirm nothing else needs those GPUs before running.
  local pids
  pids=$(pgrep -f "vllm serve" || true)
  if [ -z "$pids" ]; then
    echo "No 'vllm serve' processes found."
    return 0
  fi
  echo "Found:"
  ps -o pid,ppid,pgid,cmd -p $pids
  echo "Sending SIGTERM to each process group..."
  for pid in $pids; do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && kill -TERM "-$pgid" 2>/dev/null || true
  done
  sleep 5
  pids=$(pgrep -f "vllm serve" || true)
  if [ -n "$pids" ]; then
    echo "Still alive, sending SIGKILL: $pids"
    for pid in $pids; do
      pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
      [ -n "$pgid" ] && kill -KILL "-$pgid" 2>/dev/null || true
    done
    sleep 3
  fi
  echo "GPU memory after cleanup:"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
}

opt_1_sanity() {
  nvidia-smi
  echo
  python -c "
import torch
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())
"
  python -c "import vllm; print('vllm', vllm.__version__)"
  if ! python -c "import semantic_offload" 2>/dev/null; then
    echo "semantic_offload not importable -- running 'pip install -e .'"
    pip install -e .
    python -c "import semantic_offload; print('semantic_offload OK after install')"
  else
    echo "semantic_offload importable OK"
  fi

  if ! python -m pytest --version >/dev/null 2>&1; then
    echo "pytest not installed -- installing"
    uv pip install pytest
  else
    echo "pytest importable OK"
  fi

  # transformers imports torchaudio transitively (audio-model support this
  # text-only project doesn't use). transformers guards a *missing*
  # torchaudio, but torchaudio raises RuntimeError (not ImportError) when
  # its CUDA build doesn't match torch's -- that's uncaught and crashes
  # every real-server launch. Uninstalling it is the actual fix, not
  # reinstalling a matching build (may not exist yet for a brand-new torch
  # CUDA tag, see the wheel-lag issue sync-vllm-upstream.sh works around).
  torchaudio_err=$(python -c "import torchaudio" 2>&1 1>/dev/null)
  if echo "$torchaudio_err" | grep -q "different CUDA versions"; then
    echo "torchaudio/torch CUDA mismatch detected -- uninstalling torchaudio" \
      "(not needed for text-only serving)"
    uv pip uninstall torchaudio
  fi
}

opt_2_unit_tests() {
  python -m pytest tests/ -v
}

opt_3_latency_smoke() {
  # Small-dev-model validation run, per benchmarks/run_latency_suite.py's
  # own documented smoke usage. Real server, ~a few minutes.
  _require_free_gpu || return 1
  python benchmarks/run_latency_suite.py \
      --model "$MODEL" \
      --policies lru,arc,semantic-minmax,semantic-mean \
      --workloads chat \
      --request-rates 2.0 \
      --num-prompts 20 \
      --scale 0.05 \
      --cpu-bytes-to-use 268435456 \
      --output-dir /tmp/latency_suite_smoke
}

opt_4_splice_probe() {
  # The current open item: byte-exact proof that a spliced GPU block's
  # content is correct. Watches the server's own debug log live and fires
  # the recall the instant a real PARTIAL SPLICE references the tagged
  # needle request. Real server, real 40-filler contention, ~a few minutes.
  _require_free_gpu || return 1
  SEMANTIC_OFFLOAD_DEBUG=1 python harness/run_adaptive_probe_live.py
}

opt_5_recall_cost_experiments() {
  # Re-run the Step 0.3/0.4/1.4 recall+cost studies on real CUDA (these
  # scripts fall back to CPU but the *_cost.csv timing numbers are only
  # meaningful with a real GPU and its CUDA streams/events).
  local scripts=(
    experiments/step_0_3_lru_arc_recall.py
    experiments/step_0_3_svd_recall.py
    experiments/step_0_3_minmax_recall.py
    experiments/step_0_3_oracle_and_mean_recall.py
    experiments/step_0_4_adversarial_needle_recall.py
    experiments/step_1_4_keynorm_recall.py
    experiments/step_1_4_attention_seeding_recall.py
    experiments/step_1_4_chain_aware_multiturn.py
    experiments/step_1_4_session_aware_recall.py
    experiments/step_1_4_session_aware_concurrent_stress.py
  )
  local script rc=0
  for script in "${scripts[@]}"; do
    echo "--- $script ---"
    python "$script" || rc=1
  done
  return "$rc"
}

opt_6_grid_sweep() {
  # The still-owed Step 1.6 rigorous multi-seed benchmark. LONG (hours --
  # target-duration-s 600 per cell x policies x workloads x rates x seeds).
  # Uses every GPU in $GPUS concurrently (auto-detected via nvidia-smi,
  # override with GPUS=0,1,... env var).
  _require_free_gpu || return 1
  python benchmarks/run_grid_sweep.py \
      --model "$MODEL" \
      --policies lru,arc,semantic-minmax,semantic-mean \
      --workloads chat,rag,mixed \
      --request-rates 2.0,8.0 \
      --seeds 1,2,3 \
      --target-duration-s 600 \
      --cpu-bytes-to-use 2147483648 \
      --needle-reference-counts 0,1,2 \
      --output-dir "results/step_1_6_first_pass_$(date +%Y%m%d_%H%M%S)" \
      --gpus "$GPUS"
}

declare -A NAMES=(
  [0]="0_kill_stale"
  [1]="1_sanity"
  [2]="2_unit_tests"
  [3]="3_latency_smoke"
  [4]="4_splice_probe"
  [5]="5_recall_cost_experiments"
  [6]="6_grid_sweep"
)
declare -A DESCRIPTIONS=(
  [0]="Kill stray 'vllm serve' processes and free GPU memory. EXPLICIT ONLY -- never runs as part of 'all'."
  [1]="Sanity check -- nvidia-smi, torch/CUDA, vllm, semantic_offload import (installs if missing). Seconds."
  [2]="Unit tests -- pytest tests/. CPU-only logic but good to confirm green on this box too. Seconds."
  [3]="Latency suite smoke -- real vllm server, lru/arc/semantic-minmax/semantic-mean, chat workload, 20 prompts. ~Minutes."
  [4]="Adaptive splice live-probe -- byte-exact splice-correctness proof (the current open item). Real server. ~Minutes."
  [5]="Recall/cost experiments -- re-run Step 0.3/0.4/1.4 studies with real CUDA timing. ~Minutes."
  [6]="Grid sweep -- Step 1.6 rigorous multi-seed benchmark across all GPUs in \$GPUS ($GPUS). LONG (hours)."
)

run_option() {
  case "$1" in
    0) _run "${NAMES[0]}" opt_0_kill_stale ;;
    1) _run "${NAMES[1]}" opt_1_sanity ;;
    2) _run "${NAMES[2]}" opt_2_unit_tests ;;
    3) _run "${NAMES[3]}" opt_3_latency_smoke ;;
    4) _run "${NAMES[4]}" opt_4_splice_probe ;;
    5) _run "${NAMES[5]}" opt_5_recall_cost_experiments ;;
    6) _run "${NAMES[6]}" opt_6_grid_sweep ;;
    *) echo "Unknown option: $1 (valid: 0-6, all)"; return 1 ;;
  esac
}

print_menu() {
  echo "GPUS detected/selected: $GPUS   MODEL: $MODEL"
  echo
  for i in 0 1 2 3 4 5 6; do
    echo "  $i) ${DESCRIPTIONS[$i]}"
  done
  echo "('all' runs 1-6 only -- option 0 always requires being typed explicitly)"
  echo
}

if [ "$#" -gt 0 ]; then
  overall=0
  if [ "$1" = "all" ]; then
    set -- 1 2 3 4 5 6
  fi
  for opt in "$@"; do
    run_option "$opt" || overall=1
  done
  exit "$overall"
fi

print_menu
read -rp "Enter option numbers to run (space separated), or 'all': " -a choices
if [ "${#choices[@]}" -eq 1 ] && [ "${choices[0]}" = "all" ]; then
  choices=(1 2 3 4 5 6)
fi
overall=0
for opt in "${choices[@]}"; do
  run_option "$opt" || overall=1
done
exit "$overall"
