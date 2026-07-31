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
  # Used by opt_6 (grid sweep), which manages its own per-slot
  # CUDA_VISIBLE_DEVICES across $GPUS internally -- just checks that at
  # least one GPU in the box has room before dispatching cells.
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

_pick_free_gpu() {
  # opt_3/opt_4 launch a single vllm serve with no CUDA_VISIBLE_DEVICES of
  # their own, which means it always lands on device 0 -- so checking "is
  # SOME GPU free" isn't enough, device 0 specifically can be OOM while
  # every other GPU is idle (exactly what happened: 13/178 GiB free on
  # cuda:0 after everything else was killed). Pick the actual GPU with the
  # most free memory and print its index on stdout so callers can pin
  # CUDA_VISIBLE_DEVICES to it, instead of gambling on device 0.
  local min_free_mib="${1:-40000}"
  local idx free
  read -r idx free < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
      awk -F', *' '{print $1, $2}' | sort -k2 -rn | head -1
  )
  if [ -z "$free" ] || [ "$free" -lt "$min_free_mib" ]; then
    echo "No GPU with >= ${min_free_mib} MiB free (best: GPU ${idx:-?} with ${free:-0} MiB)." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >&2
    echo "Run option 0 to kill stray 'vllm serve' processes if that's safe" \
      "right now, or free a GPU yourself, then re-run." >&2
    return 1
  fi
  echo "$idx"
}

_vllm_pids() {
  # Two independent detection paths, unioned, since a stray process might
  # match only one: (1) cmdline pattern for vllm's own process names/
  # titles (VLLM::Worker_* etc. are setproctitle'd, so they show up in
  # `ps`'s COMMAND column even though argv never says "vllm serve"
  # literally); (2) whatever nvidia-smi itself reports as attached to a
  # GPU, filtered down to ones whose cmdline mentions vllm -- catches
  # anything launched in a way (2) doesn't have a name for, without
  # sweeping up an unrelated CUDA job that happens to share the box.
  local by_pattern by_gpu p cmd
  by_pattern=$(pgrep -f "vllm serve|VLLM::|EngineCore|vllm\.entrypoints" 2>/dev/null || true)
  by_gpu=""
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    cmd=$(ps -o cmd= -p "$p" 2>/dev/null || true)
    if echo "$cmd" | grep -qi vllm; then
      by_gpu="$by_gpu $p"
    fi
  done
  printf '%s\n%s\n' "$by_pattern" "$by_gpu" | grep -E '^[0-9]+$' | sort -u
}

_kill_vllm_processes() {
  local pids
  pids=$(_vllm_pids)
  if [ -z "$pids" ]; then
    echo "No vllm-related processes found."
    return 0
  fi
  echo "Found:"
  ps -o pid,ppid,pgid,cmd -p "$(echo "$pids" | tr '\n' ',' | sed 's/,$//')" 2>/dev/null || true
  echo "Sending SIGTERM to each process group..."
  local pid pgid
  for pid in $pids; do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ]; then kill -TERM "-$pgid" 2>/dev/null || true
    else kill -TERM "$pid" 2>/dev/null || true; fi
  done
  sleep 5
  pids=$(_vllm_pids)
  if [ -n "$pids" ]; then
    echo "Still alive, sending SIGKILL:"
    echo "$pids"
    for pid in $pids; do
      pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
      if [ -n "$pgid" ]; then kill -KILL "-$pgid" 2>/dev/null || true
      else kill -KILL "$pid" 2>/dev/null || true; fi
    done
    sleep 3
  fi
  echo "GPU memory after cleanup:"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
}

opt_0_kill_stale() {
  # Explicit only -- never part of 'all'. Confirm nothing else needs those
  # GPUs before running (opt_3/4/6 also call this before and after
  # themselves automatically -- this is for a manual box-wide cleanup).
  _kill_vllm_processes
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

  # Same category of bug as torchaudio above: flashinfer (the Python
  # package) and flashinfer-cubin (its precompiled kernel binaries) can
  # drift out of sync after an upstream jump pulls in a newer flashinfer
  # without also refreshing the cubin package. Unlike torchaudio this one
  # IS needed (it's vllm's attention backend on CUDA). Fix direction
  # matters: flashinfer-cubin's binary builds trail flashinfer-python's
  # releases (confirmed 2026-07-29 -- PyPI had flashinfer-cubin up to
  # 0.6.13 while flashinfer-python was already at 0.6.15.post1, so
  # upgrading cubin to match python 404'd with "no solution found").
  # Downgrade flashinfer-python to match whatever cubin version is
  # already installed instead -- that direction is always resolvable.
  flashinfer_err=$(python -c "import flashinfer" 2>&1 1>/dev/null)
  if echo "$flashinfer_err" | grep -q "does not match flashinfer version"; then
    cubin_ver=$(echo "$flashinfer_err" |
      sed -n 's/.*flashinfer-cubin version (\([^)]*\)) does not match.*/\1/p')
    if [ -n "$cubin_ver" ]; then
      echo "flashinfer/flashinfer-cubin version mismatch -- downgrading" \
        "flashinfer-python to $cubin_ver to match the installed cubin"
      uv pip install "flashinfer-python==$cubin_ver"
    else
      echo "flashinfer/flashinfer-cubin version mismatch detected but" \
        "couldn't parse the cubin version -- fix manually:"
      echo "$flashinfer_err"
    fi
  fi
}

opt_2_unit_tests() {
  python -m pytest tests/ -v
}

opt_3_latency_smoke() {
  # Small-dev-model validation run, per benchmarks/run_latency_suite.py's
  # own documented smoke usage. Real server, ~a few minutes.
  echo "Pre-run cleanup (stray vllm processes from earlier/crashed runs)..."
  _kill_vllm_processes
  local gpu_idx rc
  gpu_idx=$(_pick_free_gpu) || return 1
  echo "Using GPU $gpu_idx (most free memory right now)"
  CUDA_VISIBLE_DEVICES="$gpu_idx" python benchmarks/run_latency_suite.py \
      --model "$MODEL" \
      --policies lru,arc,semantic-minmax,semantic-mean \
      --workloads chat \
      --request-rates 2.0 \
      --num-prompts 20 \
      --scale 0.05 \
      --cpu-bytes-to-use 268435456 \
      --output-dir /tmp/latency_suite_smoke
  rc=$?
  echo "Post-run cleanup..."
  _kill_vllm_processes
  return "$rc"
}

opt_4_splice_probe() {
  # The current open item: byte-exact proof that a spliced GPU block's
  # content is correct. Watches the server's own debug log live and fires
  # the recall the instant a real PARTIAL SPLICE references the tagged
  # needle request. Real server, real 40-filler contention, ~a few minutes.
  echo "Pre-run cleanup (stray vllm processes from earlier/crashed runs)..."
  _kill_vllm_processes
  local gpu_idx rc
  gpu_idx=$(_pick_free_gpu) || return 1
  echo "Using GPU $gpu_idx (most free memory right now)"
  CUDA_VISIBLE_DEVICES="$gpu_idx" SEMANTIC_OFFLOAD_DEBUG=1 python harness/run_adaptive_probe_live.py
  rc=$?
  echo "Post-run cleanup..."
  _kill_vllm_processes
  return "$rc"
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
  #
  # --scale/--max-model-len/--num-gpu-blocks-override added 2026-07-30 after
  # the first pass (results/step_1_6_first_pass_20260729_234213) came back
  # mostly unusable: this command never passed any of the three, so it fell
  # back to run_grid_sweep.py's defaults (max_model_len=2048,
  # num_gpu_blocks_override=None i.e. unbounded auto-sized cache) -- the
  # same "block budget never gets exercised" class of bug test 4 hit and
  # fixed the night before (commits 4a0e740/0a4dea2/cc362df), just never
  # carried over into this option:
  #   - rag/longdoc workloads (harness/workloads.py, PRODUCTION-SCALE sizing)
  #     need ~12.5k/~48.1k input tokens at scale=1.0 -- both far exceed a
  #     2048 max_model_len, so vLLM rejected every single prompt and every
  #     rag/longdoc row in the first pass read "0/N requests completed".
  #   - num_gpu_blocks_override=None gave vLLM an auto-sized (huge) cache,
  #     so memory pressure never materialized: preemptions_delta was 0.0 in
  #     240/241 rows of the first pass. needle-v2's own built-in validity
  #     check correctly flagged this as `not_pressured` rather than faking a
  #     result (see run_latency_suite.py's needle-v2 branch comment).
  #
  # Values below (scale=0.08, max-model-len=4096, num-gpu-blocks-override=320)
  # are a worked-out-but-UNVERIFIED first attempt, not a confirmed-good
  # config -- there was no DGX access to test them at the time they were
  # written. Reasoning: at scale=0.08, longdoc lands at ~3.8k tokens and rag
  # at ~1.1k tokens (see harness/workloads.py's formulas), both comfortably
  # under a 4096 max_model_len with headroom for output; 320 GPU blocks
  # assumes vLLM's default 16-token block size, giving a ~256-block floor
  # for one full max_model_len request (vLLM's own startup validation) plus
  # ~25% headroom, the same ratio test 4's 40-block choice used over its
  # 32-block floor at max_model_len=512. During the longdoc sub-workload
  # (~241 blocks/request at this scale) that budget fits barely more than
  # one concurrent request, which should force real preemption -- but this
  # is arithmetic, not a measurement. First thing to check after running
  # this: does preemptions_delta stop being ~0 and do rag/longdoc rows stop
  # reading "0/N completed"? If not, the numbers need another pass.
  #
  # NOT addressed here: the separate semantic-mean/semantic-minmax hang at
  # chat rate=8.0 (all 3 seeds in the first pass, `lru`/`arc` unaffected at
  # the same rate) -- see .claude/docs/2026-07-30-session-handoff.md. That
  # investigation is still open; do not assume this config change fixes it,
  # and note tightening the block budget here could make it easier to
  # reproduce (more contention) or could be unrelated (the first pass's hang
  # happened with preemptions_delta=0.0, i.e. before any of today's budget
  # tightening) -- still an open question.
  echo "Pre-run cleanup (stray vllm processes from earlier/crashed runs)..."
  _kill_vllm_processes
  _require_free_gpu || return 1
  local rc
  python benchmarks/run_grid_sweep.py \
      --model "$MODEL" \
      --policies lru,arc,semantic-minmax,semantic-mean \
      --workloads chat,rag,mixed \
      --request-rates 2.0,8.0 \
      --seeds 1,2,3 \
      --target-duration-s 600 \
      --cpu-bytes-to-use 2147483648 \
      --needle-reference-counts 0,1,2 \
      --scale 0.08 \
      --max-model-len 4096 \
      --num-gpu-blocks-override 320 \
      --output-dir "results/step_1_6_second_pass_$(date +%Y%m%d_%H%M%S)" \
      --gpus "$GPUS"
  rc=$?
  echo "Post-run cleanup..."
  _kill_vllm_processes
  return "$rc"
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
  [0]="Kill stray vllm processes and free GPU memory (manual, box-wide). 3/4/6 already do this before+after themselves -- use this for a standalone cleanup. EXPLICIT ONLY -- never runs as part of 'all'."
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
