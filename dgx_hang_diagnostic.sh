#!/usr/bin/env bash
# One-shot diagnostic session for the semantic-minmax rag@rate=8.0 hang
# (see .claude/docs/2026-07-30-session-handoff.md, "Update 2"). Goal: gather
# everything needed for an offline (no-DGX) fix session in one pass instead
# of a fix-run-fix loop against the box.
#
# Run from the vllm-semantic-cache-ext repo root on the DGX/B200 box:
#   chmod +x dgx_hang_diagnostic.sh && ./dgx_hang_diagnostic.sh
#
# Everything is teed to dgx_logs/hang_diag_<timestamp>/ and also echoed to
# the terminal -- paste the terminal output back; if it's too long, paste
# 00_meta.log plus the *_trend.txt files, or scp/cat the whole directory.
#
# Runtime: ~60-90 min (run A is expected to hang and hit the harness's own
# 1800s timeout on purpose -- that's what gives us the full resident=/timing
# growth curve, not just early samples).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="dgx_logs/hang_diag_${TS}"
mkdir -p "$OUTDIR"
META="$OUTDIR/00_meta.log"
echo "Logging to $OUTDIR" | tee "$META"

{
  echo "=== provenance ==="
  git rev-parse HEAD
  git status --short
  echo "=== gpus ==="
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee -a "$META"

_pick_free_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', *' '{print $1, $2}' | sort -k2 -rn | head -1 | awk '{print $1}'
}

_kill_vllm() {
  pkill -f "vllm serve|VLLM::|EngineCore|vllm\.entrypoints" 2>/dev/null || true
  sleep 3
}

echo "=== pre-cleanup ===" | tee -a "$META"
_kill_vllm

if ! command -v py-spy >/dev/null 2>&1; then
  echo "py-spy not found -- attempting install (best-effort, non-fatal)" | tee -a "$META"
  (uv pip install py-spy || pip install py-spy) 2>&1 | tee -a "$META" || true
fi

# ---------------------------------------------------------------------------
# Run A: the confirmed 4/4 repro (semantic-minmax / rag / rate=8.0) with
# both SEMANTIC_OFFLOAD_DEBUG=1 (SEMANTIC_EVICT_DEBUG resident= lines from
# worker.py's receive_evicted_keys) and SEMANTIC_OFFLOAD_TIMING=1 (per-bucket
# timing growth). Expected to hang and hit the 1800s subprocess timeout --
# intentional, we need the growth curve across the whole hang.
# ---------------------------------------------------------------------------
GPU_A=$(_pick_free_gpu)
echo "=== Run A: semantic-minmax/rag/8.0, DEBUG+TIMING, GPU $GPU_A (expect ~30min hang) ===" | tee -a "$META"
(
  CUDA_VISIBLE_DEVICES="$GPU_A" SEMANTIC_OFFLOAD_DEBUG=1 SEMANTIC_OFFLOAD_TIMING=1 \
  python benchmarks/run_latency_suite.py \
    --model "$MODEL" \
    --policies semantic-minmax --workloads rag --request-rates 8.0 \
    --needle-reference-counts 0,1,2 --scale 0.08 --cpu-bytes-to-use 2147483648 \
    --max-model-len 4096 --num-gpu-blocks-override 320 --seed 1 \
    --target-duration-s 600 --output-dir "$OUTDIR/run_a_minmax_rag8"
) > "$OUTDIR/run_a.log" 2>&1 &
RUN_A_PID=$!

# Let contention build before profiling -- prior repros hung around the
# 1800s mark, sample midway through.
sleep 300
ENGINE_PID=$(pgrep -f "EngineCore" | head -1)
if command -v py-spy >/dev/null 2>&1 && [ -n "${ENGINE_PID:-}" ]; then
  echo "=== py-spy flamegraph on EngineCore pid $ENGINE_PID (60s sample) ===" | tee -a "$META"
  py-spy record -o "$OUTDIR/run_a_flamegraph.svg" --pid "$ENGINE_PID" --duration 60 \
    > "$OUTDIR/run_a_pyspy.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "py-spy record failed (likely ptrace perms) -- retrying with sudo -n" | tee -a "$META"
    sudo -n py-spy record -o "$OUTDIR/run_a_flamegraph.svg" --pid "$ENGINE_PID" --duration 60 \
      >> "$OUTDIR/run_a_pyspy.log" 2>&1 || \
      echo "py-spy unavailable without sudo -- skipping flamegraph, timing/debug logs still cover this" | tee -a "$META"
  fi
else
  echo "py-spy or EngineCore pid not found -- skipping flamegraph" | tee -a "$META"
fi

wait "$RUN_A_PID"
echo "Run A exit: $?" | tee -a "$META"
_kill_vllm

grep "SEMANTIC_EVICT_DEBUG" "$OUTDIR"/run_a_minmax_rag8/server_*.log > "$OUTDIR/run_a_resident_trend.txt" 2>/dev/null
grep "SEMANTIC_TIMING" "$OUTDIR"/run_a_minmax_rag8/server_*.log > "$OUTDIR/run_a_timing_trend.txt" 2>/dev/null
echo "--- run_a resident= trend (last 20) ---" | tee -a "$META"
tail -20 "$OUTDIR/run_a_resident_trend.txt" 2>/dev/null | tee -a "$META"
echo "--- run_a timing trend (last 20) ---" | tee -a "$META"
tail -20 "$OUTDIR/run_a_timing_trend.txt" 2>/dev/null | tee -a "$META"

# ---------------------------------------------------------------------------
# Run B: same repro with the prefetch-on-preemption path disabled -- rules
# that mechanism in or out directly for THIS hang (unlike the chat@8.0 case,
# preemptions_delta is very much non-zero here, so it hasn't been ruled out).
# ---------------------------------------------------------------------------
GPU_B=$(_pick_free_gpu)
echo "=== Run B: semantic-minmax/rag/8.0, DISABLE_PREFETCH control, GPU $GPU_B ===" | tee -a "$META"
CUDA_VISIBLE_DEVICES="$GPU_B" SEMANTIC_OFFLOAD_DEBUG=1 SEMANTIC_OFFLOAD_DISABLE_PREFETCH=1 \
python benchmarks/run_latency_suite.py \
  --model "$MODEL" \
  --policies semantic-minmax --workloads rag --request-rates 8.0 \
  --needle-reference-counts 0,1,2 --scale 0.08 --cpu-bytes-to-use 2147483648 \
  --max-model-len 4096 --num-gpu-blocks-override 320 --seed 1 \
  --target-duration-s 600 --output-dir "$OUTDIR/run_b_minmax_rag8_noprefetch" \
  > "$OUTDIR/run_b.log" 2>&1
echo "Run B exit: $?  (0/success + no timeout => prefetch path implicated; still hangs => rules it out)" | tee -a "$META"
_kill_vllm

# ---------------------------------------------------------------------------
# Run C: same repro on semantic-mean -- doesn't hang per prior runs, but
# shows severe TTFT-tail degradation. Comparison: same growth pattern
# without crossing into a full hang, or genuinely different behavior?
# ---------------------------------------------------------------------------
GPU_C=$(_pick_free_gpu)
echo "=== Run C: semantic-mean/rag/8.0, DEBUG+TIMING comparison, GPU $GPU_C ===" | tee -a "$META"
CUDA_VISIBLE_DEVICES="$GPU_C" SEMANTIC_OFFLOAD_DEBUG=1 SEMANTIC_OFFLOAD_TIMING=1 \
python benchmarks/run_latency_suite.py \
  --model "$MODEL" \
  --policies semantic-mean --workloads rag --request-rates 8.0 \
  --needle-reference-counts 0,1,2 --scale 0.08 --cpu-bytes-to-use 2147483648 \
  --max-model-len 4096 --num-gpu-blocks-override 320 --seed 1 \
  --target-duration-s 600 --output-dir "$OUTDIR/run_c_mean_rag8" \
  > "$OUTDIR/run_c.log" 2>&1
echo "Run C exit: $?" | tee -a "$META"
_kill_vllm

grep "SEMANTIC_EVICT_DEBUG" "$OUTDIR"/run_c_mean_rag8/server_*.log > "$OUTDIR/run_c_resident_trend.txt" 2>/dev/null
grep "SEMANTIC_TIMING" "$OUTDIR"/run_c_mean_rag8/server_*.log > "$OUTDIR/run_c_timing_trend.txt" 2>/dev/null
echo "--- run_c resident= trend (last 20) ---" | tee -a "$META"
tail -20 "$OUTDIR/run_c_resident_trend.txt" 2>/dev/null | tee -a "$META"
echo "--- run_c timing trend (last 20) ---" | tee -a "$META"
tail -20 "$OUTDIR/run_c_timing_trend.txt" 2>/dev/null | tee -a "$META"

tar czf "${OUTDIR}.tar.gz" "$OUTDIR"
echo "=== DONE: ${OUTDIR}.tar.gz ===" | tee -a "$META"
echo "Paste the terminal output above, or scp/cat back ${OUTDIR}.tar.gz if easier." | tee -a "$META"
