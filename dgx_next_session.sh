#!/usr/bin/env bash
# One-pass DGX session covering everything open per
# .claude/docs/2026-08-01-session-handoff.md's final "Update" section:
#   1. sync both branches
#   2. sanity pytest
#   3. needle-v2 multi-seed comparison, master vs
#      test-revert-stack-rebuild-on-current-master (the single-seed run was
#      inconclusive/noisy -- this is what's needed to get a real answer)
#   4. semantic-minmax/rag/rate=8.0 hang repro, reusing the already-committed
#      dgx_hang_diagnostic.sh, then pulling out the new SEMANTIC_COUNT lines
#      (query_captured_batch_size) that weren't wired in when that script
#      was written -- this directly tests the "growing concurrent batch
#      size" theory that's currently the leading (unconfirmed) lead.
#
# Run from the vllm-semantic-cache-ext repo root on the DGX/B200 box:
#   chmod +x dgx_next_session.sh && ./dgx_next_session.sh
#
# Everything is teed to dgx_logs/next_session_<timestamp>/ and tarballed at
# the end -- paste terminal output back, or the tarball if it's easier.
#
# Rough runtime budget: needle-v2 (step 3) is ~5-10 min/cell x 3 ref_counts
# x 5 policies x 3 seeds x 2 branches -- can be over an hour. The hang
# repro (step 4) is ~60-90 min on its own (dgx_hang_diagnostic.sh's own
# estimate). Total: plan for 2-3 hours of box time in one sitting.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="dgx_logs/next_session_${TS}"
mkdir -p "$OUTDIR"
META="$OUTDIR/00_meta.log"

_kill_vllm() {
  pkill -f "vllm serve|VLLM::|EngineCore|vllm\.entrypoints" 2>/dev/null || true
  sleep 3
}

echo "Logging to $OUTDIR" | tee "$META"

# ---------------------------------------------------------------------------
# Step 0: recover the ORIGINAL needle-v2 investigation command if it's still
# on this box. investigation_2_3_4.sh/.log were untracked, never committed,
# and only ever confirmed present on the DGX filesystem (2026-08-01) -- if
# they're still here, use that exact command for step 3 instead of the
# reconstruction below, so results stay comparable to the existing table in
# the handoff doc. If they're gone, the reconstruction is a best-effort
# match built from the documented working config (max_model_len=1024,
# num_gpu_blocks_override=120) and the policy/ref-count list from that
# table -- flagged inline, check NEEDLE_CMD before trusting it.
# ---------------------------------------------------------------------------
{
  echo "=== provenance ==="
  git rev-parse HEAD
  git status --short
  echo "=== checking for investigation_2_3_4.sh/.log ==="
  if [ -f investigation_2_3_4.sh ]; then
    echo "FOUND -- dumping for reference, prefer its Item #3 command over the reconstruction below:"
    cat investigation_2_3_4.sh
  else
    echo "not found -- using the reconstructed command below"
  fi
} | tee -a "$META"

_kill_vllm

# ---------------------------------------------------------------------------
# Step 1: sync both branches
# ---------------------------------------------------------------------------
echo "=== Step 1: git fetch/pull both branches ===" | tee -a "$META"
git fetch origin 2>&1 | tee -a "$META"
git checkout master 2>&1 | tee -a "$META"
git pull origin master 2>&1 | tee -a "$META"
git checkout test-revert-stack-rebuild-on-current-master 2>&1 | tee -a "$META"
git pull origin test-revert-stack-rebuild-on-current-master 2>&1 | tee -a "$META"
git checkout master 2>&1 | tee -a "$META"

# ---------------------------------------------------------------------------
# Step 2: sanity -- confirm the needle harness fix (label kwarg threading,
# response-body surfacing) didn't break anything on either branch before
# spending box time on it.
# ---------------------------------------------------------------------------
for branch in master test-revert-stack-rebuild-on-current-master; do
  echo "=== Step 2: pytest -k needle on $branch ===" | tee -a "$META"
  git checkout "$branch" 2>&1 | tee -a "$META"
  python -m pytest tests/ -k needle -v 2>&1 | tee "$OUTDIR/02_pytest_${branch}.log" | tail -30 | tee -a "$META"
done
git checkout master 2>&1 | tee -a "$META"

# ---------------------------------------------------------------------------
# Step 3: needle-v2 multi-seed, master vs revert branch.
#
# This is investigation_2_3_4.sh's real Item #3 command (recovered live
# from the DGX filesystem 2026-08-02 -- that script was never committed,
# so an earlier version of this step reconstructed the wrong config: wrong
# model, wrong entrypoint, wrong cpu-bytes-to-use, missing extra-config --
# and produced invalid all-hit data with zero discriminating power. See
# .claude/docs/2026-08-02-session-handoff.md section 1 before touching
# this again. Uses run_grid_sweep.py (not run_latency_suite.py directly)
# because that's what actually applies --extra-config per cell and
# supports --seeds as a real multi-seed loop instead of one invocation
# per seed.
# ---------------------------------------------------------------------------
NEEDLE_MODEL="Qwen/Qwen2.5-7B-Instruct"
POLICIES="lru,arc,semantic-mean,semantic-cuboid-mean,semantic-minmax"
SEEDS="1,2,3"

for branch in master test-revert-stack-rebuild-on-current-master; do
  git checkout "$branch" 2>&1 | tee -a "$META"
  echo "=== Step 3: needle-v2, branch=$branch seeds=$SEEDS ===" | tee -a "$META"
  _kill_vllm
  python benchmarks/run_grid_sweep.py \
    --model "$NEEDLE_MODEL" \
    --policies "$POLICIES" \
    --workloads needle-v2 \
    --needle-reference-counts 0,1,2 \
    --num-prompts 12 \
    --max-model-len 512 \
    --num-gpu-blocks-override 120 \
    --cpu-bytes-to-use 91750400 \
    --seeds "$SEEDS" \
    --extra-config '{"session_aware": true, "session_bonus_half_life": 8}' \
    --gpus 0,1,2,3,4,5,6,7 \
    --output-dir "$OUTDIR/needle_v2_${branch//\//_}" \
    2>&1 | tee "$OUTDIR/03_needle_${branch//\//_}.log"
  _kill_vllm
done
git checkout master 2>&1 | tee -a "$META"

echo "=== Step 3 combined results (needle_outcome by policy/branch/seed/ref_count) ===" | tee -a "$META"
for f in "$OUTDIR"/needle_v2_*/results.csv; do
  echo "--- $f ---" | tee -a "$META"
  cat "$f" | tee -a "$META"
done

# ---------------------------------------------------------------------------
# Step 4: hang repro, reusing the already-committed diagnostic script
# as-is (runs A: semantic-minmax/rag/8.0 DEBUG+TIMING, B: same +
# DISABLE_PREFETCH control, C: semantic-mean/rag/8.0 comparison). Its own
# grep only pulls SEMANTIC_EVICT_DEBUG/SEMANTIC_TIMING -- pull the new
# SEMANTIC_COUNT (query_captured_batch_size) lines out separately below
# since that instrumentation postdates this script.
# ---------------------------------------------------------------------------
echo "=== Step 4: hang repro via dgx_hang_diagnostic.sh ===" | tee -a "$META"
./dgx_hang_diagnostic.sh 2>&1 | tee "$OUTDIR/04_hang_diagnostic.log"

HANG_OUTDIR=$(grep -o 'dgx_logs/hang_diag_[0-9_]*' "$OUTDIR/04_hang_diagnostic.log" | head -1)
if [ -n "$HANG_OUTDIR" ]; then
  echo "=== Step 4: SEMANTIC_COUNT (query_captured_batch_size) trend, run A ===" | tee -a "$META"
  grep "SEMANTIC_COUNT" "$HANG_OUTDIR"/run_a_minmax_rag8/server_*.log \
    > "$OUTDIR/04_run_a_count_trend.txt" 2>/dev/null
  tail -20 "$OUTDIR/04_run_a_count_trend.txt" 2>/dev/null | tee -a "$META"
  echo "=== Step 4: SEMANTIC_COUNT trend, run C (semantic-mean comparison) ===" | tee -a "$META"
  grep "SEMANTIC_COUNT" "$HANG_OUTDIR"/run_c_mean_rag8/server_*.log \
    > "$OUTDIR/04_run_c_count_trend.txt" 2>/dev/null
  tail -20 "$OUTDIR/04_run_c_count_trend.txt" 2>/dev/null | tee -a "$META"
else
  echo "Could not locate hang_diag output dir from log -- check $OUTDIR/04_hang_diagnostic.log manually" | tee -a "$META"
fi

tar czf "${OUTDIR}.tar.gz" "$OUTDIR"
echo "=== DONE: ${OUTDIR}.tar.gz ===" | tee -a "$META"
echo "Paste terminal output back, or scp/cat ${OUTDIR}.tar.gz if easier." | tee -a "$META"
