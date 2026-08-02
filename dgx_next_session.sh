#!/usr/bin/env bash
# One-pass DGX session covering everything open on the needle-v2 revert
# comparison and the rag@8.0 hang investigation. History: this script's
# first two runs (2026-08-02) each burned a full DGX round-trip on a bad
# needle-v2 config -- see .claude/docs/2026-08-02-session-handoff.md and
# its "Update (second pass)" section for exactly what went wrong both
# times. Everything below was fixed AND locally smoke-tested (argument
# parsing, JSON parsing, the CSV validation logic against synthetic
# missing-file/error-row/clean-row cases, the pytest fail-fast PIPESTATUS
# check) before this version was committed -- see that handoff doc's
# "Update (third pass" section for what was verified and how.
#
#   0. dump investigation_2_3_4.sh for cross-reference only (informational,
#      does not change step 3's behavior -- see step 0's own comment)
#   1. sync both branches
#   2. sanity pytest on both branches -- HALTS (exit 1) before any GPU time
#      is spent if either branch's needle tests fail
#   3. needle-v2 multi-seed comparison (master vs
#      test-revert-stack-rebuild-on-current-master, 3 seeds -- the earlier
#      single-seed run was inconclusive/noisy), using the corrected,
#      verified config (see step 3's own comment for the specific bugs this
#      fixes), followed by a hard CSV validation pass that reads every
#      row's `error` column (not just "N/N cells succeeded", which only
#      means server processes exited cleanly) and loudly flags any
#      error rows OR a missing results.csv as invalid data
#   4. semantic-minmax/rag/rate=8.0 hang repro via dgx_hang_diagnostic.sh
#      (updated 2026-08-02 with SEMANTIC_GPUMEM allocator instrumentation
#      to test the current leading theory), then surfacing its
#      SEMANTIC_COUNT/SEMANTIC_GPUMEM trends in this script's own summary
#
# Run from the vllm-semantic-cache-ext repo root on the DGX/B200 box:
#   chmod +x dgx_next_session.sh && ./dgx_next_session.sh
#
# Everything is teed to dgx_logs/next_session_<timestamp>/ and tarballed at
# the end -- paste terminal output back, or the tarball if it's easier.
# If step 2 fails, the script exits before steps 3/4 run at all -- fix the
# reported failure and rerun rather than assuming later steps are safe to
# skip to.
#
# Rough runtime budget: needle-v2 (step 3) is ~5-10 min/cell x 3 ref_counts
# x 5 policies x 3 seeds x 2 branches, but cells run concurrently across
# 8 GPUs (--gpus 0-7) so wall-clock is well under the naive product. The
# hang repro (step 4) is ~60-90 min on its own (dgx_hang_diagnostic.sh's
# own estimate, single-GPU). Total: plan for 1.5-2.5 hours of box time.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# NOTE: not wired into any step. Step 3 (needle-v2) MUST use NEEDLE_MODEL=7B
# (see line ~133) -- 1.5B produced uniform all-hit invalid data 2026-08-02
# (see 2026-08-02-session-handoff.md section 1). Step 4 shells out to
# dgx_hang_diagnostic.sh, which sets its own MODEL and does NOT inherit this
# (un-exported). Left only so nothing silently defaults to 1.5B; do not feed
# it into the needle-v2 invocation.
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
# Step 0: provenance dump only -- does NOT change step 3's behavior.
# investigation_2_3_4.sh/.log were untracked, never committed, and were
# recovered once already (2026-08-02) to build step 3's real command; if
# they're still on this box, this just dumps them for cross-reference so a
# human can sanity-check step 3 still matches. Step 3 itself is hardcoded
# below with the already-verified, already-corrected config (7B model,
# run_grid_sweep.py, --max-model-len 1024, --cpu-bytes-to-use 91750400,
# session_aware extra-config) -- it does not branch on what this step finds.
# ---------------------------------------------------------------------------
{
  echo "=== provenance ==="
  git rev-parse HEAD
  git status --short
  echo "=== checking for investigation_2_3_4.sh/.log (reference only) ==="
  if [ -f investigation_2_3_4.sh ]; then
    echo "FOUND -- dumping for cross-reference against step 3's hardcoded command below:"
    cat investigation_2_3_4.sh
  else
    echo "not found -- step 3 uses its own hardcoded, already-corrected command regardless"
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
PYTEST_FAILED=0
for branch in master test-revert-stack-rebuild-on-current-master; do
  echo "=== Step 2: pytest -k needle on $branch ===" | tee -a "$META"
  git checkout "$branch" 2>&1 | tee -a "$META"
  python -m pytest tests/ -k needle -v 2>&1 | tee "$OUTDIR/02_pytest_${branch}.log" | tail -30 | tee -a "$META"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    PYTEST_FAILED=1
    echo "!!! Step 2 FAILED on $branch -- see $OUTDIR/02_pytest_${branch}.log !!!" | tee -a "$META"
  fi
done
git checkout master 2>&1 | tee -a "$META"

if [ "$PYTEST_FAILED" -ne 0 ]; then
  echo "" | tee -a "$META"
  echo "############################################################" | tee -a "$META"
  echo "# STEP 2 SANITY FAILED -- halting before any GPU time is spent." | tee -a "$META"
  echo "# Fix the failure(s) above (see 02_pytest_<branch>.log) and" | tee -a "$META"
  echo "# rerun. Not proceeding to steps 3/4 on a known-broken build." | tee -a "$META"
  echo "############################################################" | tee -a "$META"
  tar czf "${OUTDIR}.tar.gz" "$OUTDIR"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: needle-v2 multi-seed, master vs revert branch.
#
# Base command is investigation_2_3_4.sh's real Item #3 command (recovered
# live from the DGX filesystem 2026-08-02 -- that script was never
# committed, so an earlier version of this step reconstructed the wrong
# config: wrong model, wrong entrypoint, wrong cpu-bytes-to-use, missing
# extra-config -- and produced invalid all-hit data with zero
# discriminating power. See .claude/docs/2026-08-02-session-handoff.md
# section 1.)
#
# --max-model-len is 1024, NOT Item #3's literal 512: 2026-08-01 already
# root-caused and fixed this exact 400 ("prompt contains at least 497
# input tokens... max 512") for needle-v2 -- 512 is too small for the
# 200-word distractors at this model's tokenization rate, purely a
# max_model_len issue, num_gpu_blocks_override was never involved (see
# 2026-08-01-session-handoff.md). A first pass of this script replayed
# Item #3's literal 512 verbatim and reintroduced the exact same 400s
# 2026-08-02 -- don't repeat that; keep this at 1024+ for needle-v2 runs
# regardless of what any recovered investigation script says.
#
# Uses run_grid_sweep.py (not run_latency_suite.py directly) because
# that's what actually applies --extra-config per cell and supports
# --seeds as a real multi-seed loop instead of one invocation per seed.
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
    --max-model-len 1024 \
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
for branch in master test-revert-stack-rebuild-on-current-master; do
  f="$OUTDIR/needle_v2_${branch//\//_}/results.csv"
  echo "--- $f ---" | tee -a "$META"
  if [ -f "$f" ]; then
    cat "$f" | tee -a "$META"
  else
    echo "(missing -- see validation section below)" | tee -a "$META"
  fi
done

# ---------------------------------------------------------------------------
# Loud, CSV-correct pass/fail summary. The 2026-08-02 session lost two full
# DGX round-trips to "N/N cells succeeded" being misread as "the data is
# good" when every row's `error` column was actually populated (a 400 from
# a stale --max-model-len) -- that only surfaces if you actually read the
# error column. Parsed with csv.DictReader, not awk/grep, per the 2026-07-30
# handoff's own lesson (quoted error-message commas break naive splitting).
#
# Iterates the two known branch dirs explicitly rather than globbing
# results.csv -- confirmed by an actual local run (missing `requests`
# module, all 15 cells crashed before writing a single row) that
# run_grid_sweep.py does NOT create results.csv at all when every cell
# fails before its first write. A glob over a nonexistent path either
# silently skips (nullglob) or leaves the literal unexpanded pattern
# (default), and either way the previous version of this check could miss
# a total-failure branch instead of flagging it loudly.
# ---------------------------------------------------------------------------
echo "=== Step 3 validation (error-column check, not just exit codes) ===" | tee -a "$META"
NEEDLE_HAD_ERRORS=0
for branch in master test-revert-stack-rebuild-on-current-master; do
  f="$OUTDIR/needle_v2_${branch//\//_}/results.csv"
  if [ ! -f "$f" ]; then
    echo "!!! $f: MISSING -- run_grid_sweep.py produced no results.csv at all for branch=$branch (every cell likely crashed before its first write; check $OUTDIR/03_needle_${branch//\//_}.log) !!!" | tee -a "$META"
    NEEDLE_HAD_ERRORS=1
    continue
  fi
  summary=$(python3 -c "
import csv, sys
path = sys.argv[1]
total = 0
errors = []
with open(path) as fh:
    for row in csv.DictReader(fh):
        total += 1
        err = (row.get('error') or '').strip()
        if err:
            errors.append((row.get('policy'), row.get('seed'), row.get('reference_count'), err[:150]))
print(f'{path}: {total} rows, {len(errors)} with a non-empty error column')
for policy, seed, ref, err in errors[:5]:
    print(f'  SAMPLE ERROR policy={policy} seed={seed} ref_count={ref}: {err}')
if len(errors) > 5:
    print(f'  ... and {len(errors) - 5} more')
if total == 0:
    print('  WARNING: results.csv exists but has zero rows')
sys.exit(1 if (errors or total == 0) else 0)
" "$f")
  status=$?
  echo "$summary" | tee -a "$META"
  if [ "$status" -ne 0 ]; then
    NEEDLE_HAD_ERRORS=1
  fi
done

if [ "$NEEDLE_HAD_ERRORS" -ne 0 ]; then
  echo "" | tee -a "$META"
  echo "############################################################" | tee -a "$META"
  echo "# STEP 3 (needle-v2) HAS ERROR ROWS -- this is NOT valid data," | tee -a "$META"
  echo "# regardless of what the grid-sweep 'N/N cells succeeded' line" | tee -a "$META"
  echo "# said (that only means server processes exited cleanly)." | tee -a "$META"
  echo "# Do not use this run's results.csv for the master-vs-revert" | tee -a "$META"
  echo "# comparison -- read the sample errors above, fix the root" | tee -a "$META"
  echo "# cause, and rerun step 3 before trusting any of this data." | tee -a "$META"
  echo "############################################################" | tee -a "$META"
else
  echo "Step 3: no error rows in any results.csv -- data looks structurally valid." | tee -a "$META"
fi

# ---------------------------------------------------------------------------
# Step 4: hang repro, reusing the already-committed diagnostic script
# as-is (runs A: semantic-minmax/rag/8.0 DEBUG+TIMING, B: same +
# DISABLE_PREFETCH control, C: semantic-mean/rag/8.0 comparison). Its own
# grep now pulls SEMANTIC_EVICT_DEBUG/SEMANTIC_TIMING/SEMANTIC_GPUMEM
# directly (2026-08-02) -- SEMANTIC_COUNT (batch/pool size) still postdates
# that script's own grep list, so pull it out separately here for the
# top-level summary, and mirror SEMANTIC_GPUMEM here too so this script's
# own $META has the allocator data alongside it instead of only in the
# nested hang_diag output dir.
# ---------------------------------------------------------------------------
echo "=== Step 4: hang repro via dgx_hang_diagnostic.sh ===" | tee -a "$META"
./dgx_hang_diagnostic.sh 2>&1 | tee "$OUTDIR/04_hang_diagnostic.log"

HANG_OUTDIR=$(grep -o 'dgx_logs/hang_diag_[0-9_]*' "$OUTDIR/04_hang_diagnostic.log" | head -1)
if [ -n "$HANG_OUTDIR" ]; then
  for run in "run_a_minmax_rag8:run A" "run_c_mean_rag8:run C (semantic-mean comparison)"; do
    dir="${run%%:*}"
    label="${run##*:}"
    echo "=== Step 4: SEMANTIC_COUNT (batch/pool size) trend, $label ===" | tee -a "$META"
    grep "SEMANTIC_COUNT" "$HANG_OUTDIR"/"$dir"/server_*.log \
      > "$OUTDIR/04_${dir}_count_trend.txt" 2>/dev/null
    tail -20 "$OUTDIR/04_${dir}_count_trend.txt" 2>/dev/null | tee -a "$META"
    echo "=== Step 4: SEMANTIC_GPUMEM (allocator) trend, $label ===" | tee -a "$META"
    grep "SEMANTIC_GPUMEM" "$HANG_OUTDIR"/"$dir"/server_*.log \
      > "$OUTDIR/04_${dir}_gpumem_trend.txt" 2>/dev/null
    tail -20 "$OUTDIR/04_${dir}_gpumem_trend.txt" 2>/dev/null | tee -a "$META"
  done
else
  echo "Could not locate hang_diag output dir from log -- check $OUTDIR/04_hang_diagnostic.log manually" | tee -a "$META"
fi

tar czf "${OUTDIR}.tar.gz" "$OUTDIR"
echo "=== DONE: ${OUTDIR}.tar.gz ===" | tee -a "$META"
echo "Paste terminal output back, or scp/cat ${OUTDIR}.tar.gz if easier." | tee -a "$META"
