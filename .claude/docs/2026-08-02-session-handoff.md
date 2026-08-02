# Session Handoff — 2026-08-02

Ran `dgx_next_session.sh` (new, committed this session) end to end: branch
sync, needle sanity tests, a needle-v2 master-vs-revert multi-seed
comparison, and a hang repro with the new `SEMANTIC_COUNT` instrumentation.
Two of the four steps produced real signal; one step's data turned out to
be invalid due to a config mistake on my part, caught and documented below
so it isn't mistaken for a real result later.

## 1. Needle-v2 multi-seed comparison: data is INVALID, do not use it

`dgx_next_session.sh`'s needle-v2 command was a reconstruction, not a
replay of `investigation_2_3_4.sh` — that script was never committed, so
the exact Item #3 command wasn't available when the script was written.
Step 0 of the run recovered the real file from the DGX filesystem (still
sitting there, untracked, since 2026-08-01) and printed it, and it differs
from the reconstruction in every load-bearing way:

| param | reconstruction (this run) | real Item #3 |
|---|---|---|
| entrypoint | `run_latency_suite.py` | `run_grid_sweep.py` |
| model | Qwen2.5-1.5B-Instruct | **Qwen2.5-7B-Instruct** |
| `--num-prompts` | 5 | 12 |
| `--cpu-bytes-to-use` | 2147483648 (default, 2GiB) | **91750400 (~87MB)** |
| `--extra-config` | none | `{"session_aware": true, "session_bonus_half_life": 8}` |

The result: `recall_load_bytes=1376256` on **every single row** — all 5
policies (including `lru`/`arc`), all 3 ref_counts (including
`ref_count=0`, which should read `miss` for a non-preserving policy), all
3 seeds, both branches. Identical bytes-for-bytes across every condition
is the same "not under real capacity pressure" signature the 07-30 handoff
already root-caused once for the grid sweep (`opt_6`'s missing
`--num-gpu-blocks-override`) — except this time `classify_needle_outcome`
reads it as a uniform false `hit` rather than `not_pressured`, because
`recall_load_bytes > 0` for a different, more mundane reason (most likely:
a 1.5B model at this num-prompts/max-model-len scale never builds enough
real memory pressure for GPU-tier residency to differ from CPU-tier reload
across policies — the 2GiB `--cpu-bytes-to-use` alone is ~23x more
generous than the real investigation's ~87MB). Whatever the precise
mechanism, **this comparison cannot distinguish policies or branches and
should not be read as "everything hits" or as evidence either way on the
revert question.**

**Fix for next time:** rerun using the *actual* recovered command (model
7B, `run_grid_sweep.py`, `--cpu-bytes-to-use 91750400`,
`--extra-config '{"session_aware": true, "session_bonus_half_life": 8}'`)
with `--seeds 1,2,3` added, against both `master` and
`test-revert-stack-rebuild-on-current-master`. Note this also means the
2026-08-01 handoff's "confound isolated / step 2a" needle-v2 table should
be re-examined for the same model/scale mismatch before trusting it as
comparable to `investigation_2_3_4.sh`'s original session_aware
investigation — it's not yet confirmed whether that step used the 7B
model + `session_aware` extra-config or not. `investigation_2_3_4.sh`
itself is still sitting untracked on the DGX box
(`/raid/ppesic/tmp/vllm-semantic-cache-ext/investigation_2_3_4.sh`) —
commit it (or a cleaned-up version) so this stops being a recurring
recovery step.

## 2. Hang repro: another non-reproduction, softening the hang-rate estimate further

`dgx_hang_diagnostic.sh` Run A (`semantic-minmax`/`rag`/`rate=8.0`,
`SEMANTIC_OFFLOAD_DEBUG=1 SEMANTIC_OFFLOAD_TIMING=1`) completed normally —
**no hang.** Combined with 2026-08-01's non-repro, that's now at least two
non-reproductions on top of 07-30's original 3/3 — the hang is
increasingly clearly a flaky/race condition, not a monotonic, deterministic
resource leak. Don't trust any specific fraction without recounting from
the raw logs; the point is it keeps not reproducing on demand, which rules
out a simple "always happens after N steps" theory.

Because Run A didn't hang, everything below is a **healthy-operation**
profile, not a hang-state one (same caveat as 08-01's flamegraph) — still
useful for ruling theories out, but not yet a direct look at the hang
itself.

## 3. SEMANTIC_COUNT data: both current theories (pool growth, batch-size growth) ruled out on a healthy run

This was the first real run of the `query_captured_batch_size`/
`query_captured_resident_pool` instrumentation added 2026-08-01. Reading
it requires one correction to how `record_count` accumulates that's worth
flagging for future sessions: **`mean` is a cumulative mean since process
start, not a per-window average** (`_debug.py`'s `record_count`: `sum`/
`count` both accumulate across the whole process lifetime; only `max`
resets each window). So a jump like Run C's `mean=2702.21` (calls=2000)
→ `mean=3690.53` (calls=4000) is *not* "the pool grew from 2702 to 3690" —
back it out arithmetically: sum over calls 2001-4000 ≈
`3690.53*4000 - 2702.21*2000 ≈ 9,357,700`, mean over just that window ≈
**4679** — i.e. `durable_summaries` was ramping up from near-zero during
the first ~2000-call warmup and has been flat at ~4679-4681 since, which
matches `SEMANTIC_EVICT_DEBUG`'s `resident=` trend exactly (also flat at
4675-4679 throughout both runs). **No contradiction, no new pool-growth
finding — this doubly confirms 08-01's "pool growth theory is dead"
conclusion via an independent counter**, it just needs correct arithmetic
to read.

`query_captured_batch_size` also stayed small and flat: mean 3.96 (Run A,
one window) and 3.76→4.04 across two windows (Run C) — genuinely not
climbing, max capped at 10-11 both times. **This is evidence against
08-01's revised leading theory** (growing concurrent-request batch size
driving the per-call cost climb) — on this healthy run, batch size simply
isn't growing.

Yet the per-call cost climb is still there and still real: Run C's
`query_captured_total` mean went 8.25ms (calls=2000) → 12.83ms (calls=4000),
a genuine ~55% increase, with `stack_rebuild` itself only accounting for a
small fraction of that (0.07ms → 0.31ms — a bigger *relative* jump but a
tiny *absolute* one; the growth is concentrated somewhere else in
`_on_queries_captured`).

**Both of the two candidate explanations that have been instrumented so
far (candidate-pool size, concurrent-batch size) are now flat/ruled out on
a healthy run, but the cost is still climbing. Read through
`_rebuild_stack_cache` (worker.py:234-288) this session and it does NOT
match 07-30's original description** ("mask sized to the entire resident
pool, rescanned on any dirty step") — `keep_mask` is sized to
`len(self._stack_cache_keys)`, which is kept in lockstep with
`durable_summaries` via the same incremental insert/remove bookkeeping, so
it should already be O(k) in practice once the pool is flat, not O(n_total)
as originally suspected. No logical leak found in a source read.

**New leading theory, not yet instrumented or confirmed:** since neither
logical size (pool, batch) is growing, the remaining candidate is a
systemic/allocator-level cost — e.g. GPU memory fragmentation or allocator
overhead from `_rebuild_stack_cache`'s repeated `torch.cat`/boolean-mask
tensor allocation every dirty step (a fresh tensor each time, never reused
across the run's lifetime), independent of any logical structure size.
This is speculative — it has not been checked against real data, unlike
the two now-eliminated theories.

## Next DGX session, in priority order

1. **Needle-v2**: commit a cleaned copy of `investigation_2_3_4.sh` (still
   only on the DGX filesystem, section 1 above), then rerun its real Item
   #3 command with `--seeds 1,2,3` added, on both `master` and
   `test-revert-stack-rebuild-on-current-master`. Don't reuse
   `dgx_next_session.sh`'s current needle-v2 step as-is — fix its config
   first (7B model, `run_grid_sweep.py`, `--cpu-bytes-to-use 91750400`,
   the `session_aware` extra-config) or the result will be invalid again
   the same way.
2. **Hang**: instrument something beyond batch size / pool size for the
   next repro attempt — a `torch.cuda.memory_stats()` snapshot (allocated
   vs reserved bytes, fragmentation-relevant fields) printed alongside the
   existing `SEMANTIC_TIMING`/`SEMANTIC_COUNT` lines would directly test
   the allocator-overhead theory above. Keep budgeting for multiple
   attempts — 2+ non-repros now on top of the original 3/3, this really
   is flaky, not deterministic.
3. Low priority, unblocked whenever picked up: `dgx_next_session.sh`'s
   step 3 config should be fixed per item 1 rather than left as a
   standing trap for the next person who runs it unmodified.

## Update (same day, second pass): needle-v2 regressed AGAIN (own mistake), hang data got better this time

Reran the script immediately after the fix above. Two outcomes:

**Needle-v2: 30/30 cells failed, same 400 as before the fix landed.** The
"fix" in item 1 above still used Item #3's literal `--max-model-len 512`
verbatim — but 2026-08-01 had *already* root-caused and fixed this exact
error (497-input-token prompts vs. a 512 cap) by raising `max_model_len`
to 1024, established as a `max_model_len`-only issue with
`num_gpu_blocks_override` never involved. Copying Item #3's recovered
command verbatim reintroduced a bug that was already fixed and documented
one session ago. **Corrected: `--max-model-len` is now 1024 in the
script** (`num_gpu_blocks_override` unchanged at 120). Needle-v2 master-
vs-revert comparison is still unresolved — this was the second wasted
attempt in a row, for two different reasons (config reconstruction, then
reverting a known fix). Whoever runs this next should sanity-check the
actual `results.csv` for `error` columns before trusting a "done" grid
sweep, not just the "N/N cells succeeded" summary line (grid sweep
"succeeded" here just means every server process exited cleanly, not that
requests succeeded).

**Hang: still no hang (Run A exit 0 again), but this run's timing data is
the clearest replication yet of 07-30/08-01's per-call cost growth.**
`query_captured_total` mean went 17.87ms (calls=2000) -> 43.13ms
(calls=4000), a 2.4x increase within one run -- matching 08-01's own
non-hang run's growth factor almost exactly. `update_relevance` also
roughly doubled (4.08ms -> 8.83ms). Unlike the first pass today (flat
batch size), this run's `query_captured_batch_size` showed real, if
modest, growth: mean 3.96 -> 4.58 (+16%), max 11 -> 13. That's not nearly
enough growth on its own to explain a 2.4x cost increase, but it's the
first run where batch size moved at all instead of being perfectly flat --
worth tracking across future repro attempts rather than treating as fully
dead. `resident_pool`'s cumulative-mean artifact (see main body above)
reproduced identically (2694 -> 3686, same warmup-then-flat-at-~4679
interpretation, no real growth).

**Bottom line for next session:** the systemic/allocator-overhead theory
from the main body is still the best untested lead for the *cost* growth;
batch-size growth is no longer fully dead but is clearly not sufficient
alone. Needle-v2 needs a clean rerun with the now-corrected
`--max-model-len 1024`, and results should be spot-checked for `error`
columns before being trusted.

## Update (third pass, same day): two DGX round-trips burned in a row on this script, so this pass fixed it for real instead of re-guessing

After the second pass above also failed (the "fix" replayed
`investigation_2_3_4.sh`'s literal `--max-model-len 512`, reintroducing the
exact 400 that had already been fixed once), this pass did NOT ship another
guess. Two independent Opus subagents did a full adversarial audit — one
against every flag in step 3 (needle-v2) cross-referenced against all four
handoff docs, one against the hang diagnostic and the underlying
`worker.py`/`_debug.py` code — and everything both agents changed was then
locally smoke-tested (not just re-read) before committing:

- **Needle-v2 audit**: every flag in step 3 traced to a specific later fix,
  not just the recovered Item #3 command (see the agent's own citations,
  preserved in this doc's history). One dead-code landmine found and
  fixed: `dgx_next_session.sh`'s top-level `MODEL=...1.5B...` default
  (unused by any step, but sitting at the top of the file as exactly the
  wrong value that caused the first pass's failure) — commented to make
  clear it must never feed the needle-v2 invocation. `--gpus 0-7` confirmed
  NOT a memory-pressure confound (`run_grid_sweep.py` gives each cell its
  own isolated GPU/port/output-dir/process). `session_aware` in
  `--extra-config` confirmed intentional, not a leftover, held constant
  across both branches — kept as-is (see the agent's report for the
  full ablation-vs-comparability tradeoff, not revisited here).
- **Hang audit**: `dgx_hang_diagnostic.sh`'s hardcoded params confirmed to
  match 07-30's Update 2 command byte-for-byte, not stale. No logical leak
  found in `_rebuild_stack_cache` on a second independent read (agrees with
  this doc's earlier section). Stale "4/4 confirmed repro, expected to
  hang" language in the script's comments (contradicted by 2+
  non-reproductions since) rewritten to describe it as flaky. **New:
  `record_gpu_memory()` added to `_debug.py`, wired into `worker.py`'s
  `_on_queries_captured`**, gated on the same `SEMANTIC_OFFLOAD_TIMING`
  flag, printing `SEMANTIC_GPUMEM` (allocated/reserved/inactive-split MB,
  `num_alloc_retries`) on the same every-2000-calls cadence as the existing
  counters — queried only on print steps, not per-call, so it doesn't
  distort the very timing it's trying to explain. This directly tests the
  allocator-fragmentation theory that was the leading unconfirmed lead at
  the end of this doc's main body. `dgx_hang_diagnostic.sh` updated to grep
  and tail `SEMANTIC_GPUMEM` alongside the existing trends.
- **Local smoke testing** (this session, no DGX needed): `run_grid_sweep.py`
  has zero torch/vllm imports, so the exact step-3 command was actually run
  locally (Mac, no GPU) — confirmed every flag name is valid and JSON
  `--extra-config` parses. It failed fast on a missing `requests` module
  (Mac-only gap, present on the DGX), but that run exposed a **real bug**:
  when every cell crashes before its first write, `run_grid_sweep.py` never
  creates `results.csv` at all. The CSV-validation block (added in the
  second pass) would have silently skipped that case via glob-miss. Fixed
  to iterate the two known branch dirs explicitly and treat a missing file
  as an unambiguous top-level failure, then re-verified against synthetic
  data: a missing file, a row with an embedded-quote-and-comma error
  message (the exact class of string that broke naive `awk` parsing in the
  2026-07-30 handoff), and clean all-hit data — all three cases produced
  the correct banner/exit behavior. The step-2 pytest fail-fast
  (`PIPESTATUS` read immediately after a `tee | tail` pipe, no intervening
  command) was also isolated and confirmed correct in a standalone repro.
- **Also added**: step 2 now halts the whole script (exit 1, tarballs logs
  first) if pytest fails on either branch, instead of silently continuing
  into GPU-hours on a known-broken build. Step 4's own `$META` summary now
  mirrors `SEMANTIC_GPUMEM` alongside `SEMANTIC_COUNT` (previously only the
  latter). Stale comments referencing a `NEEDLE_CMD` variable that no
  longer exists (leftover from before step 3's command was hardcoded) were
  removed from step 0.

**Bottom line: `dgx_next_session.sh` is the first version of this script
that had every step's config independently cross-checked against the full
documented fix history AND had its failure-detection logic proven against
real and synthetic failure cases before being handed back for a DGX run.**
If it still fails, that's new information, not a repeat of a known bug —
paste back the output rather than assuming it's another config mistake.
