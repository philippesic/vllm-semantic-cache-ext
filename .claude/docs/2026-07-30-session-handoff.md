# Session Handoff — 2026-07-30

Follow-up to `2026-07-29-session-handoff.md`. That session kicked off
`benchmarks/run_grid_sweep.py` (Step 1.6) and left results pending. This
session read those results (`2.6-result.csv`, untracked at the ext repo
root — 241 rows from `results/step_1_6_first_pass_20260729_234213/`),
found the run was mostly unusable, root-caused why, fixed what could be
fixed without DGX access (none was available this session — everything
below is static analysis + code reading, **nothing was re-run**), and
left one investigation open.

## What was fixed

### 1. `opt_6_grid_sweep` never inherited test 4's block-budget fix

`test-dgx.sh`'s `opt_6_grid_sweep` called `run_grid_sweep.py` without
`--scale`, `--max-model-len`, or `--num-gpu-blocks-override`, so it fell
back to the script's defaults (`max_model_len=2048`,
`num_gpu_blocks_override=None`, i.e. an unbounded auto-sized cache) — the
same "block budget never gets exercised" bug test 4 hit and fixed the
night before (commits `4a0e740`/`0a4dea2`/`cc362df`), just never carried
into this option. Two separate visible failures traced back to this one
cause:

- **rag/longdoc: 100% failure, every row (48/48 rag, 24/24 longdoc).**
  `harness/workloads.py`'s production-scale sizing puts rag at ~12.5k
  input tokens and longdoc at ~48.1k at `scale=1.0` (the default) — both
  far exceed a 2048 `max_model_len`, so vLLM rejected every prompt
  outright and `vllm bench serve` completed 0/N requests every time.
  `chat` (~1.35k tokens) survived by coincidence, not by anything being
  configured correctly for it either.
- **preemptions essentially never happened: 240/241 rows read
  `preemptions_delta=0.0`.** An unbounded cache means memory pressure
  never materializes. This also explains why every `needle`/`mixed-needle`
  row read `needle_hit_rate=0.0` with outcome `miss` or `not_pressured`
  (never `hit`) — `run_latency_suite.py`'s own needle-v2 branch comment
  already documents this exact failure mode: "REQUIRES a tightened GPU
  (`--num-gpu-blocks-override`)... else every cell reads `not_pressured`."
  The harness correctly flagged its own invalid config; this wasn't a
  policy result at all.

**Fix applied:** `opt_6_grid_sweep` now passes
`--scale 0.08 --max-model-len 4096 --num-gpu-blocks-override 320`. Reasoning
(see the comment block above the function in `test-dgx.sh` for the full
version): at `scale=0.08`, longdoc lands at ~3.8k tokens and rag at ~1.1k,
both comfortably under a 4096 `max_model_len`; 320 blocks assumes vLLM's
default 16-token block size, giving a ~256-block floor for one full
`max_model_len` request (vLLM's own startup validation) plus ~25%
headroom — the same ratio test 4's 40-block choice used over its 32-block
floor at `max_model_len=512`. During the longdoc sub-workload (~241
blocks/request at this scale) that budget fits barely more than one
concurrent request, which should force real preemption.

**This is a worked-out arithmetic estimate, not a verified fix — there was
no DGX access this session to run it.** First thing to check on the next
DGX pass: does `preemptions_delta` stop being ~0, and do rag/longdoc rows
stop reading "0/N completed"? If not, the scale/block-budget numbers need
another iteration — the reasoning is documented in the script comment so
the next pass can be adjusted in place rather than re-derived from
scratch. Output dir was also renamed `step_1_6_second_pass_*` (was
`first_pass_*`) so the two runs don't get confused.

## Update (same day): second-pass results are in — fix confirmed, hang moved

The `--scale 0.08 --max-model-len 4096 --num-gpu-blocks-override 320` fix
above was run for real this session
(`results/step_1_6_second_pass_20260730_193645/`, `216` rows, parsed with
`csv.DictReader`, not eyeballed). Two outcomes:

**Fix confirmed working.** No more 0/N-completed rows on rag/longdoc, and
`preemptions_delta` is non-zero across every policy/workload/rate cell that
completed (e.g. rag@rate=2.0 now reads 0-3 preemptions, rag@rate=8.0 reads
107-2158 depending on policy — see below). The block-budget arithmetic
needs no further iteration.

**But `semantic-minmax` hung on `rag@rate=8.0` in all 3 seeds — a
different bug than the one described in section 2 below, not the same one
reproducing.** The overall sweep reported "9/12 cells succeeded"; all 3
failing cells are `semantic-minmax` (seed 1, 2, 3). Each hit
`run_latency_suite.py`'s 1800s subprocess timeout on `rag@rate=8.0`, and —
notably — every subsequent sub-workload in that same server session
*also* timed out identically (`mixed/longdoc@0.5`, `mixed/rag@0.7`,
`mixed/chat@0.8` all read the same "timed out after 1800s" error), until
the 7200s per-cell watchdog killed the process. That's why all 3
`semantic-minmax` seeds are missing the last 3 mixed-workload rows
(rate 3.2/2.8/2.0 sub-cells) — the run never got there, not a data-logging
gap.

**The originally-reported chat@rate=8.0 hang (section 2 below) did NOT
reproduce this run** — `chat@rate=8.0` reads `preemptions_delta=0.0` and
no error for every policy, every seed. Treat that investigation as stale;
the real, 3/3-reproduced hang this session is `semantic-minmax` /
`rag` / `rate=8.0`, not `semantic-mean` / `chat` / `rate=8.0`.

**Where semantic policies completed `rag@rate=8.0` (i.e. everything except
the 3 hung `semantic-minmax` cells), they show a severe, quantified
latency/preemption cliff vs. `lru`/`arc` under the identical new config:**

| policy | seed | ttft_p50 | ttft_p99 | preemptions_delta |
|---|---|---|---|---|
| lru / arc | all 6 | ~36-38 ms | ~450-565 ms | 107-167 |
| semantic-mean | 1 | 13,445 ms | 52,785 ms | 2158 |
| semantic-mean | 2 | 193 ms | 14,154 ms | 923 |
| semantic-mean | 3 | 2,174 ms | 19,555 ms | 1677 |
| semantic-minmax | 1/2/3 | — (hung, no result) | — | 333-777 (partial, before kill) |

5-20x more preemptions than lru/arc, and 30-100x worse TTFT tail where it
didn't outright hang. This is a much larger, now-quantified version of the
gap noted in `_debug.py`'s `DISABLE_PREFETCH` docstring (17 vs 5
preemptions on an earlier, smaller run).

One more isolated data point: `semantic-mean` seed=3 *only* also lost its
last 3 mixed-workload rows (`chat@3.2`/`longdoc@2.0`/`rag@2.8`) to the same
1800s-timeout pattern — seeds 1 and 2 completed those rows fine. Possibly
flaky rather than deterministic; not yet enough evidence either way.

Needle hit rate reads `1.0` for every policy including `lru`/`arc` in this
sweep's `mixed`-workload needle sub-case — expected, not a contradiction of
`step-0.4-adversarial-results.md`'s 0/16 lru/arc finding. This grid
sweep's needle case isn't the dedicated adversarial-eviction test that
produced that result; don't read anything into the two numbers agreeing or
disagreeing.

**Revised recommended next step:** the `SEMANTIC_OFFLOAD_TIMING=1` /
`SEMANTIC_OFFLOAD_DISABLE_PREFETCH=1` approach in section 2 below is still
the right tool, but point it at `semantic-minmax` / `rag` / `rate=8.0`
(the reproduced hang) rather than `semantic-mean` / `chat` / `rate=8.0`
(unreproduced). The preemption-storm angle (777+ preemptions in one run)
is a stronger lead here than it was for the original chat@8.0 report,
given `preemptions_delta` is very much non-zero this time — worth checking
first whether the prefetch-admission-backpressure path in `connector.py`
(described below) is now actually engaging and misbehaving under this much
higher preemption volume, before assuming a different mechanism.
`2.6-result.csv` at the ext repo root still holds the *first*-pass data;
the second-pass CSV (216 rows, from
`results/step_1_6_second_pass_20260730_193645/results.csv` on the DGX) has
now been copied to `2.6-second-pass-result.csv` at the ext repo root,
untracked, matching how the first-pass file is kept (not committed —
`results/` itself is gitignored and this follows the same spirit).

## Update 2 (same day): `semantic-minmax`/`rag@8.0` timing breakdown — a strong, code-level lead

Reproduced the hang standalone (outside the grid sweep, isolated
policy/workload/rate/seed) with `SEMANTIC_OFFLOAD_TIMING=1`:

```
CUDA_VISIBLE_DEVICES=<free GPU> SEMANTIC_OFFLOAD_TIMING=1 python \
  benchmarks/run_latency_suite.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --policies semantic-minmax --workloads rag --request-rates 8.0 \
  --needle-reference-counts 0,1,2 --scale 0.08 --cpu-bytes-to-use 2147483648 \
  --max-model-len 4096 --num-gpu-blocks-override 320 --seed 1 \
  --target-duration-s 600 --output-dir results/timing_repro_minmax_rag8
```

**Reproduced again** — `preemptions_delta=1113` this time (higher than any
of the 3 sweep seeds' 333-777), full 1800s timeout. 4/4 reproductions
across this session (3 sweep seeds + this standalone run); this is
deterministic under this config, not flaky.

One gotcha worth recording: `SEMANTIC_OFFLOAD_TIMING`'s output doesn't
appear in the client's terminal/log — `harness/server.py`'s `launch_server()`
redirects the `vllm serve` subprocess's stdout/stderr straight to a
per-run server log file
(`results/<output-dir>/server_<policy>_<port>_<timestamp>.log`), so
`record_timing()`'s prints (which happen inside that server process) only
show up there. `grep SEMANTIC_TIMING` on that file, not the client output.

**The timing breakdown itself is a real, quantified lead — not flat
per-call cost, but a **climbing** per-call cost** across the run, in every
instrumented bucket (`worker.py`'s `_on_queries_captured`, all tagged
`(EngineCore pid=...)` — this vLLM config runs the worker code in the
EngineCore process, not a separate worker subprocess). Comparing the
*marginal* cost of calls 2001-4000 against calls 1-2000 (each bucket's
`record_timing()` prints a cumulative summary every 2000 calls, so the
second print's total minus the first's isolates that window):

| bucket | calls 1-2000 (mean) | calls 2001-4000 (mean) | growth |
|---|---|---|---|
| `query_captured_sync` | 0.19 ms | 0.34 ms | 1.8x |
| `query_captured_total` | 8.95 ms | 20.73 ms | 2.3x |
| `update_relevance` | 1.88 ms | 4.48 ms | 2.4x |
| `stack_rebuild` | 0.11 ms | 0.67 ms | **6.0x** |

Every bucket got more expensive per call within the *same run* — the
signature of cost scaling with a growing accumulator, not a flat
per-request cost.

**Code-level candidate found in `worker.py`'s `_rebuild_stack_cache()`
(lines 234-289):** the docstring/comment (lines 235-244) describes it as
incremental — "compact out pending removals... instead of a from-scratch
rebuild over EVERY resident candidate" — but the removal branch doesn't
actually deliver that:

```python
if self._stack_pending_remove:
    keep_mask = torch.ones(len(self._stack_cache_keys), dtype=torch.bool)   # O(n_total)
    ...
    surviving_keys = [k for k, keep in zip(self._stack_cache_keys, keep_mask.tolist()) if keep]  # O(n_total)
    ...
    self._stack_cache_index = {k: i for i, k in enumerate(surviving_keys)}  # O(n_total)
```

Whenever `_stack_pending_remove` is non-empty *at all* — even one key —
this allocates a mask sized to the entire current resident pool and
rescans/rebuilds over every candidate. That's O(n_total) per dirty step,
not O(k) (k = number actually removed) as the comment claims. And
`_mark_inserted_into_stack_cache` (lines 206-215) queues a removal on
*every overwrite* — "a key that already has a synced row... is also
queued for removal of its stale row" — so under heavy preemption/
re-admission churn (which `semantic-minmax` has 5-20x more of than
lru/arc at `rag@rate=8.0`, per the "Update" section above), nearly every
dirty step likely hits this O(n_total) branch, and it gets more expensive
as the resident pool (`durable_summaries`) grows over the session —
matching the observed 1.8x-6x marginal-cost growth directly.

This closes the loop with what was already known: `semantic-minmax`
preempts far more than lru/arc → more overwrite-driven insert+remove
cycles → more O(n_total) compactions → each step gets slower → requests
queue longer → more contention/preemption → repeat. A believable death
spiral into the 1800s timeout. `query_captured_sync`/`query_captured_total`
(the batched scoring pass over the full `cache`/`keys` stack) are also
inherently O(n_candidates) per call by design, not obviously a bug on
their own, but their growth is consistent with the same underlying
cause — a growing resident pool.

**NOT YET CONFIRMED — this is a strong code-reading inference, not a
direct measurement.** Nothing this session actually measured
`len(durable_summaries)` or `len(self._stack_cache_keys)` over time to
confirm the pool is really what's growing (vs., e.g., growing purely from
more total steps at constant pool size, which wouldn't fit as cleanly but
hasn't been ruled out). **Next step before attempting a fix:** rerun with
`SEMANTIC_OFFLOAD_DEBUG=1` alongside `SEMANTIC_OFFLOAD_TIMING=1` and grep
`SEMANTIC_EVICT_DEBUG` in the server log (from `receive_evicted_keys()`'s
`debug_print` at `worker.py:201-204`, format `received=<n> removed=<n>
resident=<n>`) — if `resident=` climbs steadily over the run instead of
holding near `_max_durable_summaries` (`= num_cpu_blocks`), that's direct
confirmation the eviction signal (`receive_evicted_keys`, driven by the
real `CachePolicy`'s evictions) isn't keeping the resident pool bounded
under this load, and the O(n_total) compaction fix becomes worth doing.

If confirmed, the fix direction is to make the removal-compaction branch
actually O(k): e.g. swap-and-truncate the k removed indices with the last
k surviving entries instead of building a full-length boolean mask and
rescanning everything, or batch/defer removals instead of compacting on
every dirty step.

## What's still open

### 2. semantic-mean/semantic-minmax hang at `chat` rate=8.0 — root cause NOT confirmed, NOT reproduced in the second pass (see update above)

In the first pass, both semantic policies (never `lru`/`arc`) hit
`run_latency_suite.py`'s hardcoded 1800s subprocess timeout on the
standalone `chat` workload at `request_rate=8.0`, across all 3 seeds
(6/6 runs). Works fine at `rate=2.0`. This is real and reproducible, not
noise — but **the investigation this session did NOT land on a confirmed
mechanism**, and left a live contradiction worth reading carefully before
trusting any next hypothesis:

- **Leading candidate ruled out (or at least not applicable to the runs
  we have data for):** the natural suspect is
  `SemanticOffloadingConnectorScheduler`'s prefetch-on-preemption path
  (`connector.py`) — `get_num_new_matched_tokens` defers re-admission of a
  request while it has an in-flight prefetch job
  (`req_status.transfer_jobs` non-empty), a mechanism that only exists for
  semantic policies (`lru`/`arc` never build `TransferJob`s at all). The
  theory: under 4x arrival rate, if prefetch-job completion throughput
  doesn't scale 4x too, admitted-but-blocked requests could pile up
  monotonically instead of plateauing — a real convergence to deadlock,
  matching "hangs" rather than "is slower". There's also prior evidence in
  `_debug.py`'s `DISABLE_PREFETCH` toggle docstring that semantic-minmax
  caused *more* preemptions than lru under a tight-capacity config on a
  real B200 run (17 vs 5) — i.e. the mechanism is known to interact badly
  with contention in general.
  **BUT:** this specific hung row (`semantic-mean, seed 2, chat, rate=8.0`
  and the other 5 like it) has `preemptions_delta=0.0` — a real snapshot
  from `vllm:num_preemptions_total`, queried against the live server
  (still up after the client-side `vllm bench serve` subprocess was
  killed) via `metrics_mod.snapshot()`, not a value that could be an
  artifact of the timeout/kill. **Zero preemptions occurred during the
  entire run that hung.** The prefetch-admission-backpressure path can
  only ever engage in response to a real preemption
  (`on_request_preempted`/`_queue_preempted` are the only entry points
  into `_preempted_pending`). If nothing was ever preempted, that whole
  code path was never exercised, and can't be what caused this specific
  hang. **Do not implement a "cap concurrent prefetches" fix on the
  strength of this theory alone — it doesn't fit the evidence in hand.**
  It may still be a real bug worth fixing on its own merits (unbounded
  concurrent prefetch admission is a legitimate latent risk), just not
  the explanation for *this* symptom.
- **What wasn't ruled out:** the summary-building/query-capture path
  (`worker.py`'s `_on_queries_captured`/`_build_summaries_body`) is the
  other semantic-only, per-request-scaling code all policies don't share.
  It's already been heavily optimized against two real past perf bugs
  (issues log entries #53 and its follow-up — batched cross-request
  scoring, probe-layer-only summary building), and nothing during this
  session's read turned up an obvious remaining O(n²) or unbounded
  accumulation, but that was a read, not a profile — inconclusive either
  way.

**Recommended next step (cheap, no guessing required): the timing
instrumentation to answer this already exists in the codebase.** Rerun the
failing case (`semantic-mean`, `chat`, `rate=8.0`, one seed) with
`SEMANTIC_OFFLOAD_TIMING=1` — `_debug.py`'s `record_timing()` machinery
accumulates per-bucket wall time and prints a summary every 2000 calls of
each bucket (`query_captured_total`, `query_captured_sync`,
`stack_rebuild`, `update_relevance`, etc.), independently in the worker
and scheduler processes. Whichever bucket's mean-ms/call or call-count
is pathological identifies the actual bottleneck directly, instead of
guessing a fix and hoping. If the timing breakdown does implicate the
prefetch/preemption path after all (contradicting the preemptions_delta=0
evidence above — e.g. if preemption is happening but the metric itself is
somehow not being captured correctly), `SEMANTIC_OFFLOAD_DISABLE_PREFETCH=1`
is a second existing toggle that isolates it directly (makes
`on_request_preempted` a no-op, matching lru/arc's behavior) without
writing any new code.

### 3. Low priority, unchanged from 2026-07-29

- Aggregate splice-benefit metric (Σspliced/Σ(spliced+reloaded)) across a
  longer run vs. a splice-disabled control — still just the one 0.33
  data point from `step-1.5-partial-splice-live-proof.md`.
- Climbing filler-failure count in `4_splice_probe` runs (0→5→7→11/44,
  no surfaced error) — not yet investigated further.
- Recover/reconstruct the missing `semantic-eviction-plan.md`/
  `semantic-eviction-issues-log.md` master docs, if local copies exist
  anywhere outside git.

## For the next agent picking this up

- Read `2.6-result.csv` yourself (or its `second_pass` successor once one
  exists) with a real CSV parser before trusting any summary of it,
  including this one — several of this session's own findings only
  surfaced after re-parsing with `csv.DictReader` instead of eyeballing
  rows; a naive `awk -F','` split breaks on the quoted `error` field.
- The scale/max-model-len/block-override numbers in `opt_6_grid_sweep` are
  a first attempt, not gospel — validate against a real run before reusing
  them elsewhere (e.g. in a "third pass" or in `opt_3`/`opt_4`-style
  scripts).
- Don't chase the chat@8.0 hang as originally scoped — it didn't reproduce
  in the second pass. The confirmed, 4/4-reproduced bug going into the next
  session is `semantic-minmax` hanging on `rag@rate=8.0` (see "Update"
  above), which also poisons every later sub-workload in that same server
  session.
- The timing breakdown is done (see "Update 2" above) and points at a
  specific O(n_total)-per-dirty-step compaction in `worker.py`'s
  `_rebuild_stack_cache()` (lines 247-260) that doesn't match its own
  "incremental" docstring. **Not yet confirmed** — the next concrete step is
  a `SEMANTIC_OFFLOAD_DEBUG=1 SEMANTIC_OFFLOAD_TIMING=1` rerun of the same
  repro command, grepping `SEMANTIC_EVICT_DEBUG` for a climbing `resident=`
  count. Only attempt the O(k) compaction fix after that confirms the pool
  is actually what's growing.
- `2.6-second-pass-result.csv` at the repo root now holds the second-pass
  data (216 rows); `2.6-result.csv` is the stale first-pass data. Both are
  untracked working copies, not committed.
