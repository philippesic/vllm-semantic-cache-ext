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

## What's still open

### 2. semantic-mean/semantic-minmax hang at `chat` rate=8.0 — root cause NOT confirmed

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
- Don't re-attempt the chat@8.0 hang by tuning workload parameters again
  (block budget, rate, `max_model_len`) the way the preemption bug got
  fixed on 2026-07-29 — that pattern was tried repeatedly for *that* bug
  and turned out to be masking a dead code path, not a tuning problem.
  This one's `preemptions_delta=0.0` evidence suggests it's a different
  class of bug; get a timing breakdown before changing any config or code.
