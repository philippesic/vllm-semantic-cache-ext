# Session Handoff — 2026-08-01

Follow-up to `2026-07-30-session-handoff.md`'s open item: confirm whether
`durable_summaries`/`resident=` actually grows unbounded during the
`semantic-minmax`/`rag`/`rate=8.0` hang. Picked that up, got real B200 data,
then an unrelated dormant investigation surfaced mid-session and took over
most of the time. Net: the original hang theory took a hit, and a separate,
previously-undocumented, still-open regression was rediscovered and is now
properly written down (it wasn't before — see "How this was found" below).

## 1. The rag@8.0 hang: didn't reproduce, and the leading theory is now doubtful

Ran the exact repro from 07-30's Update 2 (`semantic-minmax`/`rag`/`rate=8.0`/
`seed=1`, `SEMANTIC_OFFLOAD_DEBUG=1 SEMANTIC_OFFLOAD_TIMING=1`, same
`--scale 0.08 --max-model-len 4096 --num-gpu-blocks-override 320`) plus two
comparison runs (same repro with `SEMANTIC_OFFLOAD_DISABLE_PREFETCH=1`; same
repro on `semantic-mean` instead of `semantic-minmax`). All three completed
normally — **no hang this time.** That makes it 4/5 reproductions of the
hang across both sessions, not 4/4 as 07-30 concluded — treat "deterministic,
not flaky" as wrong; this looks more like a race/timing-dependent condition
than a monotonic resource leak, which changes where to look next.

**`resident=` (from `SEMANTIC_EVICT_DEBUG`, i.e. `len(durable_summaries)`)
stayed flat at 4675-4679 for the entire run** — bounded, not growing. This
directly contradicts 07-30's leading theory (unbounded `durable_summaries`
growth driving an ever-more-expensive O(n_total) `_rebuild_stack_cache`
compaction). The eviction-based bound (`receive_evicted_keys`, replacing the
old FIFO cap per issues log #62-64) is working as designed.

**But the per-call timing growth from 07-30 still shows up even with a flat
pool**: `query_captured_total` mean went 18.46ms → 43.93ms (2.4x) across
calls 1-4000 on the (this time non-hanging) run, and 8.68ms → 12.56ms (1.4x)
on the `semantic-mean` comparison run. Since `resident=` is flat, the O(n_total)
compaction cost in `_rebuild_stack_cache` (worker.py:247-260, still unfixed,
still real as a per-call cost) can't be scaling from a growing *candidate*
pool. **Revised leading theory: the batch size on the query side (number of
concurrently-active requests being scored per call) is what's growing, not
the candidate pool.** Not yet confirmed — would need instrumenting
`_on_queries_captured`'s `len(req_ids)` per call, not yet done.

A 60s py-spy flamegraph was captured mid-run (`dgx_hang_diagnostic.sh`'s
`run_a_flamegraph.svg`, on the DGX under
`dgx_logs/hang_diag_20260801_153622/`) but since run A didn't hang this time,
it's a healthy-operation profile, not a hang-state one — lower value than
intended, not yet looked at.

**Next step for the hang specifically:** instrument concurrent-request count
per `_on_queries_captured` call (not candidate-pool size) and correlate with
the timing growth; then retry the repro enough times to get a real
reproduction rate (this session only had budget for one more attempt beyond
07-30's three).

## 2. How this was found: a dormant, undocumented regression from 2026-07-19

While checking DGX provenance, two untracked files turned up in the ext repo
working tree (`investigation_2_3_4.sh`/`.log`) that weren't from this session
or referenced anywhere in memory or prior handoffs — the user didn't
recognize them either. They turned out to be a fresh (same-day, 2026-08-01)
rerun of an old, still-unresolved investigation:

**`origin/diagnose-stack-rebuild-revert`** (branch, tip `cb8c0f1`, dated
**2026-07-19** — 11 days before the partial-splice/hang work this project's
memory and prior handoffs actually track) contains:

- `4e116ec`: "DIAGNOSTIC ONLY: revert stack_rebuild to isolate B200
  needle-v2 regression" — a same-session B200 grid run showed **all three
  semantic policies missing needle-v2 recall at every reference_count,
  including >=1, where they'd reliably hit in every prior validated run.**
  `session_aware` was tested and ruled out (identical misses on/off).
  `_rebuild_stack_cache`'s incrementalization (`557467b`, "make stack_rebuild
  incremental instead of a full pool rebuild") was the one remaining
  untested variable in that session's bundle.
- `cb8c0f1`: also reverts `1aeab43` ("batch cross-request query captures
  into one scored pass per step") — a second, independent optimization in
  the same suspect bundle.
- Both `557467b` and `1aeab43` are still live on `master` today, unreverted.
  This branch was never merged and never resolved — just parked.

**Today's `investigation_2_3_4.log` (run before this session started,
author/timing unclear) reruns the same needle-v2 check on current `master`
and gets the same failure**: every row (`lru`, `arc`, `semantic-minmax`,
`semantic-mean`, and a policy not previously in memory —
`semantic-cuboid-mean`) reads `needle_outcome=miss, needle_hit_rate=0.0`,
including the semantic policies that are supposed to be the entire point
(per `step-0.4-adversarial-results.md`, they should hit 0.96-1.0 on exactly
this kind of adversarial-needle case). **13 days unresolved, still
reproducing, real.**

**Also newly-noted for whoever picks this back up:** `session_aware`,
`chain_aware`, `capture_stride`, and the `semantic-cuboid-mean` policy exist
in the codebase (commits `e74a4d2`/`39f37f0`/`e72e7e0`, 2026-07-19) but
aren't mentioned in `semantic-cache-project` memory or any handoff doc before
this one — that memory/handoff trail has a real gap covering this feature
surface. Worth a dedicated read-through of what these do before next
picking up needle-v2 work.

## 3. Attempted revert-branch test — inconclusive, blocked on a separate bug

Tried to settle whether reverting `557467b`/`1aeab43` fixes the needle-v2
regression by cherry-picking `4e116ec`+`cb8c0f1` onto current `master`
(clean cherry-pick, zero conflicts — confirmed `worker.py`/`query_capture.py`
were untouched by any master commit since 07-19, so this is a faithful
revert-on-current-master, not a stale rebase). Pushed as
`test-revert-stack-rebuild-on-current-master`.

Running the exact same needle-v2 command from `investigation_2_3_4.sh` Item
#3 against this branch **failed with 400 Bad Request on every request**,
identically on `master` too when retried — including `lru`, which never
touches `semantic_offload`'s scoring/worker code at all. That ruled out the
revert (or master) as the cause of the 400s specifically.

Root cause (confirmed via response body, not guessed): **a distractor
completion request's prompt is right at (or over) the `max_model_len`
budget** — first seen as "497 input tokens + 16 output tokens = 513 >
512 max_model_len". Tried bumping `--max-model-len 512→640` and
`--num-gpu-blocks-override 120→150` (scaled together) to add headroom — **it
did not help**: the reported input-token count scaled almost exactly with
the change too (625 = 640-15, matching 497 = 512-15 from before), so
whatever's sizing that prompt is coupled to one of those two launch
parameters. **Not isolated which one** — both were changed together in the
same test, so this is genuinely unresolved, not just "needs more margin."

**This blocks any real revert-vs-master needle-v2 comparison until fixed.**
Whoever picks this up next should NOT re-attempt the master/revert
comparison until this is isolated first:
1. Re-run with `max_model_len` fixed and only `num_gpu_blocks_override`
   varied, then the reverse, to find which one the distractor sizing is
   actually coupled to (grep `harness/needle_workload.py`'s
   `make_long_distractor`/`run_needle_v2_case` call chain and
   `benchmarks/run_grid_sweep.py`'s needle-v2 wiring for anything reading
   `max_model_len` or block count when choosing `distractor_words` — nothing
   obvious was found by a source grep this session, so the coupling may be
   indirect, e.g. via tokenizer/vocab differences from a different model
   config, not literal code coupling. Not yet run to ground.)
2. Once fixed, rerun `investigation_2_3_4.sh`'s Item #3 config against both
   `master` and `test-revert-stack-rebuild-on-current-master` for a clean
   answer on whether the revert restores needle-v2 recall.

Diagnostic technique note for next time: `harness/needle_workload.py`'s
`_complete()` calls `resp.raise_for_status()` without ever reading
`resp.text` — the client-side error string ("400 Client Error: Bad Request
for url: ...") never carries the actual rejection reason. A local, uncommitted
one-line patch printing `resp.text` on non-2xx (in the CLIENT-side
`cell_<policy>_seed<seed>.log`, NOT the server log — different process) is
what actually surfaced the real message both times this session. Worth
making this a permanent (committed) improvement rather than re-patching it
ad hoc every time — cheap, high-value, already proven necessary twice now
in unrelated sessions (07-30's CSV-parsing warning is the same class of
"the summary lies, read the raw data" lesson).

## For the next agent picking this up

- Don't trust "resolved"/"deterministic" claims in prior handoffs without
  re-checking — both the hang's "4/4 deterministic" and needle-v2's
  complete absence from memory turned out to be wrong/incomplete this
  session. Memory and handoffs are a curated narrative, not a full state
  dump; a `git branch -a` + untracked-file check on the actual DGX/ext repo
  surfaced 11 days of real, unresolved work that no doc mentioned.
- Hang investigation: pool-growth theory is dead (resident= is flat).
  Query-side concurrent-batch-size growth is the new leading theory, not
  yet instrumented. Reproduction rate is 4/5 across two sessions, not
  deterministic — budget for multiple attempts, not one.
- Needle-v2 regression (13+ days old, still reproducing) and the hang
  investigation both implicate `_rebuild_stack_cache`
  (`557467b`)/cross-request batching (`1aeab43`) — plausibly the same root
  cause, plausibly not. Unconfirmed either way; the revert-branch test that
  would settle it is blocked on the token-budget bug in section 3.
- `test-revert-stack-rebuild-on-current-master` branch is pushed and ready
  to reuse once the token-budget confound is isolated — don't recreate it.
- `dgx_hang_diagnostic.sh` (repo root, committed) is reusable for the next
  hang-repro attempt as-is.

## Update (same day, no-DGX follow-up): instrumentation added, one real fix landed

No box access for this part — code-only work, done off the diagnostics
above. Three commits on `master` (`d24063d`, `7a068e1`), also merged
cleanly into `test-revert-stack-rebuild-on-current-master` (`3e93096`, no
conflicts — `worker.py`'s auto-merge picked up the new instrumentation
around the still-reverted `_rebuild_stack_cache`/scoring internals without
issue).

**1. Batch-size instrumentation for the hang (`_debug.py`/`worker.py`).**
New `record_count(bucket, value)` in `_debug.py`, same shape as
`record_timing` (accumulate, print mean+max every
`SEMANTIC_OFFLOAD_TIMING_EVERY` calls), gated on the same
`SEMANTIC_OFFLOAD_TIMING=1` flag. Wired into `_on_queries_captured` to log
`len(req_ids)` (`query_captured_batch_size`) and `len(durable_summaries)`
(`query_captured_resident_pool`) together, every call, so the next repro's
`grep SEMANTIC_COUNT` directly confirms or kills the "growing concurrent
batch size" theory in one line instead of cross-referencing separate log
streams. **Not yet run against a real repro** — next DGX pass should rerun
`semantic-minmax`/`rag`/`rate=8.0` with `SEMANTIC_OFFLOAD_TIMING=1` and
check whether `query_captured_batch_size`'s mean/max climbs alongside
`query_captured_total`'s per-call cost. If it's flat too, this theory is
also dead and the next lead is unknown — don't assume it's confirmed just
because the instrumentation exists now.

**2. Isolate the needle-v2 token-budget confound (code read, no DGX).**
Traced the full path (`make_long_distractor` → `run_needle_v2_case` →
`run_latency_suite.py` → `run_grid_sweep.py`) and confirmed
`distractor_words` is a hardcoded `200`, never derived from `max_model_len`
or `num_gpu_blocks_override` anywhere in this codebase — content generation
for a given `(seed, reference_count, i)` is fully deterministic and should
be byte-identical regardless of server launch config. **This rules out an
explicit harness-level coupling** but does NOT explain the earlier
observation (reported input-token count scaling almost exactly with
`max_model_len - 15` at both 512 and 640). Since `max_model_len` and
`num_gpu_blocks_override` were changed together in that test, it's still
unknown which one (if either) the effect is actually coupled to, or whether
it's a vLLM-server-side quirk unrelated to this project's code. **Needs a
live, single-variable test to resolve — not resolvable from code alone.**

**3. Real fix landed: `_complete()` in `harness/needle_workload.py` no
longer swallows the response body.** This is what cost two extra DGX
round-trips today (`resp.raise_for_status()`'s message never included
*why* a request was rejected). Now raises `requests.HTTPError` with the
call's label (`needle`/`probe[i]`/`distractor[i]`/`recall`), prompt
char/word count, and the response body (truncated to 300 chars) baked in
— lands directly in `run_latency_suite.py`'s `error` CSV column, no manual
patching needed next time. Threaded a `label` kwarg through both
`run_needle_case` and `run_needle_v2_case`'s local `complete()` closures.
Updated the two `test_harness.py` mocks that patch `_complete` wholesale
to accept the new kwarg. **Not run locally** (no `pytest`/`requests` in
this Mac checkout) — run `test-dgx.sh` option 2 (or plain
`pytest tests/`) on the next DGX pass before trusting it.

### Next DGX session — concrete plan, in priority order

1. **Sanity first**: `pytest tests/` on `master` — confirms the
   `test_harness.py` mock updates from step 3 above actually pass (written
   blind, no local Python env to verify against).
2. **Isolate the needle-v2 confound** (step 2 above) with two single-variable
   runs, reusing `investigation_2_3_4.sh` Item #3's exact command as the
   base: (a) `--max-model-len 1024 --num-gpu-blocks-override 120`
   (blocks held at the original value, only `max_model_len` raised well
   past any plausible overflow); (b) `--max-model-len 512
   --num-gpu-blocks-override 320` (max_model_len held at the original
   value, only blocks raised). Whichever one still 400s identifies the
   real coupling; if *neither* 400s, the earlier failures might have been
   something else entirely (worth re-reading the (now much more
   informative, per fix 3 above) `error` column either way). This no
   longer needs the manual response-body patch from today — the real fix
   is already in `_complete()`.
3. **Once needle-v2 requests complete cleanly**, rerun
   `investigation_2_3_4.sh` Item #3's config against both `master` and
   `test-revert-stack-rebuild-on-current-master` (already merged with
   today's fixes, ready to use) for the real answer on whether the revert
   restores needle-v2 recall.
4. **Hang repro**: rerun `semantic-minmax`/`rag`/`rate=8.0` with
   `SEMANTIC_OFFLOAD_TIMING=1` (now including the new
   `SEMANTIC_COUNT`/`query_captured_batch_size` lines from fix 1 above).
   Budget for multiple attempts (today's session got 1 non-repro out of 1
   try; the real rate across both sessions is 4/5) — a single non-hang
   isn't evidence the hang is gone.

## Update (same day, back on DGX): confound isolated cleanly, revert test is inconclusive at n=1

Ran the plan above, steps 1-3, same day.

**Step 1 (sanity):** `pytest tests/ -k needle` — 22/22 passed on `master`.
The blind `test_harness.py` mock updates from the no-DGX pass are good.

**Step 2 (isolate the token-budget confound) — fully resolved, and it was
never about `num_gpu_blocks_override`:**
- 2a: `--max-model-len 1024 --num-gpu-blocks-override 120` (blocks held at
  the *original* value that was failing, only `max_model_len` raised) — all
  15 cells succeeded, zero errors.
- 2b: `--max-model-len 512 --num-gpu-blocks-override 320` (max_model_len
  held at the original failing value, only blocks raised) — every cell
  failed with the *identical* "497 input tokens, max 512" error as the very
  first debug run.

**Conclusion: `max_model_len` alone gates it; `num_gpu_blocks_override` was
never involved.** The earlier apparent coupling to both params was an
artifact of scaling them together in the same test. vLLM's own error
phrasing ("at least X input tokens") was the tell in hindsight — that
reads as a lower-bound from an early-exit length check, which is exactly
why it tracked `max_model_len` almost linearly (`X ≈ max_model_len - 15`
at both 512 and 640) rather than being a fixed number. Fix for any future
needle-v2 run with `distractor_words=200` at this model's tokenization
rate (~2.5 tokens/word for the synthetic corpus): use `--max-model-len
1024` or higher, not 512.

**Step 2a also gave the first uncorrupted needle-v2 data of this
investigation, and it does NOT match either prior report:**

| policy | ref_count=0 | ref_count=1 | ref_count=2 |
|---|---|---|---|
| lru / arc | miss | miss | miss |
| semantic-mean | miss | **hit** | miss |
| semantic-cuboid-mean | miss | **hit** | **hit** |
| semantic-minmax | miss | miss | miss |

Neither the 07-19 branch's claim ("all three semantic policies missing
needle-v2 recall") nor a fully-working picture — `semantic-mean` and
`semantic-cuboid-mean` DO preserve the needle at `reference_count>=1` as
designed. **Only `semantic-minmax` fails to hit at all**, and
`semantic-minmax` is also the policy that hung on `rag@8.0` in the earlier
hang investigation — a real, narrower correlation than "the regression
affects all semantic policies" implied.

**Step 3 (revert-branch comparison on the now-clean config) — inconclusive,
looks like noise at n=1, not a signal:**

| policy | master | revert branch |
|---|---|---|
| semantic-minmax | miss, miss, miss | miss, miss, **hit** |
| semantic-mean | miss, **hit**, miss | miss, **hit**, miss (identical) |
| semantic-cuboid-mean | miss, **hit**, **hit** | miss, miss, miss (worse) |

`semantic-minmax` picked up one hit under the revert; `semantic-mean` is
byte-identical either way; `semantic-cuboid-mean` got *worse* under the
revert. With one seed per cell this doesn't look like a deterministic
effect of the code change — more consistent with hit/miss being
probabilistic near a threshold (plausibly `session_aware`'s EMA relevance
score landing close to a cutoff) than with the revert fixing or breaking
anything. **Do not conclude the revert helps or hurts from this table.**

**Next step, if anyone picks this back up:** rerun step 3's exact config
with `--seeds 1,2,3` (or more) on both `master` and
`test-revert-stack-rebuild-on-current-master` and compare *hit rate
across seeds*, not single-seed hit/miss — that's the only way to
distinguish a real effect from this noise. Given how much DGX time this
thread has already consumed across two sessions, this is a "when there's
budget for it" item, not urgent — the higher-value open thread is still
the `rag@8.0` hang itself (section 1/4 above), which `semantic-minmax`'s
isolated failure here makes a slightly more plausible shared root cause
with, but that's still unconfirmed, not a reason to chase needle-v2
further right now.
