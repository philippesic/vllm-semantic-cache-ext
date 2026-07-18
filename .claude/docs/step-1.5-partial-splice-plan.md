# Step 1.5 — Partial-Splice Support: Validity Investigation & Implementation Plan

**Author:** investigation-only pass (2026-07-17). No code written, no server run, no commit.
Static reading of the real ext-repo (`semantic_offload/`) and real vLLM offloading source
(`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
`vllm/v1/kv_offload/{base,cpu/gpu_worker}.py`). Prerequisite context: issues log entries
#23-30 and plan-doc passes 13-18.

---

## Verdict (read this first)

**VALID — with two hard preconditions and one mandatory correctness change to how keys are
matched.** Partial-splice is technically sound and should be built. It is *not* merely an
optimization of the existing exact-match path — it is the fix that makes the splice path
produce **any** benefit at all, because the current exact-match check is broken in a second,
previously-undocumented way beyond coverage (see the single most important finding below).

Preconditions the design must assert and fall through on: **(1) a single KV group**
(already scoped), and **(2) `block_size_factor == 1`** (offloaded block size == GPU block
size — the project's actual dev/test regime; all launch configs to date leave
`kv_connector_extra_config["block_size"]` unset, so `base.py:541` keeps it at 1). Outside
these, fall through to the normal full reload and release the prefetch as stale, exactly as
today.

**The single most important finding:** `prefetch.keys` and `keys_to_load` are ordered on
**different axes**, so the current `if keys_to_load != prefetch.keys` list-equality check
(`connector.py:466`) cannot succeed for any multi-block request even under perfect coverage
and even with the `_TOP_M` supply fix landed:

- `keys_to_load` (`connector.py:461`, mirrors base `scheduler.py:806`) is a **positional**
  contiguous slice `group_state.offload_keys[start_block_idx:num_blocks]` — block N is the
  N-th chunk of the request's token sequence.
- `prefetch.keys` (`connector.py:322`, `= resident_keys` from
  `manager.top_relevant_keys()`, `manager.py:106`) is **relevance-ranked** — sorted by EMA
  score descending, a completely different order, and possibly a non-contiguous subset drawn
  from the request's *whole* chain (including already-GPU-cached blocks that are not even in
  `keys_to_load`).

List equality between a positional slice and a relevance-sorted subset is satisfiable only
when both have length 1 (or by astronomically-unlikely coincidence). This is *why* every
real-server run (entries #28-30) shows `KEY MISMATCH` with `prefetch.keys` at 1-10 — the
near-hits were the single-block cases. **Partial-splice's mandatory correctness move is to
stop comparing lists positionally and instead map prefetched content to destination blocks
by `OffloadKey` identity.** `OffloadKey` is content-addressed and unique per position
(`make_offload_key(req_block_hash, group_idx)`, `scheduler.py:288`; `block_hashes` are
prefix-inclusive so no key repeats within a chain), so identity → position is well-defined.

---

## Investigation answers (the three correctness questions)

### Q1 — Is `keys_to_load` positional or an unordered set? (alignment strategy)

**Positional, and this is load-bearing.** Traced end to end:

- Base `update_state_after_alloc` (`scheduler.py:749-845`) builds `keys_to_load` and
  `dst_block_ids` as **parallel positional lists**: for group `g`,
  `keys_to_load += offload_keys[start_block_idx:num_blocks]` and
  `dst_block_ids += [b.block_id for b in group_blocks[num_locally_computed_gpu_blocks:num_gpu_blocks]]`.
  The worker (`gpu_worker.py:310-360`) pairs the CPU source blocks (from
  `prepare_load(keys_to_load)`, returned in `keys_to_load` order) with `dst_block_ids`
  **1:1 positionally within each group**. At `block_size_factor == 1`,
  `block_idx % src_block_size_factor == 0` (`gpu_worker.py:316-317`), so the per-group
  `block_indices` skip logic is a no-op and the copy is a straight positional zip. This is
  the fact that makes a *scattered* partial reload expressible (see Q2).
- The ext splice path's `dst_block_ids` (`connector.py:474-477`) is computed the same
  positional way and is already correct.
- **`prefetch.gpu_block_ids[i]` holds the KV content for `prefetch.keys[i]`**: they were
  loaded together via `prepare_load(resident_keys)` → `dst_spec = GPULoadStoreSpec(gpu_block_ids, ...)`
  (`connector.py:290-291`), same 1:1 positional worker pairing. So the prefetch reservation
  gives us a reliable `OffloadKey → src GPU block id` map.

**Alignment strategy for partial-splice** (single group, bsf==1):
1. Build `prefetch_src: dict[OffloadKey, int]` = `dict(zip(prefetch.keys, prefetch.gpu_block_ids))`.
2. Recompute `keys_to_load` (positional) and `dst_block_ids` (positional) exactly as the
   base method does — they are parallel: `dst_block_ids[i]` is the destination for
   `keys_to_load[i]`.
3. For each position `i`: if `keys_to_load[i] in prefetch_src`, add
   `(prefetch_src[keys_to_load[i]], dst_block_ids[i])` to the **splice set**; else add
   `(keys_to_load[i], dst_block_ids[i])` to the **remainder set**.
4. Splice set → GPU→GPU `index_copy_` via the existing `splice_jobs` metadata channel.
   Remainder set → one normal CPU→GPU load job (Q2). Prefetched keys **not** in
   `keys_to_load` (already-cached region) are simply unused — released with the reservation.

Never zip `prefetch.keys` against `keys_to_load` positionally. Identity mapping only.

### Q2 — Can splice + partial reload be coordinated in one `update_state_after_alloc`? (transfer coordination)

**Yes, structurally sound, and — importantly — it does NOT require "two independent tracked
jobs for one request."** The reason the historical two-transfer complexity (entries #26-27)
was dangerous does not recur here, because **the splice is untracked**:

- The splice creates **no** `TransferJob`, adds **nothing** to `req_status.transfer_jobs`
  and **nothing** to `self._jobs`. It is a fire-and-forget synchronous `index_copy_` run in
  `SemanticOffloadingConnector.bind_connector_metadata` (`connector.py:639-646`) on the
  default stream, before the step's forward pass. It carries no completion, no ref-count,
  no deferral semantics.
- The partial reload is exactly **one** normal load job through the standard `load_jobs`
  channel, registered in `req_status.transfer_jobs` — identical in kind to today's normal
  full reload, only with fewer keys. So the `assert not req_status.transfer_jobs`
  invariant (`scheduler.py:835`) still sees exactly one job added, and the prefetch job it
  replaces has already completed and cleared before re-admission (guaranteed by
  `get_num_new_matched_tokens`'s `if req_status.transfer_jobs: return None, False` deferral,
  `scheduler.py:718`; this is the same reasoning entry #28 used to prove `job_status is None`
  means "done", not "pending").

So the count of tracked jobs per request is unchanged from today (one). What changes is that
we must **not** call `super().update_state_after_alloc` (which would issue a *full* reload of
all `keys_to_load`). Instead we override it to issue a **reduced** reload of the remainder
only. This is the real cost of partial-splice: ~40 lines of base load-job construction
(`scheduler.py:823-845`) must be reproduced in the subclass for the single-group/bsf==1 case,
because the base method takes no "load only this subset" parameter. This duplication is the
main maintenance liability and the main thing to guard with a version-pinned unit test.

**Expressibility of a scattered remainder reload** (the subtle part, verified against the
real worker): because bsf==1 makes `gpu_worker.py`'s per-group `block_indices` skip-logic a
no-op, a **non-contiguous** subset of destination blocks is a legal `GPULoadStoreSpec` — the
worker pairs `prepare_load(remainder_keys)` sources with `remainder_dst_block_ids` positionally
and copies each independently. `group_sizes=[len(remainder)]`, `block_indices=[start_idx]`
(value irrelevant at bsf==1 since `% 1 == 0`), `sum(group_sizes)==len(block_ids)` holds. At
bsf>1 this breaks (the worker would compute wrong intra-block offsets for a scattered set) —
which is the second reason for the bsf==1 precondition.

### Q3 — Concrete failure modes to design against (with the specific guard for each)

Modeled on the project's three real prior defects: a KeyError from unregistered-job
completion (#26), an assert from misinterpreting a load completion (#27 Bug#2/B), and a
silent block-reuse data-corruption race (#27 Bug C + the Opus-found bug). Partial-splice's
analogous specific hazards:

1. **Wrong-content splice from positional zip (silent KV corruption — worst kind, mirrors
   Bug C).** If prefetched blocks are matched to destinations by list position instead of
   `OffloadKey` identity, the model reads another block's KV as if it were this position's.
   No crash, wrong output far downstream. **Guard:** identity map only (Q1 step 1-3); add a
   unit test that feeds a *deliberately relevance-reordered* `prefetch.keys` and asserts
   each destination receives the block whose key matches, never the positionally-parallel
   one.

2. **Double-write race on a destination block (silent corruption).** If any block is in
   *both* the splice set and the remainder reload set, the async CPU→GPU DMA (completes
   across later steps) races the synchronous splice into the same physical block. **Guard:**
   the partition in Q1-step-3 is by construction disjoint (each position goes to exactly one
   set); add an explicit `assert set(splice_dst).isdisjoint(remainder_dst)` before emitting,
   and register both dst sets in `_current_batch_allocated_block_ids` (the store fence,
   `scheduler.py:1107-1119`) as the code already does for the splice dst (`connector.py:486`).

3. **`block_size_factor > 1` misalignment (silent corruption).** A scattered remainder
   reload assumes bsf==1; at bsf>1 the worker's `block_idx % block_size_factor` skip logic
   assumes logically-contiguous blocks and would read/write wrong sub-block offsets.
   **Guard:** `if len(self.config.kv_group_configs) != 1 or self.config.block_size_factor != 1:`
   fall through to full normal reload + `_release_prefetch` (strictly widen the existing
   single-group guard at `connector.py:404`).

4. **Double ref-count decrement on covered keys (mirrors the bug avoided at
   `connector.py:493-501`).** The prefetch job already called `manager.complete_load` on
   `prefetch.keys` automatically when it completed (via the generic completion loop,
   `scheduler.py:1200`). The splice must therefore *not* be given a `TransferJob` and must
   *not* appear in the remainder reload's `keys` set. **Guard:** the remainder load job's
   `keys = set(remainder_keys)` only; splice records nothing in `self._jobs`. Never call
   `complete_load` manually (keep the existing discipline).

5. **`_blocks_being_loaded` leak (liveness bug, not corruption).** Base
   `update_state_after_alloc` adds all `keys_to_load` to `self._blocks_being_loaded`
   (`scheduler.py:844-845`); entries are removed only when a load job completes
   (`scheduler.py:1201-1202`). If covered (spliced, un-reloaded) keys are added there, they
   are never removed → future requests sharing those keys get needlessly deferred forever
   by `_lookup`'s `_blocks_being_loaded` check (`scheduler.py:646-669`). **Guard:** add only
   `remainder_keys` to `_blocks_being_loaded` (falls out naturally from constructing the
   reduced job by hand rather than calling super()).

6. **Freeing prefetch source blocks before the worker splice reads them (silent
   corruption — inherited from the current exact-match path, must be preserved carefully).**
   `update_state_after_alloc` calls `_release_prefetch` (returns src GPU blocks to the free
   pool) *before* the worker's `splice_gpu_blocks` reads them later the same step. This is
   safe today only because (a) the splice runs in `bind_connector_metadata` **before**
   `super().bind_connector_metadata()`'s `start_kv_transfers` (`connector.py:639-647`), so no
   same-step DMA writes a reused src block ahead of the splice, and (b) the step's forward
   pass runs after the splice. Partial-splice keeps the same ordering; **do not** reorder
   `_release_prefetch` ahead of recording the splice job, and keep the splice-before-loads
   ordering in `bind_connector_metadata`. Flagged as an open verification item (§Open risks)
   because it was never independently stress-tested even for the exact-match path.

---

## Q4 — Value, and sequencing relative to the `_TOP_M` fix

**Observation that changes the picture:** the current working tree already has
`worker.py:59` `_TOP_M = 64` (not 8). This is uncommitted and post-dates issues log entry
#30 (worker.py mtime 2026-07-17 22:41, after entry #30 was written referencing `_TOP_M=8`) —
i.e. the parallel supply-side workstream has already raised it. Treat the supply side as
*in progress / partially landed*, not hypothetical.

**Value reasoning:**
- With the old `_TOP_M=8`, a short request accumulated a score on only ~1-2 of its 5-20
  blocks (entries #29-30), so even a correct partial-splice would save ~1-2 blocks of
  CPU→GPU DMA per preemption — real but marginal.
- With `_TOP_M` raised (now 64), many more of a request's blocks earn scores → larger
  prefetch coverage → partial-splice can splice most/all of a reload. Supply governs the
  *magnitude* of the win; partial-splice governs whether *any* of it is realizable.
- **Crucial:** raising `_TOP_M` alone yields **zero** splices, because the exact-match check
  is order-sensitive (the Q1 finding) — perfect coverage still fails list equality for
  multi-block requests. So the two fixes are strictly complementary and **neither is
  sufficient alone**: `_TOP_M` without partial-splice = better supply feeding a matcher that
  still rejects everything; partial-splice without `_TOP_M` = a working matcher with almost
  nothing to match.

**Sequencing: build partial-splice NOW, alongside the (already-in-progress) `_TOP_M` raise.**
Partial-splice is the binding constraint on observing the first non-zero `SPLICED` event and
is independently correct regardless of `_TOP_M`'s value. Do not gate it behind the supply
fix; do measure them together (the acceptance check below only becomes non-trivially positive
once both are in). Report benefit as a **graded** metric (blocks spliced / blocks that would
have been reloaded), per entry #29's recommendation — not binary hit/miss.

---

## Implementation steps (in order)

All changes are in `semantic_offload/` (ext repo). No new vLLM source edits. Single-group,
bsf==1 scope; everything else falls through unchanged.

1. **Refactor the block/key computation out of `_try_splice_prefetch`.** Extract a helper
   `_compute_load_plan(request, blocks, num_external_tokens) -> (keys_to_load, dst_block_ids)`
   that reproduces the base positional computation (`scheduler.py:777-813`) for the single
   group. Both the splice partition and the remainder reload consume it, guaranteeing they
   agree on ordering. Assert `len(keys_to_load) == len(dst_block_ids)`.

2. **Replace exact-match with identity partition in `_try_splice_prefetch`.** Delete the
   `if keys_to_load != prefetch.keys: return False` list-equality (`connector.py:461-472`).
   Build `prefetch_src = dict(zip(prefetch.keys, prefetch.gpu_block_ids))`, then partition by
   `keys_to_load[i] in prefetch_src`. Produce `splice_pairs: list[(src_id, dst_id)]` and
   `remainder: list[(key, dst_id)]`. If `splice_pairs` is empty, return False (nothing to
   splice; let the normal path run) — preserves today's behavior when the prefetch covers
   none of the needed keys.

3. **Emit the splice.** Populate `self._pending_splice_jobs[req_id]` from `splice_pairs`
   (`src_ids`, `dst_ids` — same structure as today, `connector.py:489-492`), register
   `splice_dst` in `_current_batch_allocated_block_ids`, and `assert
   set(splice_dst).isdisjoint(remainder_dst)`.

4. **Emit the reduced remainder reload (only if `remainder` is non-empty).** Mirror
   `scheduler.py:823-845` for the single group: `src_spec =
   manager.prepare_load(remainder_keys, req_context)`; `dst_spec =
   GPULoadStoreSpec(remainder_dst, group_sizes=[len(remainder)], block_indices=[start_idx])`;
   new `job_id`; `assert not req_status.transfer_jobs`; add to
   `self._current_batch_load_jobs`, `req_status.transfer_jobs`, `self._jobs`
   (`TransferJobStatus(..., keys=set(remainder_keys), is_store=False)`); add `remainder_keys`
   (only) to `self._blocks_being_loaded` if enabled. Set
   `group_state.next_stored_block_idx = num_blocks` for BLOCK_LEVEL exactly as base does.

5. **Change `update_state_after_alloc` control flow.** When a prefetch exists and the scope
   preconditions hold, the new `_try_splice_prefetch` becomes authoritative for the whole
   load (it issues both the splice and the remainder reload itself), so on success it must
   `_release_prefetch` and `return` **without** calling `super().update_state_after_alloc`
   (`connector.py:517-521` already does the release+return; extend it so "success" now
   includes the partial case). When preconditions fail or `splice_pairs` is empty, keep the
   current stale-release + `super()` fallthrough (`connector.py:522-532`).

6. **Keep `bind_connector_metadata` ordering.** No change needed — splice already runs before
   `super().bind_connector_metadata()`'s loads (`connector.py:639-647`). Re-confirm the
   remainder reload flows through the normal `load_jobs` path (it does, since step 4 uses
   `_current_batch_load_jobs`), so the worker's stock `start_kv_transfers` handles it and its
   completion reports normally.

7. **Debug markers.** Replace `KEY MISMATCH` with `PARTIAL SPLICE spliced=<n>
   reloaded=<m> covered=<n/(n+m)>` so the graded metric is directly observable, matching the
   existing `PREFETCH_EFFECT_DEBUG` discipline (entries #28-30).

8. **Unit tests** (CPU-only, same `SemanticOffloadingWorker.__new__` / stub-scheduler
   patterns already in the suite):
   - relevance-reordered `prefetch.keys`: each dst gets its identity-matched block (Q3-1).
   - partition disjointness assert holds; splice_dst ∩ remainder_dst == ∅ (Q3-2).
   - bsf!=1 and multi-group both fall through to full reload + prefetch released (Q3-3).
   - remainder-only keys land in the load job's `keys` and in `_blocks_being_loaded`; covered
     keys land in neither (Q3-4, Q3-5).
   - `remainder` empty → behaves like today's full exact-cover splice (regression).
   - full-miss (`splice_pairs` empty) → normal reload, prefetch released.

---

## Acceptance check (real server — the discipline this project holds itself to)

Reuse entry #30's launch config and harness (Qwen2.5-1.5B-Instruct, 2080Ti,
`--num-gpu-blocks-override 200`, `--max-model-len 2048`, ~16 concurrent ~140-token requests
firing a single contended wave to force real preemptions), **with both `_TOP_M` raised and
partial-splice built**. Instrument with the step-7 markers.

**Pass when:** at least one `PARTIAL SPLICE` event reports `spliced >= 1` with `reloaded >= 1`
in the *same* re-admission (proving the two coordinated transfers coexist for one request),
across a run with real preemptions (`vllm:num_preemptions_total` delta > 0), **and** the run
is clean (zero EngineCore crashes, `/health` 200 throughout, sampled outputs coherent
English) — i.e. no regression of the entries #26-27 crash classes. **Stronger pass:** the
aggregate graded metric (Σ spliced / Σ (spliced+reloaded)) is materially > 0 across the run,
and a spot-check confirms a spliced request's output is correct (guards Q3-1 corruption,
which no crash would reveal). Compare CPU-tier `vllm:kv_offload_load_bytes_total` against an
identical `_TOP_M`-raised-but-splice-disabled control run: partial-splice should reduce it by
roughly the spliced-block fraction.

**Fail/stop signals:** any `SPLICED` accompanied by garbled output (Q3-1); a load-completion
assert or KeyError (regression of #26/#27 — audit the whole contract, do not patch one
symptom, per entry #27's lesson); or spliced==0 across the whole run despite raised `_TOP_M`
(means the identity partition or the prefetch supply is still empty — re-check
`top_relevant_keys` coverage before touching the splice code).

---

## Open risks / unknowns not resolvable by static reading

1. **Src-block-free-before-splice ordering (Q3-6).** The safety argument rests on the splice
   running before any same-step DMA that could reuse a just-freed prefetch src block, and
   before the step's forward pass. This holds under the current `bind_connector_metadata`
   ordering but was never independently stress-tested even for the exact-match path (which
   has fired 0 times on the real server, so this ordering has *never actually executed* end
   to end). First real `SPLICED` events are also the first real test of this ordering —
   watch for corruption specifically here, not just crashes.
2. **~40 lines of duplicated base load-job logic (step 4).** This is a maintenance liability:
   any upstream change to `scheduler.py:823-845` (job registration, `_blocks_being_loaded`
   semantics, `next_stored_block_idx` handling) silently desyncs. Mitigate with a unit test
   that pins the reproduced behavior, but a future vLLM bump must re-diff this region.
3. **`prepare_load` on a scattered remainder subset** is assumed equivalent to the base
   full-slice call at bsf==1. The CPU manager returns source specs in `remainder_keys` order
   and the worker pairs positionally — believed correct from reading `gpu_worker.py:310-360`,
   but not executed. The unit test can only check the scheduler-side spec shape, not the
   actual DMA; the real-server check is the true proof.
4. **Whether raised `_TOP_M` actually delivers enough coverage** for partial-splice to show a
   *material* graded win (vs. a token 1-block splice) is a supply-side empirical question
   owned by the parallel workstream — partial-splice's correctness does not depend on it, but
   its reported value does.
5. **Interaction with concurrent preemptions sharing the 5% budget** (multiple requests each
   getting a small partial prefetch) is untested; the budget accounting
   (`_prefetch_reserved_blocks`) is unchanged, but graded-benefit under many simultaneous
   partial splices is only observable on the real server.
