# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingConnector: carries relevance-score plumbing alongside
vLLM's stock OffloadingConnector.

Two extra channels, piggybacking on the existing worker<->scheduler metadata
protocol (see .claude/docs/semantic-eviction-plan.md, Step 1.3):

- scheduler -> worker: which OffloadKey(s) each store job_id represents
  (`SemanticOffloadingConnectorMetadata.store_job_keys`). The worker needs
  this to re-key its summaries by the stable OffloadKey identity instead of
  the transient GPU block_id they're computed from -- GPU blocks get reused
  for unrelated content once freed, but the CPU-tier block (and the need to
  score it) long outlives that. Granularity: job-level, not per-block within
  a job -- a store job's keys are attributed to all blocks in that job. This
  is a known simplification (see issues log); most store jobs in practice
  cover few blocks (offload is opportunistic, per-prefill-step), so the blur
  is expected to be minor.
- worker -> scheduler: computed relevance scores
  (`SemanticWorkerMetadata.pending_scores`), consumed by
  `SemanticOffloadingManager.update_relevance()` for the EMA-smoothed
  relevance state Step 1.4 will read from.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    OffloadPolicy,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.request import Request

# method -> req_id -> list[(offload_key, score)]
RelevanceScores = dict[str, dict[str, list[tuple[OffloadKey, float]]]]

# Step 1.5 (semantic prefetch, see .claude/docs/semantic-eviction-plan.md and
# issues log entries #23-#24). Bound on total GPU blocks any request's
# speculative prefetch reservation is allowed to hold, summed across all
# concurrently-preempted requests -- the plan's own "≤5% of GPU blocks"
# text, so a burst of preemptions can't starve real scheduling of blocks.
PREFETCH_BUDGET_FRACTION = 0.05


@dataclass
class _PrefetchState:
    """Scheduler-side record of one request's in-flight or completed
    speculative prefetch, dispatched through the same real `load_jobs`
    channel as a normal load (verified safe in issues log entry #23) and
    ALSO registered in `req_status.transfer_jobs` like a normal load is --
    see the registration site in `_attempt_prefetch_reservation` for why
    (issues log entry #26: not registering there crashes the engine, and
    registering it is safe because `get_num_new_matched_tokens` already
    defers re-admission while it's non-empty). This struct is our own
    lightweight index into that job (which keys, which GPU block ids) for
    the splice step -- it doesn't duplicate or replace the real
    `TransferJobStatus`/`req_status.transfer_jobs` bookkeeping."""

    job_id: int
    keys: list[OffloadKey]
    gpu_block_ids: list[int]


@dataclass
class SemanticOffloadingConnectorMetadata(OffloadingConnectorMetadata):
    store_job_keys: dict[int, list[OffloadKey]] = field(default_factory=dict)
    # req_id -> (src GPU block ids holding prefetched data, dst GPU block
    # ids the scheduler just officially allocated for the same keys).
    splice_jobs: dict[str, tuple[list[int], list[int]]] = field(default_factory=dict)
    # issues log entry #27: prefetch loads are dispatched through a SEPARATE
    # channel from `load_jobs`, not merged into it. The stock worker's
    # start_kv_transfers() registers every job in `load_jobs` into
    # self._load_jobs, which makes get_finished() report its completion via
    # `finished_recving` -- a channel the core scheduler only understands
    # for two cases (a request WAITING_FOR_REMOTE_KVS, or already finished).
    # A prefetch's request is PREEMPTED, neither of those, so merging it
    # into `load_jobs` crashes the engine (`assert
    # RequestStatus.is_finished(req.status)`) the moment the prefetch
    # completes -- reproduced on the real server. Keyed by job id, same
    # shape as `load_jobs`; SemanticOffloadingConnectorWorker.start_kv_transfers
    # submits these for real execution without registering them in
    # self._load_jobs, so their completion is still reported (unconditionally,
    # via completed_jobs) but never through finished_recving.
    prefetch_load_jobs: dict[int, TransferJob] = field(default_factory=dict)


@dataclass
class SemanticWorkerMetadata(OffloadingWorkerMetadata):
    pending_scores: RelevanceScores = field(default_factory=dict)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)
        base = super().aggregate(other)
        merged_scores: RelevanceScores = {
            method: dict(reqs) for method, reqs in self.pending_scores.items()
        }
        for method, reqs in getattr(other, "pending_scores", {}).items():
            merged_scores.setdefault(method, {}).update(reqs)
        return SemanticWorkerMetadata(
            completed_jobs=base.completed_jobs,
            transfer_stats=base.transfer_stats,
            pending_scores=merged_scores,
        )


class SemanticOffloadingConnectorScheduler(OffloadingConnectorScheduler):
    def __init__(self, spec):
        super().__init__(spec)
        self._pending_job_keys: dict[int, list[OffloadKey]] = {}
        # Step 1.5 (semantic prefetch) needs a live reference to the
        # scheduler's GPU BlockPool to reserve a small budget of blocks
        # ahead of a preempted request's official resumption. Empirically
        # verified safe on the real server (issues log entry #23): calling
        # `gpu_block_pool.get_new_blocks(n)`/`free_blocks(...)` from here
        # correctly moves the shared `free_block_queue` count that
        # `KVCacheManager`'s own admission checks read -- no desync, no
        # corruption, confirmed under real concurrent request load. This
        # reference is the real feature's foundation; the verification
        # probe itself (reserve N, assert count, release, assert restored)
        # was temporary and has been removed after confirming the result.
        self.gpu_block_pool: "BlockPool | None" = None
        # req_id -> in-flight-or-ready prefetch. See _PrefetchState's
        # docstring for why this is separate from req_status.transfer_jobs.
        self._prefetched: dict[str, _PrefetchState] = {}
        self._prefetch_reserved_blocks = 0
        self._pending_splice_jobs: dict[str, tuple[list[int], list[int]]] = {}
        # Step 1.5 retry (issues log entry #25): on_request_preempted only
        # fires ONCE, at the instant a request is preempted -- entry #24
        # found that instant structurally has zero free GPU blocks under
        # every sustained-pressure workload tested (vLLM frees just enough
        # blocks for whatever admission triggered the preemption, never
        # more). A request that failed its first reservation attempt stays
        # here and gets retried every subsequent scheduling step (from
        # build_connector_meta, which fires unconditionally every step)
        # until it either succeeds, gets re-admitted (update_state_after_alloc
        # clears it), or finishes/aborts (request_finished clears it) --
        # giving slack that appears *after* the preemption instant (e.g. an
        # unrelated request completing a moment later) a real chance to be
        # used, instead of only the single worst possible instant.
        self._preempted_pending: set[str] = set()
        # Fix A's dispatch-side counterpart: prefetch jobs are batched
        # separately from self._current_batch_load_jobs (which becomes
        # `load_jobs` -- see SemanticOffloadingConnectorMetadata.prefetch_load_jobs
        # for why), reset every step in build_connector_meta the same way.
        self._current_batch_prefetch_load_jobs: dict[int, TransferJob] = {}

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self.gpu_block_pool = gpu_block_pool

    def on_request_preempted(self, request: "Request") -> None:
        """Step 1.5: queue a just-preempted request for a prefetch
        reservation attempt -- the actual attempt always happens from
        _retry_pending_prefetches (issues log entry #27), never
        synchronously here. This is NOT just style: `build_connector_meta`'s
        base class flushes any *store* job for a request appearing in
        `scheduler_output.preempted_req_ids` this same step and asserts
        every flushed job `is_store` (offloading/scheduler.py ~line 1104).
        A prefetch is a *load*. If a reservation succeeded synchronously
        here and the request were still in this step's preempted set by
        the time that flush runs, it would trip that assert and crash the
        engine -- not yet observed live only because the tested workloads
        happened to never have free blocks at the preemption instant
        itself (issues log entry #24), not because it can't happen."""
        self._preempted_pending.add(request.request_id)

    def _retry_pending_prefetches(self, scheduler_output: SchedulerOutput) -> None:
        """Called every scheduling step (from build_connector_meta), for
        every request queued by on_request_preempted or a previously-failed
        retry. Requests preempted THIS SAME step are skipped (see
        on_request_preempted's docstring) -- by the next step they're no
        longer in `scheduler_output.preempted_req_ids`, and
        get_num_new_matched_tokens's existing "defer while transfer_jobs is
        non-empty" guard (offloading/scheduler.py ~line 718) prevents them
        from being re-preempted while a prefetch job is pending, so a
        request can only ever be in `preempted_req_ids` again after its
        prior prefetch has fully completed and cleared."""
        if not self._preempted_pending:
            return
        this_step_preempted = scheduler_output.preempted_req_ids or ()
        succeeded = [
            req_id
            for req_id in self._preempted_pending
            if req_id not in this_step_preempted
            and self._attempt_prefetch_reservation(req_id)
        ]
        self._preempted_pending.difference_update(succeeded)

    def _attempt_prefetch_reservation(self, req_id: str) -> bool:
        """One attempt to reserve GPU blocks and start loading `req_id`'s
        highest-relevance CPU-tier content. Returns True on success (or if
        there was nothing to do -- e.g. the request already has a prefetch,
        or genuinely has no candidate keys at all -- so the caller should
        NOT keep retrying it), False if it should be retried later because
        the only blocker was transient (no free GPU blocks right now)."""
        if self.gpu_block_pool is None:
            return True  # nothing to retry towards; not a real "no keys" case
        if req_id in self._prefetched:
            return True  # already have one in flight or ready
        req_status = self._req_status.get(req_id)
        if req_status is None:
            return True  # request is gone; nothing left to retry
        candidate_keys = [
            key
            for group_state in req_status.group_states
            for key in group_state.offload_keys
        ]
        if not candidate_keys:
            return True  # never will have keys to prefetch; stop retrying

        total_budget = int(
            self.gpu_block_pool.num_gpu_blocks * PREFETCH_BUDGET_FRACTION
        )
        remaining_budget = max(0, total_budget - self._prefetch_reserved_blocks)
        if remaining_budget <= 0:
            return False  # budget may free up later (another prefetch splices/releases)
        top_keys = self.manager.top_relevant_keys(candidate_keys, remaining_budget)
        # Only keys actually resident in the CPU tier can be loaded --
        # manager.prepare_load() asserts on a missing key rather than
        # skipping it, so this filter has to run first, not be assumed.
        resident_keys = [
            key
            for key in top_keys
            if self.manager.lookup(key, req_status.req_context) == LookupResult.HIT
        ]
        if not resident_keys:
            # A scored-but-not-yet-resident or not-yet-scored key may become
            # eligible on a later step (e.g. its own store job finishes, or
            # a later query scores it) -- worth retrying, not permanent.
            return False
        free_blocks = self.gpu_block_pool.get_num_free_blocks()
        n = min(len(resident_keys), free_blocks)
        if n <= 0:
            # issues log entry #24: this was the consistently-hit case under
            # sustained pressure at the preemption instant itself -- the
            # retry loop exists specifically to give a LATER step (when
            # something else frees blocks) a chance this single-shot
            # version never had.
            return False
        resident_keys = resident_keys[:n]

        reserved = self.gpu_block_pool.get_new_blocks(n)
        gpu_block_ids = [b.block_id for b in reserved]
        # issues log entry #27 (Fix C): without this, a just-reallocated
        # block that used to be a finished request's pending GPU->CPU store
        # source wouldn't be recognized by build_connector_meta's
        # store-flush fence (offloading/scheduler.py ~line 1107-1119, which
        # only checks blocks recorded here) -- the prefetch's CPU->GPU
        # write could then race an in-flight store still reading the same
        # physical block, silently corrupting the offloaded copy. No crash,
        # wrong output -- caught by tracing the fence's real precondition,
        # not by a test failing.
        self._current_batch_allocated_block_ids.update(
            bid for bid in gpu_block_ids if bid != 0
        )
        src_spec = self.manager.prepare_load(resident_keys, req_status.req_context)
        dst_spec = GPULoadStoreSpec(gpu_block_ids, group_sizes=[n], block_indices=[0])
        job_id = self._generate_job_id()
        # issues log entry #27 (Fix A): dispatched via the SEPARATE
        # prefetch_load_jobs channel, not load_jobs -- see
        # SemanticOffloadingConnectorMetadata.prefetch_load_jobs's docstring.
        self._current_batch_prefetch_load_jobs[job_id] = TransferJob(
            req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
        )
        self._jobs[job_id] = TransferJobStatus(
            req_id=req_id,
            pending_count=self.config.num_workers,
            keys=set(resident_keys),
            is_store=False,
        )
        # issues log entry #26: MUST register here. update_connector_output's
        # generic per-job completion loop unconditionally does
        # `req_status.transfer_jobs.remove(job_id)` for every job in
        # self._jobs, with no exception for jobs that were never added --
        # skipping this crashes the engine with a KeyError the moment this
        # job completes (confirmed on the real server). The original design
        # deliberately avoided this to dodge update_state_after_alloc's
        # `assert not req_status.transfer_jobs`, but that fear was
        # unfounded: get_num_new_matched_tokens already defers re-admission
        # whenever `req_status.transfer_jobs` is non-empty ("Delaying
        # request ... since it still has in-flight transfers"), so a
        # request can never reach update_state_after_alloc while this job
        # is still pending -- the existing framework already provides
        # exactly the protection the separate-tracking design was trying to
        # invent from scratch.
        req_status.transfer_jobs.add(job_id)
        self._prefetch_reserved_blocks += n
        self._prefetched[req_id] = _PrefetchState(
            job_id=job_id, keys=resident_keys, gpu_block_ids=gpu_block_ids
        )
        print(
            f"PREFETCH_EFFECT_DEBUG RESERVED req={req_id} n={n} job_id={job_id}",
            flush=True,
        )
        return True

    def _release_prefetch(self, req_id: str) -> None:
        """Return a request's reserved-but-unused/no-longer-needed prefetch
        blocks to the free pool. Safe to call for a request with no
        prefetch state (no-op)."""
        state = self._prefetched.pop(req_id, None)
        if state is None or self.gpu_block_pool is None:
            return
        self._prefetch_reserved_blocks -= len(state.gpu_block_ids)
        blocks = [self.gpu_block_pool.blocks[bid] for bid in state.gpu_block_ids]
        self.gpu_block_pool.free_blocks(blocks)

    def request_finished(self, request: "Request"):
        # A request can finish (or abort) while still preempted, in which
        # case a still-pending retry attempt would otherwise loop forever --
        # this is the one lifecycle point that always fires before a
        # request's state is torn down, regardless of how it ends.
        self._preempted_pending.discard(request.request_id)
        state = self._prefetched.get(request.request_id)
        if state is not None:
            job_status = self._jobs.get(state.job_id)
            if job_status is None or job_status.pending_count == 0:
                # The transfer already finished (or was already cleaned up)
                # -- safe to free these GPU blocks right now, nothing is
                # still writing into them.
                self._release_prefetch(request.request_id)
            # else: the prefetch's load job is still actively transferring
            # data into these reserved blocks. Freeing them now would let
            # something else reuse that GPU memory while the worker is
            # still writing to it -- a real data race, not a cleanup nit.
            # Left in self._prefetched; update_connector_output's orphan
            # sweep below releases it once the job's own completion is
            # observed, mirroring the base class's own discipline of
            # keeping req_status alive while any of its jobs are in flight
            # rather than tearing it down eagerly.
        return super().request_finished(request)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        super().update_connector_output(connector_output)
        # A prefetch whose owning request already finished/aborted while
        # its load job was still in flight has no one left to ever splice
        # it in (update_state_after_alloc, the only other place that
        # releases a prefetch, can't fire for a request that's gone).
        # req_status.transfer_jobs.remove()/req_status deletion (both in the
        # super() call just above) only happen once a job's completion is
        # actually processed, so `req_id not in self._req_status` here means
        # "this request is finished AND every one of its jobs, including
        # this prefetch, has genuinely completed" -- not just "the request
        # is gone", so this can't race the still-in-flight transfer the
        # same way freeing it in request_finished directly could have.
        orphaned = [
            req_id for req_id in self._prefetched if req_id not in self._req_status
        ]
        for req_id in orphaned:
            self._release_prefetch(req_id)

    def _compute_load_plan(
        self, request: "Request", blocks, num_external_tokens: int
    ) -> tuple[list[OffloadKey], list[int], int, int] | None:
        """Reproduces the base connector's single-group positional
        keys-to-load / destination-block computation (mirrors
        `offloading/scheduler.py`'s `update_state_after_alloc`, ~line
        749-821) so that `_try_splice_prefetch`'s splice partition and its
        remainder reload agree on ordering by construction -- both are
        derived from this one call, never recomputed independently.
        Returns `None` if there's nothing pending to load. Also returns
        `num_locally_computed_gpu_blocks` (needed as the remainder reload's
        `block_indices`) and `num_blocks` (needed for
        `next_stored_block_idx` bookkeeping) alongside the base method's
        two return values, since both callers need them."""
        req_status = self._req_status[request.request_id]
        group_config = self.config.kv_group_configs[0]
        group_state = req_status.group_states[0]
        group_blocks = blocks.blocks[0]
        num_cached_tokens = req_status.num_locally_computed_tokens + num_external_tokens
        gpu_block_size = group_config.gpu_block_size
        num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)
        num_locally_computed_gpu_blocks = num_gpu_blocks
        for i, block in enumerate(group_blocks[:num_gpu_blocks]):
            if not block.is_null and block.block_hash is None:
                num_locally_computed_gpu_blocks = i
                break
        num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks
        if num_pending_gpu_blocks == 0:
            return None

        offloaded_block_size = group_config.offloaded_block_size
        num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
        start_block_idx = (
            num_locally_computed_gpu_blocks // self.config.block_size_factor
        )
        keys_to_load = group_state.offload_keys[start_block_idx:num_blocks]
        dst_block_ids = [
            block.block_id
            for block in group_blocks[num_locally_computed_gpu_blocks:num_gpu_blocks]
        ]
        assert len(keys_to_load) == len(dst_block_ids)
        return keys_to_load, dst_block_ids, num_locally_computed_gpu_blocks, num_blocks

    def _try_splice_prefetch(
        self, request: "Request", blocks, num_external_tokens: int
    ) -> bool:
        """Step 1.5: splice whatever fraction of this request's official
        load an already-completed prefetch covers via a fast GPU-to-GPU
        copy, and issue a normal (slower) CPU-tier reload for only the
        complementary remainder in the same step -- no longer all-or-
        nothing (see issues log entries #29-32 for why the original
        exact-match design never actually spliced anything: prefetched
        content is matched to destination blocks by `OffloadKey` identity,
        never by list position, since `keys_to_load` is positionally
        ordered and `prefetch.keys` is relevance-ranked -- two different
        orderings that only coincide by chance). Returns True if it spliced
        anything at all (caller must NOT also call the normal load path for
        this request this step -- the remainder reload, if any, is already
        issued here). Returns False only when the prefetch covers none of
        what's needed, in which case the normal path handles everything.

        Deliberately scoped to the single-KV-group, `block_size_factor==1`
        case (this project's actual test/dev model, Qwen2.5-1.5B-Instruct)
        -- a known, documented simplification, same spirit as
        store_job_keys' job-level granularity. At `block_size_factor > 1` a
        scattered (non-contiguous) remainder reload would compute wrong
        intra-block offsets in the worker's block-skip logic, so those
        requests (and multi-group/sliding-window ones) always fall through
        to the normal path unmodified; their prefetch (if any) is simply
        released as stale by the caller."""
        if len(self.config.kv_group_configs) != 1 or self.config.block_size_factor != 1:
            return False
        prefetch = self._prefetched.get(request.request_id)
        if prefetch is None:
            return False
        print(
            f"PREFETCH_EFFECT_DEBUG {request.request_id}: prefetch exists at "
            f"re-admission, attempting splice",
            flush=True,
        )
        # issues log entry #28: `job_status is None` does NOT mean "not
        # ready" -- it means the OPPOSITE. Reaching this method at all (with
        # num_external_tokens > 0) already implies get_num_new_matched_tokens
        # let re-admission through, which its own unconditional first check
        # (offloading/scheduler.py ~line 717-722: `if req_status.transfer_jobs:
        # return None, False`) would never have done while OUR job's id was
        # still registered there. Since a job's removal from
        # req_status.transfer_jobs happens atomically with `del
        # self._jobs[job_id]` in update_connector_output's completion loop,
        # `self._jobs.get(job_id)` returning None at this point means the
        # job already completed and was cleaned up -- treating that as
        # "not ready" made this method return False on every real call,
        # 9/9 times on the real server, discarding a still-good, already-
        # loaded prefetch every single time. Only a still-registered job
        # with pending_count > 0 is genuinely not ready (defensive case,
        # not expected to occur given the deferral guarantee above, but
        # harmless to keep checking).
        job_status = self._jobs.get(prefetch.job_id)
        if job_status is not None and job_status.pending_count > 0:
            print(
                f"PREFETCH_EFFECT_DEBUG {request.request_id}: NOT READY "
                f"(job_status={job_status})",
                flush=True,
            )
            return False  # prefetch load hasn't completed yet

        req_status = self._req_status[request.request_id]
        plan = self._compute_load_plan(request, blocks, num_external_tokens)
        if plan is None:
            return False
        keys_to_load, dst_block_ids, num_locally_computed_gpu_blocks, num_blocks = plan

        # Identity-based partition (issues log entries #31-32; the mandatory
        # correctness move the partial-splice plan doc identified): a
        # per-key FIFO queue, not a plain dict, so duplicate keys (identical
        # block content recurring within one request's chain) each still
        # get their own distinct prefetched block rather than colliding.
        prefetched_blocks_by_key: dict[OffloadKey, list[int]] = defaultdict(list)
        for key, block_id in zip(prefetch.keys, prefetch.gpu_block_ids):
            prefetched_blocks_by_key[key].append(block_id)

        splice_src: list[int] = []
        splice_dst: list[int] = []
        remainder_keys: list[OffloadKey] = []
        remainder_dst: list[int] = []
        for key, dst in zip(keys_to_load, dst_block_ids):
            bucket = prefetched_blocks_by_key.get(key)
            if bucket:
                splice_src.append(bucket.pop(0))
                splice_dst.append(dst)
            else:
                remainder_keys.append(key)
                remainder_dst.append(dst)

        if not splice_src:
            print(
                f"PREFETCH_EFFECT_DEBUG {request.request_id}: KEY MISMATCH "
                f"spliced=0 keys_to_load={len(keys_to_load)} "
                f"prefetch.keys={len(prefetch.keys)}",
                flush=True,
            )
            return False  # nothing usable here; normal path loads everything

        # Guards against issues log entry #31's Q3-2 hazard (a block ending
        # up in both the splice and the remainder-reload sets, letting an
        # async CPU->GPU DMA race the synchronous splice into the same
        # physical block): the partition above is disjoint by construction
        # (each position goes to exactly one list), asserted here as a
        # cheap, permanent guard rather than trusted implicitly.
        assert set(splice_dst).isdisjoint(remainder_dst)

        group_state = req_status.group_states[0]
        self._current_batch_allocated_block_ids.update(
            bid for bid in splice_dst if bid != 0
        )
        self._pending_splice_jobs[request.request_id] = (splice_src, splice_dst)

        if remainder_keys:
            # A reduced reload for only the complementary remainder --
            # mirrors offloading/scheduler.py's update_state_after_alloc
            # (~line 823-845) for the single-group case, since the base
            # method takes no "load only this subset" parameter. Safe to
            # issue alongside the splice above because the splice itself is
            # untracked (no TransferJob, nothing in req_status.transfer_jobs
            # -- a fire-and-forget index_copy_ run in bind_connector_metadata
            # before any load starts, see issues log entry #31's Q2), so
            # this remains exactly one tracked job per request, identical in
            # kind to today's normal full reload.
            self._current_batch_allocated_block_ids.update(
                bid for bid in remainder_dst if bid != 0
            )
            src_spec = self.manager.prepare_load(remainder_keys, req_status.req_context)
            dst_spec = GPULoadStoreSpec(
                remainder_dst,
                group_sizes=[len(remainder_dst)],
                block_indices=[num_locally_computed_gpu_blocks],
            )
            job_id = self._generate_job_id()
            assert not req_status.transfer_jobs
            self._current_batch_load_jobs[job_id] = TransferJob(
                req_id=request.request_id, src_spec=src_spec, dst_spec=dst_spec
            )
            req_status.transfer_jobs.add(job_id)
            self._jobs[job_id] = TransferJobStatus(
                req_id=request.request_id,
                pending_count=self.config.num_workers,
                keys=set(remainder_keys),
                is_store=False,
            )
            if self._blocks_being_loaded is not None:
                self._blocks_being_loaded.update(remainder_keys)

        # NOTE: no manual manager.complete_load() call here -- checking
        # job_status.pending_count == 0 above already implies
        # update_connector_output's generic, job-id-keyed completion
        # processing (scheduler.py's update_connector_output) has already
        # called it automatically for the PREFETCH job, regardless of
        # whether that job was ever registered in req_status.transfer_jobs.
        # Calling it again here would double-decrement the CPU-tier blocks'
        # ref_cnt -- caught by tracing the real completion path before
        # writing this, not discovered by testing. The remainder reload job
        # created above (if any) is a completely normal tracked job and
        # will get its own automatic complete_load() call when it finishes,
        # same as any other load.
        if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
            group_state.next_stored_block_idx = num_blocks
        covered = len(splice_dst) / (len(splice_dst) + len(remainder_dst))
        print(
            f"PREFETCH_EFFECT_DEBUG {request.request_id}: PARTIAL SPLICE "
            f"spliced={len(splice_dst)} reloaded={len(remainder_dst)} "
            f"covered={covered:.2f}",
            flush=True,
        )
        return True

    def update_state_after_alloc(
        self, request: "Request", blocks, num_external_tokens: int
    ) -> None:
        # Being re-admitted at all means this request is no longer
        # preempted -- stop retrying it regardless of what happens below.
        self._preempted_pending.discard(request.request_id)
        if num_external_tokens > 0 and self._try_splice_prefetch(
            request, blocks, num_external_tokens
        ):
            self._release_prefetch(request.request_id)
            return
        # Prefetch either doesn't exist, isn't ready, or covers none of
        # what's needed this step (a partial cover was already spliced and
        # handled above) -- the normal path below will load everything
        # itself, making the stale prefetch redundant.
        if request.request_id in self._prefetched:
            print(
                f"PREFETCH_EFFECT_DEBUG {request.request_id}: STALE, released "
                f"unspliced, falling back to normal load",
                flush=True,
            )
            self._release_prefetch(request.request_id)
        super().update_state_after_alloc(request, blocks, num_external_tokens)

    def _build_store_jobs(self, scheduler_output: SchedulerOutput):
        store_jobs = super()._build_store_jobs(scheduler_output)
        for job_id in store_jobs:
            self._pending_job_keys[job_id] = list(self._jobs[job_id].keys)
        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # Must run BEFORE super().build_connector_meta(), which reads and
        # resets self._current_batch_load_jobs -- a job a retry adds this
        # step needs to already be in that dict when the base class
        # consumes it, not after. Needs scheduler_output itself (not just
        # its eventual effects) to know which requests were preempted THIS
        # step -- see on_request_preempted's docstring for why those must
        # be skipped here.
        self._retry_pending_prefetches(scheduler_output)
        base_meta = super().build_connector_meta(scheduler_output)
        assert isinstance(base_meta, OffloadingConnectorMetadata)
        job_keys = self._pending_job_keys
        self._pending_job_keys = {}
        splice_jobs = self._pending_splice_jobs
        self._pending_splice_jobs = {}
        prefetch_load_jobs = self._current_batch_prefetch_load_jobs
        self._current_batch_prefetch_load_jobs = {}
        return SemanticOffloadingConnectorMetadata(
            load_jobs=base_meta.load_jobs,
            store_jobs=base_meta.store_jobs,
            jobs_to_flush=base_meta.jobs_to_flush,
            store_job_keys=job_keys,
            splice_jobs=splice_jobs,
            prefetch_load_jobs=prefetch_load_jobs,
        )


class SemanticOffloadingConnectorWorker(OffloadingConnectorWorker):
    """issues log entry #27 (Fix A): submits prefetch load jobs for real
    execution via the same `self.worker.submit_load()` the stock class uses
    for normal loads, but WITHOUT registering them in `self._load_jobs`.
    That dict is what drives `get_finished()`'s `finished_recving`
    reporting -- which the core scheduler only knows how to interpret for a
    request that's `WAITING_FOR_REMOTE_KVS` or already finished, neither of
    which a preempted prefetch's request is. Their completion is still
    reported normally through `completed_jobs` (`mark_completed()` in
    `get_finished()` is unconditional, independent of `_load_jobs`), which
    is all `SemanticOffloadingConnectorScheduler`'s own bookkeeping
    (`_jobs[job_id].pending_count`, `_try_splice_prefetch`) needs."""

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata) -> None:
        super().start_kv_transfers(metadata)
        prefetch_load_jobs = getattr(metadata, "prefetch_load_jobs", None) or {}
        for job_id, entry in prefetch_load_jobs.items():
            assert self.worker is not None
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success


class SemanticOffloadingConnector(OffloadingConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        # Deliberately skip OffloadingConnector.__init__ (calling
        # KVConnectorBase_V1.__init__ directly instead): the base __init__
        # would construct its own OffloadingConnectorScheduler around a spec
        # instance we'd then discard in favor of our own scheduler subclass.
        # OffloadingConnectorScheduler.__init__ eagerly calls
        # spec.get_manager(), so doing that twice means constructing (and
        # throwing away) a whole extra SemanticOffloadingManager -- wasteful,
        # and avoidable by just not calling the redundant base constructor.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)
        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = SemanticOffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = SemanticOffloadingConnectorWorker(spec)

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if isinstance(self.connector_scheduler, SemanticOffloadingConnectorScheduler):
            self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def on_request_preempted(self, request: "Request") -> None:
        # The core scheduler calls this on the top-level connector (this
        # class), not on connector_scheduler directly -- same delegation
        # requirement as bind_gpu_block_pool above. Missing this the first
        # time round meant on_request_preempted silently no-op'd via the
        # inherited KVConnectorBase_V1 default despite real preemptions
        # happening -- caught by seeing zero debug output against a nonzero
        # vllm:num_preemptions_total on the real server, not by reasoning
        # about the code.
        if isinstance(self.connector_scheduler, SemanticOffloadingConnectorScheduler):
            self.connector_scheduler.on_request_preempted(request)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if self.connector_worker is not None and isinstance(
            connector_metadata, SemanticOffloadingConnectorMetadata
        ):
            worker = self.connector_worker.worker
            if hasattr(worker, "receive_job_keys"):
                worker.receive_job_keys(connector_metadata.store_job_keys)
            if hasattr(worker, "splice_gpu_blocks"):
                # Runs before super().bind_connector_metadata() below (which
                # hands off the load_jobs that would otherwise redundantly
                # reload the same keys -- but a spliced request has none,
                # since _try_splice_prefetch substitutes for the normal
                # load path entirely rather than running alongside it).
                for src_ids, dst_ids in connector_metadata.splice_jobs.values():
                    worker.splice_gpu_blocks(src_ids, dst_ids)
        super().bind_connector_metadata(connector_metadata)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        base_meta = super().build_connector_worker_meta()
        assert self.connector_worker is not None
        worker = self.connector_worker.worker
        scores: RelevanceScores = {}
        if hasattr(worker, "pop_pending_scores"):
            scores = worker.pop_pending_scores()
        if not scores:
            return base_meta
        base_meta = base_meta or OffloadingWorkerMetadata()
        assert isinstance(base_meta, OffloadingWorkerMetadata)
        return SemanticWorkerMetadata(
            completed_jobs=base_meta.completed_jobs,
            transfer_stats=base_meta.transfer_stats,
            pending_scores=scores,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        meta = connector_output.kv_connector_worker_meta
        scores = getattr(meta, "pending_scores", None)
        super().update_connector_output(connector_output)
        if scores and self.connector_scheduler is not None:
            manager = self.connector_scheduler.manager
            if hasattr(manager, "update_relevance"):
                manager.update_relevance(scores)
