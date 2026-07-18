# SPDX-License-Identifier: Apache-2.0
"""Step 1.5 (semantic prefetch) acceptance check: the pure-logic pieces --
top-K relevance selection (manager), the GPU-to-GPU splice primitive
(worker), and the retry-bookkeeping loop (issues log entry #25) that lets a
preempted request whose first reservation attempt fails (no free GPU blocks
right then) get retried on later scheduling steps instead of being given up
on permanently. The rest of the scheduler-side integration
(on_request_preempted's real vLLM req_status wiring, the real block-
reservation/dispatch machinery) is integration-heavy against vLLM's live
scheduler internals and is verified end-to-end on the real server instead
(issues log entries #23-#24), the same split this project used for Step
1.2/1.3's hook-point mechanics.
"""

from types import SimpleNamespace

import torch
from semantic_offload.connector import (
    SemanticOffloadingConnectorScheduler,
    _PrefetchState,
)
from semantic_offload.manager import SemanticOffloadingManager
from semantic_offload.worker import SemanticOffloadingWorker

from vllm.v1.kv_offload.base import (
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus


def to_key(n: int):
    return make_offload_key(str(n).encode(), 0)


def test_top_relevant_keys_picks_highest_scored_within_budget():
    manager = SemanticOffloadingManager(num_blocks=10)
    ema = manager.relevance_ema.setdefault("minmax", {})
    keys = [to_key(i) for i in range(5)]
    for i, key in enumerate(keys):
        ema[key] = float(i)  # key i has score i -- key 4 highest

    top = manager.top_relevant_keys(keys, k=2, method="minmax")

    assert top == [keys[4], keys[3]]


def test_top_relevant_keys_skips_unscored_candidates():
    manager = SemanticOffloadingManager(num_blocks=10)
    ema = manager.relevance_ema.setdefault("minmax", {})
    scored, unscored = to_key(1), to_key(2)
    ema[scored] = 5.0
    # `unscored` deliberately has no entry -- a preempted request's block
    # may not have earned a score yet, and there's nothing to prefer.

    top = manager.top_relevant_keys([scored, unscored], k=5, method="minmax")

    assert top == [scored]


def test_top_relevant_keys_budget_zero_returns_nothing():
    manager = SemanticOffloadingManager(num_blocks=10)
    manager.relevance_ema.setdefault("minmax", {})[to_key(1)] = 1.0

    assert manager.top_relevant_keys([to_key(1)], k=0, method="minmax") == []


def _make_layer(num_blocks, num_kv_heads, block_size, head_size, seed):
    torch.manual_seed(seed)
    kv_cache = torch.randn(num_blocks, num_kv_heads, block_size, 2 * head_size)
    return SimpleNamespace(kv_cache=kv_cache)


def _make_worker(layers: dict) -> SemanticOffloadingWorker:
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._attention_layers = layers
    return worker


def test_splice_gpu_blocks_copies_content_to_new_indices():
    layer = _make_layer(num_blocks=6, num_kv_heads=2, block_size=4, head_size=8, seed=0)
    worker = _make_worker({"layer0": layer})
    original_src = layer.kv_cache[2].clone()

    worker.splice_gpu_blocks(src_block_ids=[2], dst_block_ids=[5])

    assert torch.allclose(layer.kv_cache[5], original_src)


def test_splice_gpu_blocks_multiple_layers_and_blocks():
    layer_a = _make_layer(6, 2, 4, 8, seed=1)
    layer_b = _make_layer(6, 2, 4, 8, seed=2)
    worker = _make_worker({"a": layer_a, "b": layer_b})
    expected_a = layer_a.kv_cache[[1, 3]].clone()
    expected_b = layer_b.kv_cache[[1, 3]].clone()

    worker.splice_gpu_blocks(src_block_ids=[1, 3], dst_block_ids=[4, 5])

    assert torch.allclose(layer_a.kv_cache[[4, 5]], expected_a)
    assert torch.allclose(layer_b.kv_cache[[4, 5]], expected_b)


def test_splice_gpu_blocks_empty_is_a_noop():
    layer = _make_layer(4, 1, 4, 8, seed=3)
    worker = _make_worker({"layer0": layer})
    before = layer.kv_cache.clone()

    worker.splice_gpu_blocks(src_block_ids=[], dst_block_ids=[])

    assert torch.allclose(layer.kv_cache, before)


def test_splice_gpu_blocks_mismatched_lengths_raises():
    layer = _make_layer(4, 1, 4, 8, seed=4)
    worker = _make_worker({"layer0": layer})
    raised = False
    try:
        worker.splice_gpu_blocks(src_block_ids=[0, 1], dst_block_ids=[2])
    except AssertionError:
        raised = True
    assert raised


class _StubBlockPool:
    """Minimal stand-in for vllm's real BlockPool -- only the surface
    _attempt_prefetch_reservation actually touches. get_new_blocks() mirrors
    the real one's contract (returns objects with .block_id, decrements the
    free count) closely enough for this bookkeeping-focused test; the real
    BlockPool itself was already verified safe under real concurrent load in
    issues log entry #23."""

    def __init__(self, num_gpu_blocks: int, free_blocks: int):
        self.num_gpu_blocks = num_gpu_blocks
        self._free = free_blocks
        # Real BlockPool reserves block_id 0 as a null block, never handed
        # out for real content -- start at 1 to match, since production
        # code filters `bid != 0` when recording allocated block ids.
        self._next_id = 1

    def get_num_free_blocks(self) -> int:
        return self._free

    def get_new_blocks(self, n: int):
        assert n <= self._free
        self._free -= n
        blocks = []
        for _ in range(n):
            blocks.append(SimpleNamespace(block_id=self._next_id))
            self._next_id += 1
        return blocks

    def free_blocks(self, blocks) -> None:
        self._free += len(list(blocks))


def _insert_resident(manager: SemanticOffloadingManager, key, block_id: int) -> None:
    block = BlockStatus(block_id=block_id)
    block.ref_cnt = 0  # ready
    manager._policy.insert(key, block)
    manager._policy.mark_evictable(key)
    # Bypassing the manager's real prepare_store()->insert() completion path
    # (there's no lighter-weight public API to mark a block resident+
    # evictable for testing), so its private evictable-block counter has to
    # be kept in sync by hand too, or prepare_load()'s own internal
    # bookkeeping assertion fails on the next real insert/evict cycle.
    manager._num_evictable_cache_blocks += 1


def _make_connector_scheduler(
    manager: SemanticOffloadingManager,
    gpu_block_pool: _StubBlockPool,
    *,
    num_groups: int = 1,
    block_size_factor: int = 1,
) -> SemanticOffloadingConnectorScheduler:
    sched = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    sched.manager = manager
    sched.gpu_block_pool = gpu_block_pool
    sched._prefetched = {}
    sched._prefetch_reserved_blocks = 0
    sched._preempted_pending = set()
    sched._req_status = {}
    sched._current_batch_load_jobs = {}
    sched._current_batch_prefetch_load_jobs = {}
    sched._current_batch_allocated_block_ids = set()
    sched._pending_splice_jobs = {}
    sched._jobs = {}
    sched._job_counter = 0
    sched._blocks_being_loaded = set()
    # gpu_block_size=offloaded_block_size=1 keeps splice-plan tests'
    # tokens/blocks/keys all in 1:1 correspondence, so a test can build a
    # request needing N blocks by just handing it N keys and N stub GPU
    # blocks -- the arithmetic in _compute_load_plan degenerates cleanly.
    group_config = SimpleNamespace(
        gpu_block_size=1, offloaded_block_size=1, hash_block_size_factor=1
    )
    sched.config = SimpleNamespace(
        num_workers=1,
        kv_group_configs=tuple(group_config for _ in range(num_groups)),
        block_size_factor=block_size_factor,
    )
    return sched


def _add_req_status(
    sched,
    req_id: str,
    keys: list,
    *,
    num_locally_computed_tokens: int = 0,
    policy: OffloadPolicy = OffloadPolicy.BLOCK_LEVEL,
) -> None:
    sched._req_status[req_id] = SimpleNamespace(
        group_states=[SimpleNamespace(offload_keys=keys, next_stored_block_idx=0)],
        req_context=ReqContext(req_id=req_id),
        transfer_jobs=set(),
        num_locally_computed_tokens=num_locally_computed_tokens,
        offloading_context=RequestOffloadingContext(policy=policy),
    )


def _make_pending_blocks(n: int, start_block_id: int = 100):
    """N stub GPU blocks, none locally-cached (`block_hash=None`) -- i.e.
    all N need loading, matching a request re-admitted after preemption
    with nothing yet spliced or reloaded. `blocks.blocks[0]` is the shape
    `_compute_load_plan` reads (one list of block stand-ins per KV group)."""
    group_blocks = [
        SimpleNamespace(block_id=start_block_id + i, is_null=False, block_hash=None)
        for i in range(n)
    ]
    return SimpleNamespace(blocks=[group_blocks])


def test_attempt_prefetch_reservation_stops_retrying_when_no_candidate_keys():
    """A request with no offload keys at all can never gain any -- the
    caller must not keep retrying it forever."""
    manager = SemanticOffloadingManager(num_blocks=10)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=0))
    _add_req_status(sched, "r1", keys=[])

    assert sched._attempt_prefetch_reservation("r1") is True


def test_attempt_prefetch_reservation_retries_when_no_free_blocks():
    """issues log entry #25: a request with real, resident, scored keys but
    zero free GPU blocks right now must be retried later, not given up on."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("minmax", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=0))
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is False
    assert "r1" not in sched._prefetched


def test_attempt_prefetch_reservation_succeeds_once_blocks_free_up():
    """The exact retry scenario issues log entry #25 targets: first attempt
    fails (no free blocks), a later attempt with free blocks available
    succeeds."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("minmax", {})[key] = 1.0
    pool = _StubBlockPool(20, free_blocks=0)
    sched = _make_connector_scheduler(manager, pool)
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is False

    pool._free = 5  # simulate blocks freeing up on a later scheduling step
    assert sched._attempt_prefetch_reservation("r1") is True
    assert "r1" in sched._prefetched


def test_successful_reservation_registers_job_in_req_status_transfer_jobs():
    """issues log entry #26: a dispatched prefetch job MUST be registered in
    req_status.transfer_jobs, or update_connector_output's generic
    completion loop (`req_status.transfer_jobs.remove(job_id)`,
    unconditional for every job in self._jobs) crashes with a KeyError the
    moment the job completes -- confirmed on the real server. This also
    relies on get_num_new_matched_tokens's existing "delay re-admission
    while transfer_jobs is non-empty" guard to stay safe against
    update_state_after_alloc's `assert not req_status.transfer_jobs`."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("minmax", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is True

    job_id = sched._prefetched["r1"].job_id
    assert job_id in sched._req_status["r1"].transfer_jobs
    assert sched._prefetched["r1"].keys == [key]


def test_retry_pending_prefetches_removes_only_the_succeeded_request():
    manager = SemanticOffloadingManager(num_blocks=10)
    key1, key2 = to_key(1), to_key(2)
    _insert_resident(manager, key1, block_id=0)
    _insert_resident(manager, key2, block_id=1)
    manager.relevance_ema.setdefault("minmax", {})[key1] = 1.0
    manager.relevance_ema.setdefault("minmax", {})[key2] = 1.0
    # Only one free block: only one of the two pending requests can succeed.
    pool = _StubBlockPool(20, free_blocks=1)
    sched = _make_connector_scheduler(manager, pool)
    _add_req_status(sched, "r1", keys=[key1])
    _add_req_status(sched, "r2", keys=[key2])
    sched._preempted_pending = {"r1", "r2"}

    sched._retry_pending_prefetches(SimpleNamespace(preempted_req_ids=set()))

    succeeded = sched._preempted_pending ^ {"r1", "r2"}
    assert len(succeeded) == 1
    assert len(sched._preempted_pending) == 1  # the other one stays queued
    assert set(sched._prefetched.keys()) == succeeded


def test_on_request_preempted_queues_for_retry_on_failure():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("minmax", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=0))
    _add_req_status(sched, "r1", keys=[key])

    sched.on_request_preempted(SimpleNamespace(request_id="r1"))

    assert "r1" in sched._preempted_pending
    assert "r1" not in sched._prefetched


def test_on_request_preempted_never_reserves_synchronously():
    """issues log entry #27 (Fix B): even with free blocks available right
    now, on_request_preempted must only queue, never call
    _attempt_prefetch_reservation directly -- reserving synchronously risks
    tripping build_connector_meta's `assert is_store` flush for a request
    still in this same step's preempted_req_ids (a load job would fail that
    assert). The actual attempt only ever happens from
    _retry_pending_prefetches, on a LATER step."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("minmax", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    sched.on_request_preempted(SimpleNamespace(request_id="r1"))

    assert "r1" in sched._preempted_pending
    assert "r1" not in sched._prefetched  # not reserved yet, despite free blocks


def test_retry_pending_prefetches_skips_requests_preempted_this_step():
    """issues log entry #27 (Fix B): a request preempted THIS scheduling
    step must not be attempted even during the retry sweep -- it only
    becomes eligible on a later step, once it's no longer in
    scheduler_output.preempted_req_ids."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("minmax", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])
    sched._preempted_pending = {"r1"}

    sched._retry_pending_prefetches(SimpleNamespace(preempted_req_ids={"r1"}))
    assert "r1" in sched._preempted_pending
    assert "r1" not in sched._prefetched

    # Next step, no longer in preempted_req_ids -- now eligible.
    sched._retry_pending_prefetches(SimpleNamespace(preempted_req_ids=set()))
    assert "r1" not in sched._preempted_pending
    assert "r1" in sched._prefetched


def test_successful_reservation_records_allocated_block_ids():
    """issues log entry #27 (Fix C): reserved blocks must be recorded in
    _current_batch_allocated_block_ids, or build_connector_meta's
    store-flush fence can't see that a block a finished request's pending
    store still reads from was just reallocated to the prefetch -- a silent
    data race, not a crash."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("minmax", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is True

    gpu_block_ids = set(sched._prefetched["r1"].gpu_block_ids)
    assert gpu_block_ids
    assert gpu_block_ids <= sched._current_batch_allocated_block_ids


def test_successful_reservation_dispatches_to_prefetch_channel_not_load_jobs():
    """issues log entry #27 (Fix A): the job must go into
    _current_batch_prefetch_load_jobs, NOT _current_batch_load_jobs -- the
    latter becomes `load_jobs` in the metadata, which the stock worker's
    start_kv_transfers registers into self._load_jobs, causing its
    completion to be reported via finished_recving and crash the engine for
    a PREEMPTED (not WAITING_FOR_REMOTE_KVS, not finished) request."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("minmax", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is True

    job_id = sched._prefetched["r1"].job_id
    assert job_id in sched._current_batch_prefetch_load_jobs
    assert job_id not in sched._current_batch_load_jobs


# --- Partial-splice tests (issues log entries #31-32 + the partial-splice
# plan doc: matching prefetched content to destination blocks by OffloadKey
# identity, never by list position, and coordinating a splice + a reduced
# remainder reload for the same request in one step). `request` is a
# SimpleNamespace with only `.request_id` since that's all
# _try_splice_prefetch reads from it. ---


def test_partial_splice_identity_matches_even_with_reordered_prefetch():
    """Q3-1: `prefetch.keys` is relevance-ranked, `keys_to_load` is
    positional -- a prefetch that fully covers the request but in a
    DIFFERENT order must still splice each key's content into ITS OWN
    destination block, not the positionally-parallel one (the exact bug
    entries #31-32 found and fixed)."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1, k2 = to_key(0), to_key(1), to_key(2)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1, k2])
    # Prefetch covers all three but in reversed (relevance) order: k2's
    # content is in GPU block 52, k1's in 51, k0's in 50.
    sched._prefetched["r1"] = _PrefetchState(
        job_id=1, keys=[k2, k1, k0], gpu_block_ids=[52, 51, 50]
    )
    blocks = _make_pending_blocks(3)  # dst block_ids [100, 101, 102]

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 3) is True
    )

    # keys_to_load is positionally [k0, k1, k2] -> dst [100, 101, 102].
    # k0's real content lives in GPU block 50, k1's in 51, k2's in 52 --
    # NOT the positionally-parallel [52, 51, 50] a naive zip would produce.
    src_ids, dst_ids = sched._pending_splice_jobs["r1"]
    assert (src_ids, dst_ids) == ([50, 51, 52], [100, 101, 102])
    assert "r1" not in sched._current_batch_load_jobs  # full cover, no remainder


def test_partial_splice_splits_into_splice_and_remainder_disjointly():
    """Q3-2: a prefetch covering only SOME of the needed keys must splice
    the covered ones and issue a normal reduced reload for the rest, with
    disjoint destination sets (guards the double-write race hazard)."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1, k2 = to_key(0), to_key(1), to_key(2)
    _insert_resident(manager, k0, block_id=0)
    _insert_resident(manager, k2, block_id=1)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1, k2])
    # Prefetch covers only the MIDDLE key, k1 -- the common real-world case
    # (entries #29-32) where only a fraction of a request's blocks ever
    # earned a relevance score in time.
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[k1], gpu_block_ids=[77])
    blocks = _make_pending_blocks(3)  # dst block_ids [100, 101, 102]

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 3) is True
    )

    src_ids, dst_ids = sched._pending_splice_jobs["r1"]
    assert (src_ids, dst_ids) == ([77], [101])  # k1 spliced into its own dst

    assert len(sched._current_batch_load_jobs) == 1
    (job,) = sched._current_batch_load_jobs.values()
    remainder_dst = list(job.dst_spec.block_ids)
    assert remainder_dst == [100, 102]  # k0, k2's destinations
    assert set(dst_ids).isdisjoint(remainder_dst)

    job_id = next(iter(sched._current_batch_load_jobs))
    assert job_id in sched._req_status["r1"].transfer_jobs
    assert sched._jobs[job_id].keys == {k0, k2}


def test_partial_splice_remainder_keys_tracked_spliced_keys_not():
    """Q3-4/Q3-5: only the REMAINDER keys go into the reduced load job's
    `keys` and `_blocks_being_loaded` -- the spliced key must appear in
    neither (it was never given a TransferJob, so nothing will ever remove
    it from `_blocks_being_loaded` if it were added, leaking a permanent
    false-deferral for any future request sharing that key)."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1 = to_key(0), to_key(1)
    _insert_resident(manager, k1, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[k0], gpu_block_ids=[77])
    blocks = _make_pending_blocks(2)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 2) is True
    )

    assert sched._blocks_being_loaded == {k1}
    assert k0 not in sched._blocks_being_loaded


def test_partial_splice_full_coverage_no_remainder_job_created():
    """Regression: when the prefetch covers everything (today's original
    exact-match case, now just the `remainder` == empty branch of the same
    code path), behavior matches the pre-partial-splice mechanism -- only a
    splice job, no reload job, no transfer_jobs entry."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1 = to_key(0), to_key(1)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1])
    sched._prefetched["r1"] = _PrefetchState(
        job_id=1, keys=[k0, k1], gpu_block_ids=[50, 51]
    )
    blocks = _make_pending_blocks(2)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 2) is True
    )

    assert "r1" in sched._pending_splice_jobs
    assert not sched._current_batch_load_jobs
    assert not sched._req_status["r1"].transfer_jobs


def test_partial_splice_full_miss_returns_false():
    """Regression: a prefetch that covers none of the needed keys must
    leave everything untouched and return False, so the caller falls back
    to releasing it as stale and running a completely normal full reload."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1 = to_key(0), to_key(1)
    other = to_key(99)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[other], gpu_block_ids=[77])
    blocks = _make_pending_blocks(2)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 2) is False
    )

    assert not sched._pending_splice_jobs
    assert not sched._current_batch_load_jobs
    assert not sched._req_status["r1"].transfer_jobs


def test_partial_splice_falls_through_when_block_size_factor_not_one():
    """Q3-3: at block_size_factor > 1 a scattered remainder reload would
    compute wrong intra-block offsets in the worker's block-skip logic --
    must fall through to the normal (unmodified) path instead of
    attempting a partial splice."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1 = to_key(0), to_key(1)
    sched = _make_connector_scheduler(
        manager, _StubBlockPool(20, free_blocks=5), block_size_factor=2
    )
    _add_req_status(sched, "r1", keys=[k0, k1])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[k1], gpu_block_ids=[77])
    blocks = _make_pending_blocks(2)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 2) is False
    )
    assert not sched._pending_splice_jobs


def test_partial_splice_falls_through_when_multi_group():
    """Q3-3: multi-KV-group requests are out of scope (same as the
    original exact-match guard) -- must fall through, not attempt a
    single-group-shaped partition against a multi-group request."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0 = to_key(0)
    sched = _make_connector_scheduler(
        manager, _StubBlockPool(20, free_blocks=5), num_groups=2
    )
    _add_req_status(sched, "r1", keys=[k0])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[k0], gpu_block_ids=[77])
    blocks = _make_pending_blocks(1)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 1) is False
    )
    assert not sched._pending_splice_jobs
