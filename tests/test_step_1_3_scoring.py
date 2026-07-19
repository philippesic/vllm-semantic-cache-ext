# SPDX-License-Identifier: Apache-2.0
"""Step 1.3 acceptance check: live-query scoring against durable summaries,
correlation plumbing, and the manager's EMA -- all independent of the real
attention-op/prepare_inputs monkey-patches (those are verified end-to-end on
the real server instead; see .claude/docs/semantic-eviction-plan.md Step 1.3
and the issues log for that investigation).
"""

import pytest
import torch
from semantic_offload.index import BlockSummary, score
from semantic_offload.manager import SemanticOffloadingManager
from semantic_offload.worker import SemanticOffloadingWorker

from vllm.v1.kv_offload.base import make_offload_key


def _make_worker(method: str = "minmax") -> SemanticOffloadingWorker:
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._probe_layer_name = "layer0"
    worker.summaries = {"layer0": {}}
    worker._pending_job_keys = {}
    worker.durable_summaries = {}
    worker._max_durable_summaries = 10_000  # effectively unbounded for these tests
    worker._pending_scores = {}
    worker._method = method
    worker._stack_cache_dirty = True
    worker._stack_cache_keys = []
    worker._stack_cache = {}
    worker._stack_cache_index = {}
    worker._stack_pending_insert = set()
    worker._stack_pending_remove = set()
    return worker


def _summary(value: float, dim: int = 4) -> BlockSummary:
    v = torch.full((dim,), value)
    return BlockSummary(min=v, max=v, mean=v, mad=torch.zeros(dim))


def to_key(n: int):
    return make_offload_key(str(n).encode(), 0)


def _seed_durable_summaries(worker, mapping) -> None:
    """Test-only shortcut to populate durable_summaries directly (skipping
    the real _durably_key_summaries path some tests don't need to exercise)
    while still marking the keys for the incremental stack-cache sync --
    real code always goes through _durably_key_summaries, which does this
    marking itself; a test that bypasses it must do so explicitly or the
    cache will never pick the entries up (see _mark_inserted_into_stack_cache)."""
    worker.durable_summaries.update(mapping)
    worker._mark_inserted_into_stack_cache(mapping.keys())


def test_durably_key_summaries_aligned_job():
    worker = _make_worker()
    worker.summaries["layer0"][10] = [_summary(1.0), _summary(1.0)]
    worker.summaries["layer0"][11] = [_summary(2.0), _summary(2.0)]
    keys = [to_key(1), to_key(2)]
    worker.receive_job_keys({7: keys})

    worker._durably_key_summaries(7, [10, 11])

    assert set(worker.durable_summaries) == set(keys)
    # durable_summaries[key] is now a list[BlockSummary], one per KV head
    # (not averaged -- entry #9).
    for head_summary in worker.durable_summaries[keys[0]]:
        assert torch.allclose(head_summary.mean, torch.full((4,), 1.0))
    for head_summary in worker.durable_summaries[keys[1]]:
        assert torch.allclose(head_summary.mean, torch.full((4,), 2.0))


def test_durably_key_summaries_mismatched_length_uses_fallback():
    """3 blocks, 1 key (e.g. a job that bundled multiple blocks under one
    key due to grouping) -- falls back to the most-recent block's summary
    rather than dropping the signal or crashing."""
    worker = _make_worker()
    worker.summaries["layer0"][1] = [_summary(1.0)]
    worker.summaries["layer0"][2] = [_summary(2.0)]
    worker.summaries["layer0"][3] = [_summary(3.0)]
    key = to_key(99)
    worker.receive_job_keys({5: [key]})

    worker._durably_key_summaries(5, [1, 2, 3])

    for head_summary in worker.durable_summaries[key]:
        assert torch.allclose(head_summary.mean, torch.full((4,), 3.0))


def test_durably_key_summaries_no_job_keys_is_noop():
    worker = _make_worker()
    worker.summaries["layer0"][1] = [_summary(1.0)]
    worker._durably_key_summaries(123, [1])  # job_id never registered
    assert worker.durable_summaries == {}


def test_durably_key_summaries_prunes_oldest_past_cap():
    """Regression for issues log entries #62/#63: `durable_summaries` grew
    without bound (nothing ever pruned it), and re-stacking the whole thing
    on every store measurably became a quadratic-cost bug at real 7B/B200
    scale. Bound it to `_max_durable_summaries` (the real CPU tier's block
    capacity), FIFO-by-insertion-order."""
    worker = _make_worker()
    worker._max_durable_summaries = 2
    for i in range(4):
        worker.summaries["layer0"][i] = [_summary(float(i))]
        key = to_key(i)
        worker.receive_job_keys({i: [key]})
        worker._durably_key_summaries(i, [i])

    assert len(worker.durable_summaries) == 2
    # The two most-recently-inserted keys survive; the oldest two were
    # pruned first, not an arbitrary subset.
    assert set(worker.durable_summaries) == {to_key(2), to_key(3)}


def test_receive_evicted_keys_drops_exactly_those_keys():
    """Regression for issues log entries #62-64: the real fix. A block the
    manager reports as evicted must be dropped from durable_summaries
    immediately and precisely -- unlike the FIFO cap, a key NOT reported as
    evicted must survive no matter how old it is (the FIFO cap's own known
    failure mode: pruning still-resident, important content just because
    it's old)."""
    worker = _make_worker()
    worker._max_durable_summaries = 10_000
    for i in range(3):
        worker.summaries["layer0"][i] = [_summary(float(i))]
        key = to_key(i)
        worker.receive_job_keys({i: [key]})
        worker._durably_key_summaries(i, [i])
    assert set(worker.durable_summaries) == {to_key(0), to_key(1), to_key(2)}
    worker._stack_cache_dirty = False

    worker.receive_evicted_keys([to_key(0), to_key(2)])

    assert set(worker.durable_summaries) == {to_key(1)}
    assert worker._stack_cache_dirty is True  # a real removal happened


def test_receive_evicted_keys_no_match_is_noop():
    worker = _make_worker()
    worker._max_durable_summaries = 10_000
    worker.durable_summaries[to_key(1)] = [_summary(1.0)]
    worker._stack_cache_dirty = False

    worker.receive_evicted_keys([to_key(99)])  # never resident

    assert set(worker.durable_summaries) == {to_key(1)}
    assert worker._stack_cache_dirty is False  # nothing actually removed


def test_on_query_captured_ranks_needle_highest_only_for_configured_method():
    """Single-KV-head case (durable_summaries[key] = [BlockSummary]); the
    multi-head max-combine path (entry #9) is exercised implicitly since
    len == 1 makes max() a no-op over one element.

    Only the worker's own configured method should ever be scored --
    SemanticPolicy only ever consults its own method's relevance EMA, so
    computing the other two is pure waste (the bug behind the catastrophic
    TTFT found on a real B200 run, entry #53): confirm the other two
    methods' entries don't exist at all, not just that they're empty."""
    for method in ("minmax", "mean", "cuboid_mean"):
        worker = _make_worker(method=method)
        needle_key = to_key(1)
        distractor_keys = [to_key(i) for i in range(2, 6)]

        query = torch.tensor([[5.0, 5.0, 5.0, 5.0]])  # [num_kv_heads=1, head_dim]
        _seed_durable_summaries(worker, {needle_key: [_summary(5.0)]})  # aligned
        _seed_durable_summaries(
            worker,
            {
                key: [_summary(-5.0 - i)]  # anti-aligned
                for i, key in enumerate(distractor_keys)
            },
        )

        worker._on_query_captured("req-1", query)
        scores = worker.pop_pending_scores()

        assert set(scores.keys()) == {method}, (
            f"expected only {method!r} to be scored, got {list(scores)}"
        )
        ranked = scores[method]["req-1"]
        assert ranked[0][0] == needle_key, f"{method} did not rank needle first"

        # popping again returns nothing left to report
        assert worker.pop_pending_scores() == {}


def test_batched_scoring_matches_scalar_scoring_per_candidate():
    """Regression test for the vectorization itself: the batched worker
    path must produce the exact same ranking (and near-identical scores,
    modulo floating-point summation order) as the original scalar
    score()-per-candidate loop it replaced."""
    torch.manual_seed(0)
    num_kv_heads, head_dim, n_candidates = 3, 8, 12
    query = torch.randn(num_kv_heads, head_dim)

    keys = [to_key(i) for i in range(n_candidates)]
    summary_lists = {
        key: [
            BlockSummary(
                min=torch.randn(head_dim),
                max=torch.randn(head_dim),
                mean=torch.randn(head_dim),
                mad=torch.rand(head_dim),
            )
            for _ in range(num_kv_heads)
        ]
        for key in keys
    }

    for method in ("minmax", "mean", "cuboid_mean"):
        worker = _make_worker(method=method)
        _seed_durable_summaries(worker, summary_lists)
        worker._on_query_captured("req-1", query)
        batched_ranked = dict(worker.pop_pending_scores()[method]["req-1"])

        scalar_scores = {
            key: max(
                score(method, query[h], summary_list[h]) for h in range(num_kv_heads)
            )
            for key, summary_list in summary_lists.items()
        }

        assert set(batched_ranked) == set(scalar_scores)
        for key in keys:
            assert batched_ranked[key] == pytest.approx(scalar_scores[key], abs=1e-4)


def test_stack_cache_invalidates_on_new_insertion():
    """Correctness of the caching optimization itself (entry #53's follow-
    up): a block stored AFTER the stack cache was already built must still
    show up in scoring on the next query capture -- not silently missing
    because the cache went stale. This is the real risk a caching layer
    introduces; performance is worthless if it breaks this."""
    worker = _make_worker(method="minmax")
    worker.summaries["layer0"][1] = [_summary(1.0)]
    key1 = to_key(1)
    worker.receive_job_keys({1: [key1]})
    worker._durably_key_summaries(1, [1])
    assert worker._stack_cache_dirty is True  # insertion marks it dirty

    query = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    worker._on_query_captured("req-1", query)
    assert worker._stack_cache_dirty is False  # rebuilt on the query
    first_scores = worker.pop_pending_scores()["minmax"]["req-1"]
    assert {k for k, _ in first_scores} == {key1}

    # A new block, stored after the cache was already built and used once.
    worker.summaries["layer0"][2] = [_summary(2.0)]
    key2 = to_key(2)
    worker.receive_job_keys({2: [key2]})
    worker._durably_key_summaries(2, [2])
    assert worker._stack_cache_dirty is True  # invalidated by the insertion

    worker._on_query_captured("req-2", query)
    second_scores = worker.pop_pending_scores()["minmax"]["req-2"]
    assert {k for k, _ in second_scores} == {key1, key2}, (
        "newly-stored block missing from scoring -- stale cache bug"
    )


def test_stack_cache_eviction_compacts_and_rescoring_excludes_evicted_key():
    """Regression for the incremental stack_rebuild fix (issues log entries
    #72/#74): an evicted key must actually disappear from scoring after the
    next sync, not just flip the dirty flag -- and the surviving key's own
    score must be unaffected by the compaction (no row misalignment)."""
    worker = _make_worker(method="minmax")
    for i, val in enumerate((1.0, 5.0, 9.0)):
        worker.summaries["layer0"][i] = [_summary(val)]
        key = to_key(i)
        worker.receive_job_keys({i: [key]})
        worker._durably_key_summaries(i, [i])
    key0, key1, key2 = to_key(0), to_key(1), to_key(2)

    query = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    worker._on_query_captured("req-1", query)
    first = dict(worker.pop_pending_scores()["minmax"]["req-1"])
    assert set(first) == {key0, key1, key2}

    worker.receive_evicted_keys([key0])
    assert set(worker.durable_summaries) == {key1, key2}

    worker._on_query_captured("req-2", query)
    second = dict(worker.pop_pending_scores()["minmax"]["req-2"])
    assert set(second) == {key1, key2}, "evicted key still present after sync"
    # key1's score (exact match to the query) is unaffected by compaction.
    assert second[key1] == pytest.approx(first[key1])


def test_stack_cache_interleaved_insert_evict_insert_stays_consistent():
    """Regression: insert, sync, evict, insert again, sync -- the stacked
    cache's key list and tensor row count must always agree, across
    multiple incremental syncs, not just a single insert-then-evict pair."""
    worker = _make_worker(method="mean")
    worker.summaries["layer0"][0] = [_summary(1.0)]
    worker.receive_job_keys({0: [to_key(0)]})
    worker._durably_key_summaries(0, [0])

    query = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    worker._on_query_captured("req-1", query)
    worker.pop_pending_scores()
    assert len(worker._stack_cache_keys) == len(worker._stack_cache_index) == 1

    worker.receive_evicted_keys([to_key(0)])
    worker.summaries["layer0"][1] = [_summary(2.0)]
    worker.receive_job_keys({1: [to_key(1)]})
    worker._durably_key_summaries(1, [1])

    worker._on_query_captured("req-2", query)
    scores = dict(worker.pop_pending_scores()["mean"]["req-2"])
    assert set(scores) == {to_key(1)}
    n_rows = worker._stack_cache["mean"].shape[0]
    assert (
        n_rows == len(worker._stack_cache_keys) == len(worker._stack_cache_index) == 1
    )


def test_stack_cache_overwrite_of_resident_key_replaces_not_duplicates():
    """Regression: re-storing the SAME OffloadKey with new content (e.g. a
    job re-keying a block that's already durable) after the key has already
    been synced into the stack cache must replace its row, not leave a
    stale duplicate that a naive incremental-append would introduce."""
    worker = _make_worker(method="mean")
    key = to_key(0)
    worker.summaries["layer0"][0] = [_summary(1.0)]
    worker.receive_job_keys({0: [key]})
    worker._durably_key_summaries(0, [0])

    query = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    worker._on_query_captured("req-1", query)
    worker.pop_pending_scores()
    assert worker._stack_cache["mean"].shape[0] == 1  # synced once

    # Same key, new (very different) content, stored again before eviction.
    worker.summaries["layer0"][1] = [_summary(9.0)]
    worker.receive_job_keys({1: [key]})
    worker._durably_key_summaries(1, [1])

    worker._on_query_captured("req-2", torch.tensor([[9.0, 9.0, 9.0, 9.0]]))
    scores = dict(worker.pop_pending_scores()["mean"]["req-2"])
    assert set(scores) == {key}, "overwrite produced a duplicate row"
    assert worker._stack_cache["mean"].shape[0] == 1
    assert len(worker._stack_cache_index) == 1


def test_prune_durable_summaries_also_removes_stale_stack_cache_row():
    """Regression: the FIFO-prune backstop deletes straight from
    durable_summaries, bypassing receive_evicted_keys -- without explicit
    wiring, a pruned key's row would linger in the stack cache forever and
    keep being scored even though the manager no longer considers it
    resident."""
    worker = _make_worker(method="mean")
    worker._max_durable_summaries = 1
    worker.summaries["layer0"][0] = [_summary(1.0)]
    worker.receive_job_keys({0: [to_key(0)]})
    worker._durably_key_summaries(0, [0])

    query = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    worker._on_query_captured("req-1", query)
    worker.pop_pending_scores()  # key 0 now synced into the stack cache

    worker.summaries["layer0"][1] = [_summary(2.0)]
    worker.receive_job_keys({1: [to_key(1)]})
    worker._durably_key_summaries(1, [1])  # prunes key 0 (cap=1)
    assert set(worker.durable_summaries) == {to_key(1)}

    worker._on_query_captured("req-2", query)
    scores = dict(worker.pop_pending_scores()["mean"]["req-2"])
    assert set(scores) == {to_key(1)}, "pruned key's stale row still being scored"


def test_manager_update_relevance_ema():
    manager = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    manager.relevance_ema = {}
    key = to_key(1)

    manager.update_relevance({"mean": {"req-1": [(key, 1.0)]}})
    assert manager.relevance_ema["mean"][key] == 1.0  # first obs sets exactly

    manager.update_relevance({"mean": {"req-1": [(key, 0.0)]}})
    # alpha=0.3: 0.3*0.0 + 0.7*1.0 = 0.7
    assert abs(manager.relevance_ema["mean"][key] - 0.7) < 1e-9

    assert manager.ranked_keys("mean") == [(key, manager.relevance_ema["mean"][key])]
    assert manager.ranked_keys("minmax") == []  # untouched method stays empty


def test_update_relevance_rank_weighting_protects_top_ranked_key():
    """Regression for issues log entry #60: a needle key that one probe query
    ranked highest must survive several subsequent distractor queries that
    rank it last among a larger candidate pool -- a flat per-key alpha (the
    pre-fix behavior) collapses it to near the distractor floor within a
    handful of updates; rank-weighting should barely move it."""
    manager = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    manager.relevance_ema = {}
    needle = to_key(1)
    distractors = [to_key(i) for i in range(2, 12)]  # 10 other candidates

    # One probe query ranks the needle top of an 11-candidate pool.
    probe_ranked = [(needle, 1.0)] + [(k, 0.1) for k in distractors]
    manager.update_relevance({"mean": {"probe": probe_ranked}})
    assert manager.relevance_ema["mean"][needle] == 1.0

    # Five unrelated distractor queries each rank the needle dead last.
    for i in range(5):
        distractor_ranked = [(k, 0.9) for k in distractors] + [(needle, 0.0)]
        manager.update_relevance({"mean": {f"distractor-{i}": distractor_ranked}})

    needle_score = manager.relevance_ema["mean"][needle]
    assert needle_score > 0.9, (
        "needle's EMA collapsed toward the distractor floor -- rank weighting "
        f"should have barely touched a last-ranked key, got {needle_score}"
    )
