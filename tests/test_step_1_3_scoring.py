# SPDX-License-Identifier: Apache-2.0
"""Step 1.3 acceptance check: live-query scoring against durable summaries,
correlation plumbing, and the manager's EMA -- all independent of the real
attention-op/prepare_inputs monkey-patches (those are verified end-to-end on
the real server instead; see .claude/docs/semantic-eviction-plan.md Step 1.3
and the issues log for that investigation).
"""

import torch
from semantic_offload.index import BlockSummary
from semantic_offload.manager import SemanticOffloadingManager
from semantic_offload.worker import SemanticOffloadingWorker

from vllm.v1.kv_offload.base import make_offload_key


def _make_worker() -> SemanticOffloadingWorker:
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._probe_layer_name = "layer0"
    worker.summaries = {"layer0": {}}
    worker._pending_job_keys = {}
    worker.durable_summaries = {}
    worker._pending_scores = {}
    return worker


def _summary(value: float, dim: int = 4) -> BlockSummary:
    v = torch.full((dim,), value)
    return BlockSummary(min=v, max=v, mean=v, mad=torch.zeros(dim))


def to_key(n: int):
    return make_offload_key(str(n).encode(), 0)


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


def test_on_query_captured_ranks_needle_highest_for_all_methods():
    """Single-KV-head case (durable_summaries[key] = [BlockSummary]); the
    multi-head max-combine path (entry #9) is exercised implicitly since
    len == 1 makes max() a no-op over one element."""
    worker = _make_worker()
    needle_key = to_key(1)
    distractor_keys = [to_key(i) for i in range(2, 6)]

    query = torch.tensor([[5.0, 5.0, 5.0, 5.0]])  # [num_kv_heads=1, head_dim]
    worker.durable_summaries[needle_key] = [_summary(5.0)]  # aligned with query
    for i, key in enumerate(distractor_keys):
        worker.durable_summaries[key] = [_summary(-5.0 - i)]  # anti-aligned

    worker._on_query_captured("req-1", query)
    scores = worker.pop_pending_scores()

    for method in ("minmax", "mean", "cuboid_mean"):
        ranked = scores[method]["req-1"]
        assert ranked[0][0] == needle_key, f"{method} did not rank needle first"

    # popping again returns nothing left to report
    assert worker.pop_pending_scores() == {}


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
