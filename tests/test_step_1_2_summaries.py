# SPDX-License-Identifier: Apache-2.0
"""Step 1.2 acceptance check: summary correctness vs. reference reductions.

Builds a SemanticOffloadingWorker without going through CPUOffloadingWorker's
real CUDA-tensor allocation (that plumbing is exercised end-to-end on the real
server instead, see .claude/docs/semantic-eviction-plan.md Step 1.2 and
project memory for that run). This test only exercises the block-extraction
and summary-building logic against a synthetic kv_cache tensor with a known
layout and known values, so it runs on CPU and needs no GPU.
"""

from types import SimpleNamespace

import torch
from semantic_offload.worker import SemanticOffloadingWorker


def _make_layer(num_blocks, num_kv_heads, block_size, head_size, seed):
    torch.manual_seed(seed)
    kv_cache = torch.randn(num_blocks, num_kv_heads, block_size, 2 * head_size)

    def get_kv_cache_shape(nb, bs, nh, hs, cache_dtype_str="auto"):
        return (nb, nh, bs, 2 * hs)

    attn_backend = SimpleNamespace(get_kv_cache_shape=get_kv_cache_shape)
    return SimpleNamespace(
        kv_cache=kv_cache,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        attn_backend=attn_backend,
    )


def _make_worker(layers: dict, block_size: int) -> SemanticOffloadingWorker:
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._attention_layers = layers
    worker._block_size = block_size
    worker._summary_stream = None
    worker._layout_checked = set()
    worker.summaries = {name: {} for name in layers}
    worker._probe_layer_name = None  # Step 1.3 re-keying is out of scope here
    worker._pending_job_keys = {}
    worker.durable_summaries = {}
    return worker


def test_build_summaries_matches_reference_reductions():
    num_blocks, num_kv_heads, block_size, head_size = 8, 2, 16, 32
    layer = _make_layer(num_blocks, num_kv_heads, block_size, head_size, seed=0)
    worker = _make_worker({"layer0": layer}, block_size)

    block_id = 3
    worker._build_summaries_for_blocks(0, [block_id])

    summaries = worker.summaries["layer0"][block_id]
    assert len(summaries) == num_kv_heads

    keys_full = layer.kv_cache[block_id][..., :head_size].float()
    for h in range(num_kv_heads):
        keys = keys_full[h]
        summary = summaries[h]
        assert torch.allclose(summary.min, keys.amin(dim=0))
        assert torch.allclose(summary.max, keys.amax(dim=0))
        assert torch.allclose(summary.mean, keys.mean(dim=0))
        expected_mad = (keys - keys.mean(dim=0)).abs().mean(dim=0)
        assert torch.allclose(summary.mad, expected_mad)


def test_build_summaries_only_uses_key_half_not_value_half():
    """The last dim is [K | V] packed; corrupting only V must not change the
    computed summary, proving we sliced K and not the whole packed tensor."""
    num_blocks, num_kv_heads, block_size, head_size = 4, 1, 16, 16
    layer = _make_layer(num_blocks, num_kv_heads, block_size, head_size, seed=1)
    worker = _make_worker({"layer0": layer}, block_size)

    block_id = 0
    worker._build_summaries_for_blocks(0, [block_id])
    before = worker.summaries["layer0"][block_id][0]

    layer.kv_cache[block_id, :, :, head_size:] = 999.0  # corrupt V half only
    worker.summaries["layer0"].clear()
    worker._build_summaries_for_blocks(0, [block_id])
    after = worker.summaries["layer0"][block_id][0]

    assert torch.allclose(before.min, after.min)
    assert torch.allclose(before.max, after.max)
    assert torch.allclose(before.mean, after.mean)


def test_layout_mismatch_raises_instead_of_silently_scoring_garbage():
    layer = _make_layer(4, 1, 16, 8, seed=2)
    layer.attn_backend.get_kv_cache_shape = lambda *a, **k: (999, 1, 16, 16)
    worker = _make_worker({"layer0": layer}, block_size=16)
    try:
        worker._build_summaries_for_blocks(0, [0])
        raise AssertionError("expected AssertionError on layout mismatch")
    except AssertionError as e:
        assert "does not match get_kv_cache_shape" in str(e)


def test_no_blocks_is_a_noop():
    layer = _make_layer(4, 1, 16, 8, seed=3)
    worker = _make_worker({"layer0": layer}, block_size=16)
    worker._build_summaries_for_blocks(0, [])
    assert worker.summaries["layer0"] == {}
