# SPDX-License-Identifier: Apache-2.0
"""Step 1.1 acceptance checks.

1. The spec loads via vLLM's dynamic spec_module_path path (the mechanism
   external packages use, per vllm/v1/kv_offload/factory.py) and constructs
   end-to-end with a real VllmConfig/KVCacheConfig.
2. SemanticOffloadingManager's observable behavior (prepare_store / touch /
   evict / events) is identical to CPUOffloadingManager(cache_policy="lru")
   given the same operation sequence -- this is Step 1.1's "behavior
   identical to Step 0.1" requirement, checked at the manager level since
   that's the cheapest level that actually catches a divergence (see
   AGENTS.md test-design guidance: unit over integration over e2e).
3. index.py's scoring interface (stubbed now, wired in at Step 1.3) dispatches
   all three methods and rejects an unknown one.

Full end-to-end server verification (CPU-cache hit via /metrics with our spec
selected, mirroring Step 0.1's demonstration) is run manually against real
GPU hardware -- see .claude/docs/semantic-eviction-plan.md Step 1.1 and the
project memory for that run's result.
"""

import torch
from semantic_offload.index import BlockSummary, build_summary, score
from semantic_offload.manager import SemanticOffloadingManager
from semantic_offload.spec import SemanticOffloadingSpec

from vllm.v1.kv_offload.base import OffloadingSpec, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

# ---------------------------------------------------------------------------
# 1. Spec loads via the real factory dynamic-import path
# ---------------------------------------------------------------------------


def _make_vllm_config(cpu_bytes_to_use: int = 65536):
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=10000,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "SemanticOffloadingSpec",
            "spec_module_path": "semantic_offload.spec",
            "cpu_bytes_to_use": cpu_bytes_to_use,
        },
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def _make_kv_cache_config():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    num_blocks = 16
    num_kv_heads = 1
    head_size = 1
    dtype = torch.float32
    page_size = 2 * num_kv_heads * head_size * torch.finfo(dtype).bits // 8
    kv_tensor = KVCacheTensor(
        size=num_blocks * page_size, shared_by=["layer"], block_stride=0
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )


def test_spec_loads_via_spec_module_path():
    """SemanticOffloadingSpec is not pre-registered; it must resolve via the
    same spec_module_path dynamic-import path documented for external
    packages in vllm/v1/kv_offload/factory.py."""
    config = _make_vllm_config()
    spec_cls = OffloadingSpecFactory.get_spec_cls(config)
    assert spec_cls is SemanticOffloadingSpec
    assert issubclass(spec_cls, OffloadingSpec)


def test_spec_constructs_end_to_end_and_serves_manager():
    """Full factory -> spec construction, then get_manager() returns a live
    SemanticOffloadingManager -- the "spec loads and serves a request" check,
    at the manager-construction granularity."""
    config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    spec = OffloadingSpecFactory.create_spec(config, kv_cache_config)
    assert isinstance(spec, SemanticOffloadingSpec)
    assert spec.num_blocks > 0

    manager = spec.get_manager()
    assert isinstance(manager, SemanticOffloadingManager)
    # get_manager() must be idempotent, like the base CPUOffloadingSpec.
    assert spec.get_manager() is manager


# ---------------------------------------------------------------------------
# 2. Behavior identical to plain LRU (Step 1.1's pass-through requirement)
# ---------------------------------------------------------------------------

_EMPTY_REQ_CTX = ReqContext(req_id="")


def to_keys(int_hashes: list[int]):
    return [make_offload_key(str(i).encode(), 0) for i in int_hashes]


def _run_scenario(manager) -> list:
    """A store/touch/evict/store scenario exercising every CachePolicy method."""
    manager.prepare_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)
    manager.touch(to_keys([1]), _EMPTY_REQ_CTX)  # block 1 now most-recently-used
    # Only 1 free block (capacity 4, 3 used) -- forces eviction of the LRU
    # candidate among {2, 3} (1 was just touched, so it's protected by recency).
    result = manager.prepare_store(to_keys([4, 5]), _EMPTY_REQ_CTX)
    return result.evicted_keys if result is not None else None


def test_semantic_manager_matches_lru_manager_behavior():
    """SemanticOffloadingManager (SemanticPolicy, delegating to LRU) must
    produce identical eviction decisions to CPUOffloadingManager(cache_policy=
    "lru") given the same operations -- the literal "behavior identical to
    Step 0.1" acceptance check."""
    lru_manager = CPUOffloadingManager(num_blocks=4, cache_policy="lru")
    semantic_manager = SemanticOffloadingManager(num_blocks=4)

    lru_evicted = _run_scenario(lru_manager)
    semantic_evicted = _run_scenario(semantic_manager)

    assert semantic_evicted == lru_evicted
    assert semantic_evicted is not None


# ---------------------------------------------------------------------------
# 3. index.py's scoring interface stub
# ---------------------------------------------------------------------------


def test_build_summary_shapes():
    keys = torch.randn(16, 8)  # [block_tokens, head_dim]
    summary = build_summary(keys)
    assert isinstance(summary, BlockSummary)
    for field in (summary.min, summary.max, summary.mean, summary.mad):
        assert field.shape == (8,)


def test_score_dispatches_all_three_methods():
    keys = torch.randn(16, 8)
    query = torch.randn(8)
    summary = build_summary(keys)
    for method in ("minmax", "mean", "cuboid_mean"):
        value = score(method, query, summary)
        assert isinstance(value, float)


def test_score_rejects_unknown_method():
    keys = torch.randn(16, 8)
    query = torch.randn(8)
    summary = build_summary(keys)
    try:
        score("not_a_method", query, summary)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "not_a_method" in str(e)
