# SPDX-License-Identifier: Apache-2.0
"""Version-tolerance shims for a real upstream vLLM refactor (#48150,
"Define clean backend configuration boundary") that changed several
offload-connector construction signatures.

Why this exists rather than just updating call sites: different checkouts
of this project run against different vLLM commits (confirmed in practice
-- a dev-box checkout on the new API, a rented B200 box's checkout on a
commit that predates the refactor entirely, `config.py` module absent).
Hardcoding either shape breaks the other. Every function here inspects the
real installed vLLM's actual signatures at call time and adapts, instead of
assuming one API generation.
"""

import inspect

from vllm.v1.kv_offload.base import OffloadingSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
except ImportError:
    build_offloading_config = None


def create_offloading_spec(vllm_config, kv_cache_config) -> OffloadingSpec:
    """OffloadingSpecFactory.create_spec() takes a single translated
    OffloadingConfig on post-#48150 vLLM, or (vllm_config, kv_cache_config)
    positionally on pre-#48150 vLLM.

    Also stashes the real vllm_config onto the returned spec as
    `.vllm_config`, unconditionally. Pre-#48150 OffloadingSpec.__init__
    stored this itself (constructed directly from the raw vllm_config);
    post-#48150 it only ever sees the narrower, deliberately-decoupled
    OffloadingConfig translation, which has no live VllmConfig reference at
    all (no compilation_config.static_forward_context, no cache_config) --
    SemanticOffloadingWorker genuinely needs the real one (Step 1.2's
    kv_cache read mechanism), so attach it here rather than relying on
    whichever base class happens to set it. Harmless on pre-#48150 vLLM:
    just overwrites the base class's own copy with the identical object.
    """
    if build_offloading_config is not None:
        offloading_config = build_offloading_config(vllm_config, kv_cache_config)
        spec = OffloadingSpecFactory.create_spec(offloading_config)
    else:
        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)
    spec.vllm_config = vllm_config
    return spec


def _filtered_kwargs(func, **candidates):
    """Keep only the kwargs `func`'s real signature actually accepts."""
    params = inspect.signature(func).parameters
    return {k: v for k, v in candidates.items() if k in params}


def construct_scheduler_base(scheduler, spec, vllm_config, kv_cache_config) -> None:
    """Call the real OffloadingConnectorScheduler.__init__ with whichever
    subset of (spec, vllm_config, kv_cache_config) its actual signature
    accepts, in the base class's own declared order."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnectorScheduler,
    )

    kwargs = _filtered_kwargs(
        OffloadingConnectorScheduler.__init__,
        spec=spec,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
    )
    OffloadingConnectorScheduler.__init__(scheduler, **kwargs)


def construct_worker(worker_cls, spec, kv_cache_config):
    """Construct an OffloadingConnectorWorker (or subclass) with whichever
    subset of (spec, kv_cache_config) its real __init__ accepts."""
    kwargs = _filtered_kwargs(
        worker_cls.__init__, spec=spec, kv_cache_config=kv_cache_config
    )
    return worker_cls(**kwargs)


def spec_blocks_per_chunk(spec) -> int:
    """OffloadingSpec's block_size_factor attribute was renamed
    blocks_per_chunk upstream (#48150); read whichever exists."""
    if hasattr(spec, "blocks_per_chunk"):
        return spec.blocks_per_chunk
    return spec.block_size_factor


def init_cpu_offloading_worker_base(
    worker, kv_caches, blocks_per_chunk, num_cpu_blocks
):
    """CPUOffloadingWorker.__init__'s blocks_per_chunk param was named
    block_size_factor before #48150 -- call with whichever name the real
    installed base class's own signature declares, a rename not a filter
    (both old and new signatures require this param, just under a
    different name, so a plain _filtered_kwargs drop-if-absent would leave
    it missing rather than translated)."""
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

    params = inspect.signature(CPUOffloadingWorker.__init__).parameters
    key = "blocks_per_chunk" if "blocks_per_chunk" in params else "block_size_factor"
    CPUOffloadingWorker.__init__(
        worker,
        kv_caches=kv_caches,
        num_cpu_blocks=num_cpu_blocks,
        **{key: blocks_per_chunk},
    )
