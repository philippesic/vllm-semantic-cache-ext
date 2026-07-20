# SPDX-License-Identifier: Apache-2.0
"""Live per-step query capture for Step 1.3.

Composes two independently-verified mechanisms (full investigation trail,
including two dead ends, is in .claude/docs/semantic-eviction-issues-log.md
entries #5-#6):

1. Row-boundary/req_id info: patches `GPUModelRunner.prepare_inputs` (the V2
   runner -- this project's target model defaults to V2, confirmed via
   `vllm_config.use_v2_model_runner`; a future session targeting the legacy
   runner needs a different patch point, see the issues log) at the class
   level. `prepare_inputs` is looked up via plain attribute access on every
   call (never a cached/bound reference), so this patch fires reliably on
   every step, including pure single-token decode steps, since it's
   orchestration code that itself feeds the (possibly compiled/graph-
   replayed) model forward rather than being part of what gets replayed.
2. Query data: `TorchDispatchMode` on `torch.ops.vllm.unified_attention_with_
   output`. This one only fires on steps that include prefill or mixed
   prefill-decode batches -- vLLM's default `cudagraph_mode` fully captures
   pure decode-only batches with zero Python touchpoints during replay
   (verified empirically). Relevance updates are therefore opportunistic,
   not guaranteed every step; the manager's EMA is designed to tolerate that.

Zero vLLM source modifications -- both are class-level monkey-patches
applied from this out-of-tree package.
"""

from collections.abc import Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from semantic_offload._debug import debug_print
from vllm.config import VllmConfig
from vllm.utils.torch_utils import _resolve_layer_name


class _BatchLayout:
    __slots__ = ("req_ids", "boundaries", "num_tokens")

    def __init__(self, req_ids: list[str], boundaries: list[tuple[int, int]]):
        self.req_ids = req_ids
        self.boundaries = boundaries
        self.num_tokens = boundaries[-1][1] if boundaries else 0


def _patch_prepare_inputs(state: dict) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_semantic_prepare_inputs_patched", False):
        return
    original = GPUModelRunner.prepare_inputs

    def wrapper(self_runner, scheduler_output, batch_desc):
        result = original(self_runner, scheduler_output, batch_desc)
        req_ids = list(result.req_ids)
        token_counts = [int(c) for c in result.num_scheduled_tokens]
        boundaries = []
        start = 0
        for count in token_counts:
            boundaries.append((start, start + count))
            start += count
        state["layout"] = _BatchLayout(req_ids=req_ids, boundaries=boundaries)
        return result

    GPUModelRunner.prepare_inputs = wrapper
    GPUModelRunner._semantic_prepare_inputs_patched = True


def _should_sample_step(step_index: int, stride: int) -> bool:
    """True on every `stride`-th eligible query-capture step (stride<=1:
    every step, the historical default). Counted once per real query-
    capture-eligible dispatch event (one per step), not per request within
    it, so concurrent requests sharing a step are throttled together as a
    unit rather than independently -- a coarser cadence relies on the
    manager's own EMA staleness-tolerance (Step 1.4) to carry relevance
    signal forward across skipped steps, not on every step being scored.
    See semantic-eviction-plan.md's TTFT-tax follow-up investigation for
    why this knob exists (leads #1/#3): stack_rebuild/update_relevance's
    per-call cost is bounded but non-trivial (issues log entries #62-65),
    and their aggregate cost scales with how often query-capture fires, not
    just candidate-pool size."""
    return stride <= 1 or step_index % stride == 0


def install(
    vllm_config: VllmConfig,
    probe_layer_name: str,
    on_query: Callable[[str, torch.Tensor], None],
    num_queries_per_kv: int = 1,
    capture_stride: int = 1,
) -> TorchDispatchMode:
    """Install both patches. `on_query(req_id, query_repr)` fires once per
    request per step whenever live query data is captured for the probe
    layer -- `query_repr` is a `[num_kv_heads, head_dim]` tensor: the
    request's last real token's query, grouped by which KV head each query
    head attends against (GQA group, contiguous per
    triton_unified_attention.py's `query_offset_1 = kv_head_idx *
    num_queries_per_kv + ...` convention -- verified against this backend's
    kernel, not assumed) and averaged only within the group, not across all
    heads. See issues log entry #8 for why last-token (vs. whole-step mean)
    was adopted, and entry #9 for why per-KV-head (vs. fully-pooled) was
    adopted on top of that. Returns the installed dispatch mode; caller must
    keep a reference alive for the duration of the process.

    `capture_stride`: only every `capture_stride`-th eligible step actually
    fires `on_query` (default 1: unchanged, every step) -- see
    `_should_sample_step`."""
    assert vllm_config.use_v2_model_runner, (
        "query capture targets the V2 GPUModelRunner only -- a model/config "
        "defaulting to the legacy runner needs a different patch point, see "
        "semantic-eviction-issues-log.md entry #6"
    )
    state: dict = {"layout": None, "step_index": 0}
    _patch_prepare_inputs(state)

    class ProbeMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if "unified_attention_with_output" in str(func):
                layout: _BatchLayout | None = state["layout"]
                query = args[0] if len(args) > 0 else kwargs.get("query")
                layer_name_arg = args[4] if len(args) > 4 else kwargs.get("layer_name")
                resolved_name = (
                    _resolve_layer_name(layer_name_arg)
                    if layer_name_arg is not None
                    else None
                )
                if resolved_name == probe_layer_name:
                    debug_print(
                        "SEMANTIC_QUERY_CAPTURE_DEBUG "
                        f"query_shape0={query.shape[0] if query is not None else None} "
                        f"layout_num_tokens={layout.num_tokens if layout else None} "
                        f"layout_req_ids={layout.req_ids if layout else None}"
                    )
                if (
                    layout is not None
                    and query is not None
                    and layer_name_arg is not None
                    and resolved_name == probe_layer_name
                    and query.shape[0] >= layout.num_tokens
                ):
                    # Counted once per eligible step (not per request in it)
                    # so a throttled cadence applies uniformly to whichever
                    # requests happen to share this step -- see
                    # `_should_sample_step`.
                    step_index = state["step_index"]
                    state["step_index"] = step_index + 1
                    if not _should_sample_step(step_index, capture_stride):
                        return func(*args, **kwargs)
                    # CUDA-graph replay pads short batches up to a fixed
                    # capture size; real tokens always occupy the prefix
                    # [0, num_tokens) and padding is appended after (verified
                    # against gpu/model_runner.py's is_padding buffer
                    # construction) -- an exact-length check here silently
                    # dropped every short (needle-sized) prefill, since only
                    # large prefills happen to land on an unpadded shape.
                    for req_id, (start, end) in zip(layout.req_ids, layout.boundaries):
                        if end > start:
                            # Last-token query repr, grouped per KV head
                            # (entry #9) -- shape [num_query_heads, head_dim]
                            # -> [num_kv_heads, num_queries_per_kv, head_dim]
                            # -> mean within group -> [num_kv_heads, head_dim].
                            last_q = query[end - 1]
                            num_query_heads, head_dim = last_q.shape
                            num_kv_heads = num_query_heads // num_queries_per_kv
                            grouped = last_q.view(
                                num_kv_heads, num_queries_per_kv, head_dim
                            ).mean(dim=1)
                            on_query(req_id, grouped)
            return func(*args, **kwargs)

    mode = ProbeMode()
    mode.__enter__()
    return mode
