# SPDX-License-Identifier: Apache-2.0
"""Live per-step query capture for Step 1.3.

Composes two independently-verified mechanisms (full investigation trail,
including two dead ends, is in .claude/docs/semantic-eviction-issues-log.md
entries #5-#6):

1. Execution ownership: patches `GPUModelRunner.execute_model` (the V2
   runner) to scope a fresh dispatch mode and per-execution context to one
   exact runner instance. Shutdown waits for active executions before
   restoring the class methods.
2. Row-boundary/req_id info: patches `GPUModelRunner.prepare_inputs` (the V2
   runner -- this project's target model defaults to V2, confirmed via
   `vllm_config.use_v2_model_runner`; a future session targeting the legacy
   runner needs a different patch point, see the issues log) at the class
   level. `prepare_inputs` is looked up via plain attribute access on every
   call (never a cached/bound reference), so this patch fires reliably on
   every step, including pure single-token decode steps, since it's
   orchestration code that itself feeds the (possibly compiled/graph-
   replayed) model forward rather than being part of what gets replayed.
3. Query data: `TorchDispatchMode` on `torch.ops.vllm.unified_attention_with_
   output`. This one only fires on steps that include prefill or mixed
   prefill-decode batches -- vLLM's default `cudagraph_mode` fully captures
   pure decode-only batches with zero Python touchpoints during replay
   (verified empirically). Relevance updates are therefore opportunistic,
   not guaranteed every step; the manager's EMA is designed to tolerate that.

Zero vLLM source modifications -- both are class-level monkey-patches
applied from this out-of-tree package.
"""

from collections.abc import Callable
from contextvars import ContextVar
from threading import Condition, RLock, get_ident

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.torch_utils import _resolve_layer_name

from semantic_offload._debug import debug_print


class _BatchLayout:
    __slots__ = ("boundaries", "num_tokens", "req_ids")

    def __init__(self, req_ids: list[str], boundaries: list[tuple[int, int]]):
        self.req_ids = req_ids
        self.boundaries = boundaries
        self.num_tokens = boundaries[-1][1] if boundaries else 0


_current_handle: ContextVar["QueryCaptureHandle | None"] = ContextVar(
    "semantic_query_capture_handle", default=None
)
_current_layout: ContextVar[_BatchLayout | None] = ContextVar(
    "semantic_query_capture_layout", default=None
)
_runner_patch: tuple[type, Callable, Callable, Callable, Callable] | None = None
_active_handle: "QueryCaptureHandle | None" = None
_lifecycle_lock = RLock()
_lifecycle_condition = Condition(_lifecycle_lock)


class QueryCaptureHandle:
    """Owns the single supported runner installation in this process."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        mode_factory: Callable[[], TorchDispatchMode],
    ):
        self.vllm_config = vllm_config
        self._mode_factory = mode_factory
        self._runner = None
        self._active_executions = 0
        self._execution_threads: dict[int, int] = {}
        self._closing = False
        self._closed = False

    def _begin_execution(self, runner) -> None:
        with _lifecycle_condition:
            if self._closing or self._closed:
                raise RuntimeError("query capture is closing")
            if self._runner is None:
                self._runner = runner
            elif self._runner is not runner:
                raise RuntimeError("query capture received an unexpected runner")
            self._active_executions += 1
            thread_id = get_ident()
            self._execution_threads[thread_id] = (
                self._execution_threads.get(thread_id, 0) + 1
            )

    def _end_execution(self) -> None:
        with _lifecycle_condition:
            self._active_executions -= 1
            thread_id = get_ident()
            remaining = self._execution_threads[thread_id] - 1
            if remaining:
                self._execution_threads[thread_id] = remaining
            else:
                del self._execution_threads[thread_id]
            _lifecycle_condition.notify_all()

    def close(self) -> None:
        global _active_handle

        with _lifecycle_condition:
            if self._closed:
                return
            if _active_handle is not self:
                raise RuntimeError("query capture handle no longer owns installation")
            if get_ident() in self._execution_threads:
                raise RuntimeError(
                    "cannot close query capture from an active execution"
                )
            self._closing = True
            while self._active_executions:
                _lifecycle_condition.wait()
            try:
                _unpatch_runner(self)
            finally:
                _active_handle = None
                self._runner = None
                self._closed = True
                _lifecycle_condition.notify_all()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _patch_runner(handle: QueryCaptureHandle) -> None:
    global _runner_patch

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if _runner_patch is not None or hasattr(
        GPUModelRunner, "_semantic_query_capture_owner"
    ):
        raise RuntimeError("semantic query capture patch is already installed")
    original_prepare = GPUModelRunner.prepare_inputs
    original_execute = GPUModelRunner.execute_model

    def prepare_wrapper(self_runner, scheduler_output, batch_desc):
        result = original_prepare(self_runner, scheduler_output, batch_desc)
        if _current_handle.get() is not handle or self_runner is not handle._runner:
            return result
        req_ids = list(result.req_ids)
        token_counts = [int(c) for c in result.num_scheduled_tokens]
        boundaries = []
        start = 0
        for count in token_counts:
            boundaries.append((start, start + count))
            start += count
        _current_layout.set(_BatchLayout(req_ids=req_ids, boundaries=boundaries))
        return result

    def execute_wrapper(self_runner, *args, **kwargs):
        if getattr(self_runner, "vllm_config", None) is not handle.vllm_config:
            return original_execute(self_runner, *args, **kwargs)
        handle._begin_execution(self_runner)
        handle_token = _current_handle.set(handle)
        layout_token = _current_layout.set(None)
        try:
            with handle._mode_factory():
                return original_execute(self_runner, *args, **kwargs)
        finally:
            _current_layout.reset(layout_token)
            _current_handle.reset(handle_token)
            handle._end_execution()

    GPUModelRunner.prepare_inputs = prepare_wrapper
    GPUModelRunner.execute_model = execute_wrapper
    GPUModelRunner._semantic_query_capture_owner = handle
    _runner_patch = (
        GPUModelRunner,
        original_prepare,
        prepare_wrapper,
        original_execute,
        execute_wrapper,
    )


def _unpatch_runner(handle: QueryCaptureHandle) -> None:
    global _runner_patch

    patch = _runner_patch
    if patch is None:
        return
    runner_cls, original_prepare, prepare_wrapper, original_execute, execute_wrapper = (
        patch
    )
    conflicts = []
    if runner_cls.prepare_inputs is prepare_wrapper:
        runner_cls.prepare_inputs = original_prepare
    else:
        conflicts.append("prepare_inputs")
    if runner_cls.execute_model is execute_wrapper:
        runner_cls.execute_model = original_execute
    else:
        conflicts.append("execute_model")
    if getattr(runner_cls, "_semantic_query_capture_owner", None) is handle:
        delattr(runner_cls, "_semantic_query_capture_owner")
    _runner_patch = None
    if conflicts:
        raise RuntimeError(
            "GPUModelRunner methods changed while semantic capture owned them: "
            + ", ".join(conflicts)
        )


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


def _validate_cudagraph_mode(vllm_config: VllmConfig) -> None:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation_config, "cudagraph_mode", None)
    mixed_mode = getattr(mode, "mixed_mode", None)
    if callable(mixed_mode):
        mode = mixed_mode()
    if getattr(mode, "name", str(mode)) == "FULL":
        raise ValueError(
            "semantic query capture requires prefill and mixed batches outside "
            "FULL CUDA graphs; use NONE, PIECEWISE, FULL_DECODE_ONLY, or "
            "FULL_AND_PIECEWISE"
        )


def preflight_install(vllm_config: VllmConfig) -> None:
    """Validate capture compatibility without changing process state."""
    if not vllm_config.use_v2_model_runner:
        raise ValueError("semantic query capture requires the V2 GPUModelRunner")
    _validate_cudagraph_mode(vllm_config)

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    with _lifecycle_condition:
        if (
            _active_handle is not None
            or _runner_patch is not None
            or hasattr(GPUModelRunner, "_semantic_query_capture_owner")
        ):
            raise RuntimeError("query capture is already installed in this process")


def install(
    vllm_config: VllmConfig,
    probe_layer_name: str,
    on_query: Callable[[list[str], torch.Tensor], None],
    num_queries_per_kv: int = 1,
    capture_stride: int = 1,
) -> QueryCaptureHandle:
    """Install both patches. `on_query(req_ids, query_reprs)` fires ONCE per
    step whenever live query data is captured for the probe layer, covering
    every request scheduled in that step -- `query_reprs` is a
    `[len(req_ids), num_kv_heads, head_dim]` tensor: each request's last
    real token's query, grouped by which KV head each query
    head attends against (GQA group, contiguous per
    triton_unified_attention.py's `query_offset_1 = kv_head_idx *
    num_queries_per_kv + ...` convention -- verified against this backend's
    kernel, not assumed) and averaged only within the group, not across all
    heads. See issues log entry #8 for why last-token (vs. whole-step mean)
    was adopted, and entry #9 for why per-KV-head (vs. fully-pooled) was
    adopted on top of that. Returns an owning handle; caller must keep it alive
    and call `close()` during worker shutdown.

    Only one runner is supported per process. A concurrent installation or a
    second runner using the owning config fails before publishing capture state.

    `capture_stride`: only every `capture_stride`-th eligible step actually
    fires `on_query` (default 1: unchanged, every step) -- see
    `_should_sample_step`."""
    if capture_stride < 1:
        raise ValueError("capture_stride must be >= 1")
    if num_queries_per_kv < 1:
        raise ValueError("num_queries_per_kv must be >= 1")
    global _active_handle

    state: dict = {"step_index": 0}

    def capture_query(layout: _BatchLayout, query: torch.Tensor) -> None:
        step_index = state["step_index"]
        state["step_index"] = step_index + 1
        if not _should_sample_step(step_index, capture_stride):
            return
        req_ids = []
        last_indices = []
        for req_id, (start, end) in zip(layout.req_ids, layout.boundaries):
            if end > start:
                req_ids.append(req_id)
                last_indices.append(end - 1)
        if not req_ids:
            return
        last_q = query[torch.tensor(last_indices, device=query.device)]
        n_reqs, num_query_heads, head_dim = last_q.shape
        if num_query_heads % num_queries_per_kv:
            raise ValueError("query heads must be divisible by num_queries_per_kv")
        num_kv_heads = num_query_heads // num_queries_per_kv
        grouped = last_q.view(n_reqs, num_kv_heads, num_queries_per_kv, head_dim).mean(
            dim=2
        )
        on_query(req_ids, grouped)

    class ProbeMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if "unified_attention_with_output" in str(func):
                query = args[0] if len(args) > 0 else kwargs.get("query")
                layer_name_arg = args[4] if len(args) > 4 else kwargs.get("layer_name")
                resolved_name = (
                    _resolve_layer_name(layer_name_arg)
                    if layer_name_arg is not None
                    else None
                )
                if resolved_name == probe_layer_name:
                    layout = _current_layout.get()
                    try:
                        debug_print(
                            "SEMANTIC_QUERY_CAPTURE_DEBUG "
                            f"query_shape0={query.shape[0] if query is not None else None} "
                            f"layout_num_tokens={layout.num_tokens if layout else None} "
                            f"layout_req_ids={layout.req_ids if layout else None}"
                        )
                        # CUDA-graph replay can pad the query tensor, but real
                        # tokens occupy its prefix. Consume each prepare-inputs
                        # layout once at the configured probe layer.
                        if (
                            layout is not None
                            and query is not None
                            and query.shape[0] >= layout.num_tokens
                        ):
                            capture_query(layout, query)
                    finally:
                        _current_layout.set(None)
            return func(*args, **kwargs)

    with _lifecycle_condition:
        preflight_install(vllm_config)
        handle = QueryCaptureHandle(vllm_config, ProbeMode)
        try:
            _patch_runner(handle)
        except BaseException:
            _unpatch_runner(handle)
            raise
        _active_handle = handle
        return handle
