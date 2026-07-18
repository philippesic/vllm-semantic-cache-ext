# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingWorker: builds per-block key summaries during offload.

Hook-point rationale (see .claude/docs/semantic-eviction-plan.md, Step 1.2 --
full investigation trail, including two dead ends, is in project memory):

By the time KV blocks reach CPUOffloadingWorker, they are opaque
(num_blocks, page_size_bytes) int8 byte pages -- vllm/v1/kv_offload/base.py's
CanonicalKVCacheTensor deliberately erases per-head/per-token structure so the
generic offloading connector works across attention backends without knowing
any backend's physical KV layout. Reconstructing that layout from raw bytes
would be backend-specific and fragile.

nn.Module forward hooks and TorchDispatchMode were both tried and both fail
under vLLM's default torch.compile + CUDA-graph execution mode (verified
empirically): compiled/graph-replayed execution never returns to the Python
interpreter for the replayed portion, so no Python-side interception can see
per-step values once graphs are captured.

What actually works: each Attention instance holds `self.kv_cache`, the real,
already-correctly-shaped per-layer KV cache tensor (populated by
`bind_kv_cache`, vllm/v1/worker/utils.py). This is persistent GPU memory, not
a transient forward-pass activation, so reading it has nothing to do with
hooks or graph capture -- it works identically whether the model ran eager or
compiled. Its shape is not reverse-engineered: every attention backend
implements a standard `get_kv_cache_shape()` classmethod
(vllm/v1/attention/backends/*.py) as part of the public AttentionBackend
interface. Verified on the real server (default compiled+cudagraph config):
kv_cache.shape matched get_kv_cache_shape()'s prediction exactly, with real
non-degenerate values.

Layout assumption (asserted at runtime, not just trusted): Triton,
FlashAttention, and FlashInfer's `get_kv_cache_shape` all document the same
logical shape `(num_blocks, num_kv_heads, block_size, 2*head_size)`, K and V
packed into the last dimension (first half K, second half V). This is the
convention this file relies on; the assert in `_check_layout` catches any
backend that doesn't match it, rather than silently producing wrong
summaries. Does not apply to MLA (no per-head keys exist there at all).
"""

import time

import torch

from semantic_offload._debug import debug_print
from semantic_offload.index import (
    BlockSummary,
    build_summary,
    score_cuboid_mean_batch,
    score_mean_batch,
    score_minmax_batch,
)
from semantic_offload.query_capture import install as install_query_capture
from vllm.config import VllmConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadKey,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

_SCORING_METHODS = ("minmax", "mean", "cuboid_mean")
# Which BlockSummary fields each method's batched scorer actually needs --
# only these get stacked into the cache, not all four.
_METHOD_FIELDS = {
    "minmax": ("max", "min"),
    "mean": ("mean",),
    "cuboid_mean": ("mean", "mad"),
}


class SemanticOffloadingWorker(CPUOffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        vllm_config: VllmConfig,
        method: str = "minmax",
    ):
        super().__init__(
            kv_caches=kv_caches,
            block_size_factor=block_size_factor,
            num_cpu_blocks=num_cpu_blocks,
        )
        if method not in _SCORING_METHODS:
            raise ValueError(
                f"Unknown scoring method: {method!r}. Supported: {_SCORING_METHODS}"
            )
        self._method = method
        static_forward_context = vllm_config.compilation_config.static_forward_context
        self._attention_layers = {
            layer_name: layer
            for layer_name, layer in static_forward_context.items()
            if hasattr(layer, "num_kv_heads") and hasattr(layer, "head_size")
        }
        self._block_size = vllm_config.cache_config.block_size
        self._summary_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._layout_checked: set[str] = set()
        # summaries[layer_name][block_id] = list[BlockSummary], one per KV head.
        # Keyed by physical GPU block ID, which is naturally bounded and
        # reused/overwritten as blocks are freed and reallocated -- no
        # separate eviction-cleanup hook needed to keep this bounded.
        self.summaries: dict[str, dict[int, list[BlockSummary]]] = {
            layer_name: {} for layer_name in self._attention_layers
        }

        # Step 1.3: durable, OffloadKey-keyed summaries (survive GPU block
        # reuse, unlike `self.summaries` above) + live query scoring.
        # One probe layer only, per the plan -- picked deterministically so
        # it's stable across process restarts, not for any semantic reason.
        self._probe_layer_name = (
            min(self._attention_layers) if self._attention_layers else None
        )
        # One BlockSummary per KV head (not averaged across heads -- see
        # issues log entry #9; entry #7's pooled-average version is
        # superseded).
        self.durable_summaries: dict[OffloadKey, list[BlockSummary]] = {}
        self._pending_job_keys: dict[int, list[OffloadKey]] = {}
        self._pending_scores: dict[str, dict[str, list[tuple[OffloadKey, float]]]] = {}
        # Cache of durable_summaries stacked into batched tensors, rebuilt
        # only when new entries are added (_durably_key_summaries sets the
        # dirty flag) instead of on every query capture -- query captures
        # fire on nearly every prefill/mixed step, while insertions happen
        # far less often, so rebuilding fresh each time was itself a real
        # Python-level O(n_candidates) cost even after the scoring math
        # itself was vectorized (issues log entry #53's follow-up).
        self._stack_cache_dirty = True
        self._stack_cache_keys: list[OffloadKey] = []
        self._stack_cache: dict[str, torch.Tensor] = {}
        self._query_capture_mode = None
        if self._probe_layer_name is not None:
            probe_layer = self._attention_layers[self._probe_layer_name]
            num_attn_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            num_queries_per_kv = num_attn_heads // probe_layer.num_kv_heads
            self._query_capture_mode = install_query_capture(
                vllm_config,
                self._probe_layer_name,
                self._on_query_captured,
                num_queries_per_kv=num_queries_per_kv,
            )

    def receive_job_keys(self, store_job_keys: dict[int, list[OffloadKey]]) -> None:
        """Called by SemanticOffloadingConnector before this step's
        submit_store() calls, with the OffloadKey(s) each pending store
        job_id represents (scheduler-side info the worker has no other way
        to see -- see connector.py and the issues log entry #6/7)."""
        self._pending_job_keys.update(store_job_keys)

    def pop_pending_scores(
        self,
    ) -> dict[str, dict[str, list[tuple[OffloadKey, float]]]]:
        scores = self._pending_scores
        self._pending_scores = {}
        return scores

    def _rebuild_stack_cache(self) -> None:
        """Stack every resident candidate's summaries (only the fields
        `self._method` actually needs) into batched tensors once, cached
        until the next insertion. See the dirty-flag comment in __init__
        for why this is cached rather than rebuilt on every query capture."""
        keys = list(self.durable_summaries.keys())
        summary_lists = list(self.durable_summaries.values())
        cache: dict[str, torch.Tensor] = {}
        for field in _METHOD_FIELDS[self._method]:
            # [n_candidates, num_kv_heads, head_dim]
            cache[field] = torch.stack(
                [
                    torch.stack([getattr(s, field) for s in summary_list])
                    for summary_list in summary_lists
                ]
            )
        self._stack_cache_keys = keys
        self._stack_cache = cache
        self._stack_cache_dirty = False

    def _on_query_captured(self, req_id: str, query_repr: torch.Tensor) -> None:
        if not self.durable_summaries:
            return
        # query_repr: [num_kv_heads, head_dim]. summaries are fp32 (Step 1.2
        # upcast).
        query_repr = query_repr.float()

        # Score and keep EVERY resident summary, not just a top-M slice. A
        # prior `_TOP_M=8` cap here meant most of a request's own blocks
        # never accumulated a relevance-EMA entry at all (fewer than 8 of
        # them would ever rank in some single query's own top-8, regardless
        # of how small the resident pool was -- confirmed unconditional, not
        # a crowding effect, empirically at pool sizes 21-645), which was
        # the dominant reason Step 1.5's prefetch splice could never find a
        # fully-scored request to exact-match against (issues log entries
        # #29-31).
        #
        # This used to score ALL THREE methods here via a Python loop with
        # one `.item()`-synchronizing GPU call per (method, candidate,
        # head) -- correct, but only `self._method` is ever actually read
        # (SemanticPolicy only consults its own configured method), so the
        # other two were pure waste, and per-candidate `.item()` calls force
        # a CPU/GPU sync on every single one, serializing what should be one
        # batched op. Measured "no regression" for dropping the top-M cap
        # (entry #31) was on the tiny dev model with a small (~dozens-to-
        # hundreds candidates) resident pool -- at production scale (larger
        # model, larger CPU tier, thousands of resident candidates) this
        # per-candidate-sync design turned multi-second per query capture,
        # showing up as catastrophic TTFT with completely normal ITL (the
        # ongoing-decode path never touches this code) -- found via a real
        # B200 calibration run, not assumed. Fixed by scoring the whole
        # pool as one vectorized pass (one sync total instead of one per
        # candidate) -- and, since even the batched version still measured
        # a large TTFT gap on a real B200 retest, by caching the stacked
        # tensors themselves (_rebuild_stack_cache) so rebuilding them from
        # the durable_summaries dict -- itself a real Python-level cost at
        # thousands of candidates -- only happens on actual insertions, not
        # on every query capture.
        # TEMPORARY timing instrumentation (issues log entry #53's follow-
        # up): two real fixes already landed here but a real B200 retest
        # still shows a large TTFT gap vs. lru after both -- rather than
        # guess at a third cause, measure where the remaining time actually
        # goes. Remove once the bottleneck is identified and fixed.
        t0 = time.perf_counter()
        rebuilt = self._stack_cache_dirty
        if self._stack_cache_dirty:
            self._rebuild_stack_cache()
        t1 = time.perf_counter()
        keys = self._stack_cache_keys
        cache = self._stack_cache

        query = query_repr.unsqueeze(0)  # [1, num_kv_heads, head_dim]
        if self._method == "minmax":
            per_head = score_minmax_batch(query, cache["max"], cache["min"])
        elif self._method == "mean":
            per_head = score_mean_batch(query, cache["mean"])
        else:  # cuboid_mean
            per_head = score_cuboid_mean_batch(query, cache["mean"], cache["mad"])
        # Per-head score, combined via max across heads -- different KV
        # heads may specialize on different content, so the head most
        # aligned with this query should drive the block's relevance
        # (entry #9). One sync for the whole batch (.tolist()), not one
        # per candidate.
        scores = per_head.max(dim=-1).values.tolist()
        t2 = time.perf_counter()
        ranked = sorted(zip(keys, scores), key=lambda kv: kv[1], reverse=True)
        t3 = time.perf_counter()
        self._pending_scores.setdefault(self._method, {})[req_id] = ranked
        debug_print(
            f"SEMANTIC_STEP1_3_TIMING req={req_id} n_summaries={len(keys)} "
            f"rebuilt={rebuilt} rebuild_ms={(t1 - t0) * 1000:.2f} "
            f"score_ms={(t2 - t1) * 1000:.2f} sort_ms={(t3 - t2) * 1000:.2f} "
            f"total_ms={(t3 - t0) * 1000:.2f}"
        )
        debug_print(
            f"SEMANTIC_STEP1_3_DEBUG req={req_id} method={self._method} "
            f"n_summaries={len(self.durable_summaries)} "
            f"ranked_keys={[k.hex()[:8] for k, _ in ranked]} "
            f"scores={[round(s, 4) for _, s in ranked]}"
        )

    def _check_layout(self, layer_name: str, layer, kv_cache: torch.Tensor) -> None:
        if layer_name in self._layout_checked:
            return
        self._layout_checked.add(layer_name)
        expected = layer.attn_backend.get_kv_cache_shape(
            kv_cache.shape[0], self._block_size, layer.num_kv_heads, layer.head_size
        )
        assert tuple(kv_cache.shape) == expected, (
            f"layer {layer_name}: kv_cache.shape={tuple(kv_cache.shape)} does not "
            f"match get_kv_cache_shape()={expected} -- this backend does not "
            "match the (num_blocks, num_kv_heads, block_size, 2*head_size) "
            "layout this module assumes; do not trust summaries built from it."
        )

    def _build_summaries_for_blocks(self, job_id: int, block_ids) -> None:
        """Overlaps the reduction work with inference via a side stream when
        CUDA is available; falls back to running it inline (still correct,
        just not overlapped) so this logic stays testable without a GPU.
        Durable re-keying (by OffloadKey) happens on the same stream, right
        after, so it observes the just-computed summaries in issue order --
        no separate cross-stream synchronization needed."""
        if len(block_ids) == 0:
            return
        if self._summary_stream is not None:
            current_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self._summary_stream):
                self._summary_stream.wait_stream(current_stream)
                self._build_summaries_body(block_ids)
                self._durably_key_summaries(job_id, block_ids)
        else:
            self._build_summaries_body(block_ids)
            self._durably_key_summaries(job_id, block_ids)

    def _build_summaries_body(self, block_ids) -> None:
        for layer_name, layer in self._attention_layers.items():
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is None or kv_cache.numel() == 0:
                continue
            self._check_layout(layer_name, layer, kv_cache)
            head_size = layer.head_size
            layer_summaries = self.summaries[layer_name]
            for block_id in block_ids:
                block_id = int(block_id)
                # block: [num_kv_heads, block_size, 2 * head_size]
                block = kv_cache[block_id]
                keys = block[..., :head_size].float()  # K half, upcast for MAD
                layer_summaries[block_id] = [
                    build_summary(keys[h]) for h in range(keys.shape[0])
                ]

    def _durably_key_summaries(self, job_id: int, block_ids) -> None:
        """Re-key this job's probe-layer summaries by OffloadKey so they
        outlive GPU block reuse. Job-level granularity: a store job's keys
        are attributed to all of that job's blocks (see connector.py's
        module docstring and the issues log for why, and the known blur this
        causes when one job covers many blocks)."""
        if self._probe_layer_name is None:
            return
        keys = self._pending_job_keys.pop(job_id, None)
        if not keys:
            return
        probe_summaries = self.summaries.get(self._probe_layer_name, {})
        # list[block] of list[BlockSummary] (one per KV head, unaveraged).
        block_summaries = [
            probe_summaries[int(b)] for b in block_ids if int(b) in probe_summaries
        ]
        if not block_summaries:
            return
        debug_print(
            f"SEMANTIC_STEP1_3_DEBUG store job={job_id} "
            f"keys={[k.hex()[:8] for k in keys]} block_ids={list(block_ids)}"
        )
        if len(block_summaries) == len(keys):
            for key, summary in zip(keys, block_summaries):
                self.durable_summaries[key] = summary
        else:
            # Can't establish precise per-block correspondence for this job;
            # fall back to the job's most-recent block as a best-effort
            # proxy for all of its keys rather than dropping the signal.
            for key in keys:
                self.durable_summaries[key] = block_summaries[-1]
        self._stack_cache_dirty = True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        self._build_summaries_for_blocks(job_id, src_spec.block_ids)
        return super().submit_store(job_id, src_spec, dst_spec)

    def splice_gpu_blocks(
        self, src_block_ids: list[int], dst_block_ids: list[int]
    ) -> None:
        """Step 1.5: copy already-GPU-resident block content from
        `src_block_ids` to `dst_block_ids`, across every attention layer --
        moves a speculatively-prefetched block into a request's officially
        allocated one, avoiding a redundant CPU-tier reload for content
        that's already warm. Reuses the same `layer.kv_cache` tensor access
        validated in Step 1.2. Unlike vllm's own `copy_kv_blocks`
        (`CopyBlocksOp`), which is typed only for host<->device transfers,
        this is a same-tensor, same-device index copy -- ordinary torch
        indexing, no platform-specific op needed.

        Deliberately runs on the current (default) stream, not a side
        stream: this must complete, in program order, before the
        subsequent forward pass reads these blocks, and same-stream
        execution guarantees that ordering for free. Step 1.2's summary
        computation uses a side stream because its timing is not
        correctness-critical (a slightly-stale summary is harmless); a
        stale/incomplete splice would serve corrupted KV data, so ordinary
        in-order execution is the safer choice here, not an oversight.
        """
        if not src_block_ids or not dst_block_ids:
            return
        assert len(src_block_ids) == len(dst_block_ids)
        for layer in self._attention_layers.values():
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is None or kv_cache.numel() == 0:
                continue
            src_idx = torch.tensor(
                src_block_ids, device=kv_cache.device, dtype=torch.long
            )
            dst_idx = torch.tensor(
                dst_block_ids, device=kv_cache.device, dtype=torch.long
            )
            kv_cache.index_copy_(0, dst_idx, kv_cache.index_select(0, src_idx))
