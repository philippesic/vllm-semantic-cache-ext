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

import torch

from semantic_offload.index import (
    BlockSummary,
    build_summary,
    score,
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


class SemanticOffloadingWorker(CPUOffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        vllm_config: VllmConfig,
    ):
        super().__init__(
            kv_caches=kv_caches,
            block_size_factor=block_size_factor,
            num_cpu_blocks=num_cpu_blocks,
        )
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

    def _on_query_captured(self, req_id: str, query_repr: torch.Tensor) -> None:
        if not self.durable_summaries:
            return
        # query_repr: [num_kv_heads, head_dim]. summaries are fp32 (Step 1.2
        # upcast).
        query_repr = query_repr.float()
        for method in _SCORING_METHODS:
            # Score and keep EVERY resident summary, not just a top-M slice.
            # A prior `_TOP_M=8` cap here meant most of a request's own
            # blocks never accumulated a relevance-EMA entry at all (fewer
            # than 8 of them would ever rank in some single query's own
            # top-8, regardless of how small the resident pool was --
            # confirmed unconditional, not a crowding effect, empirically
            # at pool sizes 21-645), which was the dominant reason Step
            # 1.5's prefetch splice could never find a fully-scored request
            # to exact-match against (issues log entries #29-31). The
            # per-candidate score computation below already costs the same
            # either way (it runs before any truncation); measured real-
            # server ITL showed no regression from dropping the cap
            # (entry #31: ~10.95-11.0 ms/token, unbounded vs. capped at 8,
            # statistically indistinguishable).
            ranked = sorted(
                (
                    (
                        key,
                        # Per-head score, combined via max across heads --
                        # different KV heads may specialize on different
                        # content, so the head most aligned with this query
                        # should drive the block's relevance (entry #9).
                        max(
                            score(method, query_repr[h], summary_list[h])
                            for h in range(len(summary_list))
                        ),
                    )
                    for key, summary_list in self.durable_summaries.items()
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
            self._pending_scores.setdefault(method, {})[req_id] = ranked
            print(
                f"SEMANTIC_STEP1_3_DEBUG req={req_id} method={method} "
                f"n_summaries={len(self.durable_summaries)} "
                f"ranked_keys={[k.hex()[:8] for k, _ in ranked]} "
                f"scores={[round(s, 4) for _, s in ranked]}",
                flush=True,
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
        print(
            f"SEMANTIC_STEP1_3_DEBUG store job={job_id} "
            f"keys={[k.hex()[:8] for k in keys]} block_ids={list(block_ids)}",
            flush=True,
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
