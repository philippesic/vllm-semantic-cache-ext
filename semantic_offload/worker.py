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

import itertools
import time

import torch

from semantic_offload._debug import TIMING as _TIMING
from semantic_offload._debug import debug_print, record_timing
from semantic_offload._vllm_compat import init_cpu_offloading_worker_base
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
        blocks_per_chunk: int,
        num_cpu_blocks: int,
        vllm_config: VllmConfig,
        method: str = "minmax",
        capture_stride: int = 1,
    ):
        # CPUOffloadingWorker's own param is named block_size_factor or
        # blocks_per_chunk depending on the installed vLLM's version
        # (#48150 renamed it on some checkouts, not others still in use
        # across this project's machines) -- call via whichever name the
        # real base class actually declares. See _vllm_compat.py.
        init_cpu_offloading_worker_base(
            self,
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
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
        #
        # DEFENSIVE BACKSTOP, not the primary mechanism anymore (issues log
        # entries #62-64): receive_evicted_keys() now gets told precisely
        # which keys the real CachePolicy evicted, each step, and removes
        # exactly those -- the correct fix. This FIFO-by-insertion-order cap
        # was the original stopgap (bounds the unbounded growth but can
        # diverge from the manager's real eviction order, e.g. dropping a
        # block the real tier still holds); it's kept only to bound worst-
        # case growth from any removal path receive_evicted_keys doesn't yet
        # cover (a failed store's cleanup, or a full cache reset -- both
        # rare relative to the normal evict() path this now handles
        # precisely). Should rarely-to-never trigger in practice once
        # receive_evicted_keys is wired up correctly.
        self._max_durable_summaries = max(num_cpu_blocks, 1)
        self.durable_summaries: dict[OffloadKey, list[BlockSummary]] = {}
        self._pending_job_keys: dict[int, list[OffloadKey]] = {}
        self._pending_scores: dict[str, dict[str, list[tuple[OffloadKey, float]]]] = {}
        # Cache of durable_summaries stacked into batched tensors, kept in
        # sync incrementally (see _rebuild_stack_cache) rather than rebuilt
        # from scratch on every query capture -- query captures fire on
        # nearly every prefill/mixed step, while any single step only ever
        # inserts/evicts a handful of candidates, so a from-scratch rebuild
        # over the WHOLE resident pool on every dirty step was the dominant
        # cost in the entire hot path at real production scale (issues log
        # entries #53's follow-up, #72, #74).
        self._stack_cache_dirty = True
        self._stack_cache_keys: list[OffloadKey] = []
        self._stack_cache: dict[str, torch.Tensor] = {}
        self._stack_cache_index: dict[OffloadKey, int] = {}
        self._stack_pending_insert: set[OffloadKey] = set()
        self._stack_pending_remove: set[OffloadKey] = set()
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
                self._on_queries_captured,
                num_queries_per_kv=num_queries_per_kv,
                capture_stride=capture_stride,
            )

    def receive_job_keys(self, store_job_keys: dict[int, list[OffloadKey]]) -> None:
        """Called by SemanticOffloadingConnector before this step's
        submit_store() calls, with the OffloadKey(s) each pending store
        job_id represents (scheduler-side info the worker has no other way
        to see -- see connector.py and the issues log entry #6/7)."""
        self._pending_job_keys.update(store_job_keys)

    def receive_evicted_keys(self, evicted_keys: list[OffloadKey]) -> None:
        """Called by SemanticOffloadingConnector with exactly the keys the
        manager's real CachePolicy evicted this step (issues log entries
        #62-64) -- the precise fix for durable_summaries' unbounded growth.
        Replaces the FIFO cap (`_prune_durable_summaries`) as the primary
        mechanism; that cap stays as a defensive backstop for any removal
        path this doesn't cover (known gap: a failed store's cleanup in the
        manager's complete_store(), and reset_cache()'s full clear -- both
        rare relative to the evict() path that dominated the growth, not
        yet wired through)."""
        if not evicted_keys:
            return
        removed = [
            k for k in evicted_keys if self.durable_summaries.pop(k, None) is not None
        ]
        if removed:
            self._mark_removed_from_stack_cache(removed)
        debug_print(
            f"SEMANTIC_EVICT_DEBUG received={len(evicted_keys)} "
            f"removed={len(removed)} resident={len(self.durable_summaries)}"
        )

    def _mark_inserted_into_stack_cache(self, keys) -> None:
        """Record keys as needing a row in the stacked-tensor cache. A key
        that already has a synced row (an overwrite -- same OffloadKey
        re-stored with new content) is also queued for removal of its stale
        row, so the sync ends up with exactly one, up-to-date row per key."""
        for key in keys:
            if key in self._stack_cache_index:
                self._stack_pending_remove.add(key)
            self._stack_pending_insert.add(key)
        self._stack_cache_dirty = True

    def _mark_removed_from_stack_cache(self, keys) -> None:
        """Record keys as needing to be dropped from the stacked-tensor
        cache. A key never yet synced in (still only pending-insert) is
        simply un-queued -- no row was ever created for it."""
        for key in keys:
            self._stack_pending_insert.discard(key)
            if key in self._stack_cache_index:
                self._stack_pending_remove.add(key)
        self._stack_cache_dirty = True

    def pop_pending_scores(
        self,
    ) -> dict[str, dict[str, list[tuple[OffloadKey, float]]]]:
        scores = self._pending_scores
        self._pending_scores = {}
        return scores

    def _rebuild_stack_cache(self) -> None:
        """Bring the stacked-tensor cache back in sync with
        durable_summaries incrementally: compact out pending removals with
        one vectorized boolean-mask op per field, then stack and append only
        the newly-inserted candidates -- instead of a from-scratch Python
        getattr+torch.stack rebuild over EVERY resident candidate on every
        dirty query capture. See the dirty-flag comment in __init__: on
        `chat`-style traffic the cache is dirtied on nearly every step while
        only a handful of candidates actually change per step, so the old
        full rebuild was by far the dominant cost in the whole hot path
        (40-100x every other bucket -- issues log entries #72/#74)."""
        fields = _METHOD_FIELDS[self._method]

        if self._stack_pending_remove:
            keep_mask = torch.ones(len(self._stack_cache_keys), dtype=torch.bool)
            for key in self._stack_pending_remove:
                idx = self._stack_cache_index.pop(key, None)
                if idx is not None:
                    keep_mask[idx] = False
            surviving_keys = [
                k for k, keep in zip(self._stack_cache_keys, keep_mask.tolist()) if keep
            ]
            for field in fields:
                self._stack_cache[field] = self._stack_cache[field][keep_mask]
            self._stack_cache_keys = surviving_keys
            self._stack_cache_index = {k: i for i, k in enumerate(surviving_keys)}
            self._stack_pending_remove.clear()

        if self._stack_pending_insert:
            new_keys = [
                k for k in self._stack_pending_insert if k in self.durable_summaries
            ]
            self._stack_pending_insert.clear()
            if new_keys:
                new_summary_lists = [self.durable_summaries[k] for k in new_keys]
                for field in fields:
                    # [n_new, num_kv_heads, head_dim]
                    new_stack = torch.stack(
                        [
                            torch.stack([getattr(s, field) for s in summary_list])
                            for summary_list in new_summary_lists
                        ]
                    )
                    existing = self._stack_cache.get(field)
                    self._stack_cache[field] = (
                        torch.cat([existing, new_stack], dim=0)
                        if existing is not None and existing.numel()
                        else new_stack
                    )
                base = len(self._stack_cache_keys)
                for offset, key in enumerate(new_keys):
                    self._stack_cache_index[key] = base + offset
                self._stack_cache_keys.extend(new_keys)

        self._stack_cache_dirty = False

    def _on_query_captured(self, req_id: str, query_repr: torch.Tensor) -> None:
        """Single-request convenience wrapper over `_on_queries_captured`
        (tests and any external caller; the live capture path calls the
        batched form directly, once per step)."""
        self._on_queries_captured([req_id], query_repr.unsqueeze(0))

    def _on_queries_captured(self, req_ids: list[str], queries: torch.Tensor) -> None:
        if not self.durable_summaries or not req_ids:
            return
        _t_call = time.perf_counter() if _TIMING else 0.0
        # queries: [n_reqs, num_kv_heads, head_dim] -- one row per request
        # scheduled in this step. All concurrent requests are scored in ONE
        # batched pass with ONE GPU sync per step, instead of one pass-plus-
        # sync per request: per-call cost is flat (~1ms), but the capture
        # loop fires once per scheduled request per eligible step, so at
        # saturation the per-request loop's aggregate (~n_reqs x ~1ms/step,
        # n_reqs measured ~56 on the dev box) was the largest per-step cost
        # bucket -- see the cross-request batching entry in the issues log.
        # summaries are fp32 (Step 1.2 upcast).
        queries = queries.float()

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
        # Scores only `self._method` (not all three) via one vectorized
        # batched pass over the whole candidate pool (index.py's
        # score_*_batch functions) -- one GPU sync total, not one per
        # candidate -- against a cache of the pool stacked into tensors,
        # rebuilt only when new candidates are inserted
        # (_rebuild_stack_cache), not on every query capture. Found and
        # fixed on a real B200 production-scale run (issues log entry #53):
        # the original per-candidate `.item()`-synchronizing design and the
        # per-query stack rebuild were both measured as "near-zero cost" on
        # the tiny dev model's small resident pool, but turned multi-second
        # per query capture at production scale (larger model, larger CPU
        # tier, thousands of resident candidates) -- see entry #53 for the
        # full investigation, including the third contributing bug (worker
        # was building summaries for every model layer, not just this one)
        # fixed in _build_summaries_body below.
        if self._stack_cache_dirty:
            _t_rebuild = time.perf_counter() if _TIMING else 0.0
            self._rebuild_stack_cache()
            if _TIMING:
                record_timing("stack_rebuild", time.perf_counter() - _t_rebuild)
        keys = self._stack_cache_keys
        cache = self._stack_cache

        # [n_reqs, 1, num_kv_heads, head_dim] broadcasts against the
        # [n_candidates, num_kv_heads, head_dim] stacks to score every
        # (request, candidate) pair in one op -- the score_*_batch functions
        # are broadcast-shape-agnostic, so no index.py change is needed.
        query = queries.unsqueeze(1)
        if self._method == "minmax":
            per_head = score_minmax_batch(query, cache["max"], cache["min"])
        elif self._method == "mean":
            per_head = score_mean_batch(query, cache["mean"])
        else:  # cuboid_mean
            per_head = score_cuboid_mean_batch(query, cache["mean"], cache["mad"])
        # Per-head score, combined via max across heads -- different KV
        # heads may specialize on different content, so the head most
        # aligned with this query should drive the block's relevance
        # (entry #9). One sync for the whole step's batch (.tolist()), not
        # one per request or per candidate.
        _t_sync = time.perf_counter() if _TIMING else 0.0
        # [n_reqs, n_candidates] -> list of per-request score lists.
        all_scores = per_head.max(dim=-1).values.tolist()
        if _TIMING:
            record_timing("query_captured_sync", time.perf_counter() - _t_sync)
        method_scores = self._pending_scores.setdefault(self._method, {})
        for req_id, scores in zip(req_ids, all_scores):
            ranked = sorted(zip(keys, scores), key=lambda kv: kv[1], reverse=True)
            method_scores[req_id] = ranked
            debug_print(
                f"SEMANTIC_STEP1_3_DEBUG req={req_id} method={self._method} "
                f"n_summaries={len(self.durable_summaries)} "
                f"ranked_keys={[k.hex()[:8] for k, _ in ranked]} "
                f"scores={[round(s, 4) for _, s in ranked]}"
            )
        if _TIMING:
            record_timing("query_captured_total", time.perf_counter() - _t_call)

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
        # Real bug found on a B200 production-scale run (issues log entry
        # #53's follow-up): this used to loop over EVERY attention layer in
        # the model (self._attention_layers, ~28 for a 7B model), but
        # self.summaries is only ever READ for self._probe_layer_name
        # (_durably_key_summaries below, the only call site) -- every other
        # layer's summary was real, non-trivial tensor work (4 reductions
        # per KV head per block) computed and thrown away, unread, on
        # every single block stored. Restricting to just the probe layer
        # cuts this by roughly the model's layer count.
        if self._probe_layer_name is None:
            return
        layer = self._attention_layers.get(self._probe_layer_name)
        if layer is None:
            return
        kv_cache = getattr(layer, "kv_cache", None)
        if kv_cache is None or kv_cache.numel() == 0:
            return
        self._check_layout(self._probe_layer_name, layer, kv_cache)
        head_size = layer.head_size
        layer_summaries = self.summaries[self._probe_layer_name]
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
        self._mark_inserted_into_stack_cache(keys)
        pruned = self._prune_durable_summaries()
        if pruned:
            self._mark_removed_from_stack_cache(pruned)

    def _prune_durable_summaries(self) -> list[OffloadKey]:
        """Bound `durable_summaries` to (approximately) the real CPU tier's
        capacity -- see the `_max_durable_summaries` comment in __init__ for
        why this exists and its known imprecision. Evicts oldest-inserted
        first: plain dicts preserve insertion order, so this needs no extra
        bookkeeping, and insertion order is a reasonable proxy for "still
        resident" absent a real eviction signal. Returns the pruned keys so
        the caller can also drop their (possibly already-synced) rows from
        the stack cache -- this path bypasses receive_evicted_keys, so
        without this a pruned key's stale row would otherwise linger in the
        cache and keep being scored forever."""
        overflow = len(self.durable_summaries) - self._max_durable_summaries
        if overflow <= 0:
            return []
        pruned = list(itertools.islice(self.durable_summaries, overflow))
        for key in pruned:
            del self.durable_summaries[key]
        return pruned

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
