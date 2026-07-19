# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingManager: CPUOffloadingManager with a SemanticPolicy.

CachePolicy is not pluggable by name in vLLM core -- CPUOffloadingManager
only resolves "lru" and "arc" via its own _CACHE_POLICIES dict (see
vllm/v1/kv_offload/cpu/manager.py). We build via the "lru" path (so the base
class's block-pool/ref-count setup runs untouched) and then swap in our own
policy instance afterward. See .claude/docs/semantic-eviction-plan.md,
Step 1.1.
"""

from collections.abc import Collection

from typing_extensions import override

from semantic_offload.policy import SemanticPolicy
from vllm.v1.kv_offload.base import OffloadKey, PrepareStoreOutput, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

_EMA_ALPHA = 0.3  # ceiling weight on the newest observation; see update_relevance
_EMA_RANK_POWER = 1.0  # unturned, like _DEFAULT_ALPHA below
# minmax was the strongest per-KV-head scoring method in the real needle
# test (issues log entry #9: rank 2/580 for the answer-critical block, vs.
# cuboid_mean's 1/580 and mean's 73/580) -- picking minmax over cuboid_mean
# here is not yet A/B'd against cuboid_mean for eviction quality specifically,
# just carried over as "best observed ranking signal so far."
_DEFAULT_METHOD = "minmax"
_DEFAULT_ALPHA = 0.5  # unturned; plan's Step 1.4 explicitly warns against 1.0.
# A diagnostic run at alpha=0.95 gave an identical outcome to 0.5 (issues log
# entry #10) -- the real bottleneck there was a timing/cold-start issue
# (relevance signal arriving after the block was already evicted), not
# insufficient semantic weighting, so raising the default wouldn't help.


class SemanticOffloadingManager(CPUOffloadingManager):
    def __init__(
        self,
        num_blocks: int,
        enable_events: bool = False,
        store_threshold: int = 0,
        max_tracker_size: int = 64_000,
        grace_window_blocks: int = 0,
        eviction_mode: str = "blend",
        chain_aware: bool = False,
        session_aware: bool = False,
        session_bonus_half_life: int = 0,
        method: str = _DEFAULT_METHOD,
    ) -> None:
        super().__init__(
            num_blocks=num_blocks,
            cache_policy="lru",
            enable_events=enable_events,
            store_threshold=store_threshold,
            max_tracker_size=max_tracker_size,
        )
        # method -> OffloadKey -> EMA-smoothed relevance score (Step 1.3).
        # Passed into SemanticPolicy by reference -- update_relevance()
        # mutates this dict in place and evict() reads it live (Step 1.4).
        self.relevance_ema: dict[str, dict[OffloadKey, float]] = {}
        # Real eviction notifications for the worker's durable_summaries
        # (issues log entries #62-64) -- see prepare_store() below. Drained
        # by SemanticOffloadingConnectorScheduler.build_connector_meta()
        # each step, the same accumulate-then-drain pattern already used for
        # _pending_job_keys.
        self._pending_evicted_keys: list[OffloadKey] = []
        self._policy = SemanticPolicy(
            cache_capacity=num_blocks,
            relevance_ema=self.relevance_ema,
            method=method,
            alpha=_DEFAULT_ALPHA,
            grace_window_blocks=grace_window_blocks,
            mode=eviction_mode,
            chain_aware=chain_aware,
            session_aware=session_aware,
            session_bonus_half_life=session_bonus_half_life,
        )

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """Wraps the base CPUOffloadingManager.prepare_store() purely to
        observe `PrepareStoreOutput.evicted_keys` -- the base class already
        computes exactly which keys the real CachePolicy just evicted to
        make room for this store (manager.py's own `evict()` call), it just
        never surfaced that anywhere. Buffered here (scheduler-side) and
        drained by the connector scheduler into per-step metadata, mirroring
        _pending_job_keys' accumulate-then-drain pattern, so the worker can
        learn precisely which blocks are gone instead of approximating with
        a FIFO cap (issues log entries #62-64 -- that cap remains as a
        defensive backstop in worker.py, not the primary mechanism anymore)."""
        result = super().prepare_store(keys, req_context)
        if result is not None and result.evicted_keys:
            self._pending_evicted_keys.extend(result.evicted_keys)
        return result

    def drain_evicted_keys(self) -> list[OffloadKey]:
        """Pop and return all evicted keys accumulated since the last drain
        (per-step, from SemanticOffloadingConnectorScheduler.build_connector_meta)."""
        keys = self._pending_evicted_keys
        self._pending_evicted_keys = []
        return keys

    def update_relevance(
        self, scores: dict[str, dict[str, list[tuple[OffloadKey, float]]]]
    ) -> None:
        """Fold in this step's worker-reported scores via an EMA, so one
        noisy step can't thrash the cache (plan's Step 1.3 requirement).
        `scores` is method -> req_id -> ranked [(key, score), ...], already
        sorted highest-score-first per query (worker.py's `_on_query_captured`);
        req_id isn't needed for the EMA itself (relevance is per-block, not
        per-requester) so it's flattened away here.

        Every candidate in `ranked` is touched (not just a top-K slice) --
        an earlier `_TOP_M` cap that scored/updated only the top-8 per query
        was removed because most resident blocks would then never earn any
        EMA entry at all, regardless of pool size (issues log entries
        #29-31). But touching every candidate with the *same* flat alpha
        turned out to have its own failure mode (entry #60): a topically
        unrelated distractor query ranks the needle near the bottom of its
        own candidate list and, at full alpha, actively drags an
        already-high needle score back down -- a handful of irrelevant
        queries erase one strong probe's signal well before real capacity
        pressure ever forces an eviction decision. Fix: scale the per-key
        weight by that key's *rank within this query's own ranked list*
        (not an absolute top-K cutoff, so coverage is preserved) -- a
        candidate this query considers most-relevant gets close to the
        full ceiling weight, one it considers least-relevant gets a weight
        near zero, so it keeps its prior score rather than being pulled
        toward this query's near-irrelevant one."""
        for method, per_req in scores.items():
            ema = self.relevance_ema.setdefault(method, {})
            for ranked in per_req.values():
                n = len(ranked)
                denom = max(n - 1, 1)
                for rank, (key, new_score) in enumerate(ranked):
                    frac = rank / denom  # 0.0 = this query's top pick
                    weight = _EMA_ALPHA * (1.0 - frac) ** _EMA_RANK_POWER
                    prev = ema.get(key)
                    ema[key] = (
                        new_score
                        if prev is None
                        else weight * new_score + (1 - weight) * prev
                    )

    def ranked_keys(self, method: str) -> list[tuple[OffloadKey, float]]:
        """Current EMA-ranked relevance for a scoring method, highest first.
        Read-only introspection for Step 1.3's accept check; Step 1.4 is
        where SemanticPolicy.evict() actually consumes this."""
        ema = self.relevance_ema.get(method, {})
        return sorted(ema.items(), key=lambda kv: kv[1], reverse=True)

    def top_relevant_keys(
        self, candidate_keys: list[OffloadKey], k: int, method: str = _DEFAULT_METHOD
    ) -> list[OffloadKey]:
        """Step 1.5: among `candidate_keys` (typically one request's own
        chain), the `k` with the highest current relevance score -- unlike
        `ranked_keys`, scoped to a caller-given candidate set rather than
        every key ever scored, and silently skips unscored keys (a
        preempted request's own blocks may not have earned a score yet;
        prefetching an arbitrary unscored block isn't better-informed than
        picking any other block at random, so there's nothing to prefer)."""
        if k <= 0:
            return []
        ema = self.relevance_ema.get(method, {})
        scored = [(key, ema[key]) for key in candidate_keys if key in ema]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [key for key, _ in scored[:k]]
