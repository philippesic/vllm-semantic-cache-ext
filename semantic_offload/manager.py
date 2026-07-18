# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingManager: CPUOffloadingManager with a SemanticPolicy.

CachePolicy is not pluggable by name in vLLM core -- CPUOffloadingManager
only resolves "lru" and "arc" via its own _CACHE_POLICIES dict (see
vllm/v1/kv_offload/cpu/manager.py). We build via the "lru" path (so the base
class's block-pool/ref-count setup runs untouched) and then swap in our own
policy instance afterward. See .claude/docs/semantic-eviction-plan.md,
Step 1.1.
"""

from semantic_offload.policy import SemanticPolicy
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

_EMA_ALPHA = 0.3  # weight on the newest observation; see update_relevance
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
        self._policy = SemanticPolicy(
            cache_capacity=num_blocks,
            relevance_ema=self.relevance_ema,
            method=_DEFAULT_METHOD,
            alpha=_DEFAULT_ALPHA,
            grace_window_blocks=grace_window_blocks,
            mode=eviction_mode,
            chain_aware=chain_aware,
            session_aware=session_aware,
            session_bonus_half_life=session_bonus_half_life,
        )

    def update_relevance(
        self, scores: dict[str, dict[str, list[tuple[OffloadKey, float]]]]
    ) -> None:
        """Fold in this step's worker-reported scores via an EMA, so one
        noisy step can't thrash the cache (plan's Step 1.3 requirement).
        `scores` is method -> req_id -> ranked [(key, score), ...]; req_id
        isn't needed for the EMA itself (relevance is per-block, not
        per-requester) so it's flattened away here."""
        for method, per_req in scores.items():
            ema = self.relevance_ema.setdefault(method, {})
            for ranked in per_req.values():
                for key, new_score in ranked:
                    prev = ema.get(key)
                    ema[key] = (
                        new_score
                        if prev is None
                        else _EMA_ALPHA * new_score + (1 - _EMA_ALPHA) * prev
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
