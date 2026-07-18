# SPDX-License-Identifier: Apache-2.0
"""SemanticPolicy: CachePolicy that blends semantic relevance with recency.

get/insert/remove/touch/clear/mark_evictable/mark_non_evictable all delegate
to an internal LRUCachePolicy unchanged -- LRU's own bookkeeping (evictable
order, ref counts) stays correct and is the source of truth for recency.
Only evict() is overridden: it reads LRUCachePolicy's real recency order
non-destructively (`self._lru.evictable_blocks`, a plain `OrderedDict` --
reading a composed instance's own attribute from our own code, not a vLLM
source edit, same class of technique as reading `layer.kv_cache` directly in
Step 1.2) rather than calling `LRUCachePolicy.evict()` as a "peek", which
would mutate its state as a side effect and corrupt the comparison. This is a
first, minimal slice of the plan's Step 1.4 -- not its full design (no
per-method selection surfaced yet, alpha is a constructor arg not a runtime
config knob). See .claude/docs/semantic-eviction-plan.md Step 1.4 and
issues log entries #10 and #11 (grace period -- temporary eviction immunity
for freshly-inserted, as-yet-unscored blocks).
"""

import time
from collections.abc import Iterable

from typing_extensions import override

from semantic_offload._debug import debug_print
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class SemanticPolicy(CachePolicy):
    """CachePolicy blending semantic relevance with recency (Step 1.4 slice).

    `score = alpha * relevance_norm + (1 - alpha) * recency_norm`, both
    normalized to [0, 1] over the current evictable set; blocks with no
    relevance score yet fall back to pure recency (plan's Step 1.4 text).
    Blocks with the lowest blended score are evicted first.
    """

    def __init__(
        self,
        cache_capacity: int,
        relevance_ema: dict[str, dict[OffloadKey, float]] | None = None,
        method: str = "minmax",
        alpha: float = 0.5,
        grace_window_blocks: int = 0,
        mode: str = "blend",
        chain_aware: bool = False,
        session_aware: bool = False,
        session_bonus_half_life: int = 0,
    ) -> None:
        self._lru = LRUCachePolicy(cache_capacity=cache_capacity)
        # Shared mutable reference into SemanticOffloadingManager.relevance_ema
        # -- updated in place by update_relevance() each step; we just read it.
        self._relevance_ema = relevance_ema if relevance_ema is not None else {}
        self._method = method
        self._alpha = alpha
        # "blend" (default): score = alpha*relevance + (1-alpha)*recency, one
        # continuous ranking. "unscored_last" (issues log entry #11/#18): a
        # strict two-tier ordering with no timer -- every unscored block is
        # evicted before any scored block is touched at all, recency-ordered
        # within the unscored tier, relevance-ordered (lowest first) within
        # the scored tier. Never fixes entry #10-#12's cold-start MISS (the
        # whole candidate pool is unscored at the critical moment there, so
        # the "unscored tier" degenerates to plain LRU, same root cause as
        # grace's failure) -- its value proposition is different: once *some*
        # content legitimately earns a score, never evict it in favor of
        # never-referenced throwaway content, with no window to tune at all.
        assert mode in ("blend", "unscored_last"), mode
        self._mode = mode
        # Grace (entry #11) is only meaningful combined with "blend" -- it
        # was proven unable to fix the cold-start gap on its own, and
        # combining two already-closed-for-that-purpose ideas is out of
        # scope here; "unscored_last" ignores grace_window_blocks entirely.
        # Grace period (issues log entry #11): a freshly-inserted, as-yet-
        # unscored block is immune to eviction for `grace_window_blocks`
        # *blocks inserted* (not wall-clock -- capacity pressure here is
        # driven by store volume) after its own insertion. 0 disables it
        # entirely (matches entry #10's "regular" no-grace behavior exactly).
        # `insert()` is called once per block by manager.py's prepare_store
        # loop (verified against vllm/v1/kv_offload/cpu/manager.py, not
        # assumed), so a simple call counter is a correct proxy for "blocks
        # inserted so far."
        self._grace_window_blocks = grace_window_blocks
        self._insert_count = 0
        self._grace_expiry: dict[OffloadKey, int] = {}
        # Chain-aware ordering (issues log entry #15; upstream precedent
        # PR #47744, which is what put a real req_context onto touch() in
        # the first place). vllm's offloading scheduler calls
        # manager.touch(offload_keys, req_context) with a request's full
        # ordered block chain every time it's stored or hits
        # (offloading/scheduler.py's _touch) -- so consecutive keys in one
        # touch() call are adjacent chain positions. We remember, per key,
        # the key observed right after it. At eviction time a block whose
        # recorded successor is still resident gets a bonus: this empties a
        # chain tail-first instead of letting an early block get evicted
        # while later blocks of the *same* chain (now useless without it --
        # a broken prefix can't be partially reused) survive it. Only
        # wired into "blend" mode below; "unscored_last" is out of scope
        # for this slice.
        self._chain_aware = chain_aware
        self._chain_successor: dict[OffloadKey, OffloadKey] = {}
        # Session-proven priority (issues log entry #19's follow-up, after
        # the per-block chain-successor bonus above was found redundant with
        # LRU's own touch() semantics and closed). Targets a different gap
        # than content scoring or chain ordering: an idle-but-active
        # multi-turn chat session has no content evidence yet (that's the
        # already-closed cold-start gap, entries #10-#12) and no chain-
        # internal ordering problem (closed above) -- but it DOES have a
        # structural signal content scoring can't see: has a *different*
        # request already revisited (touched) this exact chain? Real multi-
        # turn conversations get a new req_id every turn (a fresh HTTP
        # request) while reusing the SAME content-addressed OffloadKeys for
        # the shared prefix -- so a touch() call whose keys were last
        # touched by a *different* req_id is direct, real evidence of cross-
        # request continuation, not a speculative bet on the future (unlike
        # a blanket grace period for all fresh blocks, which entry #11
        # proved mathematically can't work: the window long enough to
        # protect a genuine return visit is also long enough to blanket-
        # protect the flood of equally-fresh one-shot traffic). Once
        # detected, the *entire* current touch's key set (the request's full
        # known chain) is marked proven, persistently -- not time-limited,
        # since it's based on an observed fact, not a countdown.
        self._session_aware = session_aware
        self._last_touch_req_id: dict[OffloadKey, str] = {}
        self._session_proven: set[OffloadKey] = set()
        # Recency-weighted variant (issues log entry #21's follow-up): a flat
        # bonus is provably unable to distinguish an actively-used proven
        # session from an idle-but-proven one once eviction is choosing only
        # among proven candidates (adding the same constant to every
        # candidate can't change their relative order) -- entry #21 measured
        # this concretely as active sessions losing protection they'd have
        # had under plain LRU, to equalize with idle-but-proven sessions
        # that no longer need it. `session_bonus_half_life > 0` scales the
        # bonus down by how many touch() *calls* (not wall-clock, matching
        # entry #12's precedent of using event counts as the proxy unit)
        # have passed since this specific key was last touched, so a
        # recently-active proven session keeps close to the full bonus while
        # a long-idle one's bonus decays toward (but never reaches) zero --
        # 0 (the default) preserves entry #20's original flat-bonus
        # behavior unchanged.
        self._session_bonus_half_life = session_bonus_half_life
        self._touch_seq = 0
        self._last_touch_seq: dict[OffloadKey, int] = {}

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self._lru.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self._lru.insert(key, block)
        # Counter kept live unconditionally (not just when grace is enabled)
        # -- issues log entry #12 measured pipeline latency in these units
        # even in the no-grace ("regular") configuration, and a future
        # session may want the same "blocks inserted so far" timeline again
        # without re-adding this increment.
        self._insert_count += 1
        if self._grace_window_blocks > 0:
            self._grace_expiry[key] = self._insert_count + self._grace_window_blocks

    @override
    def remove(self, key: OffloadKey) -> None:
        self._lru.remove(key)
        self._grace_expiry.pop(key, None)
        self._chain_successor.pop(key, None)
        self._last_touch_req_id.pop(key, None)
        self._session_proven.discard(key)
        self._last_touch_seq.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        self._lru.touch(keys, req_context)
        if self._chain_aware:
            ordered = list(keys)
            for prev_key, next_key in zip(ordered, ordered[1:]):
                self._chain_successor[prev_key] = next_key
        if self._session_aware:
            ordered = list(keys)
            req_id = req_context.req_id
            cross_request_hit = any(
                self._last_touch_req_id.get(k) not in (None, req_id) for k in ordered
            )
            if cross_request_hit:
                self._session_proven.update(ordered)
            for k in ordered:
                self._last_touch_req_id[k] = req_id
            if self._session_bonus_half_life > 0:
                self._touch_seq += 1
                for k in ordered:
                    self._last_touch_seq[k] = self._touch_seq

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        # TEMPORARY timing instrumentation (issues log entry #53's follow-
        # up): all three costs measured so far in the worker's own scoring
        # path (_on_query_captured) are sub-millisecond on a real B200
        # calibration run, yet TTFT is still ~40x worse than lru -- this is
        # the one major remaining unmeasured code path (scheduler-side
        # eviction decisions), and real preemptions are confirmed happening
        # in the same run. Remove once the bottleneck is identified.
        t0 = time.perf_counter()
        if n == 0:
            return []
        # Non-destructive read of LRU's real recency order (oldest first --
        # touch() moves accessed keys to the end of this OrderedDict).
        candidates = [
            (key, self._lru.blocks[key])
            for key in self._lru.evictable_blocks
            if key not in protected
        ]
        if len(candidates) < n:
            return None
        t1 = time.perf_counter()

        if self._mode == "unscored_last":
            result = self._evict_unscored_last(n, candidates)
            debug_print(
                f"SEMANTIC_EVICT_TIMING mode=unscored_last n={n} "
                f"n_candidates={len(candidates)} "
                f"candidates_ms={(t1 - t0) * 1000:.2f} "
                f"total_ms={(time.perf_counter() - t0) * 1000:.2f}"
            )
            return result

        # Grace period (entry #11): a *soft* bonus, not a hard exclude. A
        # hard exclude-then-fallback-to-oldest-in-grace design was tried
        # first and found to degenerate to plain LRU order under sustained
        # capacity pressure: once enough blocks are resident that the whole
        # evictable set is still within its own grace window (e.g. a large
        # grace_window relative to workload churn), the "fallback" pool *is*
        # the whole candidate set, and picking oldest-first from it is
        # exactly what pure LRU does -- silently defeating the mechanism
        # (see issues log entry #11's diagnostic run). Instead, grace adds a
        # large constant bonus to a still-protected block's keep_score, so
        # it is essentially never evicted while any non-grace candidate
        # exists, but if eviction is ever forced to dip into the grace pool
        # anyway, relative ordering *within* that pool still follows the
        # real relevance+recency blend rather than raw recency.
        if self._grace_window_blocks > 0:
            in_grace = {
                key
                for key, _ in candidates
                if self._grace_expiry.get(key, -1) > self._insert_count
            }
        else:
            in_grace = set()

        relevance = self._relevance_ema.get(self._method, {})
        scored_vals = [relevance[k] for k, _ in candidates if k in relevance]
        lo = min(scored_vals) if scored_vals else 0.0
        hi = max(scored_vals) if scored_vals else 0.0
        span = (hi - lo) or 1.0
        total = len(candidates)
        # keep_score without the bonus is always in [0, 1]; +2.0 guarantees
        # a grace-protected block outranks (survives longer than) any
        # non-grace-protected block, while preserving relative order among
        # grace-protected blocks themselves.
        GRACE_BONUS = 2.0
        # Strictly less than GRACE_BONUS: grace's temporary immunity for any
        # freshly-inserted block should still win a tie-break against chain
        # protection, since grace covers a case (unscored blocks) chain
        # protection doesn't reason about at all. Strictly greater than the
        # base score's [0, 1] range, so a chain-protected block always
        # outranks every non-grace, non-chain candidate.
        CHAIN_BONUS = 1.5
        # Strictly greater than GRACE_BONUS: session-proven status is real,
        # observed cross-request evidence (a second distinct request
        # revisited this exact chain), not a speculative freshness bet --
        # it should outrank grace's temporary immunity for merely-unproven
        # fresh blocks in a direct tie-break.
        SESSION_BONUS = 2.5

        def has_resident_successor(key: OffloadKey) -> bool:
            successor = self._chain_successor.get(key)
            return successor is not None and successor in self._lru.blocks

        def session_bonus_for(key: OffloadKey) -> float:
            # Flat (issues log entry #20's original design, still the
            # default at half_life=0): every proven key gets the full
            # bonus regardless of how recently it was actually used. Entry
            # #21 found this can't distinguish an active proven session
            # from an idle one once eviction is choosing only among proven
            # candidates -- adding the same constant to every candidate
            # can't change their relative order -- costing active sessions
            # protection plain LRU would have given them for free. Decayed
            # variant (half_life > 0): scale by touch()-call staleness so an
            # active proven session keeps close to the full bonus while an
            # idle one's decays toward, but never reaches, zero -- it should
            # still edge out an equally-idle NEVER-proven candidate (which
            # gets no bonus at all), just not swamp a currently-active one.
            if self._session_bonus_half_life <= 0:
                return SESSION_BONUS
            staleness = self._touch_seq - self._last_touch_seq.get(key, self._touch_seq)
            decay = 1.0 / (1.0 + staleness / self._session_bonus_half_life)
            return SESSION_BONUS * decay

        def keep_score(idx: int, key: OffloadKey) -> float:
            # idx 0 = LRU's next eviction pick (least recent) -> 0.0;
            # idx (total-1) = most recent -> 1.0.
            recency_norm = idx / (total - 1) if total > 1 else 1.0
            if key not in relevance:
                base = recency_norm  # no signal yet -- pure recency
            else:
                relevance_norm = (relevance[key] - lo) / span
                base = self._alpha * relevance_norm + (1 - self._alpha) * recency_norm
            if key in in_grace:
                base += GRACE_BONUS
            if self._chain_aware and has_resident_successor(key):
                base += CHAIN_BONUS
            if self._session_aware and key in self._session_proven:
                base += session_bonus_for(key)
            return base

        t2 = time.perf_counter()
        scored = [
            (keep_score(idx, key), key, block)
            for idx, (key, block) in enumerate(candidates)
        ]
        t3 = time.perf_counter()
        scored.sort(key=lambda t: t[0])  # lowest keep_score evicted first
        chosen = scored[:n]
        for _, key, _ in chosen:
            self._lru.remove(key)
        debug_print(
            f"SEMANTIC_EVICT_TIMING mode=blend n={n} n_candidates={total} "
            f"candidates_ms={(t1 - t0) * 1000:.2f} "
            f"setup_ms={(t2 - t1) * 1000:.2f} "
            f"score_ms={(t3 - t2) * 1000:.2f} "
            f"sort_ms={(time.perf_counter() - t3) * 1000:.2f} "
            f"total_ms={(time.perf_counter() - t0) * 1000:.2f}"
        )
        return [(key, block) for _, key, block in chosen]

    def _evict_unscored_last(
        self, n: int, candidates: list[tuple[OffloadKey, BlockStatus]]
    ) -> list[tuple[OffloadKey, BlockStatus]]:
        """Strict two-tier eviction, no timer (entry #11/#18): every unscored
        block is evicted before any scored block, recency-ordered (oldest
        first) within the unscored tier -- `candidates` already arrives in
        that order from `evictable_blocks`. Within the scored tier, lowest
        relevance is evicted first, since real scores are available there and
        ignoring them would waste the whole point of having a scored tier."""
        relevance = self._relevance_ema.get(self._method, {})
        unscored = [(key, block) for key, block in candidates if key not in relevance]
        rest = [(key, block) for key, block in candidates if key in relevance]
        rest.sort(key=lambda kb: relevance[kb[0]])
        chosen = (unscored + rest)[:n]
        for key, _ in chosen:
            self._lru.remove(key)
        return chosen

    @override
    def clear(self) -> None:
        self._lru.clear()

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._lru.mark_evictable(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._lru.mark_non_evictable(key)
