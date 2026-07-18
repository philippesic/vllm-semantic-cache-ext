# SPDX-License-Identifier: Apache-2.0
"""Step 1.4 (preliminary slice) acceptance check: SemanticPolicy.evict()
blends relevance with recency instead of pure LRU pass-through. Real
capacity-constrained LRU-vs-semantic comparison is done end-to-end on the
live server (see .claude/docs/semantic-eviction-issues-log.md entry #10);
these tests cover the blending arithmetic and the LRU-equivalence fallback
in isolation.
"""

from semantic_offload.policy import SemanticPolicy
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus


def to_key(n: int):
    return make_offload_key(str(n).encode(), 0)


def _insert(policy: SemanticPolicy, key, block_id: int = 0) -> None:
    block = BlockStatus(block_id=block_id)
    block.ref_cnt = 0  # ready + not participating in any transfer
    policy.insert(key, block)
    policy.mark_evictable(key)


def test_no_relevance_scores_matches_pure_recency_order():
    """With an empty relevance_ema (nothing scored yet), eviction order must
    exactly match plain LRU -- this is what keeps Step 1.1's
    manager-matches-LRU test passing unmodified."""
    policy = SemanticPolicy(cache_capacity=10, relevance_ema={})
    keys = [to_key(i) for i in range(4)]
    for key in keys:
        _insert(policy, key)

    evicted = policy.evict(2, protected=set())

    assert [k for k, _ in evicted] == keys[:2]  # oldest two, LRU order


def test_high_relevance_block_survives_eviction_despite_being_oldest():
    """A block that's oldest by recency but has a high relevance score
    should be kept over a newer, low-relevance block once alpha > 0."""
    relevance_ema = {"minmax": {}}
    policy = SemanticPolicy(
        cache_capacity=10, relevance_ema=relevance_ema, method="minmax", alpha=0.8
    )
    old_important = to_key(1)
    old_unimportant = to_key(2)
    newer = to_key(3)
    for key in (old_important, old_unimportant, newer):
        _insert(policy, key)

    relevance_ema["minmax"][old_important] = 10.0
    relevance_ema["minmax"][old_unimportant] = -10.0

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    assert evicted[0][0] == old_unimportant


def test_grace_protected_key_survives_eviction_despite_being_oldest():
    """A freshly-inserted key within its grace window must survive even
    though it's the oldest/only candidate by recency -- issues log #11."""
    policy = SemanticPolicy(cache_capacity=10, relevance_ema={}, grace_window_blocks=5)
    protected_key = to_key(1)
    _insert(policy, protected_key)  # insert_count -> 1, expiry = 1 + 5 = 6
    newer = to_key(2)
    _insert(policy, newer)  # insert_count -> 2, expiry = 2 + 5 = 7

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    # Both keys are still within their grace window (insert_count=2 < 6, 7)
    # -- escape valve must kick in since neither is a "legal" candidate, and
    # the oldest-in-grace (protected_key) is picked as the fallback.
    assert evicted[0][0] == protected_key


def test_grace_expires_after_enough_inserts():
    """Once enough other blocks have been inserted, an old key's grace
    period lapses and it becomes a normal (recency-ranked) candidate."""
    policy = SemanticPolicy(cache_capacity=20, relevance_ema={}, grace_window_blocks=3)
    old_key = to_key(1)
    _insert(policy, old_key)  # insert_count=1, expiry=4
    for i in range(2, 6):  # insert_count -> 5, past old_key's expiry (4)
        _insert(policy, to_key(i))

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    assert evicted[0][0] == old_key  # oldest, grace lapsed, pure LRU order


def test_grace_window_zero_matches_no_grace_baseline():
    """grace_window_blocks=0 (the default) must reproduce entry #10's exact
    no-grace behavior -- oldest evicted first, no immunity at all."""
    policy = SemanticPolicy(cache_capacity=10, relevance_ema={}, grace_window_blocks=0)
    keys = [to_key(i) for i in range(4)]
    for key in keys:
        _insert(policy, key)

    evicted = policy.evict(2, protected=set())

    assert [k for k, _ in evicted] == keys[:2]


def test_unscored_last_evicts_all_unscored_before_any_scored_block():
    """mode='unscored_last' (entry #11/#18): a scored block must survive
    even if it is by far the oldest, as long as any unscored block exists."""
    relevance_ema = {"minmax": {}}
    policy = SemanticPolicy(
        cache_capacity=10,
        relevance_ema=relevance_ema,
        method="minmax",
        mode="unscored_last",
    )
    old_scored = to_key(1)
    newer_unscored_a = to_key(2)
    newer_unscored_b = to_key(3)
    for key in (old_scored, newer_unscored_a, newer_unscored_b):
        _insert(policy, key)
    relevance_ema["minmax"][old_scored] = -100.0  # very low score, still scored

    evicted = policy.evict(2, protected=set())

    assert evicted is not None
    evicted_keys = {k for k, _ in evicted}
    assert evicted_keys == {newer_unscored_a, newer_unscored_b}
    assert old_scored not in evicted_keys


def test_unscored_last_orders_unscored_tier_by_recency():
    policy = SemanticPolicy(
        cache_capacity=10, relevance_ema={"minmax": {}}, mode="unscored_last"
    )
    keys = [to_key(i) for i in range(4)]
    for key in keys:
        _insert(policy, key)

    evicted = policy.evict(2, protected=set())

    assert [k for k, _ in evicted] == keys[:2]  # oldest two, recency order


def test_unscored_last_orders_scored_tier_by_lowest_relevance_first():
    """Once eviction must dip into the scored tier, it should evict the
    least-relevant scored block first, not just the oldest scored one."""
    relevance_ema = {"minmax": {}}
    policy = SemanticPolicy(
        cache_capacity=10,
        relevance_ema=relevance_ema,
        method="minmax",
        mode="unscored_last",
    )
    newer_low_relevance = to_key(1)
    older_high_relevance = to_key(2)
    for key in (newer_low_relevance, older_high_relevance):
        _insert(policy, key)
    relevance_ema["minmax"][newer_low_relevance] = -5.0
    relevance_ema["minmax"][older_high_relevance] = 10.0

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    assert evicted[0][0] == newer_low_relevance


def test_unscored_last_matches_lru_when_nothing_scored():
    policy = SemanticPolicy(cache_capacity=10, relevance_ema={}, mode="unscored_last")
    keys = [to_key(i) for i in range(4)]
    for key in keys:
        _insert(policy, key)

    evicted = policy.evict(2, protected=set())

    assert [k for k, _ in evicted] == keys[:2]


def test_chain_aware_protects_head_block_while_tail_still_resident():
    """Issues log entry #15: with chain_aware=True, an early chain block
    must survive eviction over a later, otherwise-unrelated block, as long
    as its recorded successor (from touch()) is still resident -- even
    though the head block is the oldest by recency and neither has a
    relevance score."""
    from vllm.v1.kv_offload.base import ReqContext

    policy = SemanticPolicy(cache_capacity=10, relevance_ema={}, chain_aware=True)
    head, tail, unrelated = to_key(1), to_key(2), to_key(3)
    for key in (head, tail, unrelated):
        _insert(policy, key)
    policy.touch([head, tail], ReqContext(req_id="r1"))

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    # Without chain awareness, `unrelated` (inserted last, most recent) would
    # never be the pick over `head`/`tail` (both older) under pure recency --
    # the real test is that `head` specifically is skipped over despite
    # being the single oldest candidate.
    assert evicted[0][0] != head


def test_chain_aware_allows_head_eviction_once_tail_gone():
    """Once a chain's tail has actually been evicted (successor no longer
    resident), the head block loses its chain-protection bonus and reverts
    to being a normal candidate."""
    from vllm.v1.kv_offload.base import ReqContext

    policy = SemanticPolicy(cache_capacity=10, relevance_ema={}, chain_aware=True)
    head, tail = to_key(1), to_key(2)
    _insert(policy, head)
    _insert(policy, tail)
    policy.touch([head, tail], ReqContext(req_id="r1"))

    first = policy.evict(1, protected=set())
    assert first is not None
    assert first[0][0] == tail  # tail evicted first, exactly as intended

    second = policy.evict(1, protected=set())
    assert second is not None
    assert second[0][0] == head  # now the only candidate left


def test_chain_aware_disabled_by_default_matches_blend_baseline():
    """chain_aware=False (the default) must not change eviction order at
    all versus plain LRUCachePolicy.touch()'s own (reverse-order) recency
    update -- no chain-successor bonus applied regardless of the touch."""
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

    policy = SemanticPolicy(cache_capacity=10, relevance_ema={})
    reference = LRUCachePolicy(cache_capacity=10)
    head, tail, unrelated = to_key(1), to_key(2), to_key(3)
    for key in (head, tail, unrelated):
        _insert(policy, key)
        block = BlockStatus(block_id=0)
        block.ref_cnt = 0
        reference.insert(key, block)
        reference.mark_evictable(key)
    policy.touch([head, tail], ReqContext(req_id="r1"))
    reference.touch([head, tail], ReqContext(req_id="r1"))

    evicted = policy.evict(1, protected=set())
    expected = reference.evict(1, protected=set())

    assert evicted is not None
    assert expected is not None
    assert evicted[0][0] == expected[0][0]


def test_session_aware_protects_chain_once_a_different_request_revisits_it():
    """Issues log entry #19's follow-up: a chain touched a second time under
    a DIFFERENT req_id (real cross-request continuation) must survive
    eviction over an equally-old chain that has only ever been touched by
    its own original req_id (never proven), with no relevance scores at
    all -- pure structural evidence, no content signal."""
    from vllm.v1.kv_offload.base import ReqContext

    policy = SemanticPolicy(cache_capacity=10, relevance_ema={}, session_aware=True)
    proven = [to_key(1), to_key(2)]
    unproven = [to_key(3), to_key(4)]
    for key in proven + unproven:
        _insert(policy, key)
    policy.touch(proven, ReqContext(req_id="turn-0"))  # first store, own req_id
    policy.touch(unproven, ReqContext(req_id="turn-0-b"))  # first store, own req_id
    # A genuine continuation: a NEW req_id touches `proven`'s existing keys.
    policy.touch(proven, ReqContext(req_id="turn-1"))

    evicted = policy.evict(2, protected=set())

    assert evicted is not None
    evicted_keys = {k for k, _ in evicted}
    assert evicted_keys == set(unproven)


def test_session_aware_same_req_id_repeated_touch_does_not_prove():
    """Touching the same chain multiple times from the SAME req_id (e.g. a
    single generation's own store-then-touch sequence) must NOT be mistaken
    for cross-request continuation."""
    from vllm.v1.kv_offload.base import ReqContext

    policy = SemanticPolicy(cache_capacity=10, relevance_ema={}, session_aware=True)
    keys = [to_key(i) for i in range(4)]
    for key in keys:
        _insert(policy, key)
    policy.touch(keys, ReqContext(req_id="r1"))
    policy.touch(keys, ReqContext(req_id="r1"))  # same req_id again

    evicted = policy.evict(2, protected=set())

    assert evicted is not None
    # No proof of continuation exists -- must match the reference LRU order,
    # not be protected by a false-positive session bonus.
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

    reference = LRUCachePolicy(cache_capacity=10)
    for key in keys:
        block = BlockStatus(block_id=0)
        block.ref_cnt = 0
        reference.insert(key, block)
        reference.mark_evictable(key)
    reference.touch(keys, ReqContext(req_id="r1"))
    reference.touch(keys, ReqContext(req_id="r1"))
    expected = reference.evict(2, protected=set())
    assert expected is not None
    assert {k for k, _ in evicted} == {k for k, _ in expected}


def test_session_aware_disabled_by_default_ignores_cross_request_touches():
    """session_aware=False (the default) must not grant any protection even
    when the exact cross-request pattern that would prove a session is
    present -- confirms the bonus is opt-in only."""
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

    policy = SemanticPolicy(cache_capacity=10, relevance_ema={})
    reference = LRUCachePolicy(cache_capacity=10)
    proven_pattern = [to_key(1), to_key(2)]
    other = [to_key(3), to_key(4)]
    for key in proven_pattern + other:
        _insert(policy, key)
        block = BlockStatus(block_id=0)
        block.ref_cnt = 0
        reference.insert(key, block)
        reference.mark_evictable(key)
    for touch_keys, req in (
        (proven_pattern, "turn-0"),
        (other, "turn-0-b"),
        (proven_pattern, "turn-1"),
    ):
        policy.touch(touch_keys, ReqContext(req_id=req))
        reference.touch(touch_keys, ReqContext(req_id=req))

    evicted = policy.evict(2, protected=set())
    expected = reference.evict(2, protected=set())

    assert evicted is not None
    assert expected is not None
    assert {k for k, _ in evicted} == {k for k, _ in expected}


def test_session_bonus_decay_favors_recently_touched_proven_session():
    """Issues log entry #21's follow-up: with session_bonus_half_life set, a
    proven session that was touched RECENTLY should be strictly preferred
    over an equally-proven session that has gone stale, when the policy is
    forced to evict from within the proven population alone."""
    from vllm.v1.kv_offload.base import ReqContext

    policy = SemanticPolicy(
        cache_capacity=10,
        relevance_ema={},
        session_aware=True,
        session_bonus_half_life=2,
    )
    stale = [to_key(1)]
    fresh = [to_key(2)]
    for key in stale + fresh:
        _insert(policy, key)
    # Both proven via a cross-request touch.
    policy.touch(stale, ReqContext(req_id="stale-a"))
    policy.touch(stale, ReqContext(req_id="stale-b"))  # proven, staleness resets to 0
    policy.touch(fresh, ReqContext(req_id="fresh-a"))
    policy.touch(fresh, ReqContext(req_id="fresh-b"))  # proven, staleness resets to 0
    # Advance the touch_seq clock via unrelated touches, immediately
    # `remove()`d (not evicted) so they never compete as eviction
    # candidates themselves -- isolates the staleness gap between `stale`
    # and `fresh` from any tie-breaking noise the filler keys would add.
    for i in range(6):
        filler = to_key(100 + i)
        _insert(policy, filler)
        policy.touch([filler], ReqContext(req_id=f"filler-{i}"))
        policy.remove(filler)

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    assert evicted[0][0] == stale[0]  # decayed bonus lost the tie to `fresh`


def test_session_bonus_half_life_zero_matches_flat_bonus_baseline():
    """session_bonus_half_life=0 (the default) must reproduce entry #20's
    exact flat-bonus behavior -- no decay at all."""
    from vllm.v1.kv_offload.base import ReqContext

    policy = SemanticPolicy(cache_capacity=10, relevance_ema={}, session_aware=True)
    proven = [to_key(1), to_key(2)]
    unproven = [to_key(3), to_key(4)]
    for key in proven + unproven:
        _insert(policy, key)
    policy.touch(proven, ReqContext(req_id="turn-0"))
    policy.touch(unproven, ReqContext(req_id="turn-0-b"))
    policy.touch(proven, ReqContext(req_id="turn-1"))

    evicted = policy.evict(2, protected=set())

    assert evicted is not None
    assert {k for k, _ in evicted} == set(unproven)


def test_session_bonus_decay_never_reaches_exactly_zero():
    """However stale, a proven key's decayed bonus must stay strictly
    positive -- it should never fully vanish and become indistinguishable
    from an unproven key's zero bonus, however long it's been idle. This is
    a mathematical property of the decay formula (1 / (1 + staleness /
    half_life) is asymptotic, never hits 0 for finite staleness) -- tested
    directly rather than via an eviction-order scenario, since constructing
    two candidates with genuinely EQUAL recency through the public API is
    not possible here: proving a key requires an extra touch, which always
    makes it strictly more recent than a single-touch unproven key, so any
    black-box comparison is confounded by the base recency term too."""
    from vllm.v1.kv_offload.base import ReqContext

    policy = SemanticPolicy(
        cache_capacity=10,
        relevance_ema={},
        session_aware=True,
        session_bonus_half_life=1,
    )
    key = to_key(1)
    _insert(policy, key)
    policy.touch([key], ReqContext(req_id="a"))
    policy.touch([key], ReqContext(req_id="b"))  # proven
    assert key in policy._session_proven

    policy._touch_seq += 1_000_000  # simulate enormous elapsed staleness
    staleness = policy._touch_seq - policy._last_touch_seq[key]
    decay = 1.0 / (1.0 + staleness / policy._session_bonus_half_life)

    assert decay > 0.0


def test_unscored_block_falls_back_to_recency_alongside_scored_blocks():
    """Plan's Step 1.4 text: blocks with no summary/score rank by recency
    alone, even when other blocks in the same eviction batch do have scores."""
    relevance_ema = {"minmax": {}}
    policy = SemanticPolicy(
        cache_capacity=10, relevance_ema=relevance_ema, method="minmax", alpha=0.9
    )
    oldest_unscored = to_key(1)
    newer_low_relevance = to_key(2)
    for key in (oldest_unscored, newer_low_relevance):
        _insert(policy, key)

    relevance_ema["minmax"][newer_low_relevance] = -5.0
    # oldest_unscored has no entry -- falls back to its recency_norm (0.0,
    # since it's first/oldest), which is still the lowest keep_score here.

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    assert evicted[0][0] == oldest_unscored
