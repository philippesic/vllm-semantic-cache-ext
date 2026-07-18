# SPDX-License-Identifier: Apache-2.0
"""Concurrent-proven-sessions stress test (issues log entry #20's flagged
open question): does `SemanticPolicy(session_aware=True)`'s unconditional,
permanent per-session immunity remain sound once MANY sessions are proven
simultaneously and must compete against *each other* (not just against
unproven flood traffic, entry #20's original test), for a capacity too
small to hold all of them?

Structural fact worth stating up front (reasoned through before writing any
code, then confirmed empirically below): `SESSION_BONUS` is a flat additive
constant applied identically to every proven key. Sorting by keep_score when
the same constant is added to every candidate being compared doesn't change
their RELATIVE order at all -- so once eviction is forced to choose among
*only* proven candidates (no unproven ones left to sacrifice first), it
necessarily reduces to plain recency (LRU) ordering among them. This means
the real question isn't "is the bonus miscalibrated" (entry #19's mistake,
not repeated here) -- it's "is plain-recency-among-many-established-sessions
itself a fair way to allocate capacity," and specifically: does session-aware
still give an idle-but-proven session real protection over an equally-idle
NEVER-proven one (yes, expected, this is entry #20's original result) WITHOUT
that protection concentrating catastrophically on an arbitrary unlucky
subset of the many proven sessions, versus degrading them roughly evenly.

Design: two groups of already-proven sessions (each proven the same way as
entry #20: 2 early turns, second under a different req_id), sized so their
COMBINED footprint exceeds every tested capacity -- real competition is
guaranteed, not assumed.
- ACTIVE group: re-touched (no new blocks, just a fresh req_id) once per
  subsequent round -- models an ongoing conversation still in use.
- IDLE group: proven once, then never touched again -- models a
  conversation that finished right after its first continuation.
A modest one-shot flood also runs each round (real servers have other
traffic besides established sessions).

Metric: per-session usable-prefix-fraction at the end, reported both as
group averages AND as the full per-session distribution (min/max, count at
0%/100%) -- an average alone can hide "half the group is fully evicted, half
is untouched" behind a deceptively reasonable-looking mean.

No model or GPU needed -- pure eviction-policy bookkeeping, same discipline
as entries #15/#19/#20.

Run: python3 experiments/step_1_4_session_aware_concurrent_stress.py
Writes: experiments/step_1_4_session_aware_concurrent_results.csv
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from pathlib import Path

from semantic_offload.policy import SemanticPolicy
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

SEED_COUNT = 8
N_ACTIVE_SESSIONS = 6
N_IDLE_SESSIONS = 6
BLOCKS_PER_TURN = 3  # 2 establishing turns -> 6 blocks/session
ROUNDS = 6  # subsequent rounds after establishment
FLOOD_BLOCKS_PER_ROUND = 4
# Combined proven footprint = (6+6)*6 = 72 blocks, fixed (active sessions are
# re-touched, not grown) -- capacities below are all < 72, so real
# proven-vs-proven competition is guaranteed at every point in this sweep.
CAPACITIES = [24, 36, 48, 60]

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EXPERIMENTS_DIR / "step_1_4_session_aware_concurrent_results.csv"


def make_key(tag: str) -> bytes:
    return make_offload_key(tag.encode(), 0)


def new_block() -> BlockStatus:
    b = BlockStatus(block_id=0)
    b.ref_cnt = 0
    return b


def evict_overflow(policy, capacity: int) -> int:
    blocks = policy.blocks if hasattr(policy, "blocks") else policy._lru.blocks
    overflow = len(blocks) - capacity
    if overflow <= 0:
        return 0
    evicted = policy.evict(overflow, protected=set())
    assert evicted is not None, "harness bug: asked to evict more than resident"
    return len(evicted)


def is_resident(policy, key) -> bool:
    blocks = policy.blocks if hasattr(policy, "blocks") else policy._lru.blocks
    return key in blocks


def usable_prefix_fraction(policy, chain: list) -> float:
    usable = 0
    for k in chain:
        if is_resident(policy, k):
            usable += 1
        else:
            break
    return usable / len(chain)


def run_workload(policy, seed: int, capacity: int) -> dict:
    rng = random.Random(seed)
    active_chains: dict[int, list] = {i: [] for i in range(N_ACTIVE_SESSIONS)}
    idle_chains: dict[int, list] = {i: [] for i in range(N_IDLE_SESSIONS)}

    sessions = [("active", i) for i in range(N_ACTIVE_SESSIONS)] + [
        ("idle", i) for i in range(N_IDLE_SESSIONS)
    ]
    rng.shuffle(sessions)
    for group, i in sessions:
        chain: list = []
        for turn in range(2):
            new_keys = [
                make_key(f"{group}{i}-t{turn}-b{b}") for b in range(BLOCKS_PER_TURN)
            ]
            for k in new_keys:
                policy.insert(k, new_block())
                policy.mark_evictable(k)
            chain.extend(new_keys)
            policy.touch(chain, ReqContext(req_id=f"{group}{i}-turn{turn}"))
            evict_overflow(policy, capacity)
        (active_chains if group == "active" else idle_chains)[i] = chain

    for r in range(ROUNDS):
        active_order = list(range(N_ACTIVE_SESSIONS))
        rng.shuffle(active_order)
        for i in active_order:
            # Re-touch only -- no new blocks, isolates the fairness question
            # from footprint growth. Fresh req_id each round: still a
            # legitimate, distinct continuation each time.
            policy.touch(active_chains[i], ReqContext(req_id=f"active{i}-round{r}"))
        flood_keys = [make_key(f"flood-{r}-{b}") for b in range(FLOOD_BLOCKS_PER_ROUND)]
        for k in flood_keys:
            policy.insert(k, new_block())
            policy.mark_evictable(k)
        policy.touch(flood_keys, ReqContext(req_id=f"flood-round{r}"))
        evict_overflow(policy, capacity)

    active_fracs = [usable_prefix_fraction(policy, c) for c in active_chains.values()]
    idle_fracs = [usable_prefix_fraction(policy, c) for c in idle_chains.values()]
    return {"active_fracs": active_fracs, "idle_fracs": idle_fracs}


def make_policy(variant: str, capacity: int):
    if variant == "lru":
        return LRUCachePolicy(cache_capacity=capacity)
    if variant == "semantic_session_aware":
        return SemanticPolicy(
            cache_capacity=capacity, relevance_ema={}, session_aware=True
        )
    # Recency-weighted variant (this entry's whole point): does decaying the
    # bonus by staleness-since-last-touch recover active-session protection
    # closer to plain LRU's own numbers, without collapsing idle-session
    # protection back toward LRU's near-zero baseline?
    if variant.startswith("semantic_session_aware_decay"):
        half_life = int(variant.removeprefix("semantic_session_aware_decay"))
        return SemanticPolicy(
            cache_capacity=capacity,
            relevance_ema={},
            session_aware=True,
            session_bonus_half_life=half_life,
        )
    raise ValueError(variant)


def summarize(fracs: list[float]) -> dict:
    return {
        "mean": round(sum(fracs) / len(fracs), 4),
        "min": round(min(fracs), 4),
        "max": round(max(fracs), 4),
        "stdev": round(statistics.pstdev(fracs), 4),
        "n_at_zero": sum(1 for f in fracs if f == 0.0),
        "n_at_one": sum(1 for f in fracs if f == 1.0),
    }


def main():
    variants = [
        "lru",
        "semantic_session_aware",
        "semantic_session_aware_decay5",
        "semantic_session_aware_decay15",
    ]
    rows = []
    for variant in variants:
        for capacity in CAPACITIES:
            all_active: list[float] = []
            all_idle: list[float] = []
            for seed in range(SEED_COUNT):
                policy = make_policy(variant, capacity)
                r = run_workload(policy, seed=seed, capacity=capacity)
                all_active.extend(r["active_fracs"])
                all_idle.extend(r["idle_fracs"])
            active_stats = summarize(all_active)
            idle_stats = summarize(all_idle)
            row = {"variant": variant, "capacity": capacity}
            row.update({f"active_{k}": v for k, v in active_stats.items()})
            row.update({f"idle_{k}": v for k, v in idle_stats.items()})
            rows.append(row)
            print(
                f"[{variant}] capacity={capacity} "
                f"active(mean={active_stats['mean']} min={active_stats['min']} "
                f"n0={active_stats['n_at_zero']}/{SEED_COUNT * N_ACTIVE_SESSIONS} "
                f"n1={active_stats['n_at_one']}/{SEED_COUNT * N_ACTIVE_SESSIONS}) "
                f"idle(mean={idle_stats['mean']} min={idle_stats['min']} "
                f"n0={idle_stats['n_at_zero']}/{SEED_COUNT * N_IDLE_SESSIONS} "
                f"n1={idle_stats['n_at_one']}/{SEED_COUNT * N_IDLE_SESSIONS})"
            )

    print(f"\n=== FINAL (aggregated across {SEED_COUNT} seeds per capacity) ===")
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    sys.exit(main())
