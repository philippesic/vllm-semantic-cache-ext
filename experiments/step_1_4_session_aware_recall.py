# SPDX-License-Identifier: Apache-2.0
"""Session-proven-priority eviction benchmark (issues log entry #19's
follow-up, after the per-block chain-successor bonus was closed).

Targets a gap none of this project's validated mechanisms can see: an
idle-but-active multi-turn chat session has no content evidence yet (the
already-closed cold-start gap, entries #10-#12) and doesn't suffer from a
chain-internal ordering problem either (entry #19 found vLLM's own
`LRUCachePolicy.touch()` already handles that correctly). But it DOES carry
a structural signal nothing else uses: has a *different* request already
revisited (touched) this exact chain? Real multi-turn conversations get a
new req_id every turn (a fresh HTTP request) while reusing the SAME
content-addressed OffloadKeys for the shared prefix, so this is real,
observed evidence -- not a speculative bet on the future the way a blanket
grace period is (entry #11 proved that can't work).

Design, deliberately different from entry #19's round-robin harness (which
diluted the "sustained pressure after establishment" dynamic across many
concurrently-competing chains): mirrors entry #10's CUSP scenario shape
(establish -> idle -> distractor flood -> check survival) but for whole
SESSIONS keyed on request-structure evidence instead of content-relevance
evidence.

- REAL sessions: 2 early "turns" each, second turn under a DIFFERENT req_id
  that touches the first turn's keys too -- a genuine, detectable
  continuation. Session-aware should mark these proven.
- CONTROL sessions: identical size and shape, but both touches reuse the
  SAME req_id -- never proven. This isolates the mechanism itself (session-
  aware vs not) from a confound of "bigger/more-touched chains just survive
  better regardless of why" (entry #19's own lesson: trace to a real
  mechanism, don't trust an aggregate number).
- Then a FLOOD of one-shot distractor traffic (its own req_id each, touched
  once, never revisited) large enough to force real, sustained eviction
  pressure.
- Finally: is each session's full established chain still contiguously
  resident from block 0? Compared across plain LRU, SemanticPolicy with
  session_aware=False, and SemanticPolicy with session_aware=True -- on the
  IDENTICAL event sequence per seed.

No model or GPU needed -- pure eviction-policy bookkeeping, same cheap-first
discipline as entry #19 and Step 0.3/0.4.

Run: python3 experiments/step_1_4_session_aware_recall.py
Writes: experiments/step_1_4_session_aware_results.csv
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

from semantic_offload.policy import SemanticPolicy
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

SEED_COUNT = 8
N_REAL_SESSIONS = 3
N_CONTROL_SESSIONS = 3
BLOCKS_PER_TURN = 3
# Flood total = FLOOD_BURSTS * FLOOD_BLOCKS_PER_BURST = 45 blocks. Sized to be
# comparable to (not wildly larger than) the capacities swept below, so plain
# LRU has a real chance to succeed at some capacities and fail at others --
# an initial version used a much larger flood (120 blocks) that exceeded
# every tested capacity, making LRU fail trivially (0/N) everywhere
# regardless of the real/control distinction, which taught nothing.
FLOOD_BURSTS = 15
FLOOD_BLOCKS_PER_BURST = 3
# Swept, not fixed to one value -- the honest result is a trade-off curve
# (see main()'s docstring note), not a single number.
CAPACITIES = [30, 40, 45, 50, 55, 60, 70, 80]

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EXPERIMENTS_DIR / "step_1_4_session_aware_results.csv"


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
    real_chains: list[list] = []
    control_chains: list[list] = []

    # Establishment phase: each session does 2 turns, interleaved randomly.
    sessions = [("real", i) for i in range(N_REAL_SESSIONS)] + [
        ("control", i) for i in range(N_CONTROL_SESSIONS)
    ]
    # Shuffle establishment order so real/control groups get identical
    # recency treatment under plain LRU -- otherwise whichever group is
    # listed second gets an unfair recency edge unrelated to the mechanism
    # being tested (caught by comparing LRU's real vs control numbers directly
    # before trusting the semantic-policy comparison, same discipline as
    # entries #14/#18/#19).
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
            if group == "real":
                req_id = f"{group}{i}-turn{turn}"  # new req_id every turn
            else:
                req_id = f"{group}{i}-fixed"  # same req_id both turns
            policy.touch(chain, ReqContext(req_id=req_id))
            evict_overflow(policy, capacity)
        (real_chains if group == "real" else control_chains).append(chain)

    # Flood phase: heavy one-shot distractor traffic, never revisited.
    burst_order = list(range(FLOOD_BURSTS))
    rng.shuffle(burst_order)
    for b in burst_order:
        keys = [make_key(f"flood-{b}-{i}") for i in range(FLOOD_BLOCKS_PER_BURST)]
        for k in keys:
            policy.insert(k, new_block())
            policy.mark_evictable(k)
        policy.touch(keys, ReqContext(req_id=f"flood-{b}"))
        evict_overflow(policy, capacity)

    real_fracs = [usable_prefix_fraction(policy, c) for c in real_chains]
    control_fracs = [usable_prefix_fraction(policy, c) for c in control_chains]
    return {
        "real_avg_prefix_fraction": sum(real_fracs) / len(real_fracs),
        "real_complete": sum(1 for f in real_fracs if f == 1.0),
        "control_avg_prefix_fraction": sum(control_fracs) / len(control_fracs),
        "control_complete": sum(1 for f in control_fracs if f == 1.0),
    }


def make_policy(variant: str, capacity: int):
    if variant == "lru":
        return LRUCachePolicy(cache_capacity=capacity)
    if variant == "semantic_no_session":
        return SemanticPolicy(
            cache_capacity=capacity, relevance_ema={}, session_aware=False
        )
    if variant == "semantic_session_aware":
        return SemanticPolicy(
            cache_capacity=capacity, relevance_ema={}, session_aware=True
        )
    # Recency-weighted variant (issues log entry #21's follow-up): does
    # decaying the bonus by staleness-since-last-touch still preserve THIS
    # script's original win (protecting a genuinely idle-since-establishment
    # proven session against a continuous flood) even though decay reduces
    # the bonus for exactly that kind of long-idle session? Not assumed --
    # tested directly below alongside the flat-bonus variant.
    if variant.startswith("semantic_session_aware_decay"):
        half_life = int(variant.removeprefix("semantic_session_aware_decay"))
        return SemanticPolicy(
            cache_capacity=capacity,
            relevance_ema={},
            session_aware=True,
            session_bonus_half_life=half_life,
        )
    raise ValueError(variant)


def main():
    """Sweeps CAPACITIES rather than reporting one number -- the honest
    result here is a trade-off curve: session_aware gives proven sessions an
    unconditional protection floor (by design -- SESSION_BONUS strictly
    dominates the base [0,1] blend range, so a proven session is never
    evicted while any unproven candidate exists), which necessarily comes at
    the expense of OTHER competing traffic's share of a fixed capacity
    (visible here as control-session survival dropping below plain LRU's at
    higher capacities, where LRU was already doing fine without any help).
    Reporting only a capacity where this looks like a free win would hide
    that real cost."""
    variants = [
        "lru",
        "semantic_no_session",
        "semantic_session_aware",
        "semantic_session_aware_decay10",
        "semantic_session_aware_decay30",
    ]
    rows = []
    for variant in variants:
        for capacity in CAPACITIES:
            real_complete_total = 0
            control_complete_total = 0
            real_frac_sum = 0.0
            control_frac_sum = 0.0
            for seed in range(SEED_COUNT):
                policy = make_policy(variant, capacity)
                r = run_workload(policy, seed=seed, capacity=capacity)
                real_complete_total += r["real_complete"]
                control_complete_total += r["control_complete"]
                real_frac_sum += r["real_avg_prefix_fraction"]
                control_frac_sum += r["control_avg_prefix_fraction"]
            row = {
                "variant": variant,
                "capacity": capacity,
                "real_complete_rate": round(
                    real_complete_total / (SEED_COUNT * N_REAL_SESSIONS), 4
                ),
                "real_avg_prefix_fraction": round(real_frac_sum / SEED_COUNT, 4),
                "control_complete_rate": round(
                    control_complete_total / (SEED_COUNT * N_CONTROL_SESSIONS), 4
                ),
                "control_avg_prefix_fraction": round(control_frac_sum / SEED_COUNT, 4),
            }
            rows.append(row)
            print(
                f"[{variant}] capacity={capacity} "
                f"real_complete_rate={row['real_complete_rate']} "
                f"control_complete_rate={row['control_complete_rate']}"
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
