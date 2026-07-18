# SPDX-License-Identifier: Apache-2.0
"""Multi-turn chain-aware eviction benchmark (issues log entry #15).

Every prior real-scale test in this project (entries #10-#14, #17) used
standalone one-shot requests. Chain-aware ordering targets a workload shape
this project has never actually built: a real multi-turn conversation, whose
KV blocks form a growing chain (turn N's prefix = turn N-1's full chain plus
new blocks), interleaved with distractor traffic from other sessions -- the
same "cyclic multi-turn" shape vLLM issue #45405 measured (LRU/ARC recovered
0/0/0 complete prefixes across 3 runs; a chain-aware prototype recovered
3/3/3).

This harness needs no model, no GPU, no attention computation -- chain-aware
ordering is pure eviction-policy bookkeeping (which block succeeds which in a
request's own touch() ordering), so it's tested the same cheap way Step 0.3
killed SVD before any live integration: drive the real `SemanticPolicy` and
`LRUCachePolicy` classes directly against a synthetic event stream, no vLLM
server needed.

Workload: N_SESSIONS conversations, each TURNS_PER_SESSION turns long, run in
round-robin (session 0 turn 1, distractor burst, session 1 turn 1, ..., wrap
to session 0 turn 2, ...) under a cache capacity well below the total blocks
ever stored, so real eviction pressure is continuous, not a one-off cliff.
Each turn's request stores its new blocks then touch()es its FULL chain so
far (matching vllm's real offloading/scheduler.py `_touch`, which always
passes a request's complete ordered `offload_keys`). Distractor bursts are
single-shot, unrelated, and never touched again after their own store.

Metric: at the end, for each session, is its complete final-turn chain still
resident? (The same "complete prefix" metric #45405 used.) Compared across
three policies run on the IDENTICAL event sequence (same seed): plain
LRUCachePolicy, SemanticPolicy(chain_aware=False) (expected ~identical to
LRU -- a sanity check, per test_no_relevance_scores_matches_pure_recency_order),
and SemanticPolicy(chain_aware=True). No relevance scores are populated in
this run -- this isolates the chain-awareness lever specifically, the same
way entry #10's diagnostic isolated alpha from timing. Combining chain
awareness with real content scoring is future work, not this experiment.

Run: python3 experiments/step_1_4_chain_aware_multiturn.py
Writes: experiments/step_1_4_chain_aware_results.csv
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

SEED = 0
N_SESSIONS = 6
TURNS_PER_SESSION = 4
BLOCKS_PER_TURN = 3  # new blocks added to the chain each turn
DISTRACTOR_BURST_SIZE = 5  # single-shot blocks injected between each turn
CACHE_CAPACITY = 100  # comfortably fits all 6 sessions' final chains (72
# blocks total) plus some distractor headroom -- a regime where pressure is
# real but not overwhelming (see script docstring update: capacity=40 wiped
# out every policy uniformly and taught nothing; this value is where the
# policies actually diverge).
N_SEEDS = 5  # repeat the whole run under this many random distractor orderings

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EXPERIMENTS_DIR / "step_1_4_chain_aware_results.csv"


def make_key(tag: str) -> bytes:
    return make_offload_key(tag.encode(), 0)


def new_block() -> BlockStatus:
    b = BlockStatus(block_id=0)
    b.ref_cnt = 0
    return b


def evict_overflow(policy) -> int:
    """Mimic the real manager's job: keep resident count at/under capacity
    by evicting from the back after every store batch. Returns blocks
    evicted (0 if already under capacity)."""
    resident = (
        len(policy.blocks) if hasattr(policy, "blocks") else len(policy._lru.blocks)
    )
    overflow = resident - CACHE_CAPACITY
    if overflow <= 0:
        return 0
    evicted = policy.evict(overflow, protected=set())
    assert evicted is not None, "harness bug: asked to evict more than resident"
    return len(evicted)


def is_resident(policy, key) -> bool:
    blocks = policy.blocks if hasattr(policy, "blocks") else policy._lru.blocks
    return key in blocks


def run_workload(policy, seed: int) -> dict:
    """Drive one policy through the full multi-turn + distractor event
    stream. Returns per-session final-chain recovery stats."""
    rng = random.Random(seed)
    session_chains: list[list] = [[] for _ in range(N_SESSIONS)]
    distractor_counter = 0
    n_stores = 0
    n_evicted = 0

    for turn in range(TURNS_PER_SESSION):
        session_order = list(range(N_SESSIONS))
        rng.shuffle(session_order)
        for s in session_order:
            new_keys = []
            for b in range(BLOCKS_PER_TURN):
                key = make_key(f"s{s}-t{turn}-b{b}")
                policy.insert(key, new_block())
                policy.mark_evictable(key)
                new_keys.append(key)
                n_stores += 1
            session_chains[s].extend(new_keys)
            policy.touch(session_chains[s], ReqContext(req_id=f"session-{s}"))
            n_evicted += evict_overflow(policy)

            # Distractor burst: unrelated single-shot traffic between turns,
            # touched once at store time (matching real request behavior)
            # and never referenced again.
            burst_keys = []
            for i in range(DISTRACTOR_BURST_SIZE):
                key = make_key(f"distractor-{turn}-{s}-{distractor_counter}-{i}")
                policy.insert(key, new_block())
                policy.mark_evictable(key)
                burst_keys.append(key)
                n_stores += 1
            distractor_counter += 1
            policy.touch(
                burst_keys, ReqContext(req_id=f"distractor-{distractor_counter}")
            )
            n_evicted += evict_overflow(policy)

    complete = 0
    total_block_fraction = 0.0
    total_prefix_fraction = 0.0
    for s in range(N_SESSIONS):
        chain = session_chains[s]
        recovered = sum(1 for k in chain if is_resident(policy, k))
        total_block_fraction += recovered / len(chain)
        if recovered == len(chain):
            complete += 1
        # Usable prefix length: how many leading blocks are CONTIGUOUSLY
        # resident from block 0 -- this is what actually matters for KV
        # prefix-cache reuse. A block surviving eviction past the first gap
        # is individually resident but functionally useless (vLLM's prefix
        # match stops at the first miss), so raw block-count recovery
        # (above) can look fine while genuinely usable prefix is near zero
        # -- this is the metric that distinguishes "evicted the tail" (good,
        # what chain-awareness targets) from "evicted a hole in the middle"
        # (bad, breaks reuse regardless of how many blocks survive after it).
        usable_prefix = 0
        for k in chain:
            if is_resident(policy, k):
                usable_prefix += 1
            else:
                break
        total_prefix_fraction += usable_prefix / len(chain)

    return {
        "sessions_complete": complete,
        "sessions_total": N_SESSIONS,
        "avg_chain_fraction_recovered": round(total_block_fraction / N_SESSIONS, 4),
        "avg_usable_prefix_fraction": round(total_prefix_fraction / N_SESSIONS, 4),
        "n_stores": n_stores,
        "n_evicted": n_evicted,
    }


def make_policy(variant: str):
    if variant == "lru":
        return LRUCachePolicy(cache_capacity=CACHE_CAPACITY)
    if variant == "semantic_no_chain":
        return SemanticPolicy(
            cache_capacity=CACHE_CAPACITY, relevance_ema={}, chain_aware=False
        )
    if variant == "semantic_chain_aware":
        return SemanticPolicy(
            cache_capacity=CACHE_CAPACITY, relevance_ema={}, chain_aware=True
        )
    raise ValueError(variant)


def main():
    variants = ["lru", "semantic_no_chain", "semantic_chain_aware"]
    rows = []
    for variant in variants:
        agg = {"sessions_complete": 0, "sessions_total": 0, "chain_fracs": []}
        for seed in range(N_SEEDS):
            policy = make_policy(variant)
            result = run_workload(policy, seed=seed)
            agg["sessions_complete"] += result["sessions_complete"]
            agg["sessions_total"] += result["sessions_total"]
            agg["chain_fracs"].append(result["avg_chain_fraction_recovered"])
            print(
                f"[{variant}] seed={seed} "
                f"complete={result['sessions_complete']}/{result['sessions_total']} "
                f"avg_fraction={result['avg_chain_fraction_recovered']} "
                f"stores={result['n_stores']} evicted={result['n_evicted']}"
            )
        row = {
            "variant": variant,
            "sessions_complete_total": agg["sessions_complete"],
            "sessions_total": agg["sessions_total"],
            "complete_rate": round(agg["sessions_complete"] / agg["sessions_total"], 4),
            "avg_chain_fraction_recovered": round(
                sum(agg["chain_fracs"]) / len(agg["chain_fracs"]), 4
            ),
        }
        rows.append(row)

    print("\n=== FINAL (aggregated across %d seeds) ===" % N_SEEDS)
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
