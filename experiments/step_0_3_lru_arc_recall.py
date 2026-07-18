"""Step 0.3 follow-up #3: LRU and ARC recall, for a full comparison against
the semantic-scoring methods (SVD, min/max, mean, oracle) already measured.

Uses the REAL production policy classes from the vllm-semantic-cache fork
(`vllm.v1.kv_offload.cpu.policies.lru.LRUCachePolicy`,
`...arc.ARCCachePolicy`) rather than reimplementing LRU/ARC — same
correctness principle as reusing the real model's attention weights instead
of a synthetic proxy.

Methodology: for each probe (same prompts/probes/ground truth as the other
step_0_3 scripts), the only "access signal" available is the single linear
prefill order (block 0 stored first, ..., block num_blocks-1 stored last,
immediately before the query). This is the ONE thing an eviction policy in
this synthetic single-pass setup can see: no request in this experiment ever
re-references an earlier block, so there is no repeat-access signal to hand
either policy. For each (probe, N) pair: build a fresh policy, insert blocks
0..num_blocks-1 in order (each immediately marked evictable, ref_cnt=0 —
simulating a completed, ready-to-evict CPU-tier store), then call
`policy.evict(num_blocks - N, protected=set())`. The N blocks NOT evicted are
the policy's "keep" prediction; recall@N is computed against the same
true-top-N attention-mass ground truth as the other methods.

Expected and important to state up front: with no repeated block references,
ARC's promotion logic (T1->T2) never fires (every block is touched exactly
once, at insert time), so ARC is expected to degenerate to plain
insertion-order (== LRU) here. This is not a bug in the measurement — it's a
real, documented property of ARC (see arc.py's docstring: T2/frequency logic
only activates on a second access) and is itself a finding worth stating
plainly: this synthetic single-pass workload gives ARC no information
advantage over LRU. Distinguishing them would require an actual shared-prefix
reuse workload (repeated access to the same block across separate
"requests"), which is what the plan's Step 1.4 benchmark workload is for,
not this recall@N harness.

Run: python3 experiments/step_0_3_lru_arc_recall.py
Writes: experiments/step_0_3_lru_arc_results.csv
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_TOKENS = 16
TARGET_LAYER = 14
TOP_NS = [4, 8]
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EXPERIMENTS_DIR / "step_0_3_lru_arc_results.csv"

random.seed(SEED)
torch.manual_seed(SEED)

_captured: dict[str, torch.Tensor] = {}
_orig_eager_attention_forward = qwen2_mod.eager_attention_forward


def _capturing_eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    attn_output, attn_weights = _orig_eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout, **kwargs
    )
    if getattr(module, "layer_idx", None) == TARGET_LAYER:
        _captured["attn_weights"] = attn_weights.detach().to("cpu", torch.float32)
    return attn_output, attn_weights


qwen2_mod.eager_attention_forward = _capturing_eager_attention_forward

SEED_PASSAGES = [
    "Stellar nucleosynthesis proceeds through a sequence of fusion stages inside "
    "a star's core, beginning with hydrogen burning via the proton-proton chain "
    "or the CNO cycle, and progressing through helium burning via the triple-alpha "
    "process once the core temperature and density cross the required threshold.",
    "Deep-sea hydrothermal vent ecosystems derive their primary productivity from "
    "chemosynthetic bacteria that oxidize hydrogen sulfide rather than relying on "
    "photosynthesis, supporting dense communities of tube worms, vent crabs, and "
    "specialized bivalves adapted to extreme pressure and total darkness.",
    "Lattice-based cryptographic schemes rest on the conjectured hardness of "
    "problems like Learning With Errors and Shortest Vector Problem over "
    "high-dimensional lattices, which are believed to resist attacks from both "
    "classical and quantum adversaries, unlike factoring-based schemes such as RSA.",
    "Distributed consensus protocols such as Raft and Paxos must tolerate node "
    "failures and network partitions while still guaranteeing that all correct "
    "replicas agree on the same sequence of committed operations, typically by "
    "requiring a quorum of acknowledgments before a value is considered durable.",
    "Volcanic eruptions are broadly classified by their explosivity index, which "
    "depends on magma viscosity, dissolved gas content, and the rate of magma "
    "ascent; silica-rich magmas trap gas more effectively and tend to produce "
    "far more violent eruptions than the comparatively fluid basaltic magmas.",
]

PROMPT_TARGET_TOKENS = [2000, 2300, 2600, 2900, 3200]


def make_prompt(tokenizer, seed_text: str, target_tokens: int, variant: int) -> torch.Tensor:
    pieces = []
    total = 0
    i = 0
    while total < target_tokens + 64:
        piece = f"[passage {variant}.{i}] {seed_text}"
        pieces.append(piece)
        total += len(tokenizer.encode(piece, add_special_tokens=False))
        i += 1
    full_text = " ".join(pieces)
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return torch.tensor([ids], dtype=torch.long)


def policy_topn_kept(policy_cls, num_blocks: int, n: int) -> set[int]:
    """Insert blocks 0..num_blocks-1 in prefill order, evict down to n kept,
    return the set of block indices the policy chose to keep."""
    policy = policy_cls(cache_capacity=num_blocks)
    keys = [i.to_bytes(4, "big") for i in range(num_blocks)]
    for i, key in enumerate(keys):
        block = BlockStatus(block_id=i)
        block.ref_cnt = 0  # ready + evictable, as if its store already completed
        policy.insert(key, block)
    evicted = policy.evict(num_blocks - n, protected=set())
    assert evicted is not None, "policy could not evict enough blocks"
    evicted_keys = {key for key, _ in evicted}
    kept = [i for i, key in enumerate(keys) if key not in evicted_keys]
    assert len(kept) == n
    return set(kept)


def run(model, tokenizer):
    recall_sums = {"lru": {n: 0.0 for n in TOP_NS}, "arc": {n: 0.0 for n in TOP_NS}}
    recall_counts = {"lru": {n: 0 for n in TOP_NS}, "arc": {n: 0 for n in TOP_NS}}
    identical_predictions = 0
    total_predictions = 0

    for prompt_idx, (seed_text, target_tokens) in enumerate(
        zip(SEED_PASSAGES, PROMPT_TARGET_TOKENS)
    ):
        input_ids = make_prompt(tokenizer, seed_text, target_tokens, prompt_idx).to(DEVICE)
        seq_len = input_ids.shape[1]
        print(f"prompt {prompt_idx}: seq_len={seq_len}")

        _captured.clear()
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)

        attn_weights = _captured["attn_weights"][0]  # [num_heads, seq, seq]

        probe_fracs = [0.3, 0.5, 0.7, 0.9, 1.0]
        probe_positions = sorted(
            {min(int(f * (seq_len - 1)), seq_len - 1) for f in probe_fracs}
        )

        for q_pos in probe_positions:
            num_blocks = (q_pos + 1) // BLOCK_TOKENS
            if num_blocks < max(TOP_NS) * 2:
                continue

            true_scores = torch.zeros(num_blocks)
            for b in range(num_blocks):
                lo, hi = b * BLOCK_TOKENS, (b + 1) * BLOCK_TOKENS
                true_scores[b] = attn_weights[:, q_pos, lo:hi].sum(dim=-1).mean()
            true_topn = {n: set(torch.topk(true_scores, n).indices.tolist()) for n in TOP_NS}

            for n in TOP_NS:
                lru_kept = policy_topn_kept(LRUCachePolicy, num_blocks, n)
                arc_kept = policy_topn_kept(ARCCachePolicy, num_blocks, n)

                total_predictions += 1
                if lru_kept == arc_kept:
                    identical_predictions += 1

                recall_sums["lru"][n] += len(true_topn[n] & lru_kept) / n
                recall_sums["arc"][n] += len(true_topn[n] & arc_kept) / n
                recall_counts["lru"][n] += 1
                recall_counts["arc"][n] += 1

    rows = []
    for method in ["lru", "arc"]:
        row = {"method": method}
        for n in TOP_NS:
            c = recall_counts[method][n]
            row[f"recall@{n}"] = round(recall_sums[method][n] / c, 4) if c else float("nan")
            row[f"n_probes@{n}"] = c
        rows.append(row)

    print(
        f"\nLRU/ARC identical top-N predictions: {identical_predictions}/{total_predictions} "
        f"({100.0 * identical_predictions / total_predictions:.1f}%)"
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main():
    print(f"device={DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    print("\n=== LRU / ARC recall@N (real vLLM policy classes) ===")
    rows = run(model, tokenizer)
    print_table(rows)
    write_csv(RESULTS_CSV, rows)
    print(f"wrote {RESULTS_CSV}")


if __name__ == "__main__":
    sys.exit(main())
