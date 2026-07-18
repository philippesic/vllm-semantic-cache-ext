"""Step 0.3 follow-up #4 (adversarial workload): the actual test the plan's
Step 1.4 accept criterion calls for — "a synthetic workload where LRU
provably evicts the wrong blocks (long-idle but semantically-needed prefix,
plus cache-filling noise traffic)". Every prior step_0_3_* experiment used a
single coherent document, where recency and relevance are naturally
correlated (LRU tied or nearly tied with semantic methods). This experiment
is designed specifically to decorrelate them.

Construction: a short, distinctive "needle" fact is placed at the very start
of the prompt (maximum idle time — the most adversarial position for a
recency-based policy), followed by ~1500-3000 tokens of topically unrelated
"noise" filler, followed by a question that can only be answered by
recalling the needle. The needle block(s) are known by construction (not
inferred from attention). Two ground truths are measured:

  - attn_recall@N: recall of the approximate top-N against the REAL
    attention-mass top-N (same metric as every other step_0_3_* script) —
    also reveals whether the base model actually attends back to the needle
    at all (a necessary precondition; if the model itself fails the
    needle-in-haystack task, no eviction policy can be blamed for missing it).
  - needle_hit@N: whether the known needle block(s) are contained in the
    method's predicted top-N — the direct, construction-grounded test of the
    plan's hypothesis ("LRU evicts the wrong block; semantic doesn't").

Methods compared, all reusing formulas/policy classes from the earlier
step_0_3_* scripts for exact consistency: LRU, ARC (real production classes
from vllm.v1.kv_offload.cpu.policies), min/max pooling, mean-only, oracle
(real uncompressed max-key dot product).

Run: python3 experiments/step_0_4_adversarial_needle_recall.py
Writes: experiments/step_0_4_adversarial_results.csv
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
RESULTS_CSV = EXPERIMENTS_DIR / "step_0_4_adversarial_results.csv"

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
        _captured["query"] = query.detach().to("cpu", torch.float32)
        _captured["key"] = key.detach().to("cpu", torch.float32)
        _captured["attn_weights"] = attn_weights.detach().to("cpu", torch.float32)
    return attn_output, attn_weights


qwen2_mod.eager_attention_forward = _capturing_eager_attention_forward

# ---- needle facts: short, distinctive, answerable by exact recall ----
NEEDLES = [
    (
        "Classified briefing note: the secret access code for Project Nightingale "
        "is 47392-Delta. Memorize this code.",
        "What is the secret access code for Project Nightingale?",
    ),
    (
        "Field report: the hidden rendezvous point for the resistance cell is the "
        "old lighthouse on Pelican Point. This location must not be written down elsewhere.",
        "What is the hidden rendezvous point for the resistance cell?",
    ),
    (
        "Security memo: the password for the encrypted backup vault is CrimsonHawk88. "
        "Do not share this password over unsecured channels.",
        "What is the password for the encrypted backup vault?",
    ),
    (
        "Expedition log: the coordinates of the sunken treasure wreck are 21.4N, 157.8W. "
        "These coordinates were confirmed by sonar survey.",
        "What are the coordinates of the sunken treasure wreck?",
    ),
]

# ---- noise: topically unrelated filler, reused from earlier step_0_3 scripts ----
NOISE_SEEDS = [
    "Stellar nucleosynthesis proceeds through a sequence of fusion stages inside "
    "a star's core, beginning with hydrogen burning via the proton-proton chain "
    "or the CNO cycle, and progressing through helium burning via the triple-alpha "
    "process once the core temperature and density cross the required threshold.",
    "Deep-sea hydrothermal vent ecosystems derive their primary productivity from "
    "chemosynthetic bacteria that oxidize hydrogen sulfide rather than relying on "
    "photosynthesis, supporting dense communities of tube worms, vent crabs, and "
    "specialized bivalves adapted to extreme pressure and total darkness.",
    "Distributed consensus protocols such as Raft and Paxos must tolerate node "
    "failures and network partitions while still guaranteeing that all correct "
    "replicas agree on the same sequence of committed operations, typically by "
    "requiring a quorum of acknowledgments before a value is considered durable.",
    "Volcanic eruptions are broadly classified by their explosivity index, which "
    "depends on magma viscosity, dissolved gas content, and the rate of magma "
    "ascent; silica-rich magmas trap gas more effectively and tend to produce "
    "far more violent eruptions than the comparatively fluid basaltic magmas.",
    "Medieval trade networks across the Silk Road connected East Asia, Central "
    "Asia, the Middle East, and Europe, moving not only silk and spices but also "
    "technologies, religions, and diseases along overlapping caravan and maritime "
    "routes that shifted over centuries in response to political fragmentation.",
]

NOISE_TARGET_TOKENS = [1500, 3000]  # two idle durations


def build_prompt(tokenizer, needle_text, question, noise_seed, noise_target_tokens, variant):
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    needle_tokens = len(needle_ids)
    needle_blocks = list(range((needle_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS))

    pieces = [needle_text]
    total = needle_tokens
    i = 0
    while total < noise_target_tokens:
        piece = f"[filler {variant}.{i}] {noise_seed}"
        pieces.append(piece)
        total += len(tokenizer.encode(piece, add_special_tokens=False))
        i += 1
    pieces.append(f"\nQuestion: {question} Answer:")
    full_text = " ".join(pieces)
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    return torch.tensor([ids], dtype=torch.long), needle_blocks


def policy_topn_kept(policy_cls, num_blocks: int, n: int) -> set[int]:
    policy = policy_cls(cache_capacity=num_blocks)
    keys = [i.to_bytes(4, "big") for i in range(num_blocks)]
    for i, key in enumerate(keys):
        block = BlockStatus(block_id=i)
        block.ref_cnt = 0
        policy.insert(key, block)
    evicted = policy.evict(num_blocks - n, protected=set())
    assert evicted is not None
    evicted_keys = {key for key, _ in evicted}
    return {i for i, key in enumerate(keys) if key not in evicted_keys}


def run(model, tokenizer):
    num_attention_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_groups = num_attention_heads // num_kv_heads

    methods = ["lru", "arc", "minmax", "mean", "oracle"]
    attn_recall_sums = {m: {n: 0.0 for n in TOP_NS} for m in methods}
    needle_hit_sums = {m: {n: 0.0 for n in TOP_NS} for m in methods}
    n_cases = 0

    for needle_text, question in NEEDLES:
        for noise_seed in NOISE_SEEDS[:2]:  # keep runtime modest: 4 needles x 2 noise topics
            for variant, noise_target in enumerate(NOISE_TARGET_TOKENS):
                input_ids, needle_blocks = build_prompt(
                    tokenizer, needle_text, question, noise_seed, noise_target, variant
                )
                input_ids = input_ids.to(DEVICE)
                seq_len = input_ids.shape[1]
                q_pos = seq_len - 1
                num_blocks = (q_pos + 1) // BLOCK_TOKENS
                if num_blocks < max(TOP_NS) * 2 or not needle_blocks:
                    continue
                if max(needle_blocks) >= num_blocks:
                    continue  # needle got merged into the trailing partial block; skip

                _captured.clear()
                with torch.no_grad():
                    model(input_ids=input_ids, use_cache=False)

                query = _captured["query"][0]
                key = _captured["key"][0]
                attn_weights = _captured["attn_weights"][0]

                true_scores = torch.zeros(num_blocks)
                for b in range(num_blocks):
                    lo, hi = b * BLOCK_TOKENS, (b + 1) * BLOCK_TOKENS
                    true_scores[b] = attn_weights[:, q_pos, lo:hi].sum(dim=-1).mean()
                true_topn = {
                    n: set(torch.topk(true_scores, n).indices.tolist()) for n in TOP_NS
                }

                key_blocks = torch.stack(
                    [key[:, b * BLOCK_TOKENS : (b + 1) * BLOCK_TOKENS, :] for b in range(num_blocks)],
                    dim=0,
                )
                q_vec = query[:, q_pos, :]
                block_max = key_blocks.max(dim=2).values
                block_min = key_blocks.min(dim=2).values
                block_mean = key_blocks.mean(dim=2)

                per_head = {m: torch.zeros(num_blocks, num_attention_heads) for m in ["minmax", "mean", "oracle"]}
                for h in range(num_attention_heads):
                    kv_h = h // num_groups
                    qh = q_vec[h]
                    prod_max = qh.unsqueeze(0) * block_max[:, kv_h, :]
                    prod_min = qh.unsqueeze(0) * block_min[:, kv_h, :]
                    per_head["minmax"][:, h] = torch.maximum(prod_max, prod_min).sum(dim=-1)
                    per_head["mean"][:, h] = torch.einsum("bd,d->b", block_mean[:, kv_h, :], qh)
                    real_dots = torch.einsum("btd,d->bt", key_blocks[:, kv_h, :, :], qh)
                    per_head["oracle"][:, h] = real_dots.max(dim=-1).values

                needle_set = set(needle_blocks)
                n_cases += 1

                for n in TOP_NS:
                    lru_kept = policy_topn_kept(LRUCachePolicy, num_blocks, n)
                    arc_kept = policy_topn_kept(ARCCachePolicy, num_blocks, n)
                    for method, kept in [("lru", lru_kept), ("arc", arc_kept)]:
                        attn_recall_sums[method][n] += len(true_topn[n] & kept) / n
                        needle_hit_sums[method][n] += len(needle_set & kept) / len(needle_set)

                    for method in ["minmax", "mean", "oracle"]:
                        approx_scores = per_head[method].mean(dim=-1)
                        approx_topn = set(torch.topk(approx_scores, n).indices.tolist())
                        attn_recall_sums[method][n] += len(true_topn[n] & approx_topn) / n
                        needle_hit_sums[method][n] += len(needle_set & approx_topn) / len(needle_set)

                print(
                    f"needle_blocks={needle_blocks} num_blocks={num_blocks} "
                    f"true_top8={sorted(true_topn[8])}"
                )

    rows = []
    for m in methods:
        row = {"method": m, "n_cases": n_cases}
        for n in TOP_NS:
            row[f"attn_recall@{n}"] = round(attn_recall_sums[m][n] / n_cases, 4)
            row[f"needle_hit@{n}"] = round(needle_hit_sums[m][n] / n_cases, 4)
        rows.append(row)
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

    print("\n=== Adversarial needle-in-haystack: attn_recall@N and needle_hit@N ===")
    rows = run(model, tokenizer)
    print_table(rows)
    write_csv(RESULTS_CSV, rows)
    print(f"wrote {RESULTS_CSV}")


if __name__ == "__main__":
    sys.exit(main())
