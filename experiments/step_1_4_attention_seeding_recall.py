# SPDX-License-Identifier: Apache-2.0
"""Stress test for issues log entry #13's "store-time attention-seeded
relevance" idea, run BEFORE any live-server engineering investment (see
.claude/docs/semantic-eviction-issues-log.md entry #14).

Question: does a block's REAL self-attention received during its own,
independent, storage-time prefill (attention from later tokens WITHIN THAT
SAME REQUEST ONLY -- never from a different, later, unrelated request, which
never happens in the live system) predict whether a genuinely different,
later, unrelated recall query will actually need that block?

Methodology, and why it differs from step_0_4_adversarial_needle_recall.py:
step_0_4 built one combined mega-prompt and scored oracle/minmax/mean against
real K vectors in it -- valid for those methods since K vectors don't depend
on how blocks got batched together for this offline analysis. It would NOT
be valid for a self-attention-received signal, because attention weights DO
depend on what else is in the same forward pass -- computing "self-attention"
within one shared mega-prompt would let the needle's block see (and receive
attention from) unrelated distractor content and even the recall question
itself, which never happens in the real system (separate HTTP requests never
attend to each other's prefill). So here, self-attention-received is always
computed from a SEPARATE, standalone forward pass per "request" (matching
entries #10-#13's real live-server design: needle and each distractor are
their own isolated store events), and only THEN pooled together for the
needle_hit@N ranking test -- the recall-time oracle/minmax/mean scores are
still computed from one combined prompt, exactly as step_0_4 does, since that
part legitimately doesn't depend on batching.

Two storage-time constructions, tested and reported separately (the real
project failure mode is (1), not (2)):
  1. standalone: needle text alone, matching entries #10-#13's real design
     (a short isolated HTTP request). Self-attention-received can only ever
     come from the needle's OWN few trailing tokens -- a real signal-scarcity
     risk, not assumed away.
  2. embedded: needle text embedded inside one longer single document (real
     prefix + needle + real suffix, all one request/forward pass), giving
     self-attention more trailing tokens to work with -- a more favorable
     case, useful to know about even if (1) fails.

Run: python3 experiments/step_1_4_attention_seeding_recall.py
Writes: experiments/step_1_4_attention_seeding_results.csv
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_TOKENS = 16
TARGET_LAYER = 14
TOP_NS = [4, 8]
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

PREFIX_BLOCKS = 8  # 128 tokens of real prefix before the needle, "embedded" case
SUFFIX_BLOCKS = 20  # 320 tokens of real suffix after the needle, "embedded" case
DISTRACTOR_TARGET_TOKENS = 480  # ~30 blocks per distractor doc

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EXPERIMENTS_DIR / "step_1_4_attention_seeding_results.csv"

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

# Same needle set as step_0_4, for direct comparability.
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


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def build_block_aligned_ids(tokenizer, seed_text, target_tokens, tag):
    """Repeat seed_text until >= target_tokens, then truncate to an exact
    multiple of BLOCK_TOKENS so segment boundaries always land on block
    boundaries."""
    desired = ((target_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS) * BLOCK_TOKENS
    ids: list[int] = []
    i = 0
    while len(ids) < desired:
        piece = f"[chunk {tag}.{i}] {seed_text}"
        ids.extend(encode(tokenizer, piece))
        i += 1
    return ids[:desired]


def run_forward(model, ids: list[int]):
    """Run one standalone forward pass; return (attn_weights[heads,seq,seq],
    key[heads,seq,head_dim], query[heads,seq,head_dim], seq_len)."""
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    _captured.clear()
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    return (
        _captured["attn_weights"][0],
        _captured["key"][0],
        _captured["query"][0],
        input_ids.shape[1],
    )


def self_attention_received(attn: torch.Tensor, seq_len: int, block_range: range):
    """For each block in block_range, sum/max of real attention received from
    ALL later query positions WITHIN THIS SAME forward pass. Returns
    (sum_scores, max_scores), each [len(block_range), num_heads]."""
    num_heads = attn.shape[0]
    sum_scores = torch.zeros(len(block_range), num_heads)
    max_scores = torch.zeros(len(block_range), num_heads)
    for i, b in enumerate(block_range):
        lo, hi = b * BLOCK_TOKENS, (b + 1) * BLOCK_TOKENS
        if hi >= seq_len:
            continue  # last block: no later tokens exist at all
        later_to_block = attn[:, hi:, lo:hi].sum(dim=-1)  # [heads, num_later_q]
        sum_scores[i] = later_to_block.sum(dim=-1)
        max_scores[i] = later_to_block.max(dim=-1).values
    return sum_scores, max_scores


def needle_hit_at_n(scores: torch.Tensor, needle_idx: set[int], n: int) -> float:
    topn = set(torch.topk(scores, min(n, scores.shape[0])).indices.tolist())
    return len(needle_idx & topn) / len(needle_idx)


def run_construction(
    model, tokenizer, construction: str, num_heads, num_kv_heads, num_groups,
    distractor_target_tokens: int = DISTRACTOR_TARGET_TOKENS, pool_tag: str = "p0",
):
    num_groups_ = num_heads // num_kv_heads
    assert num_groups_ == num_groups

    # Distractor pool: computed once, reused for every needle case within
    # this construction (distractor content doesn't depend on the needle).
    # pool_tag/distractor_target_tokens let the caller vary pool size/content
    # to get independent trials (different competing-block compositions)
    # without needing new needle text.
    distractor_ids_list = [
        build_block_aligned_ids(tokenizer, seed, distractor_target_tokens, f"{pool_tag}.d{i}")
        for i, seed in enumerate(NOISE_SEEDS)
    ]
    distractor_self_attn_sum = []
    distractor_self_attn_max = []
    for ids in distractor_ids_list:
        attn, _key, _q, seq_len = run_forward(model, ids)
        nb = seq_len // BLOCK_TOKENS
        s_sum, s_max = self_attention_received(attn, seq_len, range(nb))
        distractor_self_attn_sum.append(s_sum)
        distractor_self_attn_max.append(s_max)

    methods = [
        "self_attn_sum_maxhead",
        "self_attn_sum_meanhead",
        "self_attn_max_maxhead",
        "self_attn_max_meanhead",
        "oracle_meanhead",
        "minmax_meanhead",
        "mean_meanhead",
        "minmax_maxhead",
        "random",
    ]
    hit_sums = {m: {n: 0.0 for n in TOP_NS} for m in methods}
    n_cases = 0

    for needle_text, question in NEEDLES:
        needle_ids = encode(tokenizer, needle_text)

        if construction == "standalone":
            segment_ids = list(needle_ids)
            needle_start = 0
        else:  # embedded
            prefix_ids = build_block_aligned_ids(
                tokenizer, NOISE_SEEDS[0], PREFIX_BLOCKS * BLOCK_TOKENS, "pre"
            )
            suffix_ids = build_block_aligned_ids(
                tokenizer, NOISE_SEEDS[1], SUFFIX_BLOCKS * BLOCK_TOKENS, "suf"
            )
            segment_ids = prefix_ids + needle_ids + suffix_ids
            needle_start = len(prefix_ids)

        needle_end = needle_start + len(needle_ids)
        seg_num_blocks = (len(segment_ids) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        needle_blocks = set(range(needle_start // BLOCK_TOKENS, -(-needle_end // BLOCK_TOKENS)))
        needle_blocks = {b for b in needle_blocks if b < seg_num_blocks}
        if not needle_blocks:
            continue

        # --- storage-time pass: needle segment alone (own forward pass) ---
        seg_padded_len = seg_num_blocks * BLOCK_TOKENS
        seg_ids_padded = segment_ids + [tokenizer.eos_token_id] * (
            seg_padded_len - len(segment_ids)
        )
        seg_attn, _seg_key, _seg_q, seg_seq_len = run_forward(model, seg_ids_padded)
        seg_self_sum, seg_self_max = self_attention_received(
            seg_attn, seg_seq_len, range(seg_num_blocks)
        )

        # --- pool: needle segment blocks (own pass) + distractor blocks (own passes) ---
        pool_self_sum = [seg_self_sum]
        pool_self_max = [seg_self_max]
        for s_sum, s_max in zip(distractor_self_attn_sum, distractor_self_attn_max):
            pool_self_sum.append(s_sum)
            pool_self_max.append(s_max)
        pool_self_sum = torch.cat(pool_self_sum, dim=0)  # [total_blocks, heads]
        pool_self_max = torch.cat(pool_self_max, dim=0)

        # --- recall-time pass: ONE combined prompt, oracle/minmax/mean (step_0_4 style) ---
        question_ids = encode(tokenizer, f"\nQuestion: {question} Answer:")
        combined_ids = list(segment_ids)
        for ids in distractor_ids_list:
            combined_ids.extend(ids)
        combined_ids.extend(question_ids)
        total_blocks = len(combined_ids) // BLOCK_TOKENS
        q_pos = len(combined_ids) - 1
        if total_blocks != pool_self_sum.shape[0]:
            # question tail isn't block-aligned; only the block-aligned prefix
            # (segment + distractors) is scored, matching pool_self_sum's size.
            total_blocks = pool_self_sum.shape[0]

        attn, key, query, seq_len = run_forward(model, combined_ids)
        true_scores = torch.zeros(total_blocks)
        for b in range(total_blocks):
            lo, hi = b * BLOCK_TOKENS, (b + 1) * BLOCK_TOKENS
            true_scores[b] = attn[:, q_pos, lo:hi].sum(dim=-1).mean()

        key_blocks = torch.stack(
            [key[:, b * BLOCK_TOKENS : (b + 1) * BLOCK_TOKENS, :] for b in range(total_blocks)],
            dim=0,
        )
        q_vec = query[:, q_pos, :]
        block_max = key_blocks.max(dim=2).values
        block_min = key_blocks.min(dim=2).values
        block_mean = key_blocks.mean(dim=2)

        per_head = {m: torch.zeros(total_blocks, num_heads) for m in ["minmax", "mean", "oracle"]}
        for h in range(num_heads):
            kv_h = h // num_groups
            qh = q_vec[h]
            prod_max = qh.unsqueeze(0) * block_max[:, kv_h, :]
            prod_min = qh.unsqueeze(0) * block_min[:, kv_h, :]
            per_head["minmax"][:, h] = torch.maximum(prod_max, prod_min).sum(dim=-1)
            per_head["mean"][:, h] = torch.einsum("bd,d->b", block_mean[:, kv_h, :], qh)
            real_dots = torch.einsum("btd,d->bt", key_blocks[:, kv_h, :, :], qh)
            per_head["oracle"][:, h] = real_dots.max(dim=-1).values

        n_cases += 1
        random_scores = torch.rand(total_blocks)
        final_scores = {
            "self_attn_sum_maxhead": pool_self_sum.max(dim=-1).values,
            "self_attn_sum_meanhead": pool_self_sum.mean(dim=-1),
            "self_attn_max_maxhead": pool_self_max.max(dim=-1).values,
            "self_attn_max_meanhead": pool_self_max.mean(dim=-1),
            "oracle_meanhead": per_head["oracle"].mean(dim=-1),
            "minmax_meanhead": per_head["minmax"].mean(dim=-1),
            "mean_meanhead": per_head["mean"].mean(dim=-1),
            "minmax_maxhead": per_head["minmax"].max(dim=-1).values,
            "random": random_scores,
        }
        for m, scores in final_scores.items():
            for n in TOP_NS:
                hit_sums[m][n] += needle_hit_at_n(scores, needle_blocks, n)

        print(
            f"[{construction}] needle='{needle_text[:30]}...' needle_blocks={needle_blocks} "
            f"total_blocks={total_blocks} seg_num_blocks={seg_num_blocks} "
            f"self_attn_sum(needle,maxhead)={pool_self_sum.max(dim=-1).values[list(needle_blocks)].tolist()}"
        )

    return methods, hit_sums, n_cases


def rows_from_hit_sums(construction, methods, hit_sums, n_cases):
    rows = []
    for m in methods:
        row = {"construction": construction, "method": m, "n_cases": n_cases}
        for n in TOP_NS:
            row[f"needle_hit@{n}"] = round(hit_sums[m][n] / n_cases, 4) if n_cases else None
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
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_groups = num_heads // num_kv_heads

    all_rows = []

    # standalone: the real project failure mode -- run against 3 differently
    # sized distractor pools (independent competing-block compositions) to
    # get 4 needles x 3 pools = 12 trials instead of 4, before trusting any
    # signal enough to justify a live-server Stage 2.
    combined_methods = None
    combined_hits = None
    combined_n = 0
    for pool_i, target_tokens in enumerate([300, 480, 700]):
        print(f"\n=== construction=standalone pool_variant={pool_i} target_tokens={target_tokens} ===")
        methods, hit_sums, n_cases = run_construction(
            model, tokenizer, "standalone", num_heads, num_kv_heads, num_groups,
            distractor_target_tokens=target_tokens, pool_tag=f"s{pool_i}",
        )
        print_table(rows_from_hit_sums(f"standalone_pool{pool_i}", methods, hit_sums, n_cases))
        if combined_methods is None:
            combined_methods = methods
            combined_hits = {m: {n: 0.0 for n in TOP_NS} for m in methods}
        for m in methods:
            for n in TOP_NS:
                combined_hits[m][n] += hit_sums[m][n]
        combined_n += n_cases
    all_rows.extend(rows_from_hit_sums("standalone_combined", combined_methods, combined_hits, combined_n))

    print(f"\n=== construction=embedded ===")
    methods, hit_sums, n_cases = run_construction(
        model, tokenizer, "embedded", num_heads, num_kv_heads, num_groups
    )
    rows = rows_from_hit_sums("embedded", methods, hit_sums, n_cases)
    all_rows.extend(rows)
    print_table(rows)

    write_csv(RESULTS_CSV, all_rows)
    print(f"\nwrote {RESULTS_CSV}")
    print("\n=== FINAL: standalone_combined (12 trials) ===")
    print_table(rows_from_hit_sums("standalone_combined", combined_methods, combined_hits, combined_n))


if __name__ == "__main__":
    sys.exit(main())
