# SPDX-License-Identifier: Apache-2.0
"""Compare raw, rank, and query-normalized relevance across query traces.

The trace starts with one needle-recall query, then issues unrelated queries
against the same candidate prefix. It replays those scores through the current
rank-weighted EMA and semantic/recency blend. A controlled scale-stress arm
multiplies distractor query vectors without changing within-query rankings.

Run: python experiments/step_1_4_score_calibration.py
Writes: experiments/step_1_4_score_calibration_results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYER = 14
BLOCK_TOKENS = 16
CANDIDATE_TOKENS = 1024
CAPACITIES = (8, 16)
EMA_ALPHA = 0.3
POLICY_ALPHA = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
SEED = 0

RESULTS_CSV = Path(__file__).with_name("step_1_4_score_calibration_results.csv")

NEEDLES = [
    (
        (
            "Classified briefing: Project Nightingale uses access code "
            "47392-Delta. This exact code must be remembered."
        ),
        "What is the access code for Project Nightingale?",
    ),
    (
        (
            "Field report: the resistance rendezvous is the old lighthouse "
            "on Pelican Point. Remember this exact location."
        ),
        "Where is the resistance rendezvous?",
    ),
    (
        (
            "Security memo: the encrypted backup vault password is "
            "CrimsonHawk88. Remember the exact password."
        ),
        "What is the encrypted backup vault password?",
    ),
    (
        (
            "Expedition log: the sunken wreck is at 21.4N, 157.8W. Remember "
            "these exact coordinates."
        ),
        "What are the coordinates of the sunken wreck?",
    ),
]

NOISE_PASSAGES = [
    (
        "Stellar nucleosynthesis progresses from hydrogen burning to helium and "
        "heavier-element fusion as stellar core temperature and density increase."
    ),
    (
        "Hydrothermal vent ecosystems rely on chemosynthetic microbes that "
        "oxidize sulfur compounds and support specialized deep-sea communities."
    ),
    (
        "Raft and Paxos coordinate replicated state despite failures by "
        "requiring quorum agreement before operations become durable."
    ),
    (
        "Silica-rich magma traps dissolved gases and generally produces more "
        "explosive eruptions than low-viscosity basaltic magma."
    ),
    (
        "Silk Road trade moved goods, technologies, religions, and diseases "
        "across shifting land and maritime routes for centuries."
    ),
    (
        "Coral bleaching occurs when heat stress disrupts the symbiosis between "
        "corals and photosynthetic algae, reducing reef productivity."
    ),
    (
        "Public-key cryptography uses mathematically related keys so data "
        "encrypted with one key can only be decrypted with the other."
    ),
    (
        "Plate tectonics recycles oceanic crust at subduction zones and creates "
        "new crust along divergent mid-ocean ridges."
    ),
]

DISTRACTOR_QUESTIONS = [
    "What powers biological production at hydrothermal vents?",
    "Why are silica-rich volcanic eruptions often explosive?",
    "How do quorum-based consensus protocols tolerate failures?",
    "What kinds of things moved along Silk Road trade networks?",
    "What process causes coral bleaching during marine heat waves?",
    "How are public and private keys used in public-key cryptography?",
    "Where is new oceanic crust formed?",
    "What fusion stage follows hydrogen burning in a star?",
]

SCALE_STRESS = (1.0, 0.25, 4.0, 0.5, 8.0, 2.0, 16.0, 1.0, 6.0)
QUERY_SCAFFOLD_TOKENS = (0, 64, 256, 512, 768, 128, 384, 640, 192)
METHODS = ("raw", "rank", "query_l2")

_captured: dict[str, torch.Tensor] = {}
_original_eager_attention = qwen2_mod.eager_attention_forward


def _capture_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    output, weights = _original_eager_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout,
        **kwargs,
    )
    if getattr(module, "layer_idx", None) == TARGET_LAYER:
        _captured["query"] = query.detach().to("cpu", torch.float32)
        _captured["key"] = key.detach().to("cpu", torch.float32)
        _captured["attention"] = weights.detach().to("cpu", torch.float32)
    return output, weights


@dataclass
class QueryScores:
    raw: torch.Tensor
    query_l2: torch.Tensor
    query_norm: float
    attention_mass: torch.Tensor


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_candidate_prefix(tokenizer, needle: str, case_index: int):
    needle_ids = encode(tokenizer, needle)
    ids = list(needle_ids)
    passage_index = case_index
    while len(ids) < CANDIDATE_TOKENS:
        passage = NOISE_PASSAGES[passage_index % len(NOISE_PASSAGES)]
        ids.extend(encode(tokenizer, f" [document {passage_index}] {passage}"))
        passage_index += 1
    ids = ids[:CANDIDATE_TOKENS]
    ids = ids[: len(ids) // BLOCK_TOKENS * BLOCK_TOKENS]
    needle_blocks = set(range((len(needle_ids) + BLOCK_TOKENS - 1) // BLOCK_TOKENS))
    return ids, needle_blocks


def build_query(tokenizer, question: str, scaffold: str, scaffold_tokens: int):
    scaffold_ids = []
    part_index = 0
    while len(scaffold_ids) < scaffold_tokens:
        scaffold_ids.extend(encode(tokenizer, f" [context {part_index}] {scaffold}"))
        part_index += 1
    scaffold_ids = scaffold_ids[:scaffold_tokens]
    return scaffold_ids + encode(tokenizer, f"\nQuestion: {question}\nAnswer:")


def capture_scores(model, prefix_ids, question_ids) -> QueryScores:
    input_ids = torch.tensor(
        [prefix_ids + question_ids], dtype=torch.long, device=DEVICE
    )
    _captured.clear()
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)

    query = _captured["query"][0, :, -1, :]
    key = _captured["key"][0, :, : len(prefix_ids), :]
    attention = _captured["attention"][0, :, -1, : len(prefix_ids)]
    num_attention_heads = query.shape[0]
    num_kv_heads = key.shape[0]
    groups = num_attention_heads // num_kv_heads
    query = query.reshape(num_kv_heads, groups, query.shape[-1]).mean(dim=1)

    num_blocks = len(prefix_ids) // BLOCK_TOKENS
    key_blocks = key.reshape(
        num_kv_heads, num_blocks, BLOCK_TOKENS, key.shape[-1]
    ).permute(1, 0, 2, 3)
    summaries = key_blocks.mean(dim=2)
    per_head = (summaries * query.unsqueeze(0)).sum(dim=-1)
    raw = per_head.mean(dim=-1)
    query_norm = query.flatten().norm().clamp_min(torch.finfo(query.dtype).eps)
    query_l2 = raw / query_norm

    attention_blocks = attention.reshape(
        num_attention_heads, num_blocks, BLOCK_TOKENS
    ).sum(dim=-1)
    attention_mass = attention_blocks.mean(dim=0)
    return QueryScores(
        raw=raw,
        query_l2=query_l2,
        query_norm=float(query_norm),
        attention_mass=attention_mass,
    )


def rank_signal(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 1:
        return torch.ones_like(scores)
    order = torch.argsort(scores, descending=True, stable=True)
    signal = torch.empty_like(scores)
    sorted_scores = scores[order]
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        mid_rank = (start + end - 1) / 2
        signal[order[start:end]] = 1.0 - mid_rank / (scores.numel() - 1)
        start = end
    return signal


def update_ema(ema: torch.Tensor | None, scores: torch.Tensor) -> torch.Tensor:
    if ema is None:
        return scores.clone()
    order = torch.argsort(scores, descending=True, stable=True)
    denom = max(scores.numel() - 1, 1)
    updated = ema.clone()
    for rank, index in enumerate(order.tolist()):
        weight = EMA_ALPHA * (1.0 - rank / denom)
        updated[index] = weight * scores[index] + (1.0 - weight) * ema[index]
    return updated


def normalized(values: torch.Tensor) -> torch.Tensor:
    span = values.max() - values.min()
    if span <= torch.finfo(values.dtype).eps:
        return torch.zeros_like(values)
    return (values - values.min()) / span


def survivors(ema: torch.Tensor, capacity: int) -> set[int]:
    recency = torch.linspace(0.0, 1.0, ema.numel())
    blended = POLICY_ALPHA * normalized(ema) + (1.0 - POLICY_ALPHA) * recency
    return set(torch.topk(blended, capacity).indices.tolist())


def relevance_top(ema: torch.Tensor, capacity: int) -> set[int]:
    return set(torch.topk(ema, capacity).indices.tolist())


def fraction_retained(target: set[int], kept: set[int]) -> float:
    return len(target & kept) / len(target)


def replay_case(case_index, needle_blocks, queries, scaffold_tokens, scenario, scales):
    rows = []
    attention_top = {
        capacity: set(torch.topk(queries[0].attention_mass, capacity).indices.tolist())
        for capacity in CAPACITIES
    }
    natural_norms = [query.query_norm for query in queries]
    effective_norms = [
        query.query_norm * scale for query, scale in zip(queries, scales)
    ]
    probe_order = torch.argsort(queries[0].raw, descending=True, stable=True)
    probe_ranks = torch.empty_like(probe_order)
    probe_ranks[probe_order] = torch.arange(probe_order.numel())
    best_needle_probe_rank = min(int(probe_ranks[index]) for index in needle_blocks)
    for method in METHODS:
        ema = None
        first_survivors = {}
        first_relevance_top = {}
        for step, (query, scale) in enumerate(zip(queries, scales)):
            if method == "raw":
                scores = query.raw * scale
            elif method == "rank":
                scores = rank_signal(query.raw)
            else:
                scores = query.query_l2
            ema = update_ema(ema, scores)
            if step == 0:
                first_survivors = {
                    capacity: survivors(ema, capacity) for capacity in CAPACITIES
                }
                first_relevance_top = {
                    capacity: relevance_top(ema, capacity) for capacity in CAPACITIES
                }
        assert ema is not None
        for capacity in CAPACITIES:
            final_survivors = survivors(ema, capacity)
            final_relevance_top = relevance_top(ema, capacity)
            rows.append(
                {
                    "case": case_index,
                    "scenario": scenario,
                    "method": method,
                    "capacity": capacity,
                    "num_blocks": ema.numel(),
                    "scaffold_tokens_min": min(scaffold_tokens),
                    "scaffold_tokens_max": max(scaffold_tokens),
                    "query_norm_min": min(natural_norms),
                    "query_norm_max": max(natural_norms),
                    "query_norm_ratio": max(natural_norms) / min(natural_norms),
                    "effective_query_norm_ratio": max(effective_norms)
                    / min(effective_norms),
                    "best_needle_probe_rank": best_needle_probe_rank,
                    "needle_relevance_after_probe": fraction_retained(
                        needle_blocks, first_relevance_top[capacity]
                    ),
                    "needle_relevance_final": fraction_retained(
                        needle_blocks, final_relevance_top
                    ),
                    "attention_relevance_final": fraction_retained(
                        attention_top[capacity], final_relevance_top
                    ),
                    "needle_after_probe": fraction_retained(
                        needle_blocks, first_survivors[capacity]
                    ),
                    "needle_final": fraction_retained(needle_blocks, final_survivors),
                    "attention_top_final": fraction_retained(
                        attention_top[capacity], final_survivors
                    ),
                }
            )
    return rows


def write_csv(rows, results_csv):
    with open(results_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    print(
        "scenario method capacity needle_relevance attention_relevance "
        "needle_policy attention_policy"
    )
    for scenario in ("natural", "scale_stress"):
        for method in METHODS:
            for capacity in CAPACITIES:
                selected = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["method"] == method
                    and row["capacity"] == capacity
                ]
                needle = sum(row["needle_final"] for row in selected) / len(selected)
                attention = sum(row["attention_top_final"] for row in selected) / len(
                    selected
                )
                needle_relevance = sum(
                    row["needle_relevance_final"] for row in selected
                ) / len(selected)
                attention_relevance = sum(
                    row["attention_relevance_final"] for row in selected
                ) / len(selected)
                print(
                    f"{scenario:12} {method:9} {capacity:8d} "
                    f"{needle_relevance:.4f} {attention_relevance:.4f} "
                    f"{needle:.4f} {attention:.4f}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=len(NEEDLES))
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    qwen2_mod.eager_attention_forward = _capture_attention
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    rows = []
    for case_index, (needle, related_question) in enumerate(NEEDLES[: args.max_cases]):
        prefix_ids, needle_blocks = build_candidate_prefix(
            tokenizer, needle, case_index
        )
        questions = [related_question, *DISTRACTOR_QUESTIONS]
        scaffold_tokens = [
            QUERY_SCAFFOLD_TOKENS[(event_index + case_index) % len(questions)]
            for event_index in range(len(questions))
        ]
        queries = [
            capture_scores(
                model,
                prefix_ids,
                build_query(
                    tokenizer,
                    question,
                    NOISE_PASSAGES[(event_index + case_index) % len(NOISE_PASSAGES)],
                    scaffold_tokens[event_index],
                ),
            )
            for event_index, question in enumerate(questions)
        ]
        print(
            f"case={case_index} blocks={len(prefix_ids) // BLOCK_TOKENS} "
            f"needle_blocks={sorted(needle_blocks)} "
            f"scaffold_tokens={scaffold_tokens} "
            f"query_norms={[round(query.query_norm, 4) for query in queries]}"
        )
        rows.extend(
            replay_case(
                case_index,
                needle_blocks,
                queries,
                scaffold_tokens,
                "natural",
                [1.0] * len(queries),
            )
        )
        rows.extend(
            replay_case(
                case_index,
                needle_blocks,
                queries,
                scaffold_tokens,
                "scale_stress",
                SCALE_STRESS,
            )
        )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.results_csv)
    summarize(rows)
    print(f"wrote {args.results_csv}")


if __name__ == "__main__":
    main()
