# SPDX-License-Identifier: Apache-2.0
"""Sweep relevance/recency weights on held-out model-query traces.

Each case updates the production-formula raw, rank-weighted EMA with one
needle query, one semantic-decoy query, and six unrelated queries. Eviction
happens only after those updates. Two held-out queries then evaluate the
survivors: a paraphrased needle query tests old semantic reuse, while an unseen
topic tests whether stronger semantic weighting sacrifices adaptation.

The cache recency order independently replays needle and decoy blocks in oldest,
middle, and newest strata. This isolates the blend from token position while
preserving identical model scores and candidate coverage across alpha arms.

Run: python experiments/step_1_4_blend_sweep.py
Writes: experiments/step_1_4_blend_sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from step_1_4_score_calibration import (
    BLOCK_TOKENS,
    CANDIDATE_TOKENS,
    DEVICE,
    DISTRACTOR_QUESTIONS,
    DTYPE,
    MODEL_NAME,
    NEEDLES,
    NOISE_PASSAGES,
    QUERY_SCAFFOLD_TOKENS,
    SEED,
    _capture_attention,
    build_query,
    capture_scores,
    encode,
    update_ema,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

ALPHAS = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.95, 1.0)
CAPACITIES = (8, 16, 24, 32)
RECENCY_STRATA = ("oldest", "middle", "newest")
RESULTS_CSV = Path(__file__).with_name("step_1_4_blend_sweep_results.csv")
DECOY_START_BLOCK = 16

RELATED_FOLLOWUPS = [
    "Repeat the exact Project Nightingale code from the archived briefing.",
    "Which exact Pelican Point landmark was named for the rendezvous?",
    "Repeat the exact password assigned to the encrypted backup vault.",
    "Give the exact latitude and longitude recorded for the wreck.",
]

DECOYS = [
    (
        "Museum catalog: the ceremonial brass key has inventory code AX-1904.",
        "What inventory code belongs to the ceremonial brass key?",
    ),
    (
        "Garden ledger: the rare orchid shipment arrived in crate QZ-771.",
        "Which crate contained the rare orchid shipment?",
    ),
    (
        "Aviation note: the retired survey aircraft used call sign Lumen-52.",
        "What was the retired survey aircraft's call sign?",
    ),
    (
        "Archive index: the blue cartography folio is filed as shelf R-314.",
        "Under which shelf code is the blue cartography folio filed?",
    ),
]


def build_candidate_prefix_with_decoy(tokenizer, case_index: int):
    needle_ids = encode(tokenizer, NEEDLES[case_index][0])
    decoy_ids = encode(tokenizer, DECOYS[case_index][0])
    ids = list(needle_ids)
    passage_index = case_index
    decoy_start = DECOY_START_BLOCK * BLOCK_TOKENS
    while len(ids) < decoy_start:
        passage = NOISE_PASSAGES[passage_index % len(NOISE_PASSAGES)]
        ids.extend(encode(tokenizer, f" [document {passage_index}] {passage}"))
        passage_index += 1
    ids = ids[:decoy_start]
    ids.extend(decoy_ids)
    decoy_end = len(ids)
    while len(ids) < CANDIDATE_TOKENS:
        passage = NOISE_PASSAGES[passage_index % len(NOISE_PASSAGES)]
        ids.extend(encode(tokenizer, f" [document {passage_index}] {passage}"))
        passage_index += 1
    ids = ids[:CANDIDATE_TOKENS]
    ids = ids[: len(ids) // BLOCK_TOKENS * BLOCK_TOKENS]
    needle_blocks = set(range((len(needle_ids) + BLOCK_TOKENS - 1) // BLOCK_TOKENS))
    decoy_blocks = set(
        range(
            decoy_start // BLOCK_TOKENS,
            (decoy_end + BLOCK_TOKENS - 1) // BLOCK_TOKENS,
        )
    )
    return ids, needle_blocks, decoy_blocks


def recency_order(
    num_blocks: int,
    needle_blocks: set[int],
    target_stratum: str,
    decoy_blocks: set[int] | None = None,
    decoy_stratum: str = "middle",
    rotation: int = 0,
) -> list[int]:
    decoy_blocks = decoy_blocks or set()
    if target_stratum not in RECENCY_STRATA or decoy_stratum not in RECENCY_STRATA:
        raise ValueError("unknown recency stratum")
    anchors = {"oldest": -0.1, "middle": 0.5, "newest": 1.1}
    position = {index: index / max(num_blocks - 1, 1) for index in range(num_blocks)}
    groups = [
        (needle_blocks, target_stratum),
        (decoy_blocks, decoy_stratum),
    ]
    for group_index, (group, stratum) in enumerate(groups):
        group_offset = 0.0
        if target_stratum == decoy_stratum:
            target_first = rotation % 2 == 0
            group_offset = 0.002 * (group_index if target_first else 1 - group_index)
        for within_group, index in enumerate(sorted(group)):
            position[index] = anchors[stratum] + group_offset + within_group * 0.0001
    order = sorted(range(num_blocks), key=lambda index: (position[index], index))
    assert sorted(order) == list(range(num_blocks))
    return order


def policy_survivors(
    ema: torch.Tensor, capacity: int, alpha: float, order: list[int]
) -> set[int]:
    lo = float(ema.min())
    span = float(ema.max()) - lo or 1.0
    denom = max(len(order) - 1, 1)
    scored = []
    for recency_rank, index in enumerate(order):
        relevance = (float(ema[index]) - lo) / span
        recency = recency_rank / denom if len(order) > 1 else 1.0
        scored.append((alpha * relevance + (1.0 - alpha) * recency, index))
    scored.sort(key=lambda item: item[0])
    evicted = {index for _, index in scored[: len(order) - capacity]}
    return set(order) - evicted


def top_indices(values: torch.Tensor, capacity: int) -> set[int]:
    ranked = sorted(
        range(values.numel()), key=lambda index: (-float(values[index]), index)
    )
    return set(ranked[:capacity])


def recall(reference: set[int], kept: set[int]) -> float:
    return len(reference & kept) / len(reference)


def run_case(model, tokenizer, case_index: int):
    _, related_question = NEEDLES[case_index]
    prefix_ids, needle_blocks, decoy_blocks = build_candidate_prefix_with_decoy(
        tokenizer, case_index
    )
    train_questions = [
        related_question,
        DECOYS[case_index][1],
        *DISTRACTOR_QUESTIONS[:6],
    ]
    scaffold_tokens = [
        QUERY_SCAFFOLD_TOKENS[(event + case_index) % len(QUERY_SCAFFOLD_TOKENS)]
        for event in range(len(train_questions))
    ]
    train_queries = [
        capture_scores(
            model,
            prefix_ids,
            build_query(
                tokenizer,
                question,
                NOISE_PASSAGES[(event + case_index) % len(NOISE_PASSAGES)],
                scaffold_tokens[event],
            ),
        )
        for event, question in enumerate(train_questions)
    ]

    ema = None
    for query in train_queries:
        ema = update_ema(ema, query.raw)
    assert ema is not None

    heldout_target = capture_scores(
        model,
        prefix_ids,
        build_query(
            tokenizer,
            RELATED_FOLLOWUPS[case_index],
            NOISE_PASSAGES[(case_index + 3) % len(NOISE_PASSAGES)],
            384,
        ),
    )
    heldout_novel = capture_scores(
        model,
        prefix_ids,
        build_query(
            tokenizer,
            DISTRACTOR_QUESTIONS[-1],
            NOISE_PASSAGES[0],
            640,
        ),
    )

    rows = []
    num_blocks = ema.numel()
    for target_stratum in RECENCY_STRATA:
        for decoy_stratum in RECENCY_STRATA:
            order = recency_order(
                num_blocks,
                needle_blocks,
                target_stratum,
                decoy_blocks,
                decoy_stratum,
                case_index,
            )
            for alpha in ALPHAS:
                for capacity in CAPACITIES:
                    kept = policy_survivors(ema, capacity, alpha, order)
                    target_top = top_indices(heldout_target.attention_mass, capacity)
                    novel_top = top_indices(heldout_novel.attention_mass, capacity)
                    rows.append(
                        {
                            "case": case_index,
                            "target_stratum": target_stratum,
                            "decoy_stratum": decoy_stratum,
                            "alpha": alpha,
                            "capacity": capacity,
                            "num_blocks": num_blocks,
                            "needle_fraction": recall(needle_blocks, kept),
                            "complete_needle": int(needle_blocks <= kept),
                            "decoy_fraction": recall(decoy_blocks, kept),
                            "complete_decoy": int(decoy_blocks <= kept),
                            "target_attention_recall": recall(target_top, kept),
                            "novel_attention_recall": recall(novel_top, kept),
                            "balanced_attention_recall": (
                                recall(target_top, kept) + recall(novel_top, kept)
                            )
                            / 2,
                            "train_query_norm_ratio": max(
                                query.query_norm for query in train_queries
                            )
                            / min(query.query_norm for query in train_queries),
                        }
                    )
    target_ranks = torch.empty_like(ema, dtype=torch.long)
    target_ranks[torch.argsort(ema, descending=True, stable=True)] = torch.arange(
        num_blocks
    )
    print(
        f"case={case_index} blocks={num_blocks} needle={sorted(needle_blocks)} "
        f"decoy={sorted(decoy_blocks)} "
        f"ema_needle_ranks={[int(target_ranks[i]) for i in needle_blocks]}"
    )
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    print(
        "alpha capacity needle complete decoy complete_decoy target_attention "
        "novel_attention balanced"
    )
    for alpha in ALPHAS:
        for capacity in CAPACITIES:
            selected = [
                row
                for row in rows
                if row["alpha"] == alpha and row["capacity"] == capacity
            ]
            means = {
                field: sum(row[field] for row in selected) / len(selected)
                for field in (
                    "needle_fraction",
                    "complete_needle",
                    "decoy_fraction",
                    "complete_decoy",
                    "target_attention_recall",
                    "novel_attention_recall",
                    "balanced_attention_recall",
                )
            }
            print(
                f"{alpha:5.2f} {capacity:8d} "
                f"{means['needle_fraction']:.4f} "
                f"{means['complete_needle']:.4f} "
                f"{means['decoy_fraction']:.4f} "
                f"{means['complete_decoy']:.4f} "
                f"{means['target_attention_recall']:.4f} "
                f"{means['novel_attention_recall']:.4f} "
                f"{means['balanced_attention_recall']:.4f}"
            )

    print("paired alpha=0.6 versus alpha=0.5")
    paired_fields = (
        "complete_needle",
        "complete_decoy",
        "target_attention_recall",
        "novel_attention_recall",
        "balanced_attention_recall",
    )
    for capacity in CAPACITIES:
        baseline = {
            (row["case"], row["target_stratum"], row["decoy_stratum"]): row
            for row in rows
            if row["alpha"] == 0.5 and row["capacity"] == capacity
        }
        candidate = {
            (row["case"], row["target_stratum"], row["decoy_stratum"]): row
            for row in rows
            if row["alpha"] == 0.6 and row["capacity"] == capacity
        }
        for field in paired_fields:
            deltas = [candidate[key][field] - baseline[key][field] for key in baseline]
            wins = sum(delta > 0 for delta in deltas)
            losses = sum(delta < 0 for delta in deltas)
            ties = len(deltas) - wins - losses
            print(
                f"capacity={capacity} field={field} wins={wins} "
                f"losses={losses} ties={ties} "
                f"mean_delta={sum(deltas) / len(deltas):.6f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=len(NEEDLES))
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    qwen2_mod.eager_attention_forward = _capture_attention
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    rows = []
    for case_index in range(min(args.max_cases, len(NEEDLES))):
        rows.extend(run_case(model, tokenizer, case_index))
    write_csv(rows, args.results_csv)
    summarize(rows)
    print(f"wrote {args.results_csv}")


if __name__ == "__main__":
    main()
