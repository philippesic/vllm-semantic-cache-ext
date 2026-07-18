"""Step 0.3 follow-up #2: two diagnostics recommended by the SVD salvage
analysis (`.claude/docs/step-0.3-svd-salvage-analysis.md`), run in the same
harness as step_0_3_svd_recall.py / step_0_3_minmax_recall.py so all numbers
are directly comparable (identical prompts, probes, ground truth).

1. Oracle: recall@N using the REAL, uncompressed max-per-key dot product
   (max_i dot(query, key_i) over the actual 16 keys in a block — zero
   compression at all) against the same attention-mass ground truth. Answers:
   is "max single-key dot product" even a good proxy for "total block
   softmax attention mass," independent of any summarization/compression
   error? This is a ceiling check, not a summary to ship.

2. Mean-augmented min/max: does adding a block_mean vector (still O(1) per
   block, not O(block_tokens)) to the existing min/max bounding-box score
   narrow the gap to the plan's 0.7 recall bar? Tests mean-alone and a
   min/max+mean blend alongside the already-measured min/max-alone baseline
   (0.44 / 0.445) for direct comparison.

Run: python3 experiments/step_0_3_oracle_and_mean_recall.py
Writes: experiments/step_0_3_oracle_mean_results.csv
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

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EXPERIMENTS_DIR / "step_0_3_oracle_mean_results.csv"

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


METHODS = ["oracle_real_maxkey", "mean_only", "minmax_plus_mean_avg", "minmax_only"]


def run(model, tokenizer):
    num_attention_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_groups = num_attention_heads // num_kv_heads
    head_dim = getattr(
        model.config, "head_dim", model.config.hidden_size // num_attention_heads
    )
    print(
        f"model: heads={num_attention_heads} kv_heads={num_kv_heads} "
        f"groups={num_groups} head_dim={head_dim} target_layer={TARGET_LAYER}"
    )

    recall_sums = {m: {n: 0.0 for n in TOP_NS} for m in METHODS}
    recall_counts = {m: {n: 0 for n in TOP_NS} for m in METHODS}

    for prompt_idx, (seed_text, target_tokens) in enumerate(
        zip(SEED_PASSAGES, PROMPT_TARGET_TOKENS)
    ):
        input_ids = make_prompt(tokenizer, seed_text, target_tokens, prompt_idx).to(DEVICE)
        seq_len = input_ids.shape[1]
        print(f"prompt {prompt_idx}: seq_len={seq_len}")

        _captured.clear()
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)

        query = _captured["query"][0]  # [num_heads, seq, head_dim]
        key = _captured["key"][0]  # [num_kv_heads, seq, head_dim]
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

            # [num_blocks, num_kv_heads, 16, head_dim]
            key_blocks = torch.stack(
                [
                    key[:, b * BLOCK_TOKENS : (b + 1) * BLOCK_TOKENS, :]
                    for b in range(num_blocks)
                ],
                dim=0,
            )
            q_vec = query[:, q_pos, :]  # [num_heads, head_dim]

            block_max = key_blocks.max(dim=2).values  # [num_blocks, num_kv_heads, head_dim]
            block_min = key_blocks.min(dim=2).values
            block_mean = key_blocks.mean(dim=2)  # [num_blocks, num_kv_heads, head_dim]

            per_head = {m: torch.zeros(num_blocks, num_attention_heads) for m in METHODS}

            for h in range(num_attention_heads):
                kv_h = h // num_groups
                qh = q_vec[h]  # [head_dim]

                # ---- oracle: real max-per-key dot product, zero compression ----
                # key_blocks[:, kv_h, :, :] -> [num_blocks, 16, head_dim]
                real_dots = torch.einsum("btd,d->bt", key_blocks[:, kv_h, :, :], qh)
                per_head["oracle_real_maxkey"][:, h] = real_dots.max(dim=-1).values

                # ---- mean-only: dot(query, block_mean) ----
                mean_dot = torch.einsum("bd,d->b", block_mean[:, kv_h, :], qh)
                per_head["mean_only"][:, h] = mean_dot

                # ---- min/max bound (same formula as step_0_3_minmax_recall.py) ----
                prod_max = qh.unsqueeze(0) * block_max[:, kv_h, :]
                prod_min = qh.unsqueeze(0) * block_min[:, kv_h, :]
                minmax_bound = torch.maximum(prod_max, prod_min).sum(dim=-1)
                per_head["minmax_only"][:, h] = minmax_bound

                # ---- min/max + mean, simple average of the two signals ----
                per_head["minmax_plus_mean_avg"][:, h] = 0.5 * (minmax_bound + mean_dot)

            for m in METHODS:
                approx_scores = per_head[m].mean(dim=-1)
                for n in TOP_NS:
                    approx_topn = set(torch.topk(approx_scores, n).indices.tolist())
                    recall = len(true_topn[n] & approx_topn) / n
                    recall_sums[m][n] += recall
                    recall_counts[m][n] += 1

    rows = []
    for m in METHODS:
        row = {"method": m}
        for n in TOP_NS:
            c = recall_counts[m][n]
            row[f"recall@{n}"] = round(recall_sums[m][n] / c, 4) if c else float("nan")
            row[f"n_probes@{n}"] = c
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

    print("\n=== Oracle / mean / min-max+mean recall@N comparison ===")
    rows = run(model, tokenizer)
    print_table(rows)
    write_csv(RESULTS_CSV, rows)
    print(f"wrote {RESULTS_CSV}")


if __name__ == "__main__":
    sys.exit(main())
