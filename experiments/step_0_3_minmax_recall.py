"""Step 0.3 follow-up: Quest-style min/max per-dimension key-pooling index.

Prototyped after the SVD subspace-projection approach (step_0_3_svd_recall.py)
measured recall@N at or below random chance (see
`.claude/docs/step-0.3-results.md`). Per the plan's Risk #2 fallback, this
swaps only the "build summary" and "score" steps for a per-dimension min/max
bounding-box proxy: a valid upper bound on the true max key-query dot product
within a block, unlike the SVD subspace score. Ground truth, prompts, probes,
and recall@N measurement are unchanged and reused verbatim from the SVD
script so the comparison is apples-to-apples.

No sweep over k: min/max pooling has no rank parameter, just one fixed-size
summary (a min vector + a max vector per KV head, size 2 * head_dim).

Run: python3 experiments/step_0_3_minmax_recall.py
Writes: experiments/step_0_3_minmax_results.csv (recall table, single row),
        experiments/step_0_3_minmax_cost.csv (build time + memory, single row)
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
TARGET_LAYER = 14  # mid-depth of 28 layers, same probe layer as the SVD run
TOP_NS = [4, 8]
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # fp16 produced NaNs on long sequences (SVD run finding)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RECALL_CSV = EXPERIMENTS_DIR / "step_0_3_minmax_results.csv"
COST_CSV = EXPERIMENTS_DIR / "step_0_3_minmax_cost.csv"

random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------------------
# Same capture monkeypatch as step_0_3_svd_recall.py.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Same prompts as the SVD run.
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Experiment 1: retrieval quality, min/max pooling
# --------------------------------------------------------------------------
def run_retrieval_quality(model, tokenizer):
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

    recall_sums = {n: 0.0 for n in TOP_NS}
    recall_counts = {n: 0 for n in TOP_NS}

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

            # ---- ground truth: identical to the SVD run ----
            true_scores = torch.zeros(num_blocks)
            for b in range(num_blocks):
                lo, hi = b * BLOCK_TOKENS, (b + 1) * BLOCK_TOKENS
                true_scores[b] = attn_weights[:, q_pos, lo:hi].sum(dim=-1).mean()
            true_topn = {n: set(torch.topk(true_scores, n).indices.tolist()) for n in TOP_NS}

            # ---- key blocks, pre-repeat_kv real per-kv-head layout ----
            # [num_blocks, num_kv_heads, 16, head_dim]
            key_blocks = torch.stack(
                [
                    key[:, b * BLOCK_TOKENS : (b + 1) * BLOCK_TOKENS, :]
                    for b in range(num_blocks)
                ],
                dim=0,
            )
            q_vec = query[:, q_pos, :]  # [num_heads, head_dim]

            # ---- min/max pooling summary: per block, per kv_head ----
            # [num_blocks, num_kv_heads, head_dim]
            block_max = key_blocks.max(dim=2).values
            block_min = key_blocks.min(dim=2).values

            # ---- score: sum_d max(q[d]*max[d], q[d]*min[d]) per head ----
            per_head_scores = torch.zeros(num_blocks, num_attention_heads)
            for h in range(num_attention_heads):
                kv_h = h // num_groups
                qh = q_vec[h]  # [head_dim]
                prod_max = qh.unsqueeze(0) * block_max[:, kv_h, :]  # [num_blocks, head_dim]
                prod_min = qh.unsqueeze(0) * block_min[:, kv_h, :]  # [num_blocks, head_dim]
                per_head_scores[:, h] = torch.maximum(prod_max, prod_min).sum(dim=-1)
            approx_scores = per_head_scores.mean(dim=-1)

            for n in TOP_NS:
                approx_topn = set(torch.topk(approx_scores, n).indices.tolist())
                recall = len(true_topn[n] & approx_topn) / n
                recall_sums[n] += recall
                recall_counts[n] += 1

    row = {}
    for n in TOP_NS:
        c = recall_counts[n]
        row[f"recall@{n}"] = recall_sums[n] / c if c else float("nan")
        row[f"n_probes@{n}"] = c
    return [row]


# --------------------------------------------------------------------------
# Experiment 2: build-time overlap + memory footprint, min/max pooling
# --------------------------------------------------------------------------
def run_cost_measurements(model):
    num_kv_heads = model.config.num_key_value_heads
    num_attention_heads = model.config.num_attention_heads
    head_dim = getattr(
        model.config, "head_dim", model.config.hidden_size // num_attention_heads
    )
    dtype_bytes = 2  # fp16/bf16

    full_block_bytes = BLOCK_TOKENS * num_kv_heads * head_dim * dtype_bytes
    summary_bytes = 2 * num_kv_heads * head_dim * dtype_bytes  # min vec + max vec
    ratio_pct = 100.0 * summary_bytes / full_block_bytes

    n_synthetic_blocks = 200

    if DEVICE == "cuda":
        synthetic_blocks = torch.randn(
            n_synthetic_blocks * num_kv_heads,
            BLOCK_TOKENS,
            head_dim,
            device=DEVICE,
            dtype=torch.float32,
        )

        compute_stream = torch.cuda.current_stream()
        pool_stream = torch.cuda.Stream()

        t0 = torch.cuda.Event(enable_timing=True)
        mm_start = torch.cuda.Event(enable_timing=True)
        mm_end = torch.cuda.Event(enable_timing=True)
        pool_start = torch.cuda.Event(enable_timing=True)
        pool_end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        t0.record()

        pool_stream.wait_event(t0)

        A = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
        B = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
        mm_start.record(compute_stream)
        for _ in range(60):
            C = A @ B
        mm_end.record(compute_stream)

        with torch.cuda.stream(pool_stream):
            pool_start.record(pool_stream)
            bmax = synthetic_blocks.max(dim=1).values
            bmin = synthetic_blocks.min(dim=1).values
            pool_end.record(pool_stream)

        torch.cuda.synchronize()

        def off(e):
            return t0.elapsed_time(e)

        mm_s, mm_e = off(mm_start), off(mm_end)
        pool_s, pool_e = off(pool_start), off(pool_end)
        overlap_confirmed = (pool_s < mm_e) and (mm_s < pool_e)
        pool_total_ms = pool_e - pool_s
        build_time_per_block_us = 1000.0 * pool_total_ms / n_synthetic_blocks
        mm_window = f"[{mm_s:.2f},{mm_e:.2f}]"
        pool_window = f"[{pool_s:.2f},{pool_e:.2f}]"
    else:
        overlap_confirmed = False
        build_time_per_block_us = float("nan")
        mm_window = pool_window = "n/a"

    return [
        {
            "full_block_bytes": full_block_bytes,
            "summary_bytes": summary_bytes,
            "summary_pct_of_block": round(ratio_pct, 2),
            "build_time_per_block_us": round(build_time_per_block_us, 2),
            "overlap_confirmed": overlap_confirmed,
            "mm_window_ms": mm_window,
            "pool_window_ms": pool_window,
        }
    ]


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

    print("\n=== Experiment 1: retrieval quality (recall@N, min/max pooling) ===")
    recall_rows = run_retrieval_quality(model, tokenizer)
    print_table(recall_rows)
    write_csv(RECALL_CSV, recall_rows)
    print(f"wrote {RECALL_CSV}")

    print("\n=== Experiment 2: build time (stream overlap) + memory footprint ===")
    cost_rows = run_cost_measurements(model)
    print_table(cost_rows)
    write_csv(COST_CSV, cost_rows)
    print(f"wrote {COST_CSV}")


if __name__ == "__main__":
    sys.exit(main())
