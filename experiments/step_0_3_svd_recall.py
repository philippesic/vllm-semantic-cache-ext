"""Step 0.3: standalone low-rank (SVD) index prototype. No vLLM involved.

Two independent experiments:

1. Retrieval quality (the decision-gate measurement): loads a real HF model,
   captures genuine post-RoPE Q/K and softmax attention weights for one
   mid-depth layer via a monkeypatch on `eager_attention_forward`, and
   measures recall@N of a rank-k SVD block summary against the true
   top-N most-attended blocks, swept over k.

2. Cost measurements (synthetic Key tensors, shapes matching the real
   model's KV layout): summary build time per block (on a side CUDA stream,
   overlapped with a dummy compute-stream matmul loop, overlap verified via
   CUDA events) and summary memory footprint vs. a full block.

Run: python3 experiments/step_0_3_svd_recall.py
Writes: experiments/step_0_3_results.csv (recall table),
        experiments/step_0_3_cost.csv (build time + memory table)
"""

from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_TOKENS = 16
TARGET_LAYER = 14  # mid-depth of 28 layers
RANKS = [4, 8, 16, 32]
TOP_NS = [4, 8]
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Qwen2.5 is released/trained in bf16; running it in fp16 produces NaNs in
# Q/K for long sequences (confirmed empirically during this experiment).
DTYPE = torch.bfloat16

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RECALL_CSV = EXPERIMENTS_DIR / "step_0_3_results.csv"
COST_CSV = EXPERIMENTS_DIR / "step_0_3_cost.csv"

random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------------------
# Monkeypatch: capture real post-RoPE query/key states + softmax attention
# weights for one target layer. `key` here is pre-repeat_kv, i.e. shaped
# exactly like a real per-KV-head KV-cache block: [batch, num_kv_heads,
# seq, head_dim]. `eager_attention_forward` is looked up by module-global
# name at call time inside `Qwen2Attention.forward` (it is not registered
# under the "eager" key in `ALL_ATTENTION_FUNCTIONS`, so the interface
# lookup falls back to the default arg, i.e. this module-level name), so
# patching the module attribute is sufficient.
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
# Prompt construction: diverse long prompts built from varied seed passages,
# expanded by repetition (with a running counter to avoid literal token
# repetition collapsing everything into one block pattern) and truncated to
# an exact target token count.
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
    "Medieval trade networks across the Silk Road connected East Asia, Central "
    "Asia, the Middle East, and Europe, moving not only silk and spices but also "
    "technologies, religions, and diseases along overlapping caravan and maritime "
    "routes that shifted over centuries in response to political fragmentation.",
]

PROMPT_TARGET_TOKENS = [2000, 2300, 2600, 2900, 3200]


def make_prompt(tokenizer, seed_text: str, target_tokens: int, variant: int) -> torch.Tensor:
    pieces = []
    total = 0
    i = 0
    # Rough token/word ratio overestimate to avoid too many encode() calls.
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
# Experiment 1: retrieval quality
# --------------------------------------------------------------------------
@dataclass
class ProbeResult:
    prompt_idx: int
    q_pos: int
    num_blocks: int


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

    # recall_sums[k][n] = running sum of recall@n across all probes
    recall_sums = {k: {n: 0.0 for n in TOP_NS} for k in RANKS}
    recall_counts = {k: {n: 0 for n in TOP_NS} for k in RANKS}

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
                continue  # not enough complete blocks for a meaningful top-N

            # ---- ground truth: true per-block attention mass, averaged over heads ----
            true_scores = torch.zeros(num_blocks)
            for b in range(num_blocks):
                lo, hi = b * BLOCK_TOKENS, (b + 1) * BLOCK_TOKENS
                # attn_weights[:, q_pos, lo:hi] -> [num_heads, 16]; sum over
                # keys in block, then mean over heads.
                true_scores[b] = attn_weights[:, q_pos, lo:hi].sum(dim=-1).mean()
            true_topn = {n: set(torch.topk(true_scores, n).indices.tolist()) for n in TOP_NS}

            # ---- key blocks for this probe, pre-repeat_kv (real per-kv-head layout) ----
            # [num_blocks, num_kv_heads, 16, head_dim]
            key_blocks = torch.stack(
                [
                    key[:, b * BLOCK_TOKENS : (b + 1) * BLOCK_TOKENS, :]
                    for b in range(num_blocks)
                ],
                dim=0,
            )
            q_vec = query[:, q_pos, :]  # [num_heads, head_dim]

            for k in RANKS:
                actual_rank = min(k, BLOCK_TOKENS)
                # Batch SVD across (num_blocks * num_kv_heads) tiny [16, head_dim] mats.
                flat = key_blocks.reshape(num_blocks * num_kv_heads, BLOCK_TOKENS, head_dim)
                U, S, V = torch.svd_lowrank(flat, q=actual_rank)
                # proxy[i] = S[i] * V[:, i]  -> [num_blocks*num_kv_heads, actual_rank, head_dim]
                proxy = S.unsqueeze(-1) * V.transpose(-1, -2)
                proxy = proxy.reshape(num_blocks, num_kv_heads, actual_rank, head_dim)

                approx_scores = torch.zeros(num_blocks)
                per_head_scores = torch.zeros(num_blocks, num_attention_heads)
                for h in range(num_attention_heads):
                    kv_h = h // num_groups
                    # [num_blocks, actual_rank, head_dim] @ [head_dim] -> [num_blocks, actual_rank]
                    dots = torch.einsum("brd,d->br", proxy[:, kv_h, :, :], q_vec[h])
                    per_head_scores[:, h] = dots.max(dim=-1).values
                approx_scores = per_head_scores.mean(dim=-1)

                for n in TOP_NS:
                    approx_topn = set(torch.topk(approx_scores, n).indices.tolist())
                    recall = len(true_topn[n] & approx_topn) / n
                    recall_sums[k][n] += recall
                    recall_counts[k][n] += 1

    rows = []
    for k in RANKS:
        row = {"k": k, "actual_rank": min(k, BLOCK_TOKENS)}
        for n in TOP_NS:
            c = recall_counts[k][n]
            row[f"recall@{n}"] = recall_sums[k][n] / c if c else float("nan")
            row[f"n_probes@{n}"] = c
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Experiment 2: build-time overlap + memory footprint (synthetic blocks)
# --------------------------------------------------------------------------
def run_cost_measurements(model):
    num_attention_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config, "head_dim", model.config.hidden_size // num_attention_heads
    )
    dtype_bytes = 2  # fp16

    full_block_bytes = BLOCK_TOKENS * num_kv_heads * head_dim * dtype_bytes

    n_synthetic_blocks = 200
    rows = []

    for k in RANKS:
        actual_rank = min(k, BLOCK_TOKENS)
        summary_bytes = num_kv_heads * actual_rank * head_dim * dtype_bytes
        ratio_pct = 100.0 * summary_bytes / full_block_bytes

        if DEVICE == "cuda":
            synthetic_blocks = torch.randn(
                n_synthetic_blocks * num_kv_heads,
                BLOCK_TOKENS,
                head_dim,
                device=DEVICE,
                dtype=torch.float32,
            )

            compute_stream = torch.cuda.current_stream()
            svd_stream = torch.cuda.Stream()

            t0 = torch.cuda.Event(enable_timing=True)
            mm_start = torch.cuda.Event(enable_timing=True)
            mm_end = torch.cuda.Event(enable_timing=True)
            svd_start = torch.cuda.Event(enable_timing=True)
            svd_end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            t0.record()

            svd_stream.wait_event(t0)

            A = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
            B = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
            mm_start.record(compute_stream)
            for _ in range(60):
                C = A @ B
            mm_end.record(compute_stream)

            with torch.cuda.stream(svd_stream):
                svd_start.record(svd_stream)
                U, S, V = torch.svd_lowrank(synthetic_blocks, q=actual_rank)
                svd_end.record(svd_stream)

            torch.cuda.synchronize()

            def off(e):
                return t0.elapsed_time(e)

            mm_s, mm_e = off(mm_start), off(mm_end)
            svd_s, svd_e = off(svd_start), off(svd_end)
            overlap_confirmed = (svd_s < mm_e) and (mm_s < svd_e)
            svd_total_ms = svd_e - svd_s
            build_time_per_block_us = 1000.0 * svd_total_ms / n_synthetic_blocks
        else:
            overlap_confirmed = False
            build_time_per_block_us = float("nan")
            mm_s = mm_e = svd_s = svd_e = float("nan")

        rows.append(
            {
                "k": k,
                "actual_rank": actual_rank,
                "full_block_bytes": full_block_bytes,
                "summary_bytes": summary_bytes,
                "summary_pct_of_block": round(ratio_pct, 2),
                "build_time_per_block_us": round(build_time_per_block_us, 2),
                "overlap_confirmed": overlap_confirmed,
                "mm_window_ms": f"[{mm_s:.2f},{mm_e:.2f}]" if DEVICE == "cuda" else "n/a",
                "svd_window_ms": f"[{svd_s:.2f},{svd_e:.2f}]" if DEVICE == "cuda" else "n/a",
            }
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
    if DEVICE != "cuda":
        print("WARNING: no CUDA device found; cost experiment overlap check will be skipped.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    print("\n=== Experiment 1: retrieval quality (recall@N vs rank k) ===")
    recall_rows = run_retrieval_quality(model, tokenizer)
    print_table(recall_rows)
    write_csv(RECALL_CSV, recall_rows)
    print(f"wrote {RECALL_CSV}")

    print("\n=== Experiment 2: build time (stream overlap) + memory footprint vs rank k ===")
    cost_rows = run_cost_measurements(model)
    print_table(cost_rows)
    write_csv(COST_CSV, cost_rows)
    print(f"wrote {COST_CSV}")


if __name__ == "__main__":
    sys.exit(main())
