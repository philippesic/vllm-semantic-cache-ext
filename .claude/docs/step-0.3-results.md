# Step 0.3 results: standalone low-rank (SVD) index prototype

Deliverable per `.claude/docs/semantic-eviction-plan.md` (vllm-semantic-cache repo)
Step 0.3. Script: `experiments/step_0_3_svd_recall.py` (reruns deterministically,
seed=0, writes `experiments/step_0_3_results.csv` and `experiments/step_0_3_cost.csv`).

## Methodology

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, loaded via `transformers`
  (`attn_implementation="eager"`, bf16 — fp16 produced NaN attention weights on
  long sequences and was abandoned). No vLLM involved.
- Ground truth: genuine post-RoPE Q/K and post-softmax attention weights
  captured via a monkeypatch on `eager_attention_forward` at layer 14 (mid-depth
  of 28). 5 diverse synthetic prompts (2000–3200 tokens, built from distinct
  seed passages: astrophysics, marine biology, cryptography, distributed
  systems, volcanology), 5 probe query positions per prompt (30/50/70/90/100%
  of sequence length) → 25 (prompt, query-position) probes total.
- Block size: 16 tokens (matches the vLLM offload `block_size` used in Step 0.1).
- True top-N: rank blocks by sum of softmax attention weight over keys in the
  block, averaged across the model's 12 attention heads.
- Approximate index: per block, per KV-head (GQA, 2 KV heads / 6 query heads
  per group), `torch.svd_lowrank(key_block, q=k)` on the `[16, head_dim=128]`
  key matrix. Proxy = `S * V` (k directions, each scaled by its singular
  value). Approximate score for a query head = `max` over the k proxy
  directions of `dot(query, proxy)`; averaged across heads (query heads mapped
  to their KV head via GQA grouping) to match the ground-truth aggregation.
  Recall@N = `|true_top_N ∩ approx_top_N| / N`, averaged over all 25 probes.
- Swept k ∈ {4, 8, 16, 32}. Since block_tokens=16, k=32 clamps to
  `actual_rank=16` (a block's key matrix has rank ≤ 16, so k=32 is identical
  to k=16 — this is the block's *exact*, lossless SVD, not an approximation).
- Cost measurements use synthetic Key tensors shaped `[16, num_kv_heads=2,
  head_dim=128]` (matching the real model's KV layout). Build time: SVD run
  on a dedicated CUDA stream, overlapped with a 60-iteration 4096×4096 fp16
  matmul loop on the compute stream simulating ongoing inference; overlap
  verified via `torch.cuda.Event` timestamps (not asserted, measured — see
  `overlap_confirmed` column). Memory: `full_block_bytes` counts K only (not
  K+V) at 2 bytes/element; `summary_bytes` counts the stored proxy directions
  (`num_kv_heads * k * head_dim * 2` bytes) — i.e., whatever the recall
  experiment above actually uses for scoring, so the two experiments are
  consistent.

## Results

### Recall@N vs. rank k (25 probes, real attention ground truth)

| k  | actual_rank | recall@4 | recall@8 |
|----|-------------|----------|----------|
| 4  | 4           | 0.0      | 0.03     |
| 8  | 8           | 0.0      | 0.03     |
| 16 | 16          | 0.0      | 0.005    |
| 32 | 16          | 0.0      | 0.01     |

For reference, a *uniform-random* block guesser's expected recall@8 given the
observed range of `num_blocks` per probe (roughly 60–200 completed blocks)
would be approximately 0.04–0.13 — i.e. every measured k value here is at or
below random-chance recall.

### Build time / stream overlap / memory footprint vs. rank k (synthetic blocks, n=200)

| k  | actual_rank | full_block_bytes | summary_bytes | summary_pct_of_block | build_time_per_block_us | overlap_confirmed |
|----|-------------|-------------------|----------------|------------------------|---------------------------|---------------------|
| 4  | 4           | 8192              | 2048           | 25.0%                  | 1383.83                  | True                |
| 8  | 8           | 8192              | 4096           | 50.0%                  | 891.67                   | True                |
| 16 | 16          | 8192              | 8192           | 100.0%                 | 917.18                   | True                |
| 32 | 16          | 8192              | 8192           | 100.0%                 | 921.31                   | True                |

Overlap was confirmed at every k (the SVD stream's event window overlaps the
concurrent matmul stream's window in all four runs — see raw CSV for the
window timestamps).

## Decision gate (plan §Step 0.3)

The plan's stated gate: *"if recall is poor (< ~0.7 at acceptable rank),
revisit the index design (e.g., mean/max key pooling per Quest, which may
beat SVD and is cheaper) before building any integration."*

Measured recall@4 and recall@8 are ~0.0–0.03 at every rank tested, including
k=16 (the block's exact, lossless SVD — not an approximation at all). This is
far below the 0.7 threshold, and at or below what random guessing would
achieve. **Gate triggered: revisit the index design before Phase 1
integration**, per plan Risk #2.

### Why SVD-subspace scoring fails here (not attributed to a code bug —
### reasoned from the linear algebra, worth recording for whoever picks up
### the redesign)

`torch.svd_lowrank` on a block's `[16, head_dim]` key matrix decomposes each
of the 16 actual key vectors as `key_i = Σ_j U[i,j] · S[j] · V[:,j]` — a
linear combination, with mixed-sign coefficients `U[i,j]`, of *all* the
proxy directions. No individual proxy direction `S[j]·V[:,j]` corresponds to
any actual key vector, even at full rank (k=16, where the decomposition is
exact). Scoring by `max_j dot(query, proxy_j)` therefore measures something
like "how well does the query align with the block's dominant subspace
directions," not "how well does the query align with any actual key in the
block" — and those two quantities can diverge sharply, especially since a
16-dimensional subspace inside a 128-dimensional ambient space already
captures a non-trivial fraction of a random query's energy regardless of
which block it came from, which dilutes the per-block signal further. This
matches the plan's own risk assessment and the standard justification in the
Quest paper for using per-dimension min/max key bounds (a valid upper bound
on true max-key dot product) instead of a subspace projection (which has no
such bound).

## Follow-up: min/max pooling (Quest-style) proxy

Prototyped in `experiments/step_0_3_minmax_recall.py`, same harness (identical
prompts, probes, ground truth, and recall@N measurement — only the summary
build and scoring functions changed). Per block per KV head: store the
per-dimension min and max across the 16 keys (`block_min`, `block_max`, each
`[head_dim]`). Score a query `q` against a block as
`sum_d max(q[d]*block_max[d], q[d]*block_min[d])` — a provable upper bound on
the true max key-query dot product for any key inside the block's bounding
box (unlike the SVD subspace score, which has no such guarantee). No rank
parameter — a single fixed-size summary.

### Results

| method       | summary_pct_of_block | recall@4 | recall@8 | build_time_per_block_us | overlap_confirmed |
|--------------|------------------------|----------|----------|----------------------------|----------------------|
| min/max pool | 12.5%                  | 0.44     | 0.445    | 781.74                    | True                 |
| SVD k=4      | 25.0%                  | 0.0      | 0.03     | 1383.83                   | True                 |
| SVD k=16     | 100.0%                 | 0.0      | 0.005    | 917.18                    | True                 |

Min/max pooling recall (~0.44) is roughly an order of magnitude above SVD's
(~0.0–0.03) and above the random-chance baseline, while using a *smaller*
summary than even the smallest SVD rank tested (12.5% vs. 25% of a full
block) and being cheaper to build (no iterative low-rank solve, just an
elementwise reduction).

**This does not yet clear the plan's ≥0.7 recall bar on its own** — 0.44 is a
real, well-above-chance signal, not the "revisit again" failure SVD was, but
it isn't validated as sufficient either. Plausible next refinements (not yet
tried): probing more heads/layers before generalizing from one probe layer,
tuning block size, or leaning on the plan's own Step 1.4 design (blend
semantic score with a recency term rather than requiring the semantic signal
alone to hit 0.7) — the blend was already anticipated by the plan for
exactly this reason ("pure-semantic will thrash on requests with no query
history — keep the blend").

## Follow-up 2: oracle ceiling + mean-augmented scoring

Per the SVD salvage analysis (`.claude/docs/step-0.3-svd-salvage-analysis.md`),
two more diagnostics were run in the identical harness
(`experiments/step_0_3_oracle_and_mean_recall.py`, same prompts/probes/ground
truth, seed=0 — `minmax_only` reproduces the earlier 0.44/0.445 numbers
exactly, confirming determinism):

| method                       | summary cost   | recall@4 | recall@8 |
|-------------------------------|-----------------|----------|----------|
| **oracle** (real max-key dot product, zero compression) | full block (no savings) | 0.65 | 0.66 |
| mean-only (`dot(query, block_mean)`) | O(1)/block | 0.56 | 0.575 |
| min/max + mean (50/50 blend) | O(1)/block, ~2x min/max | 0.59 | 0.54 |
| min/max alone (prior baseline) | O(1)/block, 12.5% of block | 0.44 | 0.445 |

**The oracle result is the most important number in this whole step.** Even
with *zero* compression — the real, full-precision max-per-key dot product,
computed against all 16 actual keys with no summarization at all — recall@8
is only 0.66. This means the plan's 0.7 gate, as stated ("recall of the
approximate top-N against the true top-N"), is **very likely unreachable by
any compression scheme scoring on this target**, because the target itself
("single max-key dot product as a proxy for total per-block softmax
attention mass") has an intrinsic ceiling around 0.65–0.66 on this
model/workload — no amount of better summarization (SVD done right, global
PCA, PQ, mean/min/max variants) can exceed a ceiling set by the scoring
target itself, only approach it. Practical implication: **the compression
mechanism was never the bottleneck** for the last ~5 points between min/max's
0.44 and 0.7 — the ceiling is ~10 points short of 0.7 even before compression
error enters the picture.

Secondary finding, less central but worth recording: mean-only scoring
(0.56/0.575) beats min/max alone (0.44/0.445) at the same O(1)-per-block
cost, and beats the min/max+mean blend at N=8. A plausible reason: block
attention mass is a sum over 16 keys' softmax weights, and
`dot(query, block_mean) ∝ dot(query, Σ key_i)` is a natural (unnormalized,
pre-softmax) linear proxy for *total* mass, whereas min/max's bounding-box
score and the oracle's max-key score both target the block's *peak*
key-alignment, not its total mass — apparently a moderately better fit for
these particular attention distributions at N=4/8, though the oracle (a peak
statistic) still scores highest of all four, suggesting the real
relationship between "peak alignment" and "total mass" in this model's
attention is workload/layer-dependent rather than cleanly one or the other.
Not deeply investigated further here — flagged as a candidate direction, not
a conclusion.

## Follow-up 3: LRU and ARC baselines (real production policy classes)

Requested comparison: how do vLLM's actual eviction policies do on the same
recall@N measurement? Implemented in
`experiments/step_0_3_lru_arc_recall.py` by driving the **real**
`LRUCachePolicy` / `ARCCachePolicy` classes from
`vllm/v1/kv_offload/cpu/policies/{lru,arc}.py` in this fork (not a
reimplementation): for each probe, insert blocks 0..num_blocks-1 in true
prefill order (each immediately marked ready/evictable), then call
`policy.evict(num_blocks - N, protected=set())`; the N blocks not evicted are
the policy's "keep" prediction, scored against the same true-top-N ground
truth as every other method here.

| method | recall@4 | recall@8 |
|--------|----------|----------|
| lru    | 0.45     | 0.435    |
| arc    | 0.45     | 0.435    |

**LRU and ARC produced identical top-N predictions on every single probe
(50/50, 100%).** This is expected, not a bug: ARC's T1→T2 promotion (its only
mechanism for diverging from pure recency) requires a *second* access to a
block, and this workload — a single linear document read start to finish —
never re-references an earlier block. With zero repeat accesses, ARC is
mathematically forced to behave exactly like insertion-order recency. This
is a real, important property of ARC to know, not a limitation of this
measurement: **ARC only pays for itself under actual reuse patterns**
(shared prefixes hit by separate requests, resumed/preempted requests,
etc.), which this single-prompt-per-probe harness doesn't exercise at all.

### The full comparison, and the important caveat about this workload

| method                | recall@4 | recall@8 | cost              |
|------------------------|----------|----------|-------------------|
| oracle (zero-compression max-key) | 0.65 | 0.66 | full block |
| **lru / arc**          | **0.45** | **0.435** | **free (already built)** |
| mean-only              | 0.56     | 0.575    | O(1)/block        |
| min/max + mean blend   | 0.59     | 0.54     | O(1)/block        |
| min/max alone          | 0.44     | 0.445    | O(1)/block        |
| SVD (any k)            | 0.0      | ≤0.03    | 25–100% of block  |

**Plain LRU essentially ties min/max pooling, and both mean-based methods
only modestly beat it.** This needs an honest caveat, not a quiet pass: this
specific workload (a single, topically-coherent document, probed at various
read positions) is exactly the kind of workload where recency and semantic
relevance are naturally correlated — the most recently-read context in a
coherent narrative tends to also be the most relevant to what comes next, so
LRU doing well here isn't surprising. This matches the plan's own Risk #4
("semantic ≈ LRU on realistic workloads — recency is a strong baseline!")
and is precisely why Step 1.4's accept criterion specifies a *different*,
adversarial workload: "a synthetic workload where LRU provably evicts the
wrong blocks (long-idle but semantically-needed prefix, plus cache-filling
noise traffic)" — i.e., something that deliberately decorrelates recency
from relevance (a needle-like important block, followed by unrelated
distractor traffic, followed by a query that needs the needle back). This
Step 0.3 harness has never tested that scenario. **None of the numbers above
should be read as "semantic loses to LRU" in general** — they show semantic
methods barely differentiating on a workload that doesn't stress-test the
premise. The real test is still ahead, in Step 1.4's adversarial workload.

## Verdict

SVD-subspace-projection, as scored in the original experiment, is dead for
this project — don't resurrect it without a fundamentally different scoring
rule (see the salvage analysis). Among O(1)-per-block methods tested,
**mean-only currently scores best** (0.56/0.575), ahead of min/max alone and
the naive min/max+mean blend — worth adopting as the working baseline over
min/max, pending further tuning (blend weights, per-head vs. per-KV-group
aggregation, additional cheap features). But the bigger finding supersedes
picking a specific formula: **the plan's 0.7 recall gate should be revisited
against the scoring-target ceiling (~0.65–0.66), not treated as a fixed bar
to hit via better compression.** The next high-value experiment is changing
the ground-truth/scoring *target* itself (e.g. sum over top-2/3 real keys,
or a cheap logsumexp-aware combination, instead of single-max) rather than
iterating further on summarization mechanisms — this is a plan-level
decision, not an engineering one, and is flagged here for whoever picks up
Step 1.1 planning.
