# Literature review: min/max and mean-based semantic KV-cache scoring

Date: 2026-07-14. Scope: prior art for Quest-style min/max block scoring and
mean/centroid block scoring, as used in Step 0.3/0.4's recall experiments;
duplicate-work check against vLLM, SGLang, TensorRT-LLM, llama.cpp, LMCache.

## Verdict (up front)

**Worth continuing, but reposition the target.** The min/max mechanism is not
novel — it's Quest (ICML 2024) and, more importantly, it's *already been built
for exactly this project's use case* (CPU-offload, asynchronous, recallable
block eviction) by **ArkVale** (NeurIPS 2024). Mean/centroid-only scoring is
not novel as an idea either — it's been tried by several groups as a
baseline — but the literature's own numbers on it are **inconsistent with
this project's empirical results**, and that inconsistency is the most
important finding of this review (see "The one contradiction that matters,"
below). No shipping inference server (vLLM, SGLang, TensorRT-LLM, llama.cpp)
has query-aware semantic scoring in its **CPU-offload eviction tier** — every
implementation of Quest/ClusterKV/ArkVale-style scoring found lives either in
a standalone research prototype, or (SGLang) in GPU-resident compute-side
sparse attention, not in the block-eviction decision for what to keep in fast
memory. That specific gap — a validated, cheap, training-free importance
score wired into a real offload-eviction policy, benchmarked on the
coherent-vs-adversarial workload split this project already built — appears
genuinely open. The defensible framing is **"apply and validate a hybrid
mean+bounding-box block score inside vLLM's CPU-offload eviction tier,
benchmarked against the coherent/adversarial split,"** not "invent mean
pooling for KV cache," which several papers already did, with mixed and
sometimes contradictory results that need to be reconciled, not ignored.

---

## 1. Quest — confirmed, and the min/max mechanism is exactly this project's

**Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference**,
Tang, Zhao, Zhu, Xiao, Kasikci, Han (MIT), ICML 2024.
[arXiv:2406.10774](https://arxiv.org/abs/2406.10774) ·
[code](https://github.com/mit-han-lab/quest)

Mechanism: per KV-cache page, maintain per-dimension min and max Key vectors.
Score a page against query `q` as `sum_d max(q_d * max_d, q_d * min_d)` — the
exact metric this project independently reproduced in `experiments/step_0_3_minmax_recall.py`.
This is a provable upper bound on `max_k(q . k)` over keys in the page, i.e.
scores the page by its *best possible* key, not an average one.

Reported numbers (their workloads, not ours):
- **Passkey retrieval, 10K context**: Quest 65% (32-token budget) → 99%
  (64+ budget); H2O, TOVA, StreamingLLM all 0–8% across all budgets.
- **Passkey retrieval, 100K context**: Quest 88–100% (256–4096 budget);
  baselines 1–10%.
- **LongBench** (6 datasets): near-lossless accuracy at 1/5–1/10 of full
  cache budget.
- No explicit failure-mode section; they note sparsity is low in early
  layers (<10%) and high in deep layers (>90%), and simply skip applying
  Quest to the first couple of layers.

Comparison to this project: Quest's own passkey-retrieval numbers already
show the same qualitative pattern this project found in Step 0.4 — a
recency/frequency baseline (StreamingLLM/H2O/TOVA) collapsing to near-zero on
a "buried needle" task while a query-aware min/max method stays high. That is
independent corroboration of the *shape* of the Step 0.4 result (LRU/ARC=0%,
minmax=79%) from a different lab, model, and harness. Quest does not test a
mean-only baseline at all — it isn't in their related work as a comparison
point, only as an intuition ("a page's importance depends on its best key,
not its average key") that motivates min/max over any single-vector summary.

---

## 2. Mean/centroid-based block scoring: prior art exists, and it disagrees with our numbers

This is not a novel idea in the abstract — it recurs across several lines of
work under different names — but there is no clean consensus that it works,
and the strongest apples-to-apples experiment found says it works *badly* as
a standalone method. Read this section carefully before committing to
"mean-only" as the headline result.

**ClusterKV** — [arXiv:2412.03213](https://arxiv.org/pdf/2412.03213). Groups
keys into semantic clusters and recalls at cluster granularity rather than
evicting individual tokens permanently; reports "negligible accuracy loss"
at 32K context with a 1–2K budget and 2–2.5x throughput/latency gains vs.
prior recallable methods. Abstract claims superiority over "SoTA recallable
KV compression" but doesn't name Quest or a plain-mean baseline directly in
what's accessible; full paper would need to be pulled for the actual
per-method table.

**CentroidKV** — [arXiv:2506.11418](https://arxiv.org/pdf/2506.11418).
Replaces raw KV states directly with cluster centroids as the compressed
representation used in attention (not just for *scoring* which blocks to
keep — the centroid *is* the surviving K/V). Positioned as a KVPress-style
compression baseline; this is a materially different scoring goal (compute
attention against a summary at all times) than this project's ("summary is
only for deciding what to evict, real Key wins if kept").

**Multipole Attention** — [arXiv:2506.13059](https://arxiv.org/pdf/2506.13059).
K-means clusters keys by semantic similarity, then literally takes the mean
of each cluster as the representative centroid for retrieval scoring — the
closest published analog to this project's "mean pooling" proxy. Framed as
an efficient reasoning-model optimization; worth pulling in full if mean-only
scoring becomes the chosen direction, since it's the most literal precedent.

**CTkvr** — [arXiv:2512.15550](https://arxiv.org/html/2512.15550v1).
Two-stage centroid-then-token retrieval: cheap centroid-grained index first,
token-level refinement second. Structurally similar to a
"mean-gates-then-real-attention" design, i.e. treats the centroid score as a
coarse filter rather than the final answer — an argument for hybrid/two-tier
scoring rather than mean-alone.

**InfLLM** — separates context into blocks and picks a **representative
token** (an actual Key from the block, chosen by some salience rule) rather
than a synthetic mean vector, then scores blocks by dot product with that
real token. This distinction matters: InfLLM's approach doesn't have the
"real keys are mixed-sign combinations, no synthetic direction corresponds to
a real key" failure mode this project already root-caused for SVD, because
it never synthesizes a vector — it picks a real one. Worth registering as a
third family (real-representative-token) distinct from both min/max and
literal-mean.

**CSAttention** — [arXiv:2604.08584](https://arxiv.org/html/2604.08584v1).
Explicitly *not* mean-pooling — clusters queries (not keys) into centroids
per embedding-subspace and precomputes centroid-key dot products offline;
its own related-work framing distinguishes itself from "key-centric" methods
(which would include plain key-mean pooling) precisely because key-centric
centroid lookups suffer recall problems under high sparsity. This is a
second, independent voice saying plain key-centroid scoring has known
recall weaknesses — worth weighing against this project's positive mean
results.

### The one contradiction that matters: ArkVale's own "centroid" ablation

**ArkVale: Efficient Generative LLM Inference with Recallable Key-Value
Eviction**, Chen et al. (Peking University), NeurIPS 2024.
[paper PDF](https://github.com/pku-liang/ArkVale/blob/main/media/arkvale-nips24-paper.pdf) ·
[code](https://github.com/pku-liang/ArkVale)

This paper is the single most important piece of prior art for this project
for two reasons: (1) its system design is almost exactly the shape of this
project's target — pages fill on GPU, get asynchronously backed up to CPU
memory with a small digest, and the digest is later used to rank pages for
recall/eviction — and (2) **it directly ablates a pure key-centroid ("mean")
digest against min/max bounding-box digests, on a page-ranking recall metric
very close to what Step 0.3 measured, and centroid loses badly:**

- Their "centroid" baseline: dot product of query against the plain
  element-wise average of a page's keys (i.e. exactly this project's
  mean-only scorer). Result (Figure 7, recall-accuracy of predicted top-k
  page set vs. true top-k by real attention): **under 5% accuracy at top-1,
  and does not reach 50% recall even by top-4.**
- Their six min/max ("bounding-volume") variants — both "bounding-sphere"
  (center + scalar radius) and "bounding-cuboid" (min-vector + max-vector,
  i.e. Quest's exact digest) — all reach **at least 60% recall at top-1**,
  with the best variant (cuboid + mean-derived radius, "cuboid-mean") at
  **95% top-1 recall and >80% at all other k**.
- Cuboid beats sphere consistently ("more retained information: two vectors
  vs. one vector + scalar"). Within each family, using the *mean* of
  per-key distances to set the box/sphere size (`r_mean`) beats using the
  *max* distance (`r_max`) — "potentially because [mean-derived radius]
  provide[s] a more balanced boundary estimation, avoiding overestimation of
  page importance." So mean *does* help ArkVale — but only as a tuning knob
  for how big to draw a bounding box, never as a replacement for keeping
  min/max structure at all.
- On their passkey-retrieval task (their equivalent of this project's
  needle-in-haystack workload), ArkVale (min/max-based, recallable) holds
  **95–100% accuracy at every context length (10k/20k/30k) and budget
  (512–4096)**, while StreamingLLM/H2O/TOVA (recency/frequency-based,
  non-recallable) fall to **0–20%** — the same qualitative shape as this
  project's Step 0.4 (LRU/ARC = 0/16), but again with *no* pure-mean
  comparison point on this specific task.

**Why this matters and how to reconcile it:** ArkVale's centroid ablation
(<5% top-1 recall) is close to the opposite of this project's finding
(mean-only recall@8 = 0.575 coherent, 1.0 adversarial, beating min/max in
both cases). Both can't be fully right in general, and this needs to be
resolved before committing to mean-only as the headline result rather than
brushed past:
  - Different **scoring targets**. ArkVale scores "is this page in the true
    top-k by max-attention-softmax-score," a page-ranking/recall-of-pages
    metric across many pages competing at once. This project's Step 0.3/0.4
    harness (per the plan memory) mostly measured recall@N against a
    single-max-key or oracle-style target on a much smaller candidate set.
    A summary that's bad at fine-grained top-1-of-many page ranking can
    still be fine at coarser recall@8-of-few.
  - Different **workloads**. ArkVale's centroid ablation runs on LongBench
    documents (HotpotQA, GovReport, etc.) — closer to this project's
    *coherent-document* workload (where this project's own margin for mean
    over min/max was much smaller: 0.575 vs 0.445) than to the adversarial
    needle setting (where mean hit 1.0). ArkVale did not run their centroid
    ablation on their passkey task at all — the one workload where this
    project's mean result is most dramatic and least contradicted.
  - Different **models/context lengths**: LongChat-7B at 10–32K vs.
    Qwen2.5-1.5B at block_size=16 with presumably much shorter synthetic
    probes. Head dimension, head count, and RoPE behavior all affect how
    much a mean vector's direction actually resembles any single key's
    direction.
  - **Action item, not yet done**: before writing up mean-only as the
    finding, replicate ArkVale's specific test — top-k page-recall against
    true top-k pages by max-attention, at k=1..40, many pages — in this
    project's own harness, to see whether the contradiction is about
    scoring target/workload (as hypothesized above) or is a real
    disagreement that needs a different explanation. Right now this project
    has *not* run the experiment that would settle this.

No evidence ArkVale has been merged into vLLM, SGLang, or TensorRT-LLM — it
remains a standalone academic prototype (built "inspired by vLLM['s paging
design]" per their own related work, i.e. borrows the PagedAttention *block*
abstraction as prior art, not an integration into vLLM itself). This is the
single closest piece of prior art to "has this already shipped," and the
answer for ArkVale specifically is: published, working, open-source, **not
integrated into any production server** as of this review.

---

## 3. Hybrid mean + min/max approaches

- **ArkVale's own `r_mean` cuboid/sphere radius variant** (above) is the
  closest documented hybrid: it uses mean-of-distances as a *sizing*
  statistic for a min/max-shaped digest, not mean-of-keys as a
  standalone score. This is a different hybrid than this project's Step 0.3
  "50/50 blend of min/max score and mean score" — worth trying ArkVale's
  version (mean informing the box, not blended with it) as an additional
  variant.
- **CTkvr**'s two-stage centroid-then-token design (cheap centroid filter,
  then real-token refinement) is a hybrid in the sense of "coarse mean
  score gates a finer real-key score" — structurally similar to what a
  "mean gates eviction, min/max or oracle breaks ties" policy would look
  like inside vLLM's offload manager.
- No paper found that runs a literal min/max-and-mean blended dot-product
  score (this project's own Step 0.3 "50/50 blend," recall@8=0.54, actually
  *worse* than mean-alone at 0.575) and reports it as a headline method —
  this specific blend appears to be this project's own contribution, for
  whatever that's worth; it also isn't obviously the right hybrid, per the
  0.54 vs 0.575 result already in hand.

---

## 4. Documented workload-dependence / failure modes

- **Quest passkey numbers** (Section 1): baselines 0–10%, Quest 65–100%,
  splitting cleanly by context length and budget — the same coherent-vs-adversarial-style
  split this project found, though Quest doesn't frame it that way
  explicitly (they don't test a "coherent document" condition where
  recency ties relevance).
- **ArkVale passkey numbers** (Section 2): the cleanest external replication
  of Step 0.4's exact shape — recency/frequency baselines craterto 0–20%,
  bounding-box method holds 95–100%.
- **"Taming the Fragility of KV Cache Eviction in LLM Inference,"**
  [arXiv:2510.13334](https://arxiv.org/html/2510.13334v1). Important false
  friend: this paper's "mean aggregation fails" finding is about a
  *different* mean — averaging **importance scores across decoding steps
  over time** for a single token (the "stability assumption" that a token's
  importance today predicts its importance later), not averaging **Key
  vectors within a block** at one point in time. Their finding (mean
  temporal aggregation gives 0.92 average-case but 0.34 worst-case retained
  importance; "defensive"/max-based aggregation recovers to 0.61) is about
  robustness to importance *drift over the generation*, which is closer to
  ArkVale's original motivating observation (page 256 in their Figure 4a
  regains importance ~4500 tokens after being deprioritized) than to this
  project's spatial min/max-vs-mean-of-keys question. Do not cite this paper
  as being about the same "mean" — it would be a real error if conflated in
  a writeup or PR description.
- **"When Does Value-Aware KV Eviction Help?"**
  [arXiv:2605.08234](https://arxiv.org/html/2605.08234v1). Identifies
  "exposure bias" (early tokens over-scored because more prefill queries
  see them) and "phase dilution" (prefill-time scoring doesn't match
  decode-time query distribution) as generic failure modes of *any*
  attention-score-based eviction method, value-aware or not. Relevant
  background for why this project's ground-truth harness should keep using
  real decode-time queries (which it already does, per the plan memory)
  rather than prefill attention scores.
- **SnapKV/AdaKV** reportedly show "failure bands at intermediate needle
  depths (11–89%)" on long-context needle tests per one survey found — a
  reminder that even content-aware methods aren't uniformly robust across
  needle *position*, a variable this project's Step 0.4 harness should check
  it's varying (needle depth), not just needle-vs-no-needle.

---

## 5. Duplicate-work check — has this shipped anywhere already?

Per AGENTS.md policy, checked vLLM's own issue/RFC tracker plus SGLang,
TensorRT-LLM, llama.cpp, and LMCache (the pluggable backend vLLM's
CPU-offload connector is designed to delegate to).

**vLLM (`vllm-project/vllm`), directly:**
- CPU-offload tier today ships only **LRU** and **ARC**
  (`vllm/v1/kv_offload/cpu/policies/{lru,arc}.py}`) — matches what this
  project's own Step 0.3 harness already drove directly. No query-aware or
  semantic scoring in-tree.
- [Issue #40268](https://github.com/vllm-project/vllm/issues/40268)
  ("KV Cache Eviction support ARC," open) — proposes a two-queue policy for
  the **GPU** prefix-cache tier as a step toward ARC; purely
  recency/frequency-based, no query-aware or semantic scoring discussed at
  all.
- [Issue #36311](https://github.com/vllm-project/vllm/issues/36311)
  ("Pluggable KV cache eviction policy with attention sink protection,"
  closed) — proposes a `BlockEvictionPolicy` interface, but the only
  scoring rule discussed is protecting fixed attention-sink *positions*
  (first 1–2 tokens), not any per-block query-aware or content-based score.
  This is the closest-named vLLM issue to "pluggable eviction policy" and it
  is still not the same idea.
- [RFC #38260](https://github.com/vllm-project/vllm/issues/38260)
  ("Multi-tier KV offloading via the vLLM offloading connector," open) —
  purely about tier *topology* (CPU → disk → object storage → remote);
  explicitly states "secondary tiers own their evictions" without
  specifying any scoring method. No semantic/query-aware content.
- DeepSeek-V3.2's sparse attention (DSA), shipped in vLLM
  ([blog](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)) — the closest
  thing vLLM has shipped to "query-aware sparsity," but it uses a **trained
  lightning-indexer module** (not a cheap training-free statistic like
  min/max or mean), and it operates as **GPU compute-side attention
  sparsity**, not as a CPU-offload eviction/tiering decision. Different
  layer of the stack; does not compete with or duplicate this project's
  target.
- No open or closed vLLM issue/PR found (web search; `gh` CLI unavailable in
  this environment, so this is web-search-only coverage, not exhaustive)
  mentioning "Quest," "min/max pooling," or "centroid"/"mean pooling" in
  the context of the CPU-offload eviction policy specifically.

**SGLang:** ships a pluggable hierarchical sparse-attention framework that
explicitly supports Quest-style min/max and ClusterKV-style centroid scoring
as swappable algorithms
([blog](https://www.alibabacloud.com/blog/sglang-hierarchical-sparse-attention_603162)).
This is the most "already exists" result found — but it's framed and
implemented as GPU-resident **compute-side sparse attention** (deciding
which cached blocks to *read* for a given attention call), not as the
CPU-offload **eviction** decision (deciding which blocks physically leave
fast memory). It's architecturally adjacent, worth reading the SGLang code
before building vLLM's version, but not the same mechanism this project
targets.

**TensorRT-LLM:** search turned up DSA-style trained-indexer sparse
attention support (routed through on Blackwell hardware per one SGLang
doc) — same compute-side/trained-module pattern as vLLM's DSA support, not
a cheap statistic in a CPU-offload eviction tier.

**llama.cpp:** no evidence found (via web search) of Quest-style,
centroid-style, or any query-aware block scoring in its KV-cache management;
its cache eviction remains context-shifting/recency-based as far as this
search could determine. Worth a direct code search if this becomes a
concern, but nothing surfaced to suggest duplicate work here.

**LMCache** (the backend vLLM's `OffloadingConnector` is designed to
delegate detailed tiering/eviction policy to): search surfaced adjacent
"semantic-aware eviction" research —
["Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM
Prefix Caches"](https://arxiv.org/pdf/2605.18825) — but this is a **learned
scoring model** (not a cheap training-free statistic) targeting the **GPU
prefix cache**, evaluated against LRU with its own LongBench-style and
needle-in-haystack-style splits. No evidence it's shipped inside LMCache or
vLLM itself; appears to be a separate research contribution, not
implemented in the tools this project would build on. Confirms the general
shape of this project's finding (recency fails on adversarial-retrieval-style
workloads) has been independently observed by yet another group, using yet
another mechanism (a learned model rather than mean or min/max) — three
independent groups now (Quest, ArkVale, this "semantic-aware eviction"
paper) converging on "recency-only eviction fails specifically on
needle/retrieval-shaped workloads," which is a fairly strong signal that
this project's Step 0.4 finding is a real, previously-documented phenomenon
rather than a harness artifact — good news for the workload-dependence
claim, less good news for its novelty as an observation (though the
specific mean-vs-minmax-vs-hybrid *comparison*, run on vLLM's *actual*
shipping OffloadingConnector/LRU/ARC code, appears to still be new).

---

## Sources

- Tang et al., [Quest: Query-Aware Sparsity for Efficient Long-Context LLM
  Inference](https://arxiv.org/abs/2406.10774), ICML 2024.
  [Code](https://github.com/mit-han-lab/quest)
- Chen et al., [ArkVale: Efficient Generative LLM Inference with Recallable
  Key-Value Eviction](https://github.com/pku-liang/ArkVale/blob/main/media/arkvale-nips24-paper.pdf),
  NeurIPS 2024. [Code](https://github.com/pku-liang/ArkVale)
- [ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable
  Compression](https://arxiv.org/pdf/2412.03213)
- [CentroidKV: Efficient Long-Context LLM Inference via KV Cache
  Clustering](https://arxiv.org/pdf/2506.11418)
- [Multipole Attention for Efficient Long Context
  Reasoning](https://arxiv.org/pdf/2506.13059)
- [CTkvr: Efficient KV Cache Retrieval for Long-Context LLMs via
  Centroid-then-Token Indexing](https://arxiv.org/html/2512.15550v1)
- [CSAttention: Centroid-Scoring Attention for Accelerating LLM
  Inference](https://arxiv.org/html/2604.08584v1)
- [HashAttention: Semantic Sparsity for Faster
  Inference](https://arxiv.org/pdf/2412.14468)
- [Taming the Fragility of KV Cache Eviction in LLM
  Inference](https://arxiv.org/html/2510.13334v1)
- [When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic for
  Non-Monotone Cache Compression](https://arxiv.org/html/2605.08234v1)
- [Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for
  LLM Prefix Caches](https://arxiv.org/pdf/2605.18825)
- SGLang [hierarchical sparse attention
  writeup](https://www.alibabacloud.com/blog/sglang-hierarchical-sparse-attention_603162)
- vLLM blog, [DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in
  Action](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)
- vLLM issues: [#40268](https://github.com/vllm-project/vllm/issues/40268),
  [#36311](https://github.com/vllm-project/vllm/issues/36311),
  [RFC #38260](https://github.com/vllm-project/vllm/issues/38260)

## Caveats on this review's own process

- `gh` CLI was not available in this environment; the vLLM issue/PR search
  above is web-search-based (Google-style queries plus direct issue fetches
  of the most relevant hits found), not an exhaustive `gh issue
  list`/`gh pr list` sweep. Before actually proposing a PR, re-run the
  AGENTS.md-mandated `gh issue view`/`gh pr list --search` commands directly
  once `gh` is available, per the project's own duplicate-work policy.
- ArkVale's PDF was read directly (pages 1–10 of 10) via the GitHub-hosted
  copy; all ArkVale numbers above are taken directly from its Figure 7
  (page-recall accuracy by digest type) and Table 1 (passkey retrieval
  accuracy), not from a secondary summary.
- ClusterKV, CentroidKV, Multipole Attention, CTkvr, and HashAttention were
  characterized from abstracts/search-tool summaries, not full-text reads;
  their numbers above are qualitative/directional, not verified
  page-by-page the way ArkVale's are. If any of these becomes load-bearing
  for a design decision, pull the full PDF before relying on it.
