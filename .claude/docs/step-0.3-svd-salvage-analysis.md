# Step 0.3 SVD salvage analysis: is low-rank/PCA compression viable for block-level eviction scoring?

Written in response to a request to evaluate whether the failed SVD-subspace
approach (`experiments/step_0_3_svd_recall.py`,
`.claude/docs/step-0.3-results.md`) can be salvaged in any form — not
necessarily the exact scoring rule tried, but the broader idea of low-rank
compression as the block-summary mechanism. This is reasoning-only: no code
was run, no GPU was touched, `step_0_3_svd_recall.py` was not modified. It
also draws on the already-completed min/max follow-up
(`experiments/step_0_3_minmax_recall.py`, `step_0_3_minmax_results.csv`,
`step_0_3_minmax_cost.csv`), which was not yet folded into the write-up doc
at the time of this analysis.

## Verdict

**Not worth salvaging for the job Step 0.3 actually needs (a cheap,
block-level, O(1)-relative-to-block-size eviction score). Salvageable only in
a narrow, technically-correct sense that, once fixed, stops being a
compression win at all.**

The diagnosed bug (scoring against synthetic "representative directions"
instead of real keys) is real and has a clean fix: reconstruct per-key
approximations from the SVD factors instead of treating singular directions
as pseudo-keys. That fix is mathematically sound and would very likely raise
recall substantially — at full rank it is provably exact. But making the fix
requires storing the per-key coefficients (`U`) that the original approach
deliberately discarded, and `U`'s storage cost scales with `block_tokens`
(16), which is not much smaller than `head_dim` (128) here. The result: by
the time the summary is faithful enough to be useful, it costs about as much
memory as just keeping the real keys, and at low enough rank to actually
save memory, fidelity (and thus recall) collapses again. This isn't a
tuning problem to iterate away — it's a structural property of "preserve
per-key identity" schemes at this block size. Min/max pooling doesn't have
this tax (its summary is O(1) in `block_tokens`, not O(`block_tokens`)), which
is the real reason it already measures far better (recall@4/8 ≈ 0.44) at a
much smaller memory footprint (12.5% of the block vs. even a modest-rank
reconstruction scheme). Global/shared low-rank projections (the
literature-validated version of "PCA for KV-cache") fix the *local-basis,
no-calibration* half of the problem but not the storage tax, because they
still assign a per-key code — so they belong to a different sub-problem
(compressed resident KV representations) than the one Step 0.3 is gating on
(a fast per-block reject/rank signal).

## The diagnosed bug, and why it's fixable in isolation

The write-up's own diagnosis is precise and, on inspection of
`step_0_3_svd_recall.py` lines 208–219, matches the code exactly: `proxy =
S.unsqueeze(-1) * V.transpose(-1,-2)` builds `k` vectors per block, and the
score is `max` over those `k` vectors of `dot(query, proxy_j)`. Each
`proxy_j` is `S[j] * V[:,j]` — a basis vector of the block's row space,
scaled by its singular value — not any actual key. Since
`key_i = Σ_j U[i,j] · S[j] · V[:,j]`, an actual key is a mixed-sign
combination of *all* `k` proxies simultaneously, so "does the query align
with proxy `j`" and "does the query align with real key `i`" are different
questions whose answers can diverge arbitrarily, and at k=16 the measured
recall (0.0–0.03) confirms they did diverge badly, at or below chance.

Crucially, this failure occurs **even at k=16, the exact/lossless
decomposition** (line 30 of the results doc; confirmed in code — `q=32`
clamps to `actual_rank=16` since `torch.svd_lowrank` can't exceed
`min(16, 128)=16`). That fact is the single most important piece of evidence
in this whole analysis: at full rank, `U`, `S`, `V` collectively contain
*all* the information in the original key block, zero bits discarded. If
recall is at chance even then, the failure cannot be attributed to lossy
compression — it is purely a consumption/scoring bug. This means the
"is it the scoring rule or the granularity?" question has a clean partial
answer already, straight from the existing data: **the scoring rule is
provably at fault, independent of granularity**, because the bug reproduces
with zero information loss.

### The direct fix: score against per-key reconstructions, not proxy directions

Reconstructing `key_i ≈ U[i,:] @ diag(S) @ V^T` and taking
`max_i dot(query, key_i_reconstructed)` restores per-key correspondence. At
k=16 this is not an approximation — `key_i_reconstructed == key_i` exactly,
by definition of SVD — so this scoring rule at k=16 is mathematically
identical to computing `max_i dot(query, key_i)` against the real keys. Its
recall at k=16 must equal whatever recall "the true per-key max-dot-product,
computed with zero compression" achieves against the block-attention-mass
ground truth. That's very likely much higher than 0.0–0.03 (it's a real,
meaningful quantity — the standard Quest-style proxy for attention mass —
not a synthetic artifact), though notably this project has not yet actually
measured that "oracle" number directly (see Recommendation section below;
this is the cheapest, highest-value thing to check next, cheaper than
prototyping any new compression scheme).

The reconstruction doesn't need to be materialized to be cheap: since
`dot(query, key_i_reconstructed) = dot(query, U[i,:]·S·V^T) = (S ⊙ U[i,:])
· (V^T @ query)`, you can compute `q_proj = V^T @ query` once per query
(`O(k·head_dim)`), then get all 16 per-key scores via `U @ (S ⊙ q_proj)`
(`O(block_tokens·k)`). Total: `O(k·(head_dim + block_tokens))`, versus
`O(block_tokens·head_dim)` for the naive real-key dot product — genuinely
cheaper compute for k < head_dim. So the compute side of this fix is fine.

### Why the fix doesn't rescue the memory case

The problem is storage, not compute. The original (failed) approach only
needed to store `S*V` — `k` vectors of length `head_dim`, i.e.
`k·head_dim` scalars, **independent of `block_tokens`**. That's exactly
what made it look cheap: `summary_bytes` in the results table (2048–8192
bytes at k=4..16) scales only with `k` and `head_dim`, never with the
16-token block size. The fix requires *additionally* storing `U`, which is
`block_tokens·k` scalars — and `block_tokens=16` is only 8x smaller than
`head_dim=128`, so this is not a negligible addition. Concretely, for this
model's shape (`num_kv_heads=2, head_dim=128, block_tokens=16`), the
memory cost of a reconstruction-capable summary is:

```
summary_bytes(k) = num_kv_heads · (k·head_dim + block_tokens·k) · dtype_bytes
                 = num_kv_heads · k · (head_dim + block_tokens) · dtype_bytes
```

At k=4: `2·4·144·2 = 2304` bytes (28.1% of the 8192-byte full block — worse
ratio than the *un-fixed* k=4 row in the original table, 25.0%, precisely
because `U` is now included). At k=16 (full rank, needed for the fix to be
exact and therefore trustworthy): `2·16·144·2 = 9216` bytes — **larger than
the 8192-byte full block itself.** The fixed version, at the rank where you'd
actually want to trust it, costs more memory than not compressing at all.
Between those extremes there may be a k where reconstruction fidelity is
"good enough" and memory is still under 100% of the block, but given
`block_tokens` and `head_dim` are within one order of magnitude of each
other, that window is narrow, and there's no reason to expect it beats
min/max's already-measured 12.5%-of-block footprint. This is not a
parameter-tuning problem — it is the general fact that **any summarization
scheme that preserves genuine per-key correspondence must store at least
`O(block_tokens)` worth of per-key information**, whereas min/max pooling's
summary size is `O(1)` with respect to `block_tokens` (one min vector + one
max vector *per block*, not per key — see `step_0_3_minmax_recall.py` lines
174–176, 214). That asymmetry gets *more* favorable to min/max as blocks get
bigger and is fixed by vLLM's `block_size=16` here — it isn't a knob this
project can tune away.

## Is per-block-local SVD the wrong granularity, full stop?

Partially, and for a more specific reason than "SVD is bad." Two granularity
questions got conflated in the original approach, worth separating:

1. **Local basis (per-block SVD) vs. shared/global basis (fixed
   projection fit once, e.g. via PCA over a calibration sample of keys).**
2. **Per-key-correspondence-preserving summary vs. per-block-aggregate
   summary that discards individual key identity.**

The failed approach was local-basis *and* aggregate (no per-key
correspondence) — arguably the worst combination for this application,
since it inherits the storage-doesn't-scale-down-with-info benefit of an
aggregate scheme in name only (it's not actually a valid aggregate bound,
just a subspace-alignment score with no correctness guarantee), and it
inherits none of the calibration/reuse benefits of a shared basis (must be
recomputed fresh for every single block via a real SVD call, ~900µs/block
per the cost table — a substantive per-block cost that a shared, offline-fit
projection would eliminate entirely, replacing it with one cheap
matrix-vector product per key).

A **global/shared low-rank projection** — the version of "PCA for KV-cache"
actually used in the literature (Palu, ThinK, Eigen-Attention and similar
low-rank KV-cache compression work all fit a shared per-layer/per-head
projection matrix once, offline or periodically, from a calibration
distribution of keys, then apply it uniformly) fixes real problems the
block-local version has:
- **No per-block SVD compute** — a fixed projection `P` (`head_dim × r`)
  turns "build a summary" into one matmul (`P^T @ key_i`), not an iterative
  SVD solve. This directly serves the plan's "cheap enough to not stall
  live inference" constraint far better than repeated `torch.svd_lowrank`
  calls.
- **Cross-block comparability.** Projected keys and a projected query all
  live in the same fixed `r`-dimensional space, so "how well does this
  query match this specific key" is a well-defined, consistent quantity
  across every block — unlike per-block bases, which have no shared
  coordinate system and can only be compared to a query via the
  proxy-direction hack that caused the original failure.
- **True per-key correspondence**, by construction: the compressed
  representative of a key is a shrunk version of *that exact key*, not a
  synthetic combination — structurally the same shape of object that makes
  PQ, LSH, and PCA-based ANN retrieval work (see below).

But it does **not** fix the storage-tax problem above, because it still
assigns one code per real key — `r` scalars for each of the 16 keys in a
block, i.e. `O(block_tokens · r)` total, the same asymptotic shape as the
reconstruction fix, just without needing to also store `U`/`S` per block
(those get amortized into the shared `P`, fit once across the whole cache).
So it is a genuine improvement on the *local-basis* half of the problem, and
on the *compute* side, but it answers a different question than "give me one
cheap O(1)-per-block score to decide what to evict" — it answers "give me a
compressed but individually-addressable version of each key," which is a
reasonable goal for a different part of the design space (e.g., keeping a
lossy-but-resident compressed KV format instead of evicting to CPU at all —
closer to what Palu/ThinK are actually for) but isn't the tool Step 0.3 is
gating on. It also brings real added complexity this project hasn't budgeted
for: a calibration step (representative key sample, likely per-layer/head),
a question of calibration staleness as workloads drift, and a versioning
story for `P` if the model or workload distribution changes — none of which
min/max pooling needs at all.

## The fast-reject hybrid (subspace-distance elimination + something else for ranking)

This has a real mathematical basis, at full rank: if a key lies exactly in
`span(V)` (true when k=16, the block's exact row space), then
`dot(query, key_i) = dot(query_parallel, key_i)` where `query_parallel` is
the projection of the query onto `span(V)`, since the orthogonal component
of the query contributes nothing to the dot product with any vector inside
the subspace. By Cauchy-Schwarz, `dot(query, key_i) ≤ ||query_parallel|| ·
||key_i||` — a valid, cheap, per-block upper bound usable to reject blocks
whose query-subspace overlap is small, in the same spirit as Quest's
per-dimension min/max bound (a true bound, not a heuristic).

The catch is exactly the dilution effect the original write-up already
flagged: a 16-dimensional subspace inside a 128-dimensional ambient space
retains roughly `16/128 = 12.5%` of a generic query's squared norm *in
expectation*, essentially regardless of which block's subspace you're
projecting onto, unless different blocks' key subspaces are unusually
mutually orthogonal (untested here, no strong reason to expect it holds for
generic natural-language key distributions). If every block admits roughly
the same fraction of query energy, this filter has weak discriminative
power between blocks — it would reject almost nothing at a threshold loose
enough to avoid false negatives, and reject almost uniformly at a threshold
tight enough to matter. It's not obviously useless, but it's exactly the
kind of thing that needs a direct recall/reject-rate measurement before any
engineering investment, and there's a cheaper diagnostic to run first (see
Recommendation).

## Why PQ / LSH / ScaNN-style techniques succeed where this attempt failed

The literature comparison the prompt asked about turns out to have a clean,
consistent answer. Product quantization, LSH, and ScaNN's anisotropic
quantization all share a structural property that the failed SVD approach
violated: **they assign one compressed code to each individual real vector**,
using a codebook/projection that is *shared and fixed* across the whole
collection, fit once (offline k-means for PQ, random fixed hyperplanes for
LSH, a learned anisotropic loss for ScaNN). None of them ever try to
jointly summarize a small ephemeral *group* of vectors into a handful of
"representative directions" the way block-local SVD did. Per-vector
identity is never discarded — only per-vector *precision* is. That's the
entire reason `max(query · codebook_reconstruction_i)` in PQ-style search
means the same thing as `max(query · key_i)` approximately, for every `i`
individually, which is precisely the property the original SVD scoring
rule lacked and the reconstruction fix (above) restores.

This also means PQ/LSH/ScaNN inherit the same storage tax discussed above:
a PQ code is still one code *per vector*, so applying PQ to KV-cache blocks
would cost `O(block_tokens)` codes per block, same asymptotic shape as the
reconstruction fix or a global low-rank projection. They're a legitimate
answer to "how do I compress N individual vectors for approximate retrieval
while preserving per-vector answers," which is a real and useful capability,
but it is not the same capability as "how do I get one cheap aggregate
signal per block of N vectors," which is what Step 0.3's decision gate is
actually testing. Adapting PQ here would mean: fit per-layer/head codebooks
offline (calibration), assign each new key a code as it's written (cheap
table lookup after one small matmul for the sub-vector assignment), and
score via asymmetric distance computation (query stays unquantized, codes
looked up via precomputed inner-product tables) — all plausible, but it is
materially more machinery than min/max pooling for a capability (individual
key retrieval) the block-eviction use case doesn't need; it would matter
more for a "keep a compressed resident copy instead of a real KV block" design.

## Recommendation

1. **Do not invest further engineering time in SVD/low-rank/PCA compression
   for the Step 0.3 block-scoring gate.** The specific bug is understood
   and fixable, but the fix eliminates the memory advantage that motivated
   using low-rank compression in the first place, for this block size. This
   is a structural conclusion (`block_tokens` vs. `head_dim` ratio), not a
   tuning failure, so more k-sweeps or scoring-rule variants on the same
   local-basis idea are not worth running.

2. **Before prototyping anything else, run one cheap diagnostic that isn't
   a new compression scheme at all**: compute recall@N using the *real*
   max per-key dot product (`max_i dot(query, key_i)` on the actual,
   uncompressed 16 keys per block) against the same attention-mass ground
   truth already captured in this harness. This requires no new
   summarization code — just skip the compression step in the existing
   recall harness and score directly off `key_blocks`. It answers a
   question this analysis surfaced but the existing experiments haven't
   measured: is "max single-key dot product" itself a good proxy for
   "total block softmax attention mass," independent of any compression
   error? If this oracle number is also well below the 0.7 gate, that
   proves the ceiling is in the *scoring target* (max-of-one-key as a stand-in
   for a sum over 16 softmax-weighted keys), not in the summarization
   mechanism — and it means no amount of better compression (SVD done
   right, global PCA, PQ, LSH, ScaNN) can cross the 0.7 gate on its own,
   because they'd all be faithfully approximating a proxy that has an
   intrinsic ceiling. In that world the right next move is redesigning the
   *target* (e.g., score by top-2/3 keys, or a cheap logsumexp-style
   softmax-aware combination across all 16 real keys, rather than a single
   max) rather than further work on how that single max gets computed
   cheaply.

3. **Lower-effort next tweak on the already-working min/max approach,
   before reaching for anything more exotic**: try augmenting the min/max
   pair with a mean vector (`block_mean`), still `O(head_dim)` per block
   (one more vector, not one per key), and see whether adding a
   central-tendency term narrows the gap to 0.7. This preserves every cost
   advantage of the current min/max prototype and is a much smaller
   engineering step than reintroducing any rank-k machinery.

4. **Low-rank/PCA-style KV compression is not dead as an idea for this
   project — it's aimed at the wrong sub-problem right now.** If Phase 2
   ever wants a "keep a lossy-but-resident compressed copy of cold blocks
   instead of evicting them to CPU entirely" mode (adjacent to the
   Step 2.5 approximate-mode idea already in the plan), a shared/global
   low-rank projection (Palu/ThinK-style) or PQ-style per-key codes would
   be the technically appropriate tool then, because at that point you
   actually want individually-addressable compressed keys, not a single
   block-level reject/rank score. Worth a forward pointer in the plan, not
   a redo of Step 0.3.

## If someone wants to prototype the reconstruction fix anyway (for
## completeness, or to confirm the diagnosis empirically)

For the record, here's the minimal sketch, reusing
`step_0_3_svd_recall.py`'s existing harness (ground truth, prompts, probes,
recall@N are all reusable verbatim, exactly as the min/max follow-up did):

```python
U, S, V = torch.svd_lowrank(flat, q=actual_rank)   # unchanged
# OLD (buggy): proxy = S.unsqueeze(-1) * V.transpose(-1, -2)  # k pseudo-keys
# NEW: project query once, combine with U row-wise per real key
q_proj = torch.einsum("d,brd->br", q_vec[h], V)          # V^T @ query, per block
per_key_scores = U * S.unsqueeze(1) @ q_proj... # (S ⊙ U[i,:]) · q_proj, per key i
approx_scores_per_block = per_key_scores.max(dim=-1).values  # max over real keys
```

(Indices/shapes need care for the batched `[num_blocks, num_kv_heads, ...]`
layout the existing script uses — the point is only that `U` must now be
retained and multiplied in, not discarded, and the query is projected once
via `V` rather than compared against `S·V` as if it were a key.) Expected
outcome per the reasoning above: recall at k=16 should jump close to
whatever the "oracle real max-key dot product" recall turns out to be
(item 2 above) — but the corresponding `summary_bytes` at that k exceeds
the full block's 8192 bytes (see the formula above), so a positive recall
result here would confirm the diagnosis without changing the memory
verdict. The useful k range to actually sweep, if this is prototyped, is
low k (2–6) specifically to see whether there's a knee where recall is
still meaningfully above min/max's 0.44 at a summary size still below
min/max's 1024-byte (12.5%) footprint — the analysis above predicts no,
but this is the one part of this write-up that's a prediction rather than
something already evidenced by existing measurements, and it's cheap to
check if someone wants empirical closure.
