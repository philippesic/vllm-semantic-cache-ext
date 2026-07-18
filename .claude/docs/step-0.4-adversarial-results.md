# Step 0.4: adversarial needle-in-haystack recall test

Follow-up requested after Step 0.3's LRU/ARC comparison showed semantic
methods barely beating LRU (`.claude/docs/step-0.3-results.md`, Follow-up 3)
— but that comparison used a single coherent document, where recency and
relevance are naturally correlated, and explicitly noted this was NOT the
adversarial workload the plan's own Step 1.4 accept criterion calls for:
*"a synthetic workload where LRU provably evicts the wrong blocks (long-idle
but semantically-needed prefix, plus cache-filling noise traffic)."* This
step builds and runs exactly that workload, pulled forward from Phase 1.

Script: `experiments/step_0_4_adversarial_needle_recall.py`.

## Construction

- A short, distinctive "needle" fact (e.g. "the secret access code for
  Project Nightingale is 47392-Delta") is placed at the very start of the
  prompt — maximum idle time, the worst case for a recency-based policy.
- Followed by 1500 or 3000 tokens of topically unrelated "noise" filler
  (astrophysics / marine biology passages, reused from Step 0.3).
- Followed by a question that can only be answered by recalling the needle
  ("What is the secret access code for Project Nightingale?").
- 4 distinct needle facts × 2 noise topics × 2 noise lengths = 16 cases.
- The needle's block indices are known by construction (not inferred), so
  recall of the needle specifically can be measured directly, in addition to
  recall against real attention-mass ground truth (same metric as Step 0.3).
- Same model, layer, block size (16 tokens), and probe methodology (last
  token position) as every other step_0_3_* script. LRU/ARC use the real
  production policy classes; min/max, mean, and oracle use the same formulas
  already validated in Step 0.3.

## Results (16 cases)

| method | attn_recall@4 | needle_hit@4 | attn_recall@8 | needle_hit@8 |
|--------|---------------|--------------|---------------|--------------|
| lru    | 0.25          | **0.0**      | 0.125         | **0.0**      |
| arc    | 0.25          | **0.0**      | 0.125         | **0.0**      |
| minmax | 0.73          | 0.58         | 0.66          | 0.79         |
| mean   | 0.875         | **0.96**     | 0.84          | **1.0**      |
| oracle | 1.0           | 0.96         | 0.89          | **1.0**      |

`needle_hit@N` = fraction of the known needle block(s) present in the
method's predicted top-N (or, for LRU/ARC, its "keep" set) — the direct,
construction-grounded test of the plan's hypothesis.

**Sanity check on the ground truth**: `true_top8` (real attention mass)
included the needle block(s) in every single one of the 16 cases (see run
log) — confirming the base model genuinely attends back to the needle at
retrieval time. This means the high semantic-method recall isn't an
artifact of a broken ground truth; the model really does need the needle,
and semantic scoring really does find it.

## What this shows

**LRU and ARC recovered the needle in 0 of 16 cases, at any N.** Not "low
recall" — exactly zero. This is the precise failure mode the plan's Risk #4
and Step 1.4 both predicted: a policy driven purely by recency will always
evict a block the moment something else has been touched more recently,
regardless of whether that block is about to be needed again. `attn_recall`
for LRU/ARC (0.25/0.125) is nonzero only because the few most-recent noise
blocks do carry some real attention mass too (local continuation effects) —
but they never carry the needle.

**Every semantic method tested recovers the needle far more often than
LRU/ARC, including the previously-unremarkable min/max pooling** (0.58–0.79
needle-hit, vs. LRU's 0.0). Mean-only is the standout at this workload
(0.96–1.0 needle-hit) and nearly matches the oracle ceiling — on this
adversarial construction, mean-only is not just "a bit better than LRU," it
is close to as good as having zero compression at all.

This is the inverse of Step 0.3's coherent-document result, and that
contrast is the actual finding, not a contradiction: **whether semantic
eviction helps is workload-dependent, exactly as the plan's Risk #4
anticipated** — it does essentially nothing over free recency on a
coherent single-document workload, and it is the difference between
"always fails" and "almost always succeeds" on a workload with a genuinely
idle-but-needed block. Real inference traffic is a mix of both patterns
(chat is closer to the coherent case; RAG/long-doc/multi-turn-with-context-
switching is closer to the adversarial case) — which is precisely why the
plan's benchmarking design (§Benchmarking, workload taxonomy across chat/
rag/longdoc/mixed) exists instead of a single aggregate number.

## Bearing on "is min/max worth developing through Phase 1/2"

This is the first workload in the whole Step 0.3/0.4 sequence that gives a
clear "yes, there's something real here" signal rather than an ambiguous
tie. It's a strong reason to continue rather than stop — but two honest
caveats before treating it as a green light:

1. This is still a synthetic smoke test (16 cases, one model, one layer,
   hand-built prompts), not the rigorous benchmark harness the plan
   specifies for Phase 1 (Step 1.6: multi-seed, steady-state, latency-vs-
   load frontier curves, on the `rag`/`longdoc`/`mixed` workload taxonomy).
   It shows the hypothesis is alive, not that it's proven.
2. **Mean-only, not min/max, is now the stronger candidate to carry
   forward** — it beat min/max on both this workload and the coherent-
   document workload in Step 0.3. Whoever plans Step 1.1 onward should
   default to mean-based scoring (or a mean/min/max combination, still
   unexplored) rather than pure min/max pooling as originally suggested by
   the plan's Quest citation.
