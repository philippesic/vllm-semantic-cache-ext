# Semantic eviction policy audit — 2026-08-07

## Bottom line

The policy has a credible targeted quality advantage, but it is not yet a
general replacement for LRU or ARC. Existing offline tests show mean-summary
semantic ranking beating both baselines on coherent and adversarial retrieval,
and the latest live needle-v2 run favored semantic-mean. The production path,
however, had correctness and lifecycle defects capable of corrupting that
signal, plus an O(requests × resident blocks) CPU/metadata hot path that still
needs DGX measurement and optimization.

This audit fixes the correctness defects, aligns production defaults with the
best existing evidence, disables speculative prefetch by default, and adds one
fail-closed DGX suite for quality, latency, throughput, pressure, preemptions,
and controlled ablations.

## Evidence already available

- Coherent retrieval: LRU/ARC recall was 0.45 at K=4 and 0.435 at K=8;
  mean-summary was 0.56 and 0.575. Min/max did not beat the baselines.
- Adversarial needle retrieval: LRU/ARC had zero needle hits; mean-summary was
  0.9583 at K=4 and 1.0 at K=8. Min/max was 0.5833 and 0.7917.
- Head aggregation: mean-summary + mean-head reached 1.0 needle hit at K=4 and
  K=8 in the standalone suite; min/max + max-head reached 0.4583 and 0.4861.
- Latest live needle-v2 handoff: semantic-mean hit 5/6 cells, ARC 1/6, LRU 0/6,
  semantic-cuboid-mean 2/6, and semantic-minmax 2/6.
- Chain bonus is currently harmful: complete-session rate fell from 0.70 for
  LRU/no-chain to 0.2667 with chain awareness. It remains disabled.
- Session evidence is workload-sensitive. It was very strong in the isolated
  continuation test, while a flat bonus hurt active sessions under concurrency.
  Decay restored active-session performance in that synthetic concurrent test;
  the live suite therefore tests a decayed arm instead of enabling it globally.

## Implemented findings

### Correctness and lifecycle

1. **Store-summary identity corruption — fixed.** The base scheduler stores job
   keys in a set. The old worker converted that set back to a list and zipped it
   to ordered GPU block IDs, silently assigning summaries to arbitrary keys.
   Metadata now carries an exact key → source-block layout reconstructed from
   the ordered CPU destination spec. Multi-block chunks are summarized across
   all constituent blocks. Ambiguous hybrid/sliding layouts are skipped rather
   than guessed.

2. **Stale/unbounded relevance state — fixed.** Eviction, explicit removal, and
   cache reset now remove relevance, grace, chain, and session metadata. Delayed
   worker scores cannot resurrect non-resident keys. This removes a correctness
   bug on key reinsertion and a scheduler-side memory/object-growth source.

3. **Worker reset gap — fixed.** Cache reset now tells workers to clear durable
   summaries, pending job layouts, scores, and stacked tensors. Speculative GPU
   reservations are released only after in-flight stale transfers finish.

4. **Chain-link leakage — fixed.** Reverse predecessor tracking now removes both
   incoming and outgoing chain edges without an O(cache-size) eviction scan.

5. **Cross-stream summary/store races — fixed.** Query scoring waits for
   reductions before consuming them, and GPU→CPU store submission is now
   transitively fenced behind summary completion. A store cannot complete and
   release/reuse its source GPU block while the summary stream still reads it.

6. **Sequential engine query-capture bug — fixed.** The class-level
   `prepare_inputs` patch no longer closes over only the first installed worker's
   layout state. Stride and GQA divisibility are validated explicitly.

### Evidence-aligned behavior

7. **Production signal now matches experiments.** The default probe changes
   from the lexicographically first layer to the natural-order middle layer,
   which is where the supporting experiments probed. First/middle/last, numeric
   ordinal, and exact layer-name overrides are supported.

8. **Mean is the default method and head aggregation.** Min/max, cuboid-mean,
   and max-head remain available as audit arms; this is a reversible default,
   not a claim that the search is complete.

9. **Prefetch defaults off.** Existing evidence proves byte-exact partial
   splicing but not an aggregate benefit, and prior runs associated semantic
   prefetch with more GPU preemptions. `prefetch_budget_fraction` now exposes
   0%, 1%, and 5% controlled arms. The feature should only be re-enabled by
   default after it improves TTFT without increasing preemptions.

10. **Prefetch source lifetime — fixed.** Source GPU blocks are no longer freed
    when splice metadata is queued. They remain quarantined until a CUDA event
    completes on every worker and the scheduler receives every acknowledgement.
    Reset and abort paths retain the same fence. The splice also updates the
    real upstream `next_stored_chunk_idx` field through a version-safe shim.

### Measurement integrity

11. **Diagnostics distinguish windows from cumulative history.** Count logs now
    report window mean/max and cumulative mean separately. Low-frequency RSS and
    GC counters, scheduler relevance size, ranking/metadata time, and worker pool
    sizes are emitted under `SEMANTIC_OFFLOAD_TIMING=1`.

12. **Benchmark package collision — fixed.** An explicit package marker prevents
    Python from importing vLLM's unrelated `benchmarks` package when both repos
    are on `PYTHONPATH`.

13. **Fail-closed result validation — added.** The summarizer rejects missing or
    empty result files, embedded errors, duplicate standalone cells, missing
    variants/policies/seeds, and unpressured leaderboard arms. Recall traffic
    with both loads and stores is reported as partial, not a full hit; summaries
    show both a conservative full-hit lower bound and any-load rate.

14. **Benchmark isolation and environment pinning — fixed.** Each needle
    reference-count arm gets a fresh server/cache while reusing identical
    content for the same seed. Server and client commands use the vLLM CLI from
    the selected Python environment. Mixed counters are retained only as
    approximate stress evidence. The DGX runner currently uses the four
    available GPUs (4–7); latency conclusions should be confirmed with an
    isolated one-GPU rerun because cells share host resources.

## Important open opportunities

1. **Collapse the R × C update path.** Every query currently sorts every
   resident candidate on CPU, serializes the full matrix, and repeats nested
   scheduler loops. The next optimization should compute an equivalent per-key
   batch update on the worker and send O(C) metadata. Keep legacy and batched
   modes side-by-side until rankings and EMA state are byte/numerically checked.

2. **Calibrate scores across queries.** Raw dot-product scales vary by query;
   rank only changes EMA weight, not score scale. Compare raw, rank-normalized,
   and query-normalized signals without changing coverage.

3. **Support hybrid/multi-group models explicitly.** The safe current behavior
   skips semantic summary attribution for these models. Metadata should carry
   the probe layer's KV-group identity before expanding support.

4. **Replace monkey-patch capture with an upstream hook.** The current mechanism
   targets the V2 GPU runner and unified attention operator. It is fragile across
   vLLM internals and does not cover every backend.

5. **Tune the policy, not just the scorer.** Alpha, EMA ceiling/rank power,
   capture stride, and session half-life need factorial or sequential tuning.
   Chain awareness should stay disabled unless redesigned; its current constant
   bonus overwhelms the [0,1] blend and has negative evidence.

6. **Add long-soak and reset integration checks on GPU.** Unit tests cover state
   transitions, but repeated engine reset, abort during prefetch, and multi-hour
   RSS/GC stability require the real asynchronous worker/scheduler topology.

7. **Close remaining lifecycle gaps.** Failed store completion still needs an
   explicit worker-side durable-summary removal signal. Query capture remains
   process-global and needs unregister/multi-runner/CUDA-graph lifecycle tests.

8. **Add statistical decision gates.** Report paired per-seed deltas and
   confidence intervals against LRU and ARC instead of relying only on means.

## DGX test plan

On the DGX, pull the repositories into `/raid/ppesic/tmp/vllm-semantic-cache`
and `/raid/ppesic/tmp/vllm-semantic-cache-ext`, then run
`./dgx_policy_audit.sh` from the extension repo. The script activates the
sibling vLLM `.venv` and dispatches only to GPUs 4–7 by default. It performs:

1. all CPU/unit tests;
2. isolated, same-content, multi-seed needle-v2 leaderboards across LRU, ARC,
   and all semantic methods, one fresh server per reference-count arm;
3. multi-seed chat/RAG/mixed serving tests at rates 2 and 8;
4. controlled first-vs-middle probe and max-vs-mean head ablations;
5. decayed-session, 1%/5% prefetch, and capture-stride-4 ablations;
6. strict expected-matrix CSV validation and Markdown/CSV aggregation with a
   worktree-content hash, reproducibility manifest, and tarball.

Before calling the policy competitive, require repeated multi-seed improvement
over both LRU and ARC on pressured quality cells, no material regression on
chat/RAG/mixed throughput or p99 TTFT, no increase in preemptions, and stable
RSS/relevance cardinality in a long soak. The exact acceptable latency margin is
a product decision; a reasonable initial engineering gate is ≤5% regression
unless quality improves enough to justify a documented tradeoff.
