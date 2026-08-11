# DGX semantic-policy audit handoff — 2026-08-07

## Current state

The runnable audit implementation was published first as commit `17af85b`
(`Add unified semantic policy audit`). The DGX repositories are expected at:

- `/raid/ppesic/tmp/vllm-semantic-cache`
- `/raid/ppesic/tmp/vllm-semantic-cache-ext`

From the extension repository, the operator only needs:

```bash
git pull
./dgx_policy_audit.sh
```

The script sources
`/raid/ppesic/tmp/vllm-semantic-cache/.venv/bin/activate`, pins both Python and
the `vllm` CLI to that environment, and dispatches only to GPUs 4–7. GPUs 0–3
are occupied and must not be used. The script exits before launching work if
the environment, CLI, or sibling vLLM checkout is missing.

This first DGX execution is a validation run, not yet a final competitive
claim. Local CPU/mocked verification passed, but the CUDA stream and
multi-process worker paths have not run on this exact code.

## What was implemented before this run

The audit corrected several issues that could invalidate semantic-policy
results:

1. Store metadata now preserves the exact `OffloadKey` to ordered source-GPU
   block mapping. Multi-block chunks are summarized as one logical item;
   ambiguous hybrid or sliding-window layouts fail closed.
2. Relevance, grace, chain, and session state is purged on eviction, removal,
   and reset. Delayed scores cannot resurrect non-resident keys.
3. Summary reductions are fenced before the GPU-to-CPU store can complete and
   before scoring consumes the resulting tensors.
4. Prefetched source blocks remain quarantined until every worker reports a
   completed GPU splice event. Reset and in-flight transfer paths retain the
   same source-lifetime protection.
5. The upstream request progress field is updated through a current/legacy
   compatibility setter instead of silently creating an unused attribute.
6. Production defaults now match the best current evidence: mean summary,
   natural-order middle probe layer, mean head aggregation, and semantic
   prefetch disabled.
7. Needle reference-count arms use fresh servers and identical content for a
   given seed. Mixed rows carry their parent request rate, and all content
   seeds vary with the grid seed.
8. Result validation checks the exact expected matrix, required fields, finite
   numeric values, valid needle outcomes/reference counts, duplicate rows,
   embedded errors, and insufficient pressure.

The final local gate was 174 passing tests, targeted Ruff checks and formatting,
`bash -n`, `git diff --check`, and a synthetic 288-row matrix test that accepted
the complete matrix and rejected a missing serving-rate cell.

## Policy architecture context

The policy is not a semantic response cache. It is an eviction policy for
vLLM's lossless CPU KV-offload tier:

1. The scheduler creates CPU-tier store jobs and sends exact key/source-block
   identities to the worker.
2. The worker reads the selected attention layer's live GPU KV cache and builds
   a durable per-key summary.
3. Query capture produces a per-request query representation. The worker scores
   it against all resident summaries and sends rankings back to the scheduler.
4. The manager folds scores into a relevance EMA.
5. `SemanticPolicy` combines relevance and recency when selecting CPU-tier
   victims. If no usable relevance exists, behavior falls back to recency.
6. Experimental prefetch can reserve and splice relevant CPU-tier blocks back
   into GPU memory, but it is disabled by default and appears only in controlled
   1% and 5% budget arms.

This distinction matters when reading results: model-answer correctness should
remain intact even on an eviction miss because vLLM recomputes missing KV. The
policy's benefit is avoiding recomputation and reducing latency under pressure,
not changing answer accuracy.

## Run matrix and defaults

The script records all settings in `manifest.txt`. Current defaults are:

- model: `Qwen/Qwen2.5-7B-Instruct`
- GPUs: `4,5,6,7`
- seeds: `1,2,3`
- CPU-tier bytes: `91,750,400`
- GPU blocks: `120`
- maximum model length: `2048`
- serving duration: `180` seconds per configured cell
- ablation prompts: `24`
- cell timeout: `7,200` seconds

The experiment phases are:

1. Full CPU/unit suite.
2. Isolated needle-v2 leaderboards for reference counts 0, 1, and 2 across
   LRU, ARC, semantic-minmax, semantic-mean, and semantic-cuboid-mean.
3. Chat, RAG, and mixed serving at parent request rates 2 and 8 across all five
   policies.
4. Semantic-mean ablations for first/max, middle/max, explicit middle/mean alpha
   0.5, experimental middle/mean alpha 0.6, decayed session evidence, 1%
   prefetch, 5% prefetch, and capture stride 4.

Four cells run concurrently on GPUs 4–7. This reduces wall-clock time but means
latency, CPU-tier transfer, pinned-memory, and RSS results share host resources.
Treat the leaderboard's preservation outcomes as primary policy-quality
evidence. Treat small latency differences and mixed-workload byte attribution
as provisional until reproduced in an isolated one-GPU confirmation run.

## Output layout

Each run creates:

```text
dgx_logs/policy_audit_<timestamp>/
├── manifest.txt
├── repository_state.txt
├── unit_tests.log
├── leaderboard_ref0/
├── leaderboard_ref1/
├── leaderboard_ref2/
├── serving/
├── signal_first_max/
├── signal_middle_max/
├── signal_middle_mean/
├── signal_middle_mean_alpha06/
├── session_decay8/
├── prefetch_001/
├── prefetch_005/
├── capture_stride4/
├── audit_summary.csv
├── alpha_paired_seed_deltas.csv
├── audit_summary.md
└── timing_summary.csv
```

A sibling `policy_audit_<timestamp>.tar.gz` is also created. Per-cell server and
driver logs remain under each variant/GPU directory. Each variant also contains
`variant_manifest.txt` with its exact command/config, seed/GPU contract, and
start/end semantic and vLLM repository fingerprints. The root manifests record
the same whole-run repository state plus environment paths, model, GPU
allocation, and NVIDIA device/driver information.

The summarizer exits nonzero on missing variants/cells, errors, malformed
metrics, non-finite values, duplicate standalone rows, an entirely unpressured
leaderboard arm, config/seed provenance disagreement, or repository drift. The
shell script atomically refuses reused output roots, accumulates grid failures,
still attempts to summarize available evidence, creates the archive, and exits
nonzero if any phase failed.

## How to interpret needle-v2

Needle-v2 isolates CPU-offload counters around the recall request:

- `hit`: loads occurred and no stores occurred. This is a conservative
  full-preservation lower bound.
- `partial`: both loads and stores occurred. At least part of the prefix was
  preserved, but the counters cannot prove every needed block was loaded;
  recall-tail stores can also make this category conservative.
- `miss`: no loads occurred and stores occurred. The prefix was recomputed and
  restored after CPU-tier eviction.
- `not_pressured`: neither counter moved. The prefix remained in GPU memory, so
  the cell does not measure CPU eviction quality.

`audit_summary.md` therefore reports both full-hit lower-bound rate and
any-load rate (`hit + partial`) among pressured outcomes. Do not silently merge
`not_pressured` into misses or successes.

Reference-count interpretation:

- `ref0` is the structural cold-start control. No semantic evidence exists
  before pressure, so semantic is not expected to consistently beat recency.
- `ref1` is the primary differentiator: one semantically related, token-disjoint
  probe should give semantic a useful signal without refreshing LRU recency.
- `ref2` checks whether repeated evidence strengthens or destabilizes the
  policy.

The strongest competitive result would be semantic-mean beating both LRU and
ARC on pressured `ref1` and `ref2` cells across seeds, while preserving serving
performance. A semantic win only on `ref0`, widespread `not_pressured`, or a
single seed is not sufficient.

## How to interpret serving and timing results

For standalone chat and RAG, compare paired policy/seed/rate rows on:

- p50 and p99 time to first token;
- output-token throughput;
- CPU load/store bytes;
- preemption count;
- server errors, timeouts, or zero-completion rows.

Mixed runs are stress evidence only. Their substreams share one live server and
global offload counters, so per-row byte/preemption deltas can include activity
from neighboring substreams. Use mixed results to expose stalls, crashes,
preemption explosions, and broad throughput collapse, not to make precise
per-substream cache-hit claims.

`timing_summary.csv` aggregates the last cumulative timing snapshot per server
process. Important buckets include summary construction, stack rebuild,
query-capture scoring/synchronization, ranking metadata, and scheduler
relevance updates. Correlate timing changes with resident-pool cardinality,
batch size, RSS, and preemptions in the server logs before attributing a
regression to a single bucket.

The provisional engineering gate is:

1. no validation failures, crashes, corrupted answers, or missing cells;
2. all leaderboard arms are genuinely pressured;
3. semantic-mean improves ref1/ref2 preservation over both LRU and ARC across
   seeds;
4. no material increase in preemptions;
5. no unexplained p99 TTFT or throughput regression larger than roughly 5%;
6. bounded RSS and relevance/summarization cardinality.

The 5% latency margin is a starting engineering threshold, not a final product
decision. A larger regression requires an explicit quality/latency tradeoff,
not post-hoc acceptance.

## Immediate triage procedure after the run

1. Record the shell exit code and output directory.
2. Read `audit_summary.md` and resolve every validation failure before comparing
   policies.
3. Confirm `manifest.txt` shows GPUs 4–7, the intended commits, the vLLM venv,
   and no unexpected dirty files.
4. Check `unit_tests.log` and every top-level `<variant>.log` for failures.
5. Search server logs for `Traceback`, `CUDA`, `assert`, `timeout`, `FAILED`,
   `WARNING`, `SEMANTIC_STORE_LAYOUT_SKIPPED`, and prefetch debug markers.
6. Verify ref0/ref1/ref2 pressure and compare full-hit lower-bound plus any-load
   rates by policy and seed.
7. Compare standalone chat/RAG serving metrics before using mixed results.
8. Inspect timing/RSS trends and identify whether cost scales with requests,
   resident candidates, or run duration.
9. Preserve the output directory and tarball unchanged before running follow-up
   experiments.

For the next agent, provide at minimum `manifest.txt`, `audit_summary.md`,
`audit_summary.csv`, `timing_summary.csv`, all top-level variant logs, and the
server logs for any failed or anomalous cells. Do not provide only aggregate
means; seed-level rows are needed to distinguish stable effects from outliers.

## Outcome-driven next steps

### If semantic-mean clearly beats LRU and ARC

1. Re-run the winning semantic/LRU/ARC subset sequentially on one of GPUs 4–7
   to remove shared-host contention.
2. Add paired per-seed deltas and confidence intervals.
3. Run a longer chat/RAG soak with timing enabled at lower sampling frequency.
4. Optimize the O(requests × resident candidates) ranking/metadata path while
   preserving numerical equivalence against the current implementation.
5. Test raw-score, rank-normalized, and query-normalized EMA updates.

### If quality improves but serving regresses

1. Attribute the regression among summary construction, query scoring, tensor
   stack rebuild, Python sorting/serialization, and scheduler EMA folding.
2. Compare `capture_stride4` to `signal_middle_mean` to estimate capture cost
   versus quality loss.
3. Replace full per-request rankings with an equivalent batched per-key update
   that sends O(candidates), not O(requests × candidates), metadata.
4. Keep prefetch disabled while isolating core semantic-policy cost.

### If semantic does not beat the baselines

1. Confirm pressure and exact store-layout coverage before rejecting the signal.
2. Compare middle/mean against first/max and middle/max to separate layer choice
   from head aggregation.
3. Inspect score distributions and EMA state for scale drift, saturation, or
   sparse coverage.
4. Test normalized scoring and alternative alpha/EMA settings.
5. Treat consistent parity or losses across clean ref1/ref2 cells as evidence
   against the present summary representation, not as a reason to tune only the
   benchmark.

### If prefetch helps

Prefetch remains experimental. Require a repeated TTFT improvement without more
preemptions, throughput loss, stalls, or memory growth. Confirm first at 1%; use
5% only if the extra reservation budget adds value. Do not enable it by default
from a single concurrent-grid win.

### If the run crashes or hangs

1. Identify whether failure occurred during server launch, summary store,
   query capture, prefetch splice, reset, or result collection.
2. Reproduce the smallest failing policy/seed/workload on one GPU.
3. For a CUDA ordering failure, preserve the complete server log and inspect
   summary-stream and splice-event markers before changing synchronization.
4. For timeouts, distinguish zero completed requests from a slow but progressing
   cell using the cell and server logs.
5. Do not relax the summarizer or timeout handling to make an incomplete run
   appear successful.

## Open engineering items

Prioritized remaining work after this run:

1. **Real CUDA lifecycle tests.** Exercise the summary/store fence and
   multi-worker splice acknowledgement with delayed kernels and immediate GPU
   block reuse.
2. **Failed-store cleanup.** Explicitly tell workers to remove durable summaries
   for store jobs that fail after summary creation.
3. **Query-capture lifecycle.** Remove process-global batch state, support
   unregister/reinstall, test multiple runners/engines, unsupported layouts,
   and CUDA-graph replay.
4. **Hybrid/multi-KV-group support.** Current behavior deliberately skips
   semantic summary attribution. Carry the probe layer's group identity before
   enabling these models.
5. **Hot-path complexity.** Collapse full worker rankings and scheduler nested
   updates from O(requests × candidates) transferred metadata toward an
   equivalent O(candidates) batch update.
6. **Score calibration.** Compare raw, rank-normalized, and query-normalized
   relevance without conflating scale with semantic utility.
7. **Statistical reporting.** Alpha 0.5/0.6 now has seed-level paired deltas;
   add confidence intervals and explicit pass/fail gates against LRU and ARC.
8. **Long-soak/reset coverage.** Test repeated engine reset, abort during
   prefetch, stale completion acknowledgements, and multi-hour RSS/GC behavior.
9. **Termination cleanup.** Add SIGINT/parent-interruption cleanup to the grid
   driver so interrupted runs cannot orphan server processes.
10. **Needle precision.** Derive expected prefix transfer bytes so `partial`
    can distinguish a true partial prefix reload from a full prefix hit plus
    newly stored recall-tail blocks.

## Files an agent should read first

1. `docs/policy-audit-2026-08-07.md` — audit findings and prior evidence.
2. `dgx_policy_audit.sh` — authoritative run matrix and DGX defaults.
3. `benchmarks/summarize_policy_audit.py` — validation and aggregation rules.
4. `harness/needle_workload.py` — preservation-outcome semantics.
5. `semantic_offload/connector.py` — scheduler/worker metadata, store identity,
   reset, and prefetch lifecycle.
6. `semantic_offload/worker.py` — summaries, scoring, CUDA fences, and splice
   completion events.
7. `semantic_offload/policy.py` and `semantic_offload/manager.py` — eviction
   ranking, EMA, and state cleanup.
8. `tests/test_policy_audit.py` and `tests/test_step_1_5_prefetch.py` — regression
   contracts added for this audit.

Preserve the fail-closed behavior when making follow-up changes. Unsupported or
ambiguous layouts should remain disabled rather than guessed, incomplete result
matrices should remain failures, and mixed-workload counters should remain
clearly labeled as approximate.
