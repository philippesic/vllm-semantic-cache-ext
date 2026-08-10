# Cache-policy issue tracker

This is the durable, current tracker for semantic-offload policy work. Historical
investigation details remain in `.claude/docs/`; the status and next validation
for active work belong here.

## Active

### CP-001: Failed stores can leave non-resident worker summaries

- **Priority:** P2 forward-compatibility correctness
- **State:** no current connector failure signal; design before implementation
- **Evidence:** `CPUOffloadingManager.complete_store(..., success=False)` removes
  the scheduler-side pending key without emitting an eviction. The worker may
  already have built a durable summary, so it can continue scoring a key that is
  absent from the CPU tier. However, the current offloading worker asserts that
  every transfer result succeeded and the scheduler always calls
  `complete_store()` with its default `success=True`, so the stale-summary path
  is not reachable without future upstream failure propagation or direct manager
  use.
- **Next:** when the connector gains a real failure channel, add a lifecycle test
  covering summary creation followed by failed completion, then propagate exact
  failed-store keys through the existing worker cleanup metadata.

### CP-002: Query-capture installation has process-global lifecycle state

- **Priority:** P1 correctness/compatibility
- **State:** fixed in local lifecycle tests; Linux GPU smoke pending
- **Evidence:** `semantic_offload/query_capture.py` keeps module-global patch
  state. Previously it had no unregister path, so sequential engines retained
  class patches, dispatch modes, callbacks, and request layout state.
- **Change:** installation now returns one process-owned, closeable handle;
  concurrent installs fail before mutation; worker shutdown closes the handle;
  exact prior methods are restored; external patch conflicts fail closed; and
  every execution gets a fresh dispatch mode and context-local request layout
  owned by one exact runner. Layout is consumed once at the probe layer;
  execution, mode-entry, and callback failures clear context. Close waits for
  in-flight execution, rejects same-thread close instead of deadlocking, and
  permits a clean sequential reinstall. The current supported topology is one
  runner per process; concurrent runners fail fast. FULL CUDA graphs are
  rejected because they replay attention without Python dispatch; eager,
  PIECEWISE, FULL_DECODE_ONLY, and default FULL_AND_PIECEWISE remain supported.
  Compatibility and ownership are preflighted before base offload resources
  are allocated, with cleanup if final installation loses a race.
- **Next:** on a Linux GPU host, construct/serve/shut down engine A and then
  construct/serve engine B in the same process, proving only the current
  engine's callback and scoring metadata appear.

### CP-003: Relevance updates transfer O(requests x candidates) metadata

- **Priority:** P1 performance
- **State:** compact update implemented locally; live Linux profiling pending
- **Evidence:** the worker materializes and sorts every request/candidate score,
  then the scheduler repeats nested rank/EMA loops. Prior live runs show
  `query_captured_total` cost growing substantially even with stable candidate
  pool, batch size, and CUDA allocator counters.
- **Change:** per-request rank-weighted EMA observations are composed into one
  `(decay, offset, unseen_value)` update per candidate. TP=1 sends this compact
  form when a step has multiple requests; TP>1 keeps raw scores until complete
  head reduction/global ranking, then composes before the scheduler fold. A
  one-request step stays on the legacy path because compact metadata is larger
  and slower there. Randomized seeded-state, first-observation, TP reduction,
  completeness, and connector-path oracles match the sequential implementation
  within `1e-14` and preserve ranking.
- **Measurement:** the tracked CPU benchmark uses the actual multiprocess
  response enum plus the response, `ModelRunnerOutput`, and KV-connector object
  envelope. It measures the tensor-free highest-protocol pickle payload with
  decode round trips, excluding constant queue framing. At
  56 requests and 512--2048 candidates, the full envelope was 15.7--16.6x
  smaller, the seeded scheduler fold was 109.7--113.4x faster, and the complete
  post-ranking metadata pipeline was 1.66--1.82x faster. At 16 requests it was
  4.8--5.0x smaller and 1.52--1.61x faster overall.
- **Next:** profile `query_rank_metadata`, metadata transport, and
  `update_relevance` on a Linux GPU serving run. TP>1 pre-aggregation transport
  remains O(requests x candidates) until ranking moves behind a distributed
  score reduction.

### CP-004: Raw score scale is not calibrated across queries

- **Priority:** P1 policy quality
- **State:** investigated; no implementation justified by natural traces
- **Evidence:** rank controls EMA weight, but the updated value remains a raw dot
  product. Query-norm variation can therefore change long-lived relevance even
  when ranking is identical. A four-case, 36-query model trace compared raw,
  tie-aware rank, and global query-L2-normalized EMA inputs. Deterministic
  0--768-token query scaffolds produced only 1.058--1.090 within-case query-norm
  spread, and raw/query-L2 had identical natural pre-policy and final outcomes.
  Controlled positive query scaling caused raw EMA rankings to collapse while
  rank/query-L2 remained invariant, proving the mechanism but not that it occurs
  materially in the measured workload.
- **Next:** keep raw as the only production behavior. Reopen only if a broader
  natural trace suite observes materially wider query-norm spread and paired
  pre-policy gains; synthetic scaling alone is a property test, not a launch
  gate.

### CP-005: Hybrid and multi-KV-group layouts cannot use semantic attribution

- **Priority:** P2 coverage
- **State:** deliberately fail-closed
- **Evidence:** store attribution and prefetch decline ambiguous multi-group
  layouts because the probe layer's group identity is not carried through the
  scheduler metadata.
- **Next:** add CPU layout tests for explicit probe-group identity, then run a
  small compatible-model smoke test.

### CP-006: Interrupted grid runs can orphan server processes

- **Priority:** P2 harness reliability
- **State:** open
- **Next:** add parent-interruption cleanup and prove it against synthetic child
  processes before using it in GPU runs.

### CP-007: Full fail-closed audit has no preserved local result bundle

- **Priority:** P1 evidence
- **State:** blocked on an appropriate Linux GPU host
- **Evidence:** the 2026-08-07 documents define the audit and its gates, but no
  `dgx_logs/` output is present in this checkout. The two untracked `2.6*.csv`
  files are invalid historical artifacts and are not policy evidence.
- **Next:** run the full audit on the DGX, preserve manifest/raw logs/summaries,
  then reproduce the winning subset sequentially on one GPU.

### CP-008: RTX 5080 host lacks a Linux vLLM runtime

- **Priority:** P2 infrastructure
- **State:** constrained but usable
- **Evidence:** the host is reachable and exposes an RTX 5080 with 16 GB VRAM,
  but currently has no WSL, Docker, `uv`, or regular Python environment.
- **Progress:** created a dedicated native-Windows session directory with
  portable `uv`, isolated CPython/venv/cache paths, manifests, transcripts, and
  a quarantine directory. The bootstrap script is tracked under
  `tools/rtx5080/`.
- **Next:** keep Linux-only vLLM integration on local/DGX infrastructure.
  Continue routing every native-Windows model trace through the tracked runner
  and audit the session root after material dependency or cache changes.

### CP-014: Fixed relevance/recency blend can erase strong old-block evidence

- **Priority:** P1 policy quality
- **State:** alpha is now a validated experimental opt-in; default remains 0.5
- **Evidence:** in CP-004's natural traces, all three EMA inputs retained every
  known needle block in relevance-only top-8 and top-16, but the current
  `alpha=0.5` relevance/recency keep-score retained none after placing the
  needles at the oldest positions. The oldest block's maximal relevance and a
  newest block's maximal recency can tie at exactly 0.5, leaving capacity and
  tie order to dominate a semantically clear case. A follow-up four-case model
  trace added explicit semantic decoys, two held-out attention queries, and all
  nine target/decoy oldest-middle-newest recency combinations. The exact policy
  simulator matched `SemanticPolicy` in 288/288 random comparisons plus the
  boundary tie. At capacities 8/16/24, alpha 0.6 improved complete-needle and
  held-out attention retention over 0.5 without paired attention regressions.
  Complete stale-decoy gain did not exceed complete-target gain at those
  capacities. Higher alphas retained progressively more stale semantic decoys.
- **Change:** `extra_config={"alpha": 0.6}` now reaches the policy, while the
  default remains 0.5. Non-numeric, non-finite, and out-of-range values fail
  closed, as does a non-default alpha with `unscored_last`, where alpha has no
  effect.
- **Next:** include 0.5 and 0.6 experimental cells in the Linux/DGX end-to-end
  audit and compare actual CPU-tier preservation, TTFT, throughput, and
  preemptions before considering any default change.

## Closed

### CP-009: Multi-worker score aggregation was last-writer-wins

- **Priority:** P0 correctness
- **State:** closed in `a6f4456`
- **Evidence:** `SemanticWorkerMetadata.aggregate()` used `dict.update()` for
  duplicate request IDs. Tensor-parallel workers therefore selected a score by
  arrival order instead of reducing across head shards.
- **Change:** carry per-request head counts and the configured head reduction;
  use weighted means or maxima, require the complete expected TP contributor
  set (including empty worker envelopes), intersect candidates that lack
  complete shard coverage, reject PP/PCP/DCP, external worker topologies, MLA,
  and unsafe replicated-KV layouts, and always wrap worker metadata in the
  semantic type.
- **Validation:** focused real-aggregator tests and the complete 186-test suite pass;
  Ruff and formatting checks pass. Three adversarial review passes found no
  remaining blocker after the topology restrictions were added.

## Ruled out

### CP-010: Revert incremental stack-cache rebuilding to recover needle quality

- **State:** rejected by clean live evidence
- **Evidence:** semantic policies produced 9/18 ref1/ref2 hits on master versus
  6/18 on the revert; controls were byte-identical between branches.

### CP-011: Candidate-pool growth causes the long-run query-cost increase

- **State:** ruled out on measured healthy runs

### CP-012: Concurrent query-batch growth explains the cost increase

- **State:** insufficient explanation; observed growth was too small

### CP-013: CUDA allocator fragmentation explains the cost increase

- **State:** ruled out; allocated/reserved/inactive-split and retry counters
  stayed flat while per-call cost increased
