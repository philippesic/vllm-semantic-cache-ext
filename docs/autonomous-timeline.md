# Autonomous development timeline

This file is the concise chronological entry point for unattended work. Detailed
open status lives in `docs/cache-policy-issues.md`; historical experiment reports
remain in `.claude/docs/` and the dated audit documents.

## 2026-08-09: Session bootstrap

- Read both repository instruction files, the 2026-08-07 audit/runbook, the
  2026-08-02 handoff, current code, tests, audit scripts, and recent history.
- Preserved existing user work: modified `AGENTS.md` and untracked `.DS_Store`,
  `2.6-result.csv`, and `2.6-second-pass-result.csv` were not edited.
- Confirmed the extension baseline against the sibling vLLM checkout:

  ```text
  PYTHONPATH=/Users/pippo/github/vllm-semantic-cache \
    .venv/bin/python -m pytest -q tests
  174 passed, 15 warnings
  ```

- Confirmed SSH access to `philip@192.168.4.29`: RTX 5080, 16,303 MiB, driver
  610.88. The Windows host has no WSL/Docker or project Python environment.
  Remote work will be isolated in one dedicated directory with portable `uv`, a
  local venv, copied inputs, manifests, and command/output logs. No system-wide
  provisioning is authorized or planned.
- Bootstrapped `C:\Users\philip\semantic-cache-agent\session-20260809` with
  `uv` 0.11.32, an isolated CPython 3.12.13 installation, and
  `workspace\.venv`. The reproducible bootstrap is
  `tools/rtx5080/bootstrap.ps1`; transcripts live under the remote `logs`
  directory. `uv python install` initially emitted one launcher under the user
  profile despite isolated storage; that exact `.local` tree was immediately
  moved intact to the session's `quarantine\uv-python-shims`, and the bootstrap
  now uses `--no-bin` to prevent recurrence.
- Installed PyTorch 2.12.0+cu130 only in the session venv. A tracked verification
  script completed a CUDA matrix multiplication on the RTX 5080 and reported
  compute capability 12.0. The first inline verification exposed that
  PowerShell does not automatically fail on nonzero native-process exits; both
  remote setup scripts now check every native exit code and fail closed.
- Installed pinned `numpy` 2.5.1 and `transformers` 5.14.1 in the same venv,
  with Hugging Face caches redirected under session `artifacts`. Ran the
  unchanged `step_0_3_oracle_and_mean_recall.py` model experiment on the 5080:

  | Method | Recall@4 | Recall@8 |
  |---|---:|---:|
  | uncompressed max-key oracle | 0.650 | 0.645 |
  | mean-only | 0.510 | 0.575 |
  | min/max + mean | 0.560 | 0.545 |
  | min/max only | 0.430 | 0.440 |

  The ordering reproduces the existing evidence that mean summaries outperform
  min/max. Raw CSV and transcript remain under the isolated remote session.
- Two read-only subagents independently reviewed architecture and experiment
  history. Their highest-confidence findings were the multi-worker aggregation
  defect, failed-store cleanup gap, unresolved metadata hot path, query-capture
  lifecycle risk, and missing authoritative issue/timeline records.

## 2026-08-09: Loop 1 — multi-worker score correctness

- Added failing unit coverage proving that duplicate request scores must be
  reduced across worker head shards, independent of worker order.
- Replaced last-writer-wins aggregation with weighted mean or max reduction,
  carrying the number of contributing local KV heads per request.
- Every worker now emits a score envelope, including when it has no scores.
  Requests and candidates missing any expected TP contribution are omitted so
  incomplete semantic evidence fails closed.
- Pipeline parallelism is rejected until the policy carries one global probe
  layer across stages. Replicated KV heads are rejected for nonlinear scoring
  or max-head aggregation; the linear mean/mean case remains supported.
- Prefill/decode context parallelism, external worker topologies larger than TP,
  and MLA are also rejected until query gathering, score-group topology, and
  latent-cache summary semantics are implemented explicitly.
- Equal scores use the key as a deterministic secondary ordering in both the
  worker and scheduler reductions, because rank position controls EMA weight.
- Converted base-only offload worker metadata to `SemanticWorkerMetadata` when
  emitted, preventing semantic fields from being discarded solely because an
  earlier worker had only transfer completions to report.
- Validation after implementation:

  ```text
  186 passed, 15 warnings
  Ruff check: passed
  Ruff format --check: passed
  git diff --check: passed
  ```

- A frontier review found missing-contributor, PP probe-layer, replicated-head,
  tie-order, and mixed-version hazards in the first draft. Those findings were
  addressed before the second pass.
- Two follow-up reviews identified context-parallel and MLA hazards, which were
  rejected explicitly. The final review reported no blockers. The completed
  code/test slice was committed as `a6f4456`.
- A containment audit after dependency installation and the model run confirmed
  that `C:\Users\philip\.local` is absent and all persistent session files are
  under the dedicated root (including the quarantined launcher evidence).

## 2026-08-09: Loop 2 — score-calibration trace

- Added a reproducible model-trace experiment comparing the current raw EMA
  input with tie-aware rank and global query-L2 normalization. All arms preserve
  the current complete-candidate coverage and rank-weighted EMA update.
- Added a dedicated PowerShell entry point that invokes the common contained
  runner. The copied script, Hugging Face cache, venv, CSV output, hashes, and
  transcripts remain under
  `C:\Users\philip\semantic-cache-agent\session-20260809`.
- Ran an initial short-query trace, then a stricter four-case trace with
  deterministic 0--768-token scaffolds to seek natural query-norm variation.
  The final run covered 36 query events and 64 candidate blocks per case.
- Natural query-norm ratios remained narrow (1.058--1.090). Raw and query-L2
  produced identical natural outcomes, so no normalization implementation or
  default change is justified.
- A positive query-scale stress control demonstrated the expected metamorphic
  property without being treated as workload evidence:

  | Scenario / EMA input | Needle relevance@8 | Attention relevance@8 | Needle policy@8 |
  |---|---:|---:|---:|
  | natural / raw | 1.0000 | 0.8438 | 0.0000 |
  | natural / rank | 1.0000 | 0.8438 | 0.0000 |
  | natural / query-L2 | 1.0000 | 0.8438 | 0.0000 |
  | scale stress / raw | 0.0000 | 0.0000 | 0.0000 |
  | scale stress / rank | 1.0000 | 0.8438 | 0.0000 |
  | scale stress / query-L2 | 1.0000 | 0.8438 | 0.0000 |

- The final CSV SHA-256 is
  `D740F00E80154F3AC088BF34836BA703EE47F33EA41F8D1AC0DD9DD1C087A35A`;
  its transcript is
  `logs\cp004_score_calibration_full_20260809_175307.log` on the isolated host.
- Separating relevance-only from policy outcomes exposed a more direct issue:
  every natural arm kept all needle blocks in relevance-only top-8/top-16, but
  the fixed 0.5 relevance/recency blend kept none. This is tracked as CP-014;
  score normalization is not being used to disguise that downstream tradeoff.

## 2026-08-09: Loop 3 — relevance/recency blend sweep

- Built a held-out model-trace sweep for policy alpha. Each of four cases has a
  two-block target fact, a separate two-block semantic decoy, filler, one target
  update, one decoy update, six unrelated updates, and two held-out queries.
- Replayed target and decoy recency independently across all nine
  oldest/middle/newest combinations at capacities 8/16/24/32. Model inference
  is shared across alpha arms, so every comparison has identical scores,
  candidate coverage, held-out attention labels, and recency order.
- The Windows-side simulator was checked locally against the real
  `SemanticPolicy` on all 288 alpha/capacity/recency combinations with random
  relevance plus the exact alpha-0.5 boundary tie. Every survivor set matched.
- Alpha 0.6 was selected as the conservative experimental arm over 0.5:

  | Capacity | Metric | alpha 0.5 | alpha 0.6 |
  |---:|---|---:|---:|
  | 8 | complete needle | 0.6667 | 0.8333 |
  | 8 | balanced held-out attention | 0.2865 | 0.3715 |
  | 8 | complete stale decoy | 0.4167 | 0.5833 |
  | 16 | complete needle | 0.6667 | 0.9167 |
  | 16 | balanced held-out attention | 0.2161 | 0.3568 |
  | 16 | complete stale decoy | 0.6667 | 0.6667 |
  | 24 | complete needle | 0.8611 | 1.0000 |
  | 24 | balanced held-out attention | 0.3154 | 0.4682 |
  | 24 | complete stale decoy | 0.6667 | 0.6667 |

- There were no paired target, novel-topic, or balanced-attention regressions
  at alpha 0.6. Higher values continued to improve this trace's attention
  recall but increasingly retained stale semantic evidence, so they were not
  promoted. This decoy was intentionally queried once and is a staleness
  control, not a true false-positive label.
- Exposed alpha through the existing `extra_config` path with finite `[0,1]`
  validation. Alpha 0.6 is opt-in only; the behavioral default remains 0.5
  until a Linux/DGX serving audit evaluates real preservation and
  performance.
- Final contained artifacts:

  ```text
  outputs\step_1_4_blend_sweep_full.csv
    SHA256 0C5B235061E6A2A4858BC70E18E7D00393F741A527DB981FBD80BC9F1F215644
  logs\cp014_blend_sweep_full_20260809_181209.log
  ```

- Validation after implementation: 200 tests passed with 15 known warnings;
  focused Ruff and formatting checks passed.

## 2026-08-09: Loop 4 — query-capture lifecycle ownership

- Replaced the unowned process-lifetime query-capture installation with a
  closeable `QueryCaptureHandle`. Only one semantic capture owner may exist in
  a process; a second install fails before changing hooks or callbacks.
- `SemanticOffloadingWorker.shutdown()` now closes capture before shutting down
  the inherited CPU transfer handlers. Each owning runner execution creates a
  fresh dispatch mode and context-local request layout; close waits for active
  execution before restoring the exact prior `prepare_inputs` and
  `execute_model` methods.
- The prepare-inputs wrapper accepts layouts only from the exact runner that
  claimed the installation. Layout survives earlier attention layers, is
  consumed once at the configured probe layer, and is cleared on execution,
  mode-entry, or callback failure. Same-thread close raises instead of
  deadlocking, while cross-thread close waits for quiescence.
- External monkey-patch replacement is never overwritten during close; the
  conflict is surfaced, ownership is released, and a later installation can
  safely compose with the replacement.
- FULL CUDA graphs are rejected before patching because their mixed/prefill
  replay cannot expose attention to Python dispatch. The current supported
  topology is deliberately one runner per process; concurrent runners fail
  fast instead of sharing callbacks or layout state.
- Query-capture and scoring compatibility are preflighted before the inherited
  CPU offload worker allocates tensors or transfer handlers. If final hook
  installation loses a race after allocation, construction shuts those base
  resources down before propagating the error.
- Added CPU behavior tests for active-owner rejection, exact restoration,
  sequential reinstall, exact-runner isolation, non-probe-layer preservation,
  exception cleanup, external-patch conflict, cross-thread quiescence,
  same-thread deadlock prevention, guarded worker shutdown, FULL-mode rejection,
  callback isolation, and worker shutdown order.
- Validation after implementation: 212 tests passed with 15 known warnings;
  focused Ruff and formatting checks passed.
- A same-process sequential-engine Linux GPU smoke remains required because
  native Windows cannot run the vLLM integration.

## 2026-08-09: Loop 5 — compact relevance-update metadata

- Derived an affine composition for the existing rank-weighted EMA. For each
  method/candidate, `(decay, offset, unseen_value)` reproduces the ordered
  request updates while preserving the special rule that a candidate's first
  observation initializes its EMA exactly, even when its rank weight is zero.
- Added a TP=1 compact metadata path for multi-request steps. TP>1 retains raw
  request/candidate scores until complete head reduction and deterministic
  global ranking, then composes before scheduler application. Single-request
  steps remain on the legacy path after measurement showed no compaction gain.
- Added randomized seeded-state and production-path differential tests,
  including missing candidates, zero-weight first observations, TP reduction,
  incomplete contributor groups, compact versus raw transport, and connector
  application. Values match the sequential oracle within `1e-14` and preserve
  relevance ordering.
- Added `benchmarks/relevance_update_metadata.py`, which starts both arms from
  identical ranked scores and times the production multiprocess object envelope:
  the actual response enum, `ModelRunnerOutput`, and KV metadata serialized as
  a tensor-free highest-protocol pickle payload. Queue framing is excluded. It
  also measures seeded scheduler folding and combined post-ranking CPU. Every
  cell performs a full encode/decode round trip; timing uses thread CPU time
  with GC disabled. Final command:

  ```text
  PYTHONPATH=/Users/pippo/github/vllm-semantic-cache \
    .venv/bin/python benchmarks/relevance_update_metadata.py \
    --warmups 2 --repetitions 20
  ```

  | Requests | Candidates | Raw envelope bytes | Compact envelope bytes | Byte reduction | Scheduler speedup | Pipeline speedup |
  |---:|---:|---:|---:|---:|---:|---:|
  | 16 | 512 | 139,522 | 28,968 | 4.82x | 32.33x | 1.61x |
  | 16 | 2,048 | 563,576 | 111,925 | 5.04x | 32.16x | 1.52x |
  | 56 | 512 | 466,087 | 29,648 | 15.72x | 109.75x | 1.82x |
  | 56 | 2,048 | 1,873,476 | 112,605 | 16.64x | 113.45x | 1.66x |

- Acceptance cells (`requests>=16`, `candidates>=512`) require at least 4x
  envelope reduction, 4x seeded scheduler speedup, and no post-ranking pipeline
  regression; all four passed. One-request rows are explicitly labeled
  `legacy_fallback` and execute raw metadata in production. A Linux serving run
  remains necessary to measure end-to-end queueing and worker/scheduler overlap,
  especially for TP>1.
- Validation after implementation: 221 tests passed with 15 known warnings;
  focused Ruff and formatting checks passed. Two adversarial review rounds found
  no production correctness blocker after malformed dual-format metadata and
  stale compact updates were made fail-closed.

## 2026-08-09: Loop 6 — interruption cleanup design audit

- Traced the grid-cell, vLLM-server, and mixed benchmark-client subprocess
  lifecycles. They can occupy separate sessions, while the grid only tracks the
  cell leader and uses `fuser` as a Linux-only detached-server fallback.
- Built and tested a preliminary stored-PGID TERM/KILL/reap path, readiness
  cleanup, and grid signal guard. The focused harness and full 224-test suite
  passed, but two adversarial reviews found ownership blockers that tests could
  not make safe: numeric PGID reuse, detached groups outside the cell session,
  reentrant multi-slot signal cleanup, and uncancellable mixed-client threads.
- Reverted the complete trial before commit. The durable CP-006 design now
  requires a lifetime-authenticated supervisor spanning cells, servers, and
  benchmark clients, with bounded controlled unwinding and synthetic
  descendant tests. No partial signal/PGID patch remains in the worktree.

## 2026-08-10: Loop 7 — paired alpha audit evidence

- Added an explicit alpha 0.5 control and a separately labeled alpha 0.6
  semantic-mean arm to the fail-closed DGX audit. Both use the same middle probe,
  mean head aggregation, disabled prefetch, seeds, workloads, rates, and GPU
  allocation; alpha 0.6 remains experimental and the production default stays
  0.5.
- Every variant now records its exact command/config and seed contract. The
  summarizer requires the expanded 99-row-per-seed matrix, rejects mislabeled
  alpha configs and seed overrides, and writes exact seed-level outcomes and
  alpha-0.6-minus-alpha-0.5 deltas to `alpha_paired_seed_deltas.csv`.
- Audit output roots are acquired with atomic `mkdir` and cannot be reused, so
  stale results cannot be relabeled by a later failed run. A two-process
  regression proves only one concurrent launch can claim a path.
- The runner fingerprints both semantic and vLLM revisions/worktrees at the
  whole-run and per-variant boundaries. Shell execution marks drift as failure,
  and the summarizer independently rejects missing, changed, or cross-variant
  repository state.
- The concurrent alpha arm is a screen, not a default-change gate: it covers
  ref1 plus chat/RAG and shares DGX host resources. A clean isolated one-GPU
  alpha 0.5/0.6 confirmation must add ref2 before any production decision.
- Final local validation: 233 tests passed with 15 known warnings; `bash -n`,
  ShellCheck, focused Ruff, formatting, and `git diff --check` passed. Direct
  Tester validation passed, and adversarial review reported no remaining
  actionable findings after output-root, seed-contract, and repository-state
  fail-open paths were closed.
