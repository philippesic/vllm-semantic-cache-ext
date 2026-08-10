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
