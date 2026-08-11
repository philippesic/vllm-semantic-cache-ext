# DGX semantic-policy audit session handoff — 2026-08-10

## Objective and close status

This session made the alpha 0.5 versus 0.6 policy comparison reproducible and
fail-closed before requesting new GPU evidence. The implementation is complete,
reviewed, committed as `7b5d796` (`Add paired alpha audit evidence`), and pushed
to `origin/master`.

No DGX or RTX GPU experiment ran during this session. There is no local
`dgx_logs/` directory, no current manifest, no audit summary, no timing summary,
and no seed-level raw bundle. Do not infer policy quality, latency, throughput,
or preemption results from this implementation-only session.

## Completed implementation

Commit `7b5d796` changed:

- `dgx_policy_audit.sh`
- `benchmarks/summarize_policy_audit.py`
- `tests/test_policy_audit.py`
- `docs/autonomous-timeline.md`
- `docs/cache-policy-issues.md`
- `docs/dgx-policy-audit-handoff-2026-08-07.md`

The audit now:

1. runs an explicit semantic-mean alpha 0.5 control and a separately labeled
   alpha 0.6 experimental arm with identical probe/head/prefetch settings;
2. requires the expanded 99-row-per-seed result matrix and emits exact paired
   seed rows in `alpha_paired_seed_deltas.csv`;
3. records and validates exact variant configs, seed/GPU contracts, and commands;
4. atomically refuses reused output roots, including concurrent same-path runs;
5. fingerprints the semantic and vLLM revisions/worktrees at whole-run,
   per-variant, measurement-end, and post-summary boundaries; and
6. rejects missing, inconsistent, or drifting provenance during normal or later
   independent validation. The narrowly scoped hidden `--pre-summary` mode is
   used only to bootstrap the shell's first summary before post-summary state is
   captured.

Alpha 0.6 remains opt-in. The production default remains alpha 0.5.

## Exact local validation

The selected local environment was the extension project's Python environment
with the sibling vLLM checkout on `PYTHONPATH`, because the sibling `.venv` is a
runtime-only environment on this macOS checkout and lacks the local test stack.

```bash
cd /Users/pippo/github/vllm-semantic-cache-ext

PYTHONPATH="$PWD:../vllm-semantic-cache" \
  .venv/bin/python -m pytest -q tests/test_policy_audit.py
# 22 passed, 1 known vLLM version warning

PYTHONPATH="$PWD:../vllm-semantic-cache" \
  .venv/bin/python -m pytest -q tests
# 233 passed, 15 known warnings

bash -n dgx_policy_audit.sh
# exit 0

shellcheck dgx_policy_audit.sh
# exit 0 (ShellCheck 0.11.0 in the independent Tester pass)

uvx --from ruff ruff check \
  benchmarks/summarize_policy_audit.py tests/test_policy_audit.py
# All checks passed

uvx --from ruff ruff format --check \
  benchmarks/summarize_policy_audit.py tests/test_policy_audit.py
# 2 files already formatted

git diff --check -- \
  dgx_policy_audit.sh benchmarks/summarize_policy_audit.py \
  tests/test_policy_audit.py
# exit 0
```

The Tester independently reproduced the 22-test focused and 233-test full
results. The Reviewer completed adversarial passes over output-root reuse,
atomic acquisition, matrix completeness, config/seed provenance, repository
drift, and interrupted summary-finalization paths, then reported no remaining
actionable findings.

## Repository and environment state at close

- Extension repository: `7b5d79697823368f1aef3cc85ae35db6917ebb0a`
- Extension branch: `master`, equal to `origin/master`
- Sibling vLLM revision: `dc1be79031d948d7a18c37600881e45ca708d913`
- Local Python: `3.12.13`
- Local PyTorch: `2.13.0`
- Imported vLLM source:
  `/Users/pippo/github/vllm-semantic-cache/vllm/__init__.py`
- DGX allocation: none; DGX was not contacted
- Intended future DGX allocation from the tracked runner: GPUs `4,5,6,7`
- RTX 5080 check at `philip@192.168.4.29`: unreachable with
  `No route to host`; its current GPU/process state is unknown and the IP likely
  rotated
- Local attributable processes at close: none (`vllm`, grid, latency-suite,
  audit, and pytest process search was empty after excluding the inspection
  command itself)
- Processes stopped: none; no child process attributable to this session was
  alive

The sibling vLLM checkout was not changed. It remains independently dirty and
divergent (`main...origin/main [ahead 19193, behind 18627]`) with pre-existing
deleted, modified, and untracked files. Preserve it until its owner decides how
to reconcile that checkout.

## Preserved user-owned extension worktree

The following pre-existing changes remain intentionally uncommitted and were
not modified, staged, or deleted by the implementation or stop workflow:

```text
 M AGENTS.md
 M semantic_offload/manager.py
 M sync-vllm-upstream.sh
?? .DS_Store
?? 2.6-result.csv
?? 2.6-second-pass-result.csv
?? README.md
?? docs/repository-workflow.md
```

The two `2.6*.csv` files remain invalid historical artifacts, not current audit
evidence. No raw evidence file was moved, rewritten, or removed.

## Intended audit configuration

The tracked full runner currently defaults to:

- model `Qwen/Qwen2.5-7B-Instruct`;
- GPUs `4,5,6,7`;
- seeds `1,2,3`;
- CPU tier `91,750,400` bytes;
- `120` GPU blocks;
- maximum model length `2048`;
- scale `0.04`;
- serving duration `180` seconds;
- ablation prompt count `24`; and
- cell timeout `7,200` seconds.

The alpha screen uses semantic-mean, middle probe layer, mean head aggregation,
prefetch disabled, needle-v2 ref1, and standalone chat/RAG at rate 8. Alpha 0.5
and 0.6 share the same seeds, content, workload, and GPU contract. Concurrent
four-GPU timing, byte, RSS, and preemption deltas remain provisional until an
isolated one-GPU confirmation adds ref2 and standalone chat/RAG.

## Policy-versus-LRU/ARC evidence boundary

No current fail-closed policy comparison exists in this checkout. In
particular, there are no current pressured/not-pressured seed rows to report.
The 2026-08-07 documents record historical evidence only:

- offline coherent retrieval reported mean-summary recall of `0.56` at K=4 and
  `0.575` at K=8 versus `0.45` and `0.435` for LRU/ARC;
- offline adversarial retrieval reported mean-summary recall of `0.9583` and
  `1.0` versus zero LRU/ARC needle hits; and
- the historical live needle-v2 handoff reported semantic-mean hits in `5/6`
  cells, ARC in `1/6`, and LRU in `0/6`.

Those figures do not come with a preserved current `dgx_logs` bundle here and
must not be promoted to a new competitive, latency, or default-change claim.
The next run must preserve seed-level outcomes and distinguish `hit`, `partial`,
`miss`, and `not_pressured`; incomplete, invalid, shared-host, or missing output
cannot be treated as a success.

## Remaining work and risks

1. Run the full fail-closed audit on the DGX and preserve the output directory
   plus tarball unchanged.
2. Resolve every validation, missing-cell, timeout, or repository-drift error
   before comparing policies.
3. Confirm alpha 0.5 versus 0.6 sequentially on one GPU with ref1/ref2 and
   standalone chat/RAG before considering a default change.
4. CP-002 and CP-003 still require Linux GPU lifecycle/profiling evidence.
5. CP-006 still requires the authenticated process-supervisor design; the
   rejected partial PGID/signal patch must not be revived.
6. The RTX 5080 address must be refreshed before another remote sanity check.

## Exact first restart action

First, obtain the RTX 5080 host's current IP (the recorded address is
unreachable) and update the operator context. Then, for the actual evidence run,
use the DGX extension checkout and require the committed revision before launch:

```bash
cd /raid/ppesic/tmp/vllm-semantic-cache-ext
git pull --ff-only
test "$(git rev-parse HEAD)" = \
  "7b5d79697823368f1aef3cc85ae35db6917ebb0a"
./dgx_policy_audit.sh
```

Record the shell exit code and printed `Audit output:` path immediately. Do not
rerun into the same `OUTPUT_ROOT`; the runner now refuses reuse by design.
