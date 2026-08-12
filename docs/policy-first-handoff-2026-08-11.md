# Policy-first work handoff — 2026-08-11

## Stop status

The user stopped the policy-first work loop after the first new candidate was
falsified and reverted and after the second candidate reached a locally tested
prototype. No local or remote experiment is running. Do not resume an RTX,
CP-002, compiler, or DGX lane from this handoff.

The current product decision is unchanged:

- production `semantic-mean` remains at fixed `alpha=0.5`;
- fixed `alpha=0.6` remains an opt-in experimental control, not a default;
- `adaptive_blend` was rejected and fully removed; and
- the EMA negative-evidence floor is uncommitted WIP without policy-quality
  evidence or Reviewer approval.

## Repository and dirty-tree boundary

The extension checkout is `master` at
`c5f7c0bb69611be2ca1e4b23f916c547d82a8742`, equal to `origin/master` at
stop. No file was staged, committed, or pushed during this policy loop.

Pre-existing user-owned paths remain preserved:

```text
 M AGENTS.md
 M semantic_offload/manager.py       # user docstring hunk plus WIP below
 M sync-vllm-upstream.sh
?? .DS_Store
?? .codex/
?? 2.6-result.csv
?? 2.6-second-pass-result.csv
?? README.md
?? docs/agent-session-difficulties-2026-08-10.md
?? docs/repository-workflow.md
```

The manager's lines 2–8 contain a pre-existing user-owned docstring update.
It is separate from the WIP EMA-floor hunks and must not be staged implicitly.
The sibling `../vllm-semantic-cache` checkout remains a read-only, independently
dirty/divergent dependency at
`dc1be79031d948d7a18c37600881e45ca708d913`.

## Current offline control screen

Tester compared the real vLLM `LRUCachePolicy` and `ARCCachePolicy` classes
with real `SemanticPolicy` fixed-alpha arms on a pressured bookkeeping-only
screen. Each capacity has 36 independent structural cases (nine target/decoy
recency-stratum pairs times four rotations); 25 timing repeats are not
independent accuracy trials. Each case inserted 32 blocks, touched every block
once, and evicted to capacity 8, 16, or 24. Only four of 32 blocks were scored.
There are no model-attention labels and no serving latency/throughput data.

Target complete/fraction, stale-decoy complete/fraction, and median offline
eviction time were:

| Capacity | LRU | ARC | alpha 0.5 | alpha 0.6 |
| --- | --- | --- | --- | --- |
| 8 | .333/.333; .333/.333; .00275 ms | same preservation; .01908 ms | .333/.389; 0/0; .01321 ms | .444/.556; 0/0; .01313 ms |
| 16 | .500/.500; .500/.500; .00188 ms | same preservation; .00929 ms | .778/.861; .0556/.1389; .01125 ms | 1.000/1.000; 0/0; .01125 ms |
| 24 | .667/.667; .667/.667; .00104 ms | same preservation; .00350 ms | 1.000/1.000; .5556/.6111; .00929 ms | 1.000/1.000; .3333/.4444; .00933 ms |

ARC is non-discriminating in this workload because it has no repeated access
or ghost hit. These rows are not evidence that ARC and LRU are generally equal.
Offline nanosecond timings are directional policy-operation costs, not serving
performance.

The later clean structural artifact also exposed a material novel-topic proxy
tradeoff at capacity 8: fixed alpha 0.6 retained novel blocks at complete rate
`.333` and fraction `.444`, versus `.556/.611` for alpha 0.5; LRU and ARC were
`.444/.444`. At capacities 16 and 24 all variants were 1.0 on that proxy.
Therefore the current screen supports keeping alpha 0.6 opt-in, not promoting
it.

The original aggregate transcript is warning-contaminated and must not be
parsed as clean JSONL:

`dgx_logs/policy-screen-20260811/alpha06-baseline-screen.jsonl`

SHA-256:
`e5bd55d5890a316547b24fe2892ca69a6a7a28b93c0c0bc3915d6497a64f72bc`.

## Rejected candidate 1: `adaptive_blend`

Hypothesis: choose an effective alpha in `[0.5, 0.6]` from the normalized
relevance gap at the relevance-only eviction/survival boundary, falling back
to 0.5 when score coverage is incomplete.

The successful screen contained 108 rows (36 structural cases times three
capacities). Score coverage was only 4/32, so the fail-safe selected alpha 0.5
in all 108 rows and exactly reproduced the fixed 0.5 survivor set. Relative to
fixed 0.6:

- capacity 8: four of 36 complete-target losses and target-fraction delta
  `-0.1667`, with no decoy benefit;
- capacity 16: eight of 36 complete-target losses, target-fraction delta
  `-0.1389`, and stale-decoy fraction `+0.1389`;
- capacity 24: target tie and stale-decoy fraction `+0.1667`.

The candidate failed the agreed target/decoy frontier gate. It was reverted
instead of being rescued by a riskier partial-coverage rule. Reviewer and
Brainstormer agreed: reject it, do not run a full suite for it, and spend no
GPU budget on it.

The valid current offline synthetic candidate-selection artifact is:

`dgx_logs/policy-screen-20260811/alpha06-adaptive-structural-cases-v3.jsonl`

- 108 clean JSON lines;
- SHA-256
  `3f83eb926278ee5e445d0d5242557713da9a357d06cc4fe484aeb94382846bb2`;
- warnings file SHA-256
  `f89b0d8baa6ee45b490881f8140c5b6f89c07965ceb266c7c3de55daf026808c`.

Invalid historical instrumentation attempts remain preserved and must not be
used as results:

- v1 JSON SHA
  `30b4fb81518aa8a7bc5fda2172e215d0be2524c21db11f31ad22b6d623a60a3b`,
  warnings SHA
  `b1c17067ebdf31cf0f2937fd1702c9624693ea3324ca1b6fd0a4367eec70c7a1`;
- v2 empty JSON SHA
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
  warnings SHA
  `bfb0bb53bc19b14e87483bb0e11ec4609ca8bc8c79364862c9c3c22215e899e3`.

Adaptive code has zero residue. Exact post-revert identities are:

- `semantic_offload/policy.py`:
  `8eebce3d59ee27d3513606f7ec5a3d8d9a7353ef146bc135ba0501d657e9353a`;
- `tests/test_step_1_4_eviction.py`:
  `81fe4a149124e66012f6133be8ea03e762ba4939558e843b6a7197849f29f198`.

No `adaptive_blend`, `MAX_ADAPTIVE`, or `_effective_alpha` symbol remains.

## Candidate 2 WIP: negative-evidence EMA floor

This is an uncommitted, unstaged, locally tested prototype only. No structural
or model-trace comparison has run, and there is no keep/revert or Reviewer
policy-quality disposition.

Hypothesis: the current rank-last EMA weight is exactly zero, so stale high
relevance can persist indefinitely. A small absolute per-query rank-weight
floor may preserve a strong target through a short unrelated burst while
allowing sustained negative evidence to age stale targets/decoys and recover
novel-topic adaptation. The default remains exactly zero. Candidate values are
intended to be swept over `0.01/0.03/0.05/0.1`, bounded by the existing 0.3
EMA ceiling.

The WIP currently:

- validates `ema_rank_weight_floor` in `[0, 0.3]`;
- applies the same floor to sequential and compact relevance updates;
- threads the opt-in through spec, scheduler-side TP reduction, and worker-side
  TP=1 compact metadata;
- preserves first-observation behavior and floor-zero legacy behavior; and
- adds a local formula check showing floor 0.01 retains a rank-last target
  above 0.92 after eight low observations and ages it below 0.53 after 64.

Brainstormer's pre-stop gate does not select a floor winner. The first screen
must keep decay conditional on explicit ranked observations: keys absent from a
ranked list do not decay. At horizons 0/1/4, target complete/fraction may be no
more than two percentage points below same-alpha floor zero and the
no-negative-evidence control must be unchanged. At horizons 16/64, a candidate
must reduce stale retention and improve novel preservation by at least ten
points in two capacities without losing more than two points on periodically
reaffirmed or multi-block targets. Update-plus-evict p50/p95 overhead must stay
within five percent. These are planned falsification gates, not achieved
results.

Exact WIP files and SHA-256 values:

- `semantic_offload/manager.py`:
  `a23101a73055dda36d9910eb77e217fc23d1eb8edf90551a926334fd363434f9`;
- `semantic_offload/connector.py`:
  `8ffcfe5b5d46eb6b44a51993bd4d848bd6c1eb0eb5e0e792543b58dd6d864a5b`;
- `semantic_offload/spec.py`:
  `c57eaac41188449b4046531bf559fd315c046dc570f328817df4e9ccf96b8069`;
- `tests/test_step_1_3_scoring.py`:
  `21478e3d19986a995ba5899d022cb476f9882d07b02e2841a61d26ffa7c48b2c`;
- `tests/test_step_1_1_smoke.py`:
  `6bfa7f3408804bc2742975df7d37dd9c9cd504c7e5ee464434189814fcab9860`.

The scoped diff is 150 insertions and 15 deletions across those five files;
the apparent manager deletion includes the preserved user docstring hunk and
is not wholly candidate-owned. Do not stage the manager file wholesale.

## Validation at stop

Local validation uses the extension `.venv` with the sibling source on
`PYTHONPATH` because the sibling runtime environment lacks the local test
stack. The known missing `vllm._version` warning and existing Torch warnings
remain.

Adaptive no-residue gate:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:../vllm-semantic-cache" \
  .venv/bin/python -m pytest -q tests/test_step_1_4_eviction.py
# 29 passed, 1 known warning, 1.39s (Implementer)
# 29 passed, 1 known warning, 1.38s (Tester)
```

Candidate 2 WIP focused gate:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:../vllm-semantic-cache" \
  .venv/bin/python -m pytest -q \
  tests/test_step_1_3_scoring.py tests/test_step_1_1_smoke.py
# 67 passed, 15 known warnings, 8.51s

uvx --from ruff ruff check \
  semantic_offload/manager.py semantic_offload/spec.py \
  semantic_offload/connector.py tests/test_step_1_3_scoring.py \
  tests/test_step_1_1_smoke.py
# All checks passed

uvx --from ruff ruff format --check \
  semantic_offload/manager.py semantic_offload/spec.py \
  semantic_offload/connector.py tests/test_step_1_3_scoring.py \
  tests/test_step_1_1_smoke.py
# 5 files already formatted

git diff --check -- \
  semantic_offload/manager.py semantic_offload/spec.py \
  semantic_offload/connector.py tests/test_step_1_3_scoring.py \
  tests/test_step_1_1_smoke.py
# exit 0
```

The first focused attempt exposed one missing default fallback on a
`__new__`-constructed connector (`1 failed, 66 passed`); that WIP bug was
fixed before the 67-pass result. No full suite was run on candidate 2 because
the user stopped before its policy comparison and review.

Final process inspection found no attributable pytest, policy-screen, vLLM,
SSH/SCP, Triton, or remote-transfer process. No process was killed. No RTX or
DGX resource was used in this policy loop.

## CP-002 and infrastructure boundary

Historical CP-002 `-01` and `-02` remain consumed authenticated failures with
no query-capture lifecycle claim. `-02` passed WSL pin/UVA capability and
failed during first-engine KV profiling because Triton could not find a C
compiler. Compiler discovery is a secondary, read-only lane and did not change
during this policy loop. Do not install a compiler, retry either root, reconnect
to RTX, or allocate DGX from this handoff.

## Exact restart action

Restart locally, not on the compiler lane:

1. read this handoff and recheck the five candidate-2 WIP hashes and `git
   status` without staging the manager docstring;
2. have Reviewer examine the preserved floor-zero equivalence, raw/compact/TP
   plumbing, validation, and first-observation semantics;
3. run the cheap paired phase screen across floors `0/.01/.03/.05/.1`, fixed
   semantic alpha 0.5/0.6, real LRU, and real ARC on identical traces;
4. include target, novel-topic, stale-decoy, operation cost, pressure, and a
   repeated-access/scan trace that actually promotes ready keys from ARC T1 to
   T2, then performs a ghost re-reference that changes `target_t1_size`; record
   the ARC partition/ghost counters to prove the comparator discriminated; and
5. keep and finish the WIP only if a floor preserves short-horizon targets,
   improves long-horizon stale/novel adaptation, and remains competitive with
   both LRU and ARC. Otherwise revert it and advance to the bounded decayed-
   session hypothesis.

Do not present the WIP, the adaptive rejection, or infrastructure progress as
a live policy-performance improvement.
