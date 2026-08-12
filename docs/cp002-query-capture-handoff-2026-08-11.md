# CP-002 query-capture lifecycle handoff — 2026-08-11

## Objective and stop status

This session built and reviewed a fail-closed Linux GPU evidence producer for
CP-002. The intended proof is two sequential vLLM V2 engine lifetimes in one
Python process: engine A must install and exercise semantic query capture,
shut down and release every process-global patch and owned object, then engine
B must reinstall fresh state and serve without any A callback or scoring
metadata.

The implementation and offline tests are complete. The revised WSL evidence
producer is approved by Reviewer and independently validated by Tester. No
successful live lifecycle result exists. The only authenticated live engine
attempt is a historical valid failure before engine A construction. A second
attempt stopped before launch when the RTX host became unreachable.

The user issued a stop directive after that connectivity failure. Do not
reconnect, retry an experiment, reuse an evidence root, or allocate DGX as part
of this stopped session.

## Scoped implementation

The session owns these files:

- `benchmarks/run_query_capture_lifecycle_smoke.py`
- `tests/test_query_capture_lifecycle_smoke.py`
- `benchmarks/cp002_source_identity_20260811.json`
- `docs/cp002-query-capture-handoff-2026-08-11.md`

The standalone runner preserves the existing CP-002 acceptance boundary:

1. one OS process with `VLLM_ENABLE_V1_MULTIPROCESSING=0`;
2. vLLM V2 model runner, eager execution, TP/PP/context parallelism 1;
3. real prefill callbacks, resident durable summaries, and finite score
   evidence for both A and B;
4. exact shutdown restoration of class methods, owner attributes, ContextVars,
   connector state, and callback activity;
5. garbage collection of A's engine, capture handle, worker, and model runner
   before constructing B;
6. no A request or scoring identity in B and no child processes at either
   lifecycle boundary; and
7. authenticated source/archive/runner identities, atomic output ownership,
   event-first evidence writes, post-event double provenance validation, and
   pass-result publication last.

The revised WSL contract additionally:

- requires exact `VLLM_WSL2_ENABLE_PIN_MEMORY=1` before any Torch or vLLM
  import and rejects conflicting or already-imported runtime state;
- authenticates both extracted live source trees with stdlib-only hashing
  before the capability probe, runtime imports, or evidence-root claim;
- requires and records vLLM platform pin-memory and UVA availability;
- exercises a real pinned Torch `int32` CPU tensor through its CUDA UVA view
  and requires synchronized exact readback;
- constructs a real pinned vLLM `UvaBuffer`, writes through its CPU storage,
  reads through its CUDA view, and requires synchronized exact equality; and
- records kernel, platform, Python, Torch/CUDA, vLLM source/version, GPU,
  required environment, and both UVA probes in the manifest and result. The
  exact required environment is revalidated during both completion passes.

Capability, live-source, manifest, archive, or environment failure occurs
before an engine is constructed. Capability or pre-import source failure also
occurs before the output directory is claimed.

## Exact approved identities

- Extension code-under-test revision:
  `646fea09139a5a4112a650967acc0b4394ac9c6f`
- Pinned sibling vLLM revision:
  `dc1be79031d948d7a18c37600881e45ca708d913`
- Evidence producer SHA-256:
  `13e338f042ecc93d165449204aba1c52e5606da98fd523790b8cd62b4da07193`
- Offline test SHA-256:
  `745b73e4306c3027455936acc5d8f982ce4554d7e3282d39b00806808c4f6806`
- Authenticated source-identity manifest SHA-256:
  `f309c13f740f713bc72d7c7ebd67c9a7b568603886ed75bd943caf5ca3a5699a`
- Bootstrap `SOURCE_MANIFEST.txt` SHA-256:
  `f15b6b0640088a3a4ccf98b2ea151dc1c2ae3ee8a12c55842fbc8d6a6677e78c`
- Extracted extension tree SHA-256:
  `dac69d0ac08ec6099d0f6546a743899936276e1519d9fed6d59a8c60ffca5a8a`
- Extracted vLLM tree SHA-256:
  `bdb9b4cb82c661ab1ececea44fd23bd341a220764d84cc1859b374338de498ea`
- Extension archive SHA-256:
  `2e9b0991d4577d46f51d7f0f562b3da8522af6a95a5f331914520c6c2eed7a90`
- vLLM archive SHA-256:
  `5a091d0fd86be7122aeedddcd58db84885a1f47a1c0d76110bdb6f474d938da5`

Reviewer approved the exact producer/test/identity tuple above with no
remaining P0-P3 findings. That approval covers one future controlled live run
only; it is not lifecycle evidence.

## Exact local validation

The macOS sibling vLLM environment lacks the local test stack. Validation used
the extension `.venv` with the sibling checkout on `PYTHONPATH`, as permitted
by the repository preflight rules. Local Python is 3.12.13; local PyTorch is
2.13.0 without CUDA. The known sibling-source warning about absent
`vllm._version` remains.

Implementer final exact-byte gate:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:../vllm-semantic-cache" \
  .venv/bin/python -m pytest -q \
  tests/test_query_capture_lifecycle_smoke.py tests/test_query_capture.py
# 59 passed, 15 known warnings, 3.32s

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:../vllm-semantic-cache" \
  .venv/bin/python -m pytest -q tests
# 277 passed, 15 known warnings, 15.42s

uvx --from ruff ruff check \
  benchmarks/run_query_capture_lifecycle_smoke.py \
  tests/test_query_capture_lifecycle_smoke.py
# All checks passed

uvx --from ruff ruff format --check \
  benchmarks/run_query_capture_lifecycle_smoke.py \
  tests/test_query_capture_lifecycle_smoke.py
# 2 files already formatted

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  benchmarks/run_query_capture_lifecycle_smoke.py \
  tests/test_query_capture_lifecycle_smoke.py
# exit 0

git diff --check -- \
  benchmarks/run_query_capture_lifecycle_smoke.py \
  tests/test_query_capture_lifecycle_smoke.py \
  benchmarks/cp002_source_identity_20260811.json
# exit 0
```

The identity manifest's producer hash matched the actual runner. Running the
producer under optimized Python (`python -O`) exited 1 at the assertion guard.

Tester independently validated the same final hashes:

- focused: 59 passed, 15 known warnings, 3.29s;
- full suite: 277 passed, 15 known warnings, 10.22s;
- Ruff check and format check: pass;
- `py_compile`, JSON runner binding, and `git diff --check`: pass.

Reviewer independently reran the focused suite: 59 passed, 15 known warnings,
3.01s, with JSON parsing, `py_compile`, and exact-hash checks passing.

## Historical authenticated live failure: `-01`

The consumed evidence root is:

`/home/philip/semantic-cache-agent/wsl-session-20260810/evidence/cp002-lifecycle-20260811-01`

The preserved local copy is:

`dgx_logs/rtx5080/cp002-lifecycle-20260811-01/`

Exact local bundle identities:

- `manifest.json`:
  `c8beba7db7aea3ee90ddd71798505e63df5dde0f52d151e2c21913dba89a831f`
- `result.json`:
  `c2eea6e00faacb31ff1801ff10d2b3a22ba0ef17312a67dadde389686e3b401f`
- `run.log`:
  `7e275237f4985aea3648f70f2aea82df61bc81409af8e2a06935b7558ca7af26`
- local `SHA256SUMS`:
  `934fc4f6b822e061ed7ec97128d1fd935bb3beb1a26b44b6c6c4ac72bbfc4269`

The run exited 1 with `RuntimeError: UVA is not available` while vLLM V2
constructed `RequestState -> StagedWriteTensor -> UvaBuffer`. Engine A never
finished construction; semantic query capture was never installed. There are
no callback events, completion provenance, A/B lifecycle metrics, or valid
lifecycle claim. The root is permanently consumed and must never be reused,
relabeled, or treated as a query-capture cleanup failure.

The failed constructor emitted a PyTorch NCCL warning about
`destroy_process_group()` not being called. Final observed child and GPU
compute-process inventories were empty. This is evidence of an incomplete
generic vLLM constructor path, not evidence about CP-002 capture teardown.

## Accepted WSL UVA diagnostic

Tester ran a bounded diagnostic only; it did not construct a model engine or
touch an evidence root. With `VLLM_WSL2_ENABLE_PIN_MEMORY=1` established before
imports on the pinned runtime, both pin-memory and UVA helpers returned true,
a real pinned 10x10 `int32` CPU tensor obtained a CUDA UVA view, and exact
CPU/CUDA/CPU equality passed. With the flag set to 0, both helpers returned
false as expected.

Remote diagnostic log:

`/home/philip/semantic-cache-agent/wsl-session-20260810/logs/cp002-uva-diagnostic-20260811.log`

Log SHA-256:
`328d1df62288ebee97c1930ded57a9fc725c3136a9f58502fbe549f8dd6435e8`

The diagnostic proves capability for a tiny allocation only. WSL/WDDM pinned
memory remains capacity-limited; it provides no performance or large-model
claim.

## Pinned remote runtime and intended smoke configuration

The last accepted WSL runtime inventory was:

- host address at the time: `philip@192.168.4.43`;
- WSL2 kernel: `6.18.33.2-microsoft-standard-WSL2`;
- Python: 3.12.13;
- PyTorch: `2.13.0+cu130`, CUDA runtime 13.0;
- GPU: NVIDIA GeForce RTX 5080, 16,303 MiB;
- Windows NVIDIA driver: 610.88;
- vLLM: `0.26.1rc1.dev75+gdc1be7903`, from the pinned source tree;
- stable compiled extension: `vllm/_C_stable_libtorch.abi3.so`;
- platform: `NvmlCudaPlatform`.

The approved smoke remains Qwen/Qwen2.5-1.5B-Instruct at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, max model length 512, max 2
sequences, 512 batched tokens, 40 GPU blocks, 0.5 GPU memory utilization,
64 MiB CPU tier, eager mode, V2 runner, one visible GPU, and one process.

This is a correctness smoke, not a pressured policy benchmark. It has no seed
matrix, latency/throughput result, or LRU/ARC comparison.

## Stopped pre-launch `-02` attempt

After Reviewer approval and Tester's green local gate, Tester began separate
SCP transfers to these distinct Windows inbox targets:

- `C:\Users\philip\semantic-cache-agent\wsl-session-20260810\incoming\run_query_capture_lifecycle_smoke-cp002-20260811-03.py`
- `C:\Users\philip\semantic-cache-agent\wsl-session-20260810\incoming\cp002_source_identity_20260811-03.json`

During remote pre-launch hash/root/process verification, SSH failed exactly:

```text
client_loop: ssh_packet_write_poll: Connection to 192.168.4.43 port 22: Host is down
```

Tester stopped and closed the local transfer panes. The remote `-03` inbox
files are partial or unverified and must be distrusted. No launch command was
submitted. No capability probe or model engine ran. The intended remote root
and log were not created before the disconnect:

- `/home/philip/semantic-cache-agent/wsl-session-20260810/evidence/cp002-lifecycle-20260811-02`
- `/home/philip/semantic-cache-agent/wsl-session-20260810/logs/cp002-lifecycle-20260811-02.log`

Because the host became unreachable, a future session must independently
confirm their absence rather than relying on this observation. There is no
local `dgx_logs/rtx5080/cp002-lifecycle-20260811-02/` bundle.

## Process, repository, and dirty-tree state at stop

No local lifecycle runner, vLLM, pytest, SCP, or CP-002 transfer process was
running at the final process audit. The initially observed PID 19590
(`ssh -o ConnectTimeout=10 philip@192.168.4.43`, parent zsh 19513, tmux parent
14259) belonged to Tester's CP-002 diagnostic pane `%5`. Tester closed only
that attributable pane, without reconnecting or sending a remote command. A
final process search found no matching SSH, SCP, runner, pytest, vLLM, or
CP-002 transfer process.

The extension branch started at `master` revision
`646fea09139a5a4112a650967acc0b4394ac9c6f`, equal to `origin/master` before
the scoped stop commit. The sibling vLLM checkout remains at
`dc1be79031d948d7a18c37600881e45ca708d913` and independently dirty/divergent
(`main...origin/main [ahead 19193, behind 18627]`). It was read only throughout
this work.

Preserve these pre-existing user-owned extension paths; they were not edited,
staged, deleted, or reformatted by CP-002 work:

```text
 M AGENTS.md
 M semantic_offload/manager.py
 M sync-vllm-upstream.sh
?? .DS_Store
?? .codex/
?? 2.6-result.csv
?? 2.6-second-pass-result.csv
?? README.md
?? docs/agent-session-difficulties-2026-08-10.md
?? docs/repository-workflow.md
```

## Policy evidence boundary

This session produced no semantic-versus-LRU/ARC measurement and no DGX
allocation. The historical policy figures in the 2026-08-10 policy-audit
handoff remain historical, lack a current fail-closed raw bundle, and cannot
be promoted to a new competitive, latency, throughput, preemption, or default
change claim. Missing CP-002 lifecycle and policy data are unavailable, not
zero.

## Exact first restart action

First ask the operator for the RTX desktop's current IP and confirmation that
Windows/WSL SSH is reachable. Then perform read-only connectivity and inventory
only; do not launch the smoke as the first restart action.

Once connectivity is restored in a separately authorized session:

1. distrust and hash both partial/unverified `-03` inbox files; retransfer the
   approved files with SCP if either exact digest is absent or mismatched;
2. verify the runner, identity manifest, bootstrap manifest, extracted source
   trees, archives, Python/vLLM/CUDA/GPU runtime, required environment, child
   inventory, and empty GPU compute-process inventory;
3. independently confirm that the proposed output root and log are absent. If
   either exists, do not reuse it; request a new explicit root;
4. obtain fresh authorization for exactly one bounded run; and only then
   execute the approved producer.

The exact approved runner/test/identity hashes are recorded above. Never reuse
historical `cp002-lifecycle-20260811-01`, never infer that `-02` is fresh from
this stopped session, and do not allocate DGX implicitly. If a future revised
WSL engine run fails, preserve the failure and return a DGX decision packet.
