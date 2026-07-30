# Session Handoff — 2026-07-29

Written at the end of a session that moved primary development from a
2080Ti PC to a MacBook Pro (M3 Max, no NVIDIA GPU) + a DGX B200 box for
GPU work. Covers: getting the DGX environment working from scratch,
new remote-test tooling, and a real bug found and fixed in the partial-
splice feature. A `benchmarks/run_grid_sweep.py` (Step 1.6) run was
kicked off at the end of this session; results pending.

## Current environment state

**Mac** (`/Users/pippo/github/vllm-semantic-cache` +
`/Users/pippo/github/vllm-semantic-cache-ext`, sibling directories):
no GPU. Used for git/tooling work only — never runs real-server tests.

**DGX B200** (`/raid/ppesic/tmp/vllm-semantic-cache` +
`/raid/ppesic/tmp/vllm-semantic-cache-ext`, sibling directories, same
layout): 8x B200, where all real-server work happens.

**`vllm-semantic-cache` (the vLLM fork) `main` branch, on both
machines:** hard-pinned to commit `dc1be79031d948d7a18c37600881e45ca708d913`
("Add CachePolicyFactory for pluggable/external eviction policies
(#49114)") — **not** the live upstream tip. This is deliberate: it's the
newest commit with a published `x86_64`+`cu130` precompiled wheel as of
this session (x86_64 wheel publishing on wheels.vllm.ai was observed
lagging aarch64 by 30+ commits / about a day). It also happens to be the
exact commit where the user's own `CachePolicyFactory` PR landed
upstream, so nothing was sacrificed by not chasing the tip.

Local `main` was synced via a **hard reset**, not a merge — this fork's
`main` and real upstream only share a Feb-2023 common ancestor, so a
real `git merge upstream/main` produces large unrelated conflicts in
core files (`gpu_worker.py`, `gpu_model_runner.py`, etc.) that have
nothing to do with this project. Verified safe (zero commits on `main`
actually authored by the user before resetting). **Not pushed to
`origin/main`** on either machine — local-only.

## Tooling added this session (in `vllm-semantic-cache-ext`, on `origin/master`)

### `sync-vllm-upstream.sh`
Safely syncs the sibling `vllm-semantic-cache` repo's `main` to
upstream — walks backward from `upstream/main`'s tip checking
`wheels.vllm.ai` for a commit with a published wheel matching this
machine's architecture/CUDA variant (via `uname -m` + `CUDA_VARIANT` env,
default `cu130`) before resetting, instead of blindly landing on a tip
that might not have a wheel yet. Includes a safety check (refuses to
reset if `main` has any commit authored by the user that upstream
doesn't have) and hard-resets rather than merges, for the reason above.

```bash
./sync-vllm-upstream.sh                 # repo at ../vllm-semantic-cache
VLLM_REPO=/path/to/vllm-semantic-cache ./sync-vllm-upstream.sh
CUDA_VARIANT=cu128 ./sync-vllm-upstream.sh
SKIP_WHEEL_CHECK=1 ./sync-vllm-upstream.sh
```

After running it, rebuild: `cd ../vllm-semantic-cache &&
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`.

### `test-dgx.sh`
Numbered menu of every GPU-dependent test/benchmark, meant to be driven
remotely — say "run option N" and paste back the terminal output rather
than copy-pasting full commands each time.

```bash
./test-dgx.sh            # interactive menu
./test-dgx.sh 1 3 4       # run specific options in order, no prompts
./test-dgx.sh all         # runs 1-6 (option 0 is always explicit-only)
```

| # | What | Notes |
|---|------|-------|
| 0 | Kill stray vllm processes, free GPU memory | Explicit only, never part of `all` — box-wide, not scoped to "yours" |
| 1 | Sanity check | nvidia-smi, torch/CUDA, vllm, `semantic_offload` import; auto-installs/fixes pytest, torchaudio, flashinfer if broken (see gotchas below) |
| 2 | Unit tests | `pytest tests/`, CPU-only logic |
| 3 | Latency suite smoke | Real server, lru/arc/semantic-minmax/semantic-mean, ~minutes |
| 4 | Adaptive splice live-probe | The byte-exact splice-correctness proof driver, ~minutes |
| 5 | Recall/cost experiments | Re-runs Step 0.3/0.4/1.4 studies with real CUDA timing |
| 6 | Grid sweep | Step 1.6 rigorous multi-seed benchmark, LONG (hours), all GPUs in `$GPUS` |

Options 3/4/6 each: pre-clean stray vllm processes, pin to whichever GPU
currently has the most free memory (`launch_server` never sets
`CUDA_VISIBLE_DEVICES` itself, so this matters — it'll otherwise always
try device 0 regardless of what's actually free), fail fast with a clear
diagnostic if no GPU has enough headroom, and post-clean afterward
regardless of success/failure. All output is teed to
`dgx_logs/<n>_<name>_<timestamp>.log` (gitignored) in case terminal
scrollback isn't enough.

## The bug: dead partial-splice pipeline (found and fixed this session)

**Symptom:** `4_splice_probe` (the byte-exact correctness proof driver)
produced zero `PARTIAL SPLICE`/`KEY MISMATCH` debug markers across
several consecutive real-server runs, despite three rounds of workload
tuning (block budget, `max_model_len`, filler decode length) aimed at
forcing real GPU contention.

**Root cause (not a workload problem):** vLLM removed the scheduler-side
`on_request_preempted` hook this connector's entire prefetch pipeline
was gated behind (confirmed zero matches anywhere in the current `vllm/`
tree at `dc1be79031` — replaced by `SchedulerOutput.preempted_req_ids`,
read from inside `build_connector_meta`, plus an unrelated worker-side
`handle_preemptions` callback). `_preempted_pending` was therefore never
populated, so every workload-tuning attempt was fighting a dead code
path regardless of whether real preemptions occurred.

**Fix** (`semantic_offload/connector.py`, commit `cc362df`): added
`_queue_preempted(scheduler_output)`, called first thing in
`build_connector_meta`, seeding `_preempted_pending` from
`scheduler_output.preempted_req_ids` — same queue-only contract the old
hook had. `on_request_preempted` itself was left in place unmodified
(it has its own direct unit tests and is still correct in isolation,
just no longer vLLM's actual trigger). New unit tests added in
`tests/test_step_1_5_prefetch.py`.

**Proof it works:** first real-server retry after the fix produced a
full pass of the original `step-1.5-partial-splice-plan.md` acceptance
check — `spliced=1 reloaded=2` in one re-admission, byte-exact-correct
spot-checked recall. Full writeup:
`.claude/docs/step-1.5-partial-splice-live-proof.md` (commit `59f295d`).

## Environment gotchas (all now auto-fixed by `test-dgx.sh` option 1, documented here for context)

- **`torchaudio`/`torch` CUDA build mismatch:** `transformers` pulls
  `torchaudio` in transitively; if its CUDA build doesn't match torch's,
  import raises `RuntimeError` (not `ImportError`), which
  `transformers`' normal missing-torchaudio guard doesn't catch, and it
  crashes every real-server launch. Fix: uninstall `torchaudio` entirely
  (not needed for text-only serving).
- **`flashinfer-python`/`flashinfer-cubin` version mismatch:**
  `flashinfer-cubin` (the precompiled kernel binaries) trails
  `flashinfer-python` (the pure-Python package) on PyPI. Fix direction
  matters — downgrade `flashinfer-python` to match whatever cubin is
  already installed, **never** the reverse (upgrading cubin to match a
  brand-new python release 404s, the build doesn't exist yet).
- **Single-GPU tests default to device 0** regardless of which GPU is
  actually free, since `harness/server.py`'s `launch_server` never sets
  `CUDA_VISIBLE_DEVICES`. `test-dgx.sh` options 3/4 handle this now;
  if writing a new ad-hoc script against `harness/server.py` directly,
  remember to set it yourself.
- **`vllm serve` needs enough KV cache blocks to serve one full
  `max_model_len` request**, regardless of what real traffic actually
  needs — an aggressive `num_gpu_blocks_override` combined with a large
  `max_model_len` fails engine startup with a `ValueError`, not a
  runtime error. Shrink `max_model_len` to match real workload needs
  before tightening the block override.

## Open items / next steps

1. **In progress:** `benchmarks/run_grid_sweep.py` (Step 1.6), results
   pending from the user.
2. Capture the *aggregate* graded splice-benefit metric
   (Σspliced/Σ(spliced+reloaded)) across a longer run vs. a
   splice-disabled control — the one proof recorded so far is a single
   data point (0.33 coverage on one request), not a trend.
3. Investigate the climbing filler-failure count seen in the later
   `4_splice_probe` runs (0→5→7→11 failed out of 44, no error surfaced
   server- or client-side — the harness swallows the actual exception).
   Best guess: the tight 40-block budget causing some fillers to queue
   long enough under real contention to hit the harness's 230s
   client-side result timeout. Didn't block the proof; worth a look if
   it recurs or grows further.
4. Low priority: the project's master plan/issues-log docs
   (`semantic-eviction-plan.md`, `semantic-eviction-issues-log.md`,
   referenced constantly by name in code comments) don't exist in git
   history in either repo — never committed, not lost in transit. If
   local copies exist anywhere (old PC, notes app), worth recovering
   and committing before more context erodes.
