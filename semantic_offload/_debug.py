# SPDX-License-Identifier: Apache-2.0
"""Debug-print gating for the hot-path prints scattered through
worker.py/connector.py/query_capture.py (every query-capture event, every
splice attempt) -- these were unconditional `print(..., flush=True)` since
Step 1.3, fine for the small-scale dev testing they were built for, but at
Step 1.6's benchmark scale (~1000+ runs) the per-call I/O flush and
unstructured stdout volume would both cost real time and pollute logs
(issues log entry #34). Default off; set SEMANTIC_OFFLOAD_DEBUG=1 to
restore the old always-on behavior for interactive debugging."""

import os

ENABLED = os.environ.get("SEMANTIC_OFFLOAD_DEBUG", "") not in (
    "",
    "0",
    "false",
    "False",
)

# Scoped timing instrumentation for the sustained-concurrent-load TTFT
# investigation (issues log open item #1). Unlike SEMANTIC_OFFLOAD_DEBUG (which
# prints one line per event -- far too noisy to leave on under real load), this
# accumulates per-bucket wall time and call counts and emits a compact summary
# line every SEMANTIC_OFFLOAD_TIMING_EVERY calls of each bucket. Worker and
# scheduler run in separate processes, so each keeps its own independent
# accumulators -- a bucket only ever sees calls from one process. Default off;
# set SEMANTIC_OFFLOAD_TIMING=1 to enable. Remove once the investigation closes.
TIMING = os.environ.get("SEMANTIC_OFFLOAD_TIMING", "") not in (
    "",
    "0",
    "false",
    "False",
)
_TIMING_EVERY = int(os.environ.get("SEMANTIC_OFFLOAD_TIMING_EVERY", "2000") or 2000)
# bucket -> [total_seconds, call_count]
_timing_state: dict[str, list] = {}


def record_timing(bucket: str, dt: float) -> None:
    """Accumulate `dt` seconds under `bucket`; print a cumulative summary
    (total time, call count, mean ms/call) every `_TIMING_EVERY` calls of
    that bucket. No-op unless SEMANTIC_OFFLOAD_TIMING is set."""
    if not TIMING:
        return
    slot = _timing_state.setdefault(bucket, [0.0, 0])
    slot[0] += dt
    slot[1] += 1
    if slot[1] % _TIMING_EVERY == 0:
        total, count = slot[0], slot[1]
        print(
            f"SEMANTIC_TIMING bucket={bucket} pid={os.getpid()} calls={count} "
            f"total_s={total:.3f} mean_ms={1000.0 * total / count:.4f}",
            flush=True,
        )


# TEMPORARY diagnostic toggle (issues log entry #53's follow-up): a real
# B200 run showed semantic-minmax causing MORE GPU preemptions than lru
# under an identical, tight-capacity config (17 vs 5), and each preempted
# request's real readmission wait (hundreds of ms to ~1.6s) accounts for
# most of the measured TTFT gap. Leading hypothesis: the prefetch/
# reservation mechanism speculatively holds GPU blocks aside for preempted
# requests, taking capacity away from currently-running ones and causing
# more preemptions than would happen without it. Set
# SEMANTIC_OFFLOAD_DISABLE_PREFETCH=1 to test that directly -- makes
# on_request_preempted a no-op (matching the base KVConnectorBase_V1
# default lru gets), so requests only ever resolve via normal vLLM
# readmission, with scoring still fully active. Remove once confirmed.
DISABLE_PREFETCH = os.environ.get("SEMANTIC_OFFLOAD_DISABLE_PREFETCH", "") not in (
    "",
    "0",
    "false",
    "False",
)


def debug_print(*args, **kwargs) -> None:
    if ENABLED:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)
