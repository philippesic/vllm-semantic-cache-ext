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


# bucket -> [cumulative_sum, cumulative_count, window_sum, window_count, window_max]
_count_state: dict[str, list] = {}


def record_count(bucket: str, value: int) -> None:
    """Same accumulate-and-summarize-every-N-calls shape as `record_timing`,
    but for a size/count value instead of a duration -- added 2026-08-01 to
    test whether `_on_queries_captured`'s per-call cost growth (SEMANTIC_TIMING
    bucket=query_captured_total climbing 2.4x within a single rag@8.0 run,
    07-30/08-01 handoffs) tracks a growing concurrent-request batch size
    (`len(req_ids)`) rather than the resident candidate pool, which stayed
    flat (`resident=` in SEMANTIC_EVICT_DEBUG) across the same run. Prints
    mean AND max per window so a spiky-but-flat-on-average batch size is
    still visible. No-op unless SEMANTIC_OFFLOAD_TIMING is set (reuses that
    flag rather than adding a third one)."""
    if not TIMING:
        return
    slot = _count_state.setdefault(bucket, [0, 0, 0, 0, 0])
    slot[0] += value
    slot[1] += 1
    slot[2] += value
    slot[3] += 1
    slot[4] = max(slot[4], value)
    if slot[1] % _TIMING_EVERY == 0:
        total, count, window_total, window_count, peak = slot
        print(
            f"SEMANTIC_COUNT bucket={bucket} pid={os.getpid()} calls={count} "
            f"cumulative_mean={total / count:.2f} "
            f"window_mean={window_total / window_count:.2f} window_max={peak}",
            flush=True,
        )
        slot[2:] = [0, 0, 0]


# bucket -> call_count. Only the count is kept; the allocator stats
# themselves are read fresh on each print step, never cached.
_gpu_mem_state: dict[str, int] = {}


def record_gpu_memory(bucket: str) -> None:
    """Snapshot the CUDA caching allocator's fragmentation-relevant counters
    on the same every-N-calls cadence as record_timing/record_count -- added
    2026-08-02 to test the allocator-overhead theory (08-02 handoff): both
    logical sizes (resident pool, concurrent batch) are flat, yet
    query_captured_total's per-call cost still climbs ~2.4x within one rag@8.0
    run, so the remaining suspect is allocator fragmentation from
    _rebuild_stack_cache's fresh torch.cat/boolean-mask tensor per dirty step.
    The stats are queried ONLY on print steps (every _TIMING_EVERY calls), not
    every call, so this adds no per-call cost to the timed region it sits
    beside. memory_stats() is a host-side read of allocator bookkeeping -- no
    device sync. No-op unless SEMANTIC_OFFLOAD_TIMING is set, or if CUDA is
    unavailable. Remove once the investigation closes."""
    if not TIMING:
        return
    count = _gpu_mem_state.get(bucket, 0) + 1
    _gpu_mem_state[bucket] = count
    if count % _TIMING_EVERY != 0:
        return
    import torch

    if not torch.cuda.is_available():
        return
    stats = torch.cuda.memory_stats()
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    # reserved-but-inactive-and-split is the caching allocator's fragmented
    # slack; a climbing value alongside num_alloc_retries is the signature.
    inactive_split = stats.get("inactive_split_bytes.all.current", 0)
    retries = stats.get("num_alloc_retries", 0)
    print(
        f"SEMANTIC_GPUMEM bucket={bucket} pid={os.getpid()} calls={count} "
        f"allocated_mb={allocated / 1048576:.1f} "
        f"reserved_mb={reserved / 1048576:.1f} "
        f"inactive_split_mb={inactive_split / 1048576:.1f} "
        f"num_alloc_retries={retries}",
        flush=True,
    )


_process_state_calls: dict[str, int] = {}


def record_process_state(bucket: str) -> None:
    """Emit low-frequency CPU memory and GC state for growth diagnosis."""
    if not TIMING:
        return
    count = _process_state_calls.get(bucket, 0) + 1
    _process_state_calls[bucket] = count
    if count % _TIMING_EVERY != 0:
        return
    import gc

    rss_mb = -1.0
    try:
        with open("/proc/self/statm", encoding="utf-8") as statm:
            resident_pages = int(statm.read().split()[1])
        rss_mb = resident_pages * os.sysconf("SC_PAGE_SIZE") / 1048576
    except (FileNotFoundError, IndexError, OSError, ValueError):
        pass
    generations = gc.get_stats()
    collections = ",".join(str(item["collections"]) for item in generations)
    collected = ",".join(str(item["collected"]) for item in generations)
    print(
        f"SEMANTIC_PROCESS bucket={bucket} pid={os.getpid()} calls={count} "
        f"rss_mb={rss_mb:.1f} gc_count={','.join(map(str, gc.get_count()))} "
        f"gc_collections={collections} gc_collected={collected}",
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
