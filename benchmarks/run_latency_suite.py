# SPDX-License-Identifier: Apache-2.0
"""Phase 1 latency/throughput suite (plan's Benchmarking section, Step
1.6). For each (policy, workload, request_rate) combination: launch a real
server with that policy, run the workload (`vllm bench serve` for
chat/rag/longdoc, the bespoke needle_workload for `needle`, concurrent
chat/rag/longdoc + needle sub-streams for `mixed` -- the plan's "headline
workload"), snapshot the offload-subsystem Prometheus counters before/
after, tear the server down, and append one combined result row (or, for
`mixed`, one row per sub-stream) to a CSV.

This deliberately wraps `vllm bench serve` rather than reimplementing a
load generator -- it already does open-loop Poisson arrivals, TTFT/TPOT/
ITL percentiles, and (via `prefix_repetition`) the `rag` workload's
shared-prefix structure, per the plan's own "wrap, don't rebuild" guidance.

Usage (small-dev-model validation run, matching this project's real
launch configs throughout the issues log):

  python benchmarks/run_latency_suite.py \\
      --model Qwen/Qwen2.5-1.5B-Instruct \\
      --policies lru,semantic-minmax \\
      --workloads chat \\
      --request-rates 2.0 \\
      --num-prompts 20 \\
      --scale 0.05 \\
      --cpu-bytes-to-use 268435456 \\
      --output-dir /tmp/latency_suite_smoke
"""

import argparse
import concurrent.futures
import csv
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import metrics as metrics_mod
from harness import needle_workload
from harness import policies as policies_mod
from harness import workloads as workloads_mod
from harness.server import launch_server

RESULT_FIELDS = [
    "policy",
    "seed",
    "workload",
    "sub_workload",
    "request_rate",
    "reference_count",
    "num_prompts",
    "duration_s",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "itl_p50_ms",
    "itl_p90_ms",
    "itl_p99_ms",
    "throughput_tok_s",
    "needle_hit_rate",
    # needle-v2's preservation signal (see needle_workload.classify_needle_
    # outcome): the recall's isolated CPU-tier interaction, not a string
    # match. `needle_outcome` in {hit, miss, not_pressured}; the two byte
    # columns are the recall-only counter deltas it was derived from.
    "needle_outcome",
    "recall_load_bytes",
    "recall_store_bytes",
    "load_bytes_delta",
    "store_bytes_delta",
    "preemptions_delta",
    "error",
]

# `mixed`: the plan's "headline workload" (§Benchmarking: "all four
# interleaved") -- proportions chosen so chat/rag dominate (the realistic
# high-arrival-rate and shared-prefix-reuse cases) with longdoc present at
# lower weight (its own arrival is rarer and each request is much larger).
_MIXED_SUBWORKLOAD_WEIGHTS = {"chat": 0.40, "rag": 0.35, "longdoc": 0.25}


def _seed_tag(seed: int | None) -> str:
    """Filesystem-safe tag for raw-result filenames -- without this, two
    cells sharing a policy/workload/rate but different seeds (the plan's
    own >=3-seed protocol, not optional) silently overwrite each other's
    raw JSON, and concurrently-running cells (multi-GPU dispatch) could
    race on the same file mid-write."""
    return f"seed{seed}" if seed is not None else "seedNone"


def parse_vllm_bench_result(result_json_path: str) -> dict:
    """Pure, unit-testable: extract the fields we care about from a real
    `vllm bench serve --save-result` JSON file into our flat row shape."""
    with open(result_json_path) as f:
        data = json.load(f)
    return {
        "duration_s": data.get("duration"),
        "ttft_p50_ms": data.get("p50_ttft_ms"),
        "ttft_p90_ms": data.get("p90_ttft_ms"),
        "ttft_p99_ms": data.get("p99_ttft_ms"),
        "itl_p50_ms": data.get("p50_itl_ms"),
        "itl_p90_ms": data.get("p90_itl_ms"),
        "itl_p99_ms": data.get("p99_itl_ms"),
        "throughput_tok_s": data.get("output_throughput"),
    }


def resolve_num_prompts(
    request_rate: float,
    num_prompts: int | None,
    target_duration_s: float | None,
    min_num_prompts: int | None = None,
) -> int:
    """Pure, unit-testable: either an explicit --num-prompts, or a count
    derived from a target steady-state duration at the given request rate
    (--target-duration-s) -- a flat --num-prompts produces very different
    steady-state windows at different arrival rates, which the plan's own
    protocol (>=10 min steady-state per configuration) doesn't allow (see
    the Step 1.6 grid-trim proposal's follow-up #2).

    `min_num_prompts`: some datasets have their own hard floor (e.g. `rag`'s
    `prefix_repetition` needs num_requests >= num_prefixes,
    `workloads.RAG_NUM_PREFIXES`) that a derived count can silently fall
    below at a low rate/short duration -- raise here, before launching
    `vllm bench serve`, instead of letting that cell crash with a
    less legible error deep inside the benchmark subprocess."""
    if target_duration_s is not None:
        if request_rate == float("inf"):
            raise ValueError(
                "--target-duration-s requires a finite --request-rate "
                "(inf sends everything at t=0, duration is undefined)"
            )
        resolved = max(1, round(target_duration_s * request_rate))
    elif num_prompts is not None:
        resolved = num_prompts
    else:
        raise ValueError("either --num-prompts or --target-duration-s is required")
    if min_num_prompts is not None and resolved < min_num_prompts:
        raise ValueError(
            f"resolved num_prompts ({resolved}) is below this workload's "
            f"minimum ({min_num_prompts}) at request_rate={request_rate}, "
            f"num_prompts={num_prompts}, target_duration_s={target_duration_s} "
            "-- raise --target-duration-s or --num-prompts for this cell"
        )
    return resolved


def run_vllm_bench_serve(
    base_url: str,
    model: str,
    workload: str,
    num_prompts: int,
    request_rate: float,
    scale: float,
    result_path: str,
    seed: int | None = None,
) -> dict:
    args = workloads_mod.get_args(
        workload, num_prompts=num_prompts, request_rate=request_rate, scale=scale
    )
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--model",
        model,
        "--result-filename",
        os.path.basename(result_path),
        "--result-dir",
        os.path.dirname(result_path),
        *args,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {
            "error": (
                "vllm bench serve timed out after 1800s -- likely a stalled "
                "request (see vLLM issue #45388)"
            )
        }
    if proc.returncode != 0:
        return {"error": f"vllm bench serve failed: {proc.stderr[-2000:]}"}
    if not os.path.exists(result_path):
        return {"error": f"vllm bench serve produced no result file at {result_path}"}
    return parse_vllm_bench_result(result_path)


def run_mixed_case(
    base_url: str,
    model: str,
    output_dir: str,
    policy: str,
    rate: float,
    resolved_num_prompts: int,
    scale: float,
    seed: int | None,
    ref_counts: list[int],
) -> list[dict]:
    """`mixed`: chat/rag/longdoc run concurrently (via `vllm bench serve`
    sub-streams in parallel threads, proportioned by
    `_MIXED_SUBWORKLOAD_WEIGHTS` of the total rate/prompt count) plus a
    periodic stream of needle cases cycling through `ref_counts`, all
    against the same server in the same wall-clock window. Returns one row
    per sub-stream/needle-case, tagged `sub_workload`, rather than one
    lossily-averaged row -- chat/rag/longdoc and needle have different
    metric shapes (TTFT/ITL percentiles vs. a hit/miss rate) that don't
    average meaningfully. Each sub-stream's own failure (e.g. a stalled
    request, see vLLM issue #45388) is caught independently so one bad
    sub-stream doesn't lose the others' real data."""

    def _run_sub(sub_workload: str, weight: float) -> dict:
        sub_rate = rate * weight if rate != float("inf") else rate
        sub_num_prompts = max(1, round(resolved_num_prompts * weight))
        result_path = os.path.join(
            output_dir,
            f"raw_{policy}_mixed-{sub_workload}_{rate}_{_seed_tag(seed)}.json",
        )
        try:
            if (
                sub_workload == "rag"
                and sub_num_prompts < workloads_mod.RAG_NUM_PREFIXES
            ):
                raise ValueError(
                    f"mixed's rag sub-stream num_prompts ({sub_num_prompts}, "
                    f"{resolved_num_prompts}*{weight} weight) is below "
                    f"RAG_NUM_PREFIXES ({workloads_mod.RAG_NUM_PREFIXES}) -- "
                    "raise the overall --num-prompts/--target-duration-s"
                )
            row = run_vllm_bench_serve(
                base_url,
                model,
                sub_workload,
                sub_num_prompts,
                sub_rate,
                scale,
                result_path,
                seed=seed,
            )
        except Exception as e:
            row = {"error": str(e)[:500]}
        row.update(
            {
                "policy": policy,
                "seed": seed,
                "workload": "mixed",
                "sub_workload": sub_workload,
                "request_rate": sub_rate,
                "num_prompts": sub_num_prompts,
            }
        )
        return row

    def _run_needle_stream() -> list[dict]:
        duration_estimate = (
            resolved_num_prompts / rate if rate not in (0, float("inf")) else 60.0
        )
        n_cases = max(1, round(duration_estimate / 120))  # ~1 case/2 min
        rows = []
        for i in range(n_cases):
            ref_count = ref_counts[i % len(ref_counts)]
            row = {
                "policy": policy,
                "seed": seed,
                "workload": "mixed",
                "sub_workload": "needle",
                "reference_count": ref_count,
            }
            try:
                result = needle_workload.run_needle_case(
                    base_url,
                    reference_count=ref_count,
                    num_distractors=max(1, 10 - ref_count),
                    model=model,
                    seed=1_000_000 + i,
                )
                row["needle_hit_rate"] = 1.0 if result["hit"] else 0.0
            except Exception as e:
                row["error"] = str(e)[:500]
            rows.append(row)
        return rows

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        sub_futures = {
            ex.submit(_run_sub, name, weight): name
            for name, weight in _MIXED_SUBWORKLOAD_WEIGHTS.items()
        }
        needle_future = ex.submit(_run_needle_stream)
        for fut in concurrent.futures.as_completed(sub_futures):
            rows.append(fut.result())
        rows.extend(needle_future.result())
    return rows


_active_handle: list = []


def _handle_sigterm(signum, frame):
    """Best-effort graceful shutdown when killed from outside (e.g. by
    run_grid_sweep.py's per-cell timeout after a stalled request, see
    vLLM issue #45388) -- without this, the `finally: handle.shutdown()`
    block never runs on SIGTERM and the `vllm serve` child (its own
    process group via setsid) is orphaned, leaking GPU memory and the
    port for the next cell."""
    if _active_handle:
        try:
            _active_handle[0].shutdown()
        except Exception:
            pass
    sys.exit(143)


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policies", required=True, help="comma-separated")
    parser.add_argument("--workloads", required=True, help="comma-separated")
    parser.add_argument("--request-rates", default="inf", help="comma-separated")
    parser.add_argument(
        "--needle-reference-counts", default="0,1,2", help="comma-separated"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="flat request count per run; mutually exclusive with --target-duration-s",
    )
    parser.add_argument(
        "--target-duration-s",
        type=float,
        default=None,
        help=(
            "derive num_prompts as target_duration_s * request_rate per "
            "(workload, rate) cell, so every cell gets the same steady-state "
            "window instead of a flat prompt count producing very different "
            "durations at different arrival rates (plan protocol: >=10 min)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "single seed for this invocation's `vllm bench serve` calls "
            "(passed through as-is); a multi-seed sweep loops this whole "
            "script once per seed, see benchmarks/run_grid_sweep.py"
        ),
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--cpu-bytes-to-use", type=int, default=2 * 1024**3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.num_prompts is None and args.target_duration_s is None:
        parser.error("one of --num-prompts or --target-duration-s is required")
    if args.num_prompts is not None and args.target_duration_s is not None:
        parser.error("--num-prompts and --target-duration-s are mutually exclusive")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    write_header = not os.path.exists(csv_path)

    policies = args.policies.split(",")
    workloads = args.workloads.split(",")
    request_rates = [float(r) for r in args.request_rates.split(",")]
    ref_counts = [int(r) for r in args.needle_reference_counts.split(",")]

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()

        for policy in policies:
            kv_config = policies_mod.kv_transfer_config(policy, args.cpu_bytes_to_use)
            print(f"=== policy={policy} : launching server ===", flush=True)
            try:
                handle = launch_server(
                    args.model,
                    args.port,
                    args.output_dir,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    num_gpu_blocks_override=args.num_gpu_blocks_override,
                    kv_transfer_config=kv_config,
                    log_label=policy,
                )
            except (RuntimeError, TimeoutError) as e:
                print(f"policy={policy} FAILED TO LAUNCH: {e}", flush=True)
                writer.writerow({"policy": policy, "error": str(e)[:500]})
                csv_file.flush()
                continue

            _active_handle[:] = [handle]

            # needle isn't rate-based, so --target-duration-s doesn't map
            # onto it the same way -- fall back to a rough ~15s/distractor
            # estimate when only a duration was given, matching this
            # project's own needle-case latency scale (issues log entries
            # #33/#35).
            needle_num_prompts = args.num_prompts or max(
                1, round(args.target_duration_s / 15)
            )

            try:
                for workload in workloads:
                    if workload == "needle":
                        for ref_count in ref_counts:
                            try:
                                before = metrics_mod.snapshot(handle.metrics_url())
                                result = needle_workload.run_needle_case(
                                    handle.base_url(),
                                    reference_count=ref_count,
                                    num_distractors=max(
                                        1, needle_num_prompts - ref_count
                                    ),
                                    model=args.model,
                                    seed=ref_count,
                                )
                                after = metrics_mod.snapshot(handle.metrics_url())
                                delta = metrics_mod.diff(before, after)
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "reference_count": ref_count,
                                    "needle_hit_rate": 1.0 if result["hit"] else 0.0,
                                    "duration_s": result["t_needle_s"]
                                    + result["t_recall_s"]
                                    + sum(result["probe_latencies_s"])
                                    + sum(result["distractor_latencies_s"]),
                                    "load_bytes_delta": delta[
                                        "vllm:kv_offload_load_bytes_total"
                                    ],
                                    "store_bytes_delta": delta[
                                        "vllm:kv_offload_store_bytes_total"
                                    ],
                                    "preemptions_delta": delta[
                                        "vllm:num_preemptions_total"
                                    ],
                                }
                            except Exception as e:
                                # A stalled/hung request (see vLLM issue
                                # #45388) or any other per-case failure must
                                # not abort every remaining cell in this
                                # (and every later) policy's run.
                                print(
                                    f"policy={policy} workload=needle "
                                    f"ref_count={ref_count} FAILED: {e}",
                                    flush=True,
                                )
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "reference_count": ref_count,
                                    "error": str(e)[:500],
                                }
                            writer.writerow(row)
                            csv_file.flush()
                    elif workload == "needle-v2":
                        # Preservation-aware needle: the recall's cache-hit
                        # outcome (hit/miss/not_pressured), the only signal
                        # that distinguishes semantic-preserved from
                        # lru-evicted under lossless offload (issues log
                        # entry #58). REQUIRES a tightened GPU
                        # (--num-gpu-blocks-override) and a small
                        # --cpu-bytes-to-use, else every cell reads
                        # not_pressured (the built-in validity check).
                        def _snap():
                            return metrics_mod.snapshot(handle.metrics_url())

                        for ref_count in ref_counts:
                            try:
                                before = _snap()
                                result = needle_workload.run_needle_v2_case(
                                    handle.base_url(),
                                    reference_count=ref_count,
                                    num_distractors=max(
                                        1, needle_num_prompts - ref_count
                                    ),
                                    model=args.model,
                                    seed=ref_count,
                                    snapshot_metrics=_snap,
                                )
                                after = _snap()
                                delta = metrics_mod.diff(before, after)
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "reference_count": ref_count,
                                    "needle_outcome": result["needle_outcome"],
                                    "recall_load_bytes": result["recall_load_bytes"],
                                    "recall_store_bytes": result["recall_store_bytes"],
                                    "needle_hit_rate": (
                                        1.0
                                        if result["needle_outcome"]
                                        == needle_workload.NEEDLE_HIT
                                        else 0.0
                                    ),
                                    "duration_s": result["t_needle_s"]
                                    + result["t_recall_s"]
                                    + sum(result["probe_latencies_s"])
                                    + sum(result["distractor_latencies_s"]),
                                    "load_bytes_delta": delta[
                                        "vllm:kv_offload_load_bytes_total"
                                    ],
                                    "store_bytes_delta": delta[
                                        "vllm:kv_offload_store_bytes_total"
                                    ],
                                    "preemptions_delta": delta[
                                        "vllm:num_preemptions_total"
                                    ],
                                }
                            except Exception as e:
                                print(
                                    f"policy={policy} workload=needle-v2 "
                                    f"ref_count={ref_count} FAILED: {e}",
                                    flush=True,
                                )
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "reference_count": ref_count,
                                    "error": str(e)[:500],
                                }
                            writer.writerow(row)
                            csv_file.flush()
                    elif workload == "mixed":
                        for rate in request_rates:
                            resolved_num_prompts = resolve_num_prompts(
                                rate, args.num_prompts, args.target_duration_s
                            )
                            before = metrics_mod.snapshot(handle.metrics_url())
                            try:
                                mixed_rows = run_mixed_case(
                                    handle.base_url(),
                                    args.model,
                                    args.output_dir,
                                    policy,
                                    rate,
                                    resolved_num_prompts,
                                    args.scale,
                                    args.seed,
                                    ref_counts,
                                )
                            except Exception as e:
                                print(
                                    f"policy={policy} workload=mixed "
                                    f"rate={rate} FAILED: {e}",
                                    flush=True,
                                )
                                mixed_rows = [
                                    {
                                        "policy": policy,
                                        "seed": args.seed,
                                        "workload": "mixed",
                                        "request_rate": rate,
                                        "num_prompts": resolved_num_prompts,
                                        "error": str(e)[:500],
                                    }
                                ]
                            after = metrics_mod.snapshot(handle.metrics_url())
                            delta = metrics_mod.diff(before, after)
                            for row in mixed_rows:
                                row.setdefault(
                                    "load_bytes_delta",
                                    delta.get("vllm:kv_offload_load_bytes_total"),
                                )
                                row.setdefault(
                                    "store_bytes_delta",
                                    delta.get("vllm:kv_offload_store_bytes_total"),
                                )
                                row.setdefault(
                                    "preemptions_delta",
                                    delta.get("vllm:num_preemptions_total"),
                                )
                                writer.writerow(row)
                            csv_file.flush()
                            print(
                                f"policy={policy} workload=mixed rate={rate} "
                                f"sub_rows={len(mixed_rows)} metrics_delta={delta}",
                                flush=True,
                            )
                    else:
                        for rate in request_rates:
                            resolved_num_prompts = resolve_num_prompts(
                                rate,
                                args.num_prompts,
                                args.target_duration_s,
                                min_num_prompts=workloads_mod.RAG_NUM_PREFIXES
                                if workload == "rag"
                                else None,
                            )
                            try:
                                before = metrics_mod.snapshot(handle.metrics_url())
                                result_path = os.path.join(
                                    args.output_dir,
                                    f"raw_{policy}_{workload}_{rate}_"
                                    f"{_seed_tag(args.seed)}.json",
                                )
                                row = run_vllm_bench_serve(
                                    handle.base_url(),
                                    args.model,
                                    workload,
                                    resolved_num_prompts,
                                    rate,
                                    args.scale,
                                    result_path,
                                    seed=args.seed,
                                )
                                after = metrics_mod.snapshot(handle.metrics_url())
                                delta = metrics_mod.diff(before, after)
                                row.update(
                                    {
                                        "policy": policy,
                                        "seed": args.seed,
                                        "workload": workload,
                                        "request_rate": rate,
                                        "num_prompts": resolved_num_prompts,
                                        "load_bytes_delta": delta.get(
                                            "vllm:kv_offload_load_bytes_total"
                                        ),
                                        "store_bytes_delta": delta.get(
                                            "vllm:kv_offload_store_bytes_total"
                                        ),
                                        "preemptions_delta": delta.get(
                                            "vllm:num_preemptions_total"
                                        ),
                                    }
                                )
                                print(
                                    f"policy={policy} workload={workload} "
                                    f"rate={rate} metrics_delta={delta}",
                                    flush=True,
                                )
                            except Exception as e:
                                # Same rationale as the needle branch above:
                                # one stalled cell must not abort the rest
                                # of the grid.
                                print(
                                    f"policy={policy} workload={workload} "
                                    f"rate={rate} FAILED: {e}",
                                    flush=True,
                                )
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "request_rate": rate,
                                    "num_prompts": resolved_num_prompts,
                                    "error": str(e)[:500],
                                }
                            writer.writerow(row)
                            csv_file.flush()
            finally:
                print(f"=== policy={policy} : shutting down ===", flush=True)
                handle.shutdown()
                _active_handle.clear()
                time.sleep(2.0)

    print(f"Done. Results: {csv_path}")


if __name__ == "__main__":
    main()
