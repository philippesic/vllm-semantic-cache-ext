# SPDX-License-Identifier: Apache-2.0
"""Phase 1 latency/throughput suite (plan's Benchmarking section, Step
1.6). For each (policy, workload, request_rate) combination: launch a real
server with that policy, run the workload (`vllm bench serve` for
chat/rag/longdoc, the bespoke needle_workload for `needle`), snapshot the
offload-subsystem Prometheus counters before/after, tear the server down,
and append one combined result row to a CSV.

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
import csv
import json
import os
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
    "workload",
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
    "load_bytes_delta",
    "store_bytes_delta",
    "preemptions_delta",
    "error",
]


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
    request_rate: float, num_prompts: int | None, target_duration_s: float | None
) -> int:
    """Pure, unit-testable: either an explicit --num-prompts, or a count
    derived from a target steady-state duration at the given request rate
    (--target-duration-s) -- a flat --num-prompts produces very different
    steady-state windows at different arrival rates, which the plan's own
    protocol (>=10 min steady-state per configuration) doesn't allow (see
    the Step 1.6 grid-trim proposal's follow-up #2)."""
    if target_duration_s is not None:
        if request_rate == float("inf"):
            raise ValueError(
                "--target-duration-s requires a finite --request-rate "
                "(inf sends everything at t=0, duration is undefined)"
            )
        return max(1, round(target_duration_s * request_rate))
    if num_prompts is not None:
        return num_prompts
    raise ValueError("either --num-prompts or --target-duration-s is required")


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
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        return {"error": f"vllm bench serve failed: {proc.stderr[-2000:]}"}
    if not os.path.exists(result_path):
        return {"error": f"vllm bench serve produced no result file at {result_path}"}
    return parse_vllm_bench_result(result_path)


def main():
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
                            before = metrics_mod.snapshot(handle.metrics_url())
                            result = needle_workload.run_needle_case(
                                handle.base_url(),
                                reference_count=ref_count,
                                num_distractors=max(1, needle_num_prompts - ref_count),
                                model=args.model,
                                seed=ref_count,
                            )
                            after = metrics_mod.snapshot(handle.metrics_url())
                            delta = metrics_mod.diff(before, after)
                            row = {
                                "policy": policy,
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
                            writer.writerow(row)
                            csv_file.flush()
                    else:
                        for rate in request_rates:
                            resolved_num_prompts = resolve_num_prompts(
                                rate, args.num_prompts, args.target_duration_s
                            )
                            before = metrics_mod.snapshot(handle.metrics_url())
                            result_path = os.path.join(
                                args.output_dir,
                                f"raw_{policy}_{workload}_{rate}.json",
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
                            writer.writerow(row)
                            csv_file.flush()
                            print(
                                f"policy={policy} workload={workload} rate={rate} "
                                f"metrics_delta={delta}",
                                flush=True,
                            )
            finally:
                print(f"=== policy={policy} : shutting down ===", flush=True)
                handle.shutdown()
                time.sleep(2.0)

    print(f"Done. Results: {csv_path}")


if __name__ == "__main__":
    main()
