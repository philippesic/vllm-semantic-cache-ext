# SPDX-License-Identifier: Apache-2.0
"""Multi-seed grid driver for run_latency_suite.py -- the third follow-up
flagged (not built) in `.claude/docs/step-1.6-grid-trim-proposal.md` (in
the vllm-semantic-cache repo). `run_latency_suite.py` runs one policy
(looping its own workloads/rates internally) for one seed; this loops
POLICIES x SEEDS, invoking it once per (policy, seed) pair as a
subprocess -- matching the pattern `run_latency_suite.py` itself already
uses to invoke `vllm bench serve`, rather than importing and calling its
internals directly (keeps each invocation's server process fully isolated
-- a crashed run for one policy/seed can't leave stray state for the
next).

Usage (the trimmed first-pass grid from step-1.6-grid-trim-proposal.md):

  python benchmarks/run_grid_sweep.py \\
      --model <7-8B-class model> \\
      --policies lru,arc,semantic-minmax,semantic-mean,semantic-cuboid-mean \\
      --workloads chat,rag,mixed \\
      --request-rates 2.0,8.0 \\
      --seeds 1,2,3 \\
      --target-duration-s 600 \\
      --cpu-bytes-to-use 2147483648 \\
      --needle-reference-counts 0,1,2 \\
      --output-dir results/step_1_6_first_pass
"""

import argparse
import os
import subprocess
import sys
import time


def build_run_latency_suite_args(
    *,
    model: str,
    policy: str,
    workloads: str,
    request_rates: str,
    needle_reference_counts: str,
    target_duration_s: float | None,
    num_prompts: int | None,
    scale: float,
    cpu_bytes_to_use: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    num_gpu_blocks_override: int | None,
    port: int,
    seed: int,
    output_dir: str,
) -> list[str]:
    """Pure, unit-testable: the argv for one (policy, seed) cell. Kept
    separate from the actual subprocess-launching loop below."""
    args = [
        "--model",
        model,
        "--policies",
        policy,
        "--workloads",
        workloads,
        "--request-rates",
        request_rates,
        "--needle-reference-counts",
        needle_reference_counts,
        "--scale",
        str(scale),
        "--cpu-bytes-to-use",
        str(cpu_bytes_to_use),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
    ]
    if num_gpu_blocks_override is not None:
        args += ["--num-gpu-blocks-override", str(num_gpu_blocks_override)]
    if target_duration_s is not None:
        args += ["--target-duration-s", str(target_duration_s)]
    else:
        args += ["--num-prompts", str(num_prompts)]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policies", required=True, help="comma-separated")
    parser.add_argument("--workloads", required=True, help="comma-separated")
    parser.add_argument("--request-rates", default="inf", help="comma-separated")
    parser.add_argument(
        "--needle-reference-counts", default="0,1,2", help="comma-separated"
    )
    parser.add_argument("--seeds", required=True, help="comma-separated")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--target-duration-s", type=float, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--cpu-bytes-to-use", type=int, default=2 * 1024**3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--output-dir", required=True)
    parsed = parser.parse_args()

    if parsed.num_prompts is None and parsed.target_duration_s is None:
        parser.error("one of --num-prompts or --target-duration-s is required")

    policies = parsed.policies.split(",")
    seeds = [int(s) for s in parsed.seeds.split(",")]
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_latency_suite.py"
    )

    total_cells = len(policies) * len(seeds)
    done = 0
    failures = []
    for policy in policies:
        for seed in seeds:
            done += 1
            print(
                f"=== grid sweep {done}/{total_cells}: policy={policy} seed={seed} ===",
                flush=True,
            )
            cell_args = build_run_latency_suite_args(
                model=parsed.model,
                policy=policy,
                workloads=parsed.workloads,
                request_rates=parsed.request_rates,
                needle_reference_counts=parsed.needle_reference_counts,
                target_duration_s=parsed.target_duration_s,
                num_prompts=parsed.num_prompts,
                scale=parsed.scale,
                cpu_bytes_to_use=parsed.cpu_bytes_to_use,
                gpu_memory_utilization=parsed.gpu_memory_utilization,
                max_model_len=parsed.max_model_len,
                num_gpu_blocks_override=parsed.num_gpu_blocks_override,
                port=parsed.port,
                seed=seed,
                output_dir=parsed.output_dir,
            )
            start = time.monotonic()
            proc = subprocess.run(
                [sys.executable, script_path, *cell_args],
                capture_output=True,
                text=True,
            )
            elapsed = time.monotonic() - start
            if proc.returncode != 0:
                print(
                    f"CELL FAILED policy={policy} seed={seed} "
                    f"(exit {proc.returncode}, {elapsed:.0f}s): "
                    f"{proc.stderr[-1000:]}",
                    flush=True,
                )
                failures.append((policy, seed))
            else:
                print(
                    f"cell done policy={policy} seed={seed} ({elapsed:.0f}s)",
                    flush=True,
                )

    print(
        f"\nGrid sweep done: {total_cells - len(failures)}/{total_cells} cells succeeded."
    )
    if failures:
        print(f"Failed cells: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
