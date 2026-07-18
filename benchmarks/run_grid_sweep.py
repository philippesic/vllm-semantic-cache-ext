# SPDX-License-Identifier: Apache-2.0
"""Multi-seed, multi-GPU grid driver for run_latency_suite.py. `run_latency_
suite.py` runs one policy (looping its own workloads/rates internally) for
one seed on one GPU; this loops POLICIES x SEEDS, dispatching each
(policy, seed) cell as its own subprocess -- one server process per cell,
fully isolated (a crashed run for one cell can't leave stray state for the
next) -- across a pool of GPUs (`--gpus`) so independent cells run
concurrently instead of strictly sequentially.

Each GPU gets a fixed "slot": its own CUDA_VISIBLE_DEVICES pin, its own
port (avoids bind collisions between concurrently-running cells on the
same node), and its own output subdirectory (avoids two processes
concurrently appending to the same results.csv -- a real corruption risk,
not hypothetical, since csv.DictWriter has no cross-process locking). Cells
are dispatched to whichever slot frees up next; results from every slot's
subdirectory are merged into one top-level results.csv at the end.

With `--gpus` omitted (single slot "0"), this reproduces the original
strictly-sequential single-GPU behavior -- just with results written to
`<output-dir>/gpu0/` and merged, rather than directly into `<output-dir>/`.

Usage (the trimmed first-pass grid, single GPU):

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

Usage (same grid, 8-way concurrent across an 8-GPU node):

  python benchmarks/run_grid_sweep.py \\
      ... (same as above) ... \\
      --gpus 0,1,2,3,4,5,6,7 \\
      --output-dir results/step_1_6_first_pass
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time

POLL_INTERVAL_S = 2.0
STATUS_INTERVAL_S = 60.0


def _kill_cell(proc: subprocess.Popen, port: int) -> None:
    """Escalating shutdown for a timed-out cell: SIGTERM the whole process
    group first (gives run_latency_suite.py's own SIGTERM handler a chance
    to shut its `vllm serve` child down cleanly), a short grace period,
    then SIGKILL the group, then -- belt and suspenders, since the `vllm
    serve` child runs in its OWN process group via setsid specifically so
    ServerHandle.shutdown() can manage it independently, meaning group-
    killing run_latency_suite.py's group does NOT reach it if the graceful
    path didn't get a chance to run -- kill whatever is still bound to the
    cell's port so the next cell's server launch can't collide with a
    leaked one holding the GPU."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


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


def build_slots(gpu_ids: list[str], base_port: int, output_dir: str) -> list[dict]:
    """Pure, unit-testable: one slot per GPU -- its pin, its own port (base
    + slot index, so concurrently-running cells on the same node never
    collide), and its own output subdirectory (so concurrently-running
    cells never append to the same results.csv)."""
    return [
        {
            "gpu_id": gpu_id,
            "port": base_port + i,
            "output_dir": os.path.join(output_dir, f"gpu{i}"),
        }
        for i, gpu_id in enumerate(gpu_ids)
    ]


def merge_results(output_dir: str, slots: list[dict]) -> str:
    """Concatenate every slot's own results.csv into one top-level file,
    keeping a single header. A slot with no results.csv (e.g. every cell
    dispatched to it failed before writing a header) is skipped, not an
    error -- failures are already recorded via the `failures` list."""
    merged_path = os.path.join(output_dir, "results.csv")
    fieldnames = None
    rows = []
    for slot in slots:
        slot_csv = os.path.join(slot["output_dir"], "results.csv")
        if not os.path.exists(slot_csv):
            continue
        with open(slot_csv, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)
    if fieldnames is None:
        return merged_path
    with open(merged_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return merged_path


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
    parser.add_argument(
        "--gpus",
        default="0",
        help=(
            "comma-separated GPU ids to run cells on concurrently, one "
            "cell per free GPU at a time (e.g. 0,1,2,3,4,5,6,7 on an "
            "8-GPU node). Default '0' reproduces strictly-sequential "
            "single-GPU behavior."
        ),
    )
    parser.add_argument(
        "--cell-timeout-s",
        type=float,
        default=7200.0,
        help=(
            "kill and skip a (policy, seed) cell if it doesn't finish "
            "within this many seconds -- defense-in-depth against a "
            "stalled request hanging the whole grid indefinitely "
            "(see vLLM issue #45388); default is generous (2h) since a "
            "real cell legitimately loops many workloads/rates"
        ),
    )
    parsed = parser.parse_args()

    if parsed.num_prompts is None and parsed.target_duration_s is None:
        parser.error("one of --num-prompts or --target-duration-s is required")

    policies = parsed.policies.split(",")
    seeds = [int(s) for s in parsed.seeds.split(",")]
    gpu_ids = parsed.gpus.split(",")
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_latency_suite.py"
    )

    slots = build_slots(gpu_ids, parsed.port, parsed.output_dir)
    for slot in slots:
        os.makedirs(slot["output_dir"], exist_ok=True)

    cells = [(policy, seed) for policy in policies for seed in seeds]
    total_cells = len(cells)
    queue = list(cells)
    running: dict[int, dict] = {}  # slot index -> {proc, start, policy, seed}
    done = 0
    failures = []

    print(
        f"Grid sweep: {total_cells} cells across {len(slots)} GPU slot(s) "
        f"({', '.join(gpu_ids)})",
        flush=True,
    )

    last_status = time.monotonic()
    while queue or running:
        for slot_idx, slot in enumerate(slots):
            if slot_idx in running or not queue:
                continue
            policy, seed = queue.pop(0)
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
                port=slot["port"],
                seed=seed,
                output_dir=slot["output_dir"],
            )
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": slot["gpu_id"]}
            proc = subprocess.Popen(
                [sys.executable, script_path, *cell_args],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
                env=env,
            )
            print(
                f"=== dispatched policy={policy} seed={seed} to "
                f"gpu={slot['gpu_id']} (slot {slot_idx}, port {slot['port']}) ===",
                flush=True,
            )
            running[slot_idx] = {
                "proc": proc,
                "start": time.monotonic(),
                "policy": policy,
                "seed": seed,
            }

        for slot_idx in list(running.keys()):
            entry = running[slot_idx]
            proc = entry["proc"]
            elapsed = time.monotonic() - entry["start"]
            retcode = proc.poll()
            timed_out = retcode is None and elapsed > parsed.cell_timeout_s
            if retcode is None and not timed_out:
                continue  # still running, within its budget

            if timed_out:
                _kill_cell(proc, slots[slot_idx]["port"])
                stdout = proc.stdout.read() if proc.stdout else ""
                print(
                    f"CELL TIMED OUT policy={entry['policy']} seed={entry['seed']} "
                    f"after {elapsed:.0f}s (limit {parsed.cell_timeout_s:.0f}s) -- "
                    f"likely a stalled request (see vLLM issue #45388); "
                    f"killed and continuing.",
                    flush=True,
                )
                failures.append((entry["policy"], entry["seed"]))
            elif retcode != 0:
                stdout = proc.stdout.read() if proc.stdout else ""
                print(
                    f"CELL FAILED policy={entry['policy']} seed={entry['seed']} "
                    f"(exit {retcode}, {elapsed:.0f}s): {stdout[-1000:]}",
                    flush=True,
                )
                failures.append((entry["policy"], entry["seed"]))
            else:
                print(
                    f"cell done policy={entry['policy']} seed={entry['seed']} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
            done += 1
            del running[slot_idx]

        if time.monotonic() - last_status > STATUS_INTERVAL_S:
            print(
                f"--- status: {done}/{total_cells} done, {len(running)} "
                f"running, {len(queue)} queued ---",
                flush=True,
            )
            last_status = time.monotonic()

        if queue or running:
            time.sleep(POLL_INTERVAL_S)

    merged_path = merge_results(parsed.output_dir, slots)
    print(
        f"\nGrid sweep done: {total_cells - len(failures)}/{total_cells} "
        f"cells succeeded. Results: {merged_path}"
    )
    if failures:
        print(f"Failed cells: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
