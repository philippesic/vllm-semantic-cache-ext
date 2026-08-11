# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the unified DGX policy-audit plumbing."""

import csv
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.summarize_policy_audit import (
    ABLATION_VARIANTS,
    ALPHA_CANDIDATE_VARIANT,
    EXPECTED_VARIANTS,
    LEADERBOARD_POLICIES,
    MIXED_SUBWORKLOAD_RATES,
    SERVING_POLICIES,
    aggregate,
    aggregate_timings,
    alpha_pair_rows,
    load_results,
    validate_matrix,
    validate_repository_state,
    validate_variant_manifests,
)
from semantic_offload.store_layout import build_store_job_layout
from semantic_offload.worker import SemanticOffloadingWorker


def test_store_layout_recovers_destination_order_from_unordered_keys():
    key_a = b"a"
    key_b = b"b"

    layout = build_store_job_layout(
        {key_a: 10, key_b: 20},
        dst_block_ids=[20, 10],
        src_block_ids=[3, 4, 1, 2],
        blocks_per_chunk=2,
    )

    assert layout == [(key_b, [3, 4]), (key_a, [1, 2])]


def test_store_layout_rejects_ambiguous_source_blocks():
    key = b"a"
    assert build_store_job_layout({key: 10}, [10], [1], 2) is None


def test_audit_summary_fails_rows_with_embedded_errors(tmp_path):
    variant = tmp_path / "variant"
    variant.mkdir()
    path = variant / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["policy", "workload", "needle_outcome", "error"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "policy": "semantic-mean",
                "workload": "needle-v2",
                "needle_outcome": "hit",
                "error": "server failed",
            }
        )

    rows, errors = load_results(tmp_path)

    assert len(rows) == 1
    assert errors and "server failed" in errors[0]


def test_audit_summary_computes_pressured_hit_rate():
    rows = [
        {
            "variant": "v",
            "policy": "semantic-mean",
            "workload": "needle-v2",
            "needle_outcome": outcome,
        }
        for outcome in ("hit", "hit", "miss", "not_pressured")
    ]

    summary = aggregate(rows)[0]

    assert summary["needle_hits"] == 2
    assert summary["needle_misses"] == 1
    assert summary["needle_not_pressured"] == 1
    assert summary["needle_pressured_hit_rate"] == 2 / 3


def test_audit_summary_counts_partial_as_pressured_not_full_hit():
    rows = [
        {
            "variant": "v",
            "policy": "semantic-mean",
            "workload": "needle-v2",
            "needle_outcome": outcome,
        }
        for outcome in ("hit", "partial", "miss")
    ]

    summary = aggregate(rows)[0]

    assert summary["needle_partials"] == 1
    assert summary["needle_pressured_hit_rate"] == 1 / 3
    assert summary["needle_pressured_any_load_rate"] == 2 / 3


def test_matrix_validation_rejects_missing_expected_variants():
    errors = validate_matrix([], {"1"})

    assert any("missing expected variant" in error for error in errors)


def _numeric_result(**fields) -> dict:
    return {
        "duration_s": "1",
        "ttft_p50_ms": "2",
        "ttft_p99_ms": "3",
        "throughput_tok_s": "4",
        "load_bytes_delta": "5",
        "store_bytes_delta": "6",
        "preemptions_delta": "0",
        "error": "",
        **fields,
    }


def _needle_result(**fields) -> dict:
    return {
        "needle_outcome": "hit",
        "recall_load_bytes": "1",
        "recall_store_bytes": "0",
        "error": "",
        **fields,
    }


def _complete_matrix_rows(seed: str = "1") -> list[dict]:
    rows: list[dict] = []
    for ref_count in (0, 1, 2):
        for policy in LEADERBOARD_POLICIES:
            rows.append(
                _needle_result(
                    variant=f"leaderboard_ref{ref_count}",
                    policy=policy,
                    seed=seed,
                    workload="needle-v2",
                    reference_count=str(ref_count),
                )
            )
    for policy in SERVING_POLICIES:
        for parent_rate, sub_rates in MIXED_SUBWORKLOAD_RATES.items():
            for workload in ("chat", "rag"):
                rows.append(
                    _numeric_result(
                        variant="serving",
                        policy=policy,
                        seed=seed,
                        workload=workload,
                        request_rate=parent_rate,
                    )
                )
            for sub_workload, sub_rate in sub_rates.items():
                rows.append(
                    _numeric_result(
                        variant="serving",
                        policy=policy,
                        seed=seed,
                        workload="mixed",
                        sub_workload=sub_workload,
                        request_rate=sub_rate,
                        parent_request_rate=parent_rate,
                    )
                )
            rows.append(
                _needle_result(
                    variant="serving",
                    policy=policy,
                    seed=seed,
                    workload="mixed",
                    sub_workload="needle",
                    parent_request_rate=parent_rate,
                    reference_count="0",
                )
            )
    for variant in ABLATION_VARIANTS:
        rows.append(
            _needle_result(
                variant=variant,
                policy="semantic-mean",
                seed=seed,
                workload="needle-v2",
                reference_count="1",
            )
        )
        for workload in ("chat", "rag"):
            rows.append(
                _numeric_result(
                    variant=variant,
                    policy="semantic-mean",
                    seed=seed,
                    workload=workload,
                    request_rate="8.0",
                )
            )
    return rows


def test_matrix_validation_requires_complete_alpha06_ablation_cells():
    rows = _complete_matrix_rows()
    assert validate_matrix(rows, {"1"}) == []

    incomplete = [
        row
        for row in rows
        if not (row["variant"] == ALPHA_CANDIDATE_VARIANT and row["workload"] == "rag")
    ]
    incomplete_errors = validate_matrix(incomplete, {"1"})
    assert any(
        ALPHA_CANDIDATE_VARIANT in error
        and "missing result cell" in error
        and "rag" in error
        for error in incomplete_errors
    )


def _write_alpha_manifest(root, variant: str, alpha: float) -> None:
    variant_dir = root / variant
    variant_dir.mkdir()
    (variant_dir / "variant_manifest.txt").write_text(
        f"variant={variant}\n"
        "policies=semantic-mean\n"
        "workloads=needle-v2,chat,rag\n"
        "request-rates=8.0\n"
        "needle-reference-counts=1\n"
        "num-prompts=24\n"
        "target-duration-s=\n"
        "seeds=1,2,3\n"
        "gpus=4,5,6,7\n"
        f'extra-config={{"probe_layer":"middle","head_aggregation":"mean",'
        f'"alpha":{alpha},"prefetch_budget_fraction":0}}\n'
    )


def test_variant_manifest_rejects_mislabeled_alpha_config(tmp_path):
    from benchmarks.summarize_policy_audit import (
        ALPHA_BASELINE_VARIANT,
        EXPECTED_VARIANTS,
    )

    for variant in EXPECTED_VARIANTS - {
        ALPHA_BASELINE_VARIANT,
        ALPHA_CANDIDATE_VARIANT,
    }:
        variant_dir = tmp_path / variant
        variant_dir.mkdir()
        (variant_dir / "variant_manifest.txt").write_text(
            f"variant={variant}\nseeds=1,2,3\n"
        )
    _write_alpha_manifest(tmp_path, ALPHA_BASELINE_VARIANT, 0.5)
    _write_alpha_manifest(tmp_path, ALPHA_CANDIDATE_VARIANT, 0.5)

    errors = validate_variant_manifests(tmp_path)

    assert any(
        ALPHA_CANDIDATE_VARIANT in error and "does not match expected" in error
        for error in errors
    )


def test_variant_manifests_accept_exact_paired_alpha_configs(tmp_path):
    from benchmarks.summarize_policy_audit import (
        ALPHA_BASELINE_VARIANT,
        EXPECTED_VARIANTS,
    )

    for variant in EXPECTED_VARIANTS - {
        ALPHA_BASELINE_VARIANT,
        ALPHA_CANDIDATE_VARIANT,
    }:
        variant_dir = tmp_path / variant
        variant_dir.mkdir()
        (variant_dir / "variant_manifest.txt").write_text(
            f"variant={variant}\nseeds=1,2,3\n"
        )
    _write_alpha_manifest(tmp_path, ALPHA_BASELINE_VARIANT, 0.5)
    _write_alpha_manifest(tmp_path, ALPHA_CANDIDATE_VARIANT, 0.6)

    assert validate_variant_manifests(tmp_path) == []


def test_variant_manifests_reject_partial_expected_seed_override(tmp_path):
    from benchmarks.summarize_policy_audit import (
        ALPHA_BASELINE_VARIANT,
        EXPECTED_VARIANTS,
    )

    for variant in EXPECTED_VARIANTS - {
        ALPHA_BASELINE_VARIANT,
        ALPHA_CANDIDATE_VARIANT,
    }:
        variant_dir = tmp_path / variant
        variant_dir.mkdir()
        (variant_dir / "variant_manifest.txt").write_text(
            f"variant={variant}\nseeds=1,2,3\n"
        )
    _write_alpha_manifest(tmp_path, ALPHA_BASELINE_VARIANT, 0.5)
    _write_alpha_manifest(tmp_path, ALPHA_CANDIDATE_VARIANT, 0.6)

    assert validate_matrix(_complete_matrix_rows(seed="1"), {"1"}) == []
    errors = validate_variant_manifests(tmp_path, {"1"})

    assert any("do not match expected seeds ['1']" in error for error in errors)


def _write_repository_provenance(
    root, drift_variant: str | None = None, drift_summary: bool = False
) -> None:
    fields = (
        "semantic_revision",
        "semantic_state",
        "vllm_revision",
        "vllm_state",
    )
    (root / "repository_state.txt").write_text(
        "".join(
            f"{field}_start=stable\n"
            f"{field}_end=stable\n"
            f"{field}_summary_end="
            f"{'changed' if drift_summary and field == 'semantic_state' else 'stable'}\n"
            for field in fields
        )
    )
    for variant in EXPECTED_VARIANTS:
        variant_dir = root / variant
        variant_dir.mkdir(exist_ok=True)
        with (variant_dir / "variant_manifest.txt").open("a") as handle:
            for field in fields:
                handle.write(f"{field}_start=stable\n")
                end = (
                    "changed"
                    if variant == drift_variant and field == "semantic_state"
                    else "stable"
                )
                handle.write(f"{field}_end={end}\n")


def test_repository_provenance_rejects_mid_variant_drift(tmp_path):
    _write_repository_provenance(tmp_path, drift_variant=ALPHA_CANDIDATE_VARIANT)

    errors = validate_repository_state(tmp_path)

    assert any(
        ALPHA_CANDIDATE_VARIANT in error
        and "repository state drift for semantic_state" in error
        for error in errors
    )


def test_repository_provenance_accepts_stable_variants(tmp_path):
    _write_repository_provenance(tmp_path)

    assert validate_repository_state(tmp_path) == []


def test_repository_provenance_rejects_drift_during_summary(tmp_path):
    _write_repository_provenance(tmp_path, drift_summary=True)

    errors = validate_repository_state(tmp_path)

    assert any(
        "repository state drift during summary: semantic_state" in error
        for error in errors
    )


def test_repository_provenance_requires_completed_summary_boundary(tmp_path):
    _write_repository_provenance(tmp_path)
    state_path = tmp_path / "repository_state.txt"
    state_path.write_text(
        "\n".join(
            line
            for line in state_path.read_text().splitlines()
            if "_summary_end=" not in line
        )
        + "\n"
    )

    errors = validate_repository_state(tmp_path)

    assert any("missing semantic_state_summary_end" in error for error in errors)
    assert validate_repository_state(tmp_path, require_summary_end=False) == []


def test_alpha_pair_rows_preserve_seed_outcomes_and_numeric_deltas():
    rows = [
        {
            "variant": variant,
            "seed": "1",
            "workload": "needle-v2",
            "reference_count": "1",
            "needle_outcome": outcome,
        }
        for variant, outcome in (
            ("signal_middle_mean", "partial"),
            (ALPHA_CANDIDATE_VARIANT, "hit"),
        )
    ] + [
        {
            "variant": variant,
            "seed": "1",
            "workload": "chat",
            "request_rate": "8.0",
            "ttft_p50_ms": ttft,
            "ttft_p99_ms": "20",
            "throughput_tok_s": "30",
            "load_bytes_delta": "40",
            "store_bytes_delta": "50",
            "preemptions_delta": "0",
        }
        for variant, ttft in (
            ("signal_middle_mean", "10"),
            (ALPHA_CANDIDATE_VARIANT, "8"),
        )
    ]

    paired = alpha_pair_rows(rows)

    outcome = next(row for row in paired if row["metric"] == "needle_outcome")
    assert outcome["alpha_05"] == "partial"
    assert outcome["alpha_06"] == "hit"
    ttft = next(row for row in paired if row["metric"] == "ttft_p50_ms")
    assert ttft["delta_alpha06_minus_alpha05"] == -2.0


def test_grid_sweep_forwards_alpha06_config_exactly():
    from benchmarks.run_grid_sweep import build_run_latency_suite_args

    extra_config = (
        '{"probe_layer":"middle","head_aggregation":"mean",'
        '"alpha":0.6,"prefetch_budget_fraction":0}'
    )
    args = build_run_latency_suite_args(
        model="m",
        policy="semantic-mean",
        workloads="needle-v2,chat,rag",
        request_rates="8.0",
        needle_reference_counts="1",
        target_duration_s=None,
        num_prompts=24,
        scale=1.0,
        cpu_bytes_to_use=1000,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        num_gpu_blocks_override=120,
        port=8199,
        seed=1,
        output_dir="/tmp/out",
        extra_config=extra_config,
    )

    assert args[args.index("--extra-config") + 1] == extra_config


def test_dgx_audit_refuses_to_reuse_existing_output_root(tmp_path):
    vllm_repo = tmp_path / "vllm"
    activate = vllm_repo / ".venv" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("")
    python_bin = activate.parent / "python"
    python_bin.write_text("#!/bin/sh\nexit 2\n")
    python_bin.chmod(0o755)
    vllm_cli = activate.parent / "vllm"
    vllm_cli.write_text("#!/bin/sh\nexit 2\n")
    vllm_cli.chmod(0o755)
    (vllm_repo / "vllm").mkdir()
    output_root = tmp_path / "existing-output"
    output_root.mkdir()
    script = Path(__file__).parents[1] / "dgx_policy_audit.sh"

    result = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "VLLM_REPO": str(vllm_repo),
            "VENV_DIR": str(vllm_repo / ".venv"),
            "OUTPUT_ROOT": str(output_root),
        },
        timeout=10,
    )

    assert result.returncode == 2
    assert "Refusing to reuse existing audit output root" in result.stderr


def test_dgx_audit_output_root_acquisition_is_atomic(tmp_path):
    vllm_repo = tmp_path / "vllm"
    activate = vllm_repo / ".venv" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("")
    python_bin = activate.parent / "python"
    python_bin.write_text("#!/bin/sh\nexit 2\n")
    python_bin.chmod(0o755)
    vllm_cli = activate.parent / "vllm"
    vllm_cli.write_text("#!/bin/sh\nexit 2\n")
    vllm_cli.chmod(0o755)
    (vllm_repo / "vllm").mkdir()
    output_root = tmp_path / "shared-output"
    script = Path(__file__).parents[1] / "dgx_policy_audit.sh"
    env = {
        **os.environ,
        "VLLM_REPO": str(vllm_repo),
        "VENV_DIR": str(vllm_repo / ".venv"),
        "OUTPUT_ROOT": str(output_root),
    }

    processes = [
        subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=10) for process in processes]

    assert all(process.returncode != 0 for process in processes)
    errors = [stderr for _stdout, stderr in results]
    assert (
        sum("Refusing to reuse existing audit output root" in error for error in errors)
        == 1
    )
    assert (
        sum(
            "Refusing to reuse existing audit output root" not in error
            for error in errors
        )
        == 1
    )


def test_matrix_validation_rejects_blank_mixed_needle_reference_count():
    row = {
        "variant": "serving",
        "policy": "lru",
        "seed": "1",
        "workload": "mixed",
        "sub_workload": "needle",
        "parent_request_rate": "2.0",
        "reference_count": "",
        "needle_outcome": "partial",
        "recall_load_bytes": "1",
        "recall_store_bytes": "1",
        "error": "",
    }

    errors = validate_matrix([row], set())

    assert any("invalid mixed needle reference_count" in error for error in errors)


def test_matrix_validation_rejects_non_finite_numeric_metrics():
    row = {
        "variant": "serving",
        "policy": "lru",
        "seed": "1",
        "workload": "chat",
        "request_rate": "2.0",
        "duration_s": "1",
        "ttft_p50_ms": "nan",
        "ttft_p99_ms": "2",
        "throughput_tok_s": "3",
        "load_bytes_delta": "0",
        "store_bytes_delta": "0",
        "preemptions_delta": "0",
        "error": "",
    }

    errors = validate_matrix([row], set())

    assert any("non-finite numeric field ttft_p50_ms" in error for error in errors)


def test_store_submission_fences_summary_stream_before_base_transfer():
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._pending_job_blocks = {}
    worker._summary_stream = Mock()
    worker._build_summaries_for_blocks = Mock()
    current_stream = Mock()
    src_spec = SimpleNamespace(block_ids=[])

    with (
        patch(
            "semantic_offload.worker.torch.cuda.current_stream",
            return_value=current_stream,
        ),
        patch.object(
            CPUOffloadingWorker, "submit_store", return_value=True
        ) as base_store,
    ):
        assert worker.submit_store(1, src_spec, SimpleNamespace())

    current_stream.wait_stream.assert_called_once_with(worker._summary_stream)
    base_store.assert_called_once()


def test_timing_summary_uses_last_cumulative_snapshot(tmp_path):
    log_dir = tmp_path / "variant" / "gpu0"
    log_dir.mkdir(parents=True)
    (log_dir / "server_semantic-mean_8199_123.log").write_text(
        "SEMANTIC_TIMING bucket=query_captured_total pid=10 calls=500 "
        "total_s=2.000 mean_ms=4.0000\n"
        "SEMANTIC_TIMING bucket=query_captured_total pid=10 calls=1000 "
        "total_s=5.000 mean_ms=5.0000\n"
        "SEMANTIC_PROCESS bucket=query_captured pid=10 calls=1000 rss_mb=812.5 "
        "gc_count=1,2,3 gc_collections=1,1,1 gc_collected=2,2,2\n"
    )

    summary = aggregate_timings(tmp_path)

    assert summary == [
        {
            "variant": "variant:semantic-mean",
            "bucket": "query_captured_total",
            "processes": 1,
            "calls": 1000,
            "total_s": 5.0,
            "mean_ms": 5.0,
            "max_rss_mb": 812.5,
        }
    ]
