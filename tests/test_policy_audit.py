# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the unified DGX policy-audit plumbing."""

import csv
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.summarize_policy_audit import (
    aggregate,
    aggregate_timings,
    load_results,
    validate_matrix,
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
