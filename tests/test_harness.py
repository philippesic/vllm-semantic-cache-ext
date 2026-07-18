# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the benchmark harness's pure-logic pieces (policy config
generation, workload arg generation, needle prompt construction, metrics
text parsing, vllm-bench-serve result parsing). The actual server
launch/subprocess orchestration in benchmarks/run_latency_suite.py and
harness/server.py is integration-heavy and verified against the real
2080Ti/Qwen2.5-1.5B setup instead, the same split this project has used
throughout (see Step 1.2/1.3's own test files)."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from harness import metrics, needle_workload, policies, workloads


def test_kv_transfer_config_lru_uses_stock_offloading_connector():
    config = policies.kv_transfer_config("lru", cpu_bytes_to_use=1000)
    assert config["kv_connector"] == "OffloadingConnector"
    assert "kv_connector_module_path" not in config
    assert config["kv_connector_extra_config"]["eviction_policy"] == "lru"


def test_kv_transfer_config_arc_selects_arc_eviction_policy():
    config = policies.kv_transfer_config("arc", cpu_bytes_to_use=1000)
    assert config["kv_connector_extra_config"]["eviction_policy"] == "arc"


def test_kv_transfer_config_semantic_selects_real_method_name():
    for policy, expected_method in (
        ("semantic-minmax", "minmax"),
        ("semantic-mean", "mean"),
        ("semantic-cuboid-mean", "cuboid_mean"),
    ):
        config = policies.kv_transfer_config(policy, cpu_bytes_to_use=1000)
        assert config["kv_connector"] == "SemanticOffloadingConnector"
        assert (
            config["kv_connector_extra_config"]["spec_name"] == "SemanticOffloadingSpec"
        )
        assert config["kv_connector_extra_config"]["method"] == expected_method


def test_kv_transfer_config_rejects_unknown_policy():
    with pytest.raises(ValueError):
        policies.kv_transfer_config("not-a-real-policy", cpu_bytes_to_use=1000)


def test_workload_args_reject_invalid_scale():
    with pytest.raises(ValueError):
        workloads.get_args("chat", num_prompts=10, request_rate=1.0, scale=0.0)
    with pytest.raises(ValueError):
        workloads.get_args("chat", num_prompts=10, request_rate=1.0, scale=1.5)


def test_workload_args_scale_shrinks_lengths_proportionally():
    full = workloads.get_args("rag", num_prompts=10, request_rate=1.0, scale=1.0)
    small = workloads.get_args("rag", num_prompts=10, request_rate=1.0, scale=0.1)

    def prefix_len(args):
        return int(args[args.index("--prefix-repetition-prefix-len") + 1])

    assert prefix_len(small) < prefix_len(full)


def test_workload_args_rejects_unknown_workload():
    with pytest.raises(ValueError):
        workloads.get_args("not-a-real-workload", num_prompts=10, request_rate=1.0)


def test_needle_prompt_contains_the_code_it_returns():
    prompt, code = needle_workload.make_needle(seed=7)
    assert code in prompt


def test_needle_prompt_is_deterministic_per_seed():
    prompt_a, code_a = needle_workload.make_needle(seed=3)
    prompt_b, code_b = needle_workload.make_needle(seed=3)
    assert prompt_a == prompt_b
    assert code_a == code_b


def test_needle_and_probe_and_distractor_share_no_literal_overlap():
    """The probe/distractor prompts must not accidentally contain the
    needle's own code -- would silently invalidate the "zero content
    overlap" premise reference-count sweeps rely on."""
    _, code = needle_workload.make_needle(seed=1)
    probe = needle_workload.make_probe(seed=1)
    distractor = needle_workload.make_distractor(seed=1)
    assert code not in probe
    assert code not in distractor


def test_metrics_parse_sums_across_label_combinations():
    text = """
# HELP vllm:num_preemptions_total x
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="a"} 3.0
vllm:num_preemptions_total{model_name="b"} 4.0
vllm:kv_offload_load_bytes_total 12345.0
"""
    parsed = metrics.parse_metrics(text)
    assert parsed["vllm:num_preemptions_total"] == 7.0
    assert parsed["vllm:kv_offload_load_bytes_total"] == 12345.0
    assert parsed["vllm:kv_offload_store_bytes_total"] == 0.0


def test_metrics_diff_computes_deltas():
    before = {"vllm:num_preemptions_total": 5.0}
    after = {"vllm:num_preemptions_total": 9.0}
    assert metrics.diff(before, after) == {"vllm:num_preemptions_total": 4.0}


def test_parse_vllm_bench_result_extracts_expected_fields():
    from benchmarks.run_latency_suite import parse_vllm_bench_result

    payload = {
        "duration": 12.5,
        "p50_ttft_ms": 10.0,
        "p90_ttft_ms": 20.0,
        "p99_ttft_ms": 30.0,
        "p50_itl_ms": 5.0,
        "p90_itl_ms": 6.0,
        "p99_itl_ms": 7.0,
        "output_throughput": 100.0,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        row = parse_vllm_bench_result(path)
        assert row["duration_s"] == 12.5
        assert row["ttft_p99_ms"] == 30.0
        assert row["throughput_tok_s"] == 100.0
    finally:
        os.unlink(path)
