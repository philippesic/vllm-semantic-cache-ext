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
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from harness import adaptive_splice_probe, metrics, needle_workload, policies, workloads


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


def test_distractor_prompts_are_distinct_across_a_real_case_size():
    """A real needle case calls make_distractor(seed+i) for i in
    range(num_distractors) (up to 25) -- `subject` is drawn from only 5
    values, so without a per-call unique tag, most calls collapse onto
    identical prompt text and vLLM's prefix-cache hashing then treats
    them as cache hits instead of distinct KV blocks, silently capping
    how much real content the workload generates regardless of
    num_distractors (issues log entry #56)."""
    distractors = [needle_workload.make_distractor(i) for i in range(25)]
    assert len(set(distractors)) == 25


def test_distractor_prompts_differ_within_the_first_block_of_tokens():
    """The base sentence tokenizes to under 16 tokens (one KV block), and
    the CPU-offload connector only ever stores blocks complete AT PREFILL
    TIME -- a unique tag appended at the END still lands in a later block
    that two same-`subject` calls share identically, so real content
    volume stayed capped even after the first fix (entry #56 follow-up,
    confirmed token-exact against the real tokenizer). Guard against that
    regression here without needing a real tokenizer: the tag must be a
    *prefix*, so distinct calls must differ within the first ~16
    characters, not just somewhere in the full string."""
    prefixes = {needle_workload.make_distractor(i)[:16] for i in range(25)}
    assert len(prefixes) == 25


def test_probe_prompts_are_distinct_across_a_real_case_size():
    probes = [needle_workload.make_probe(i) for i in range(5)]
    assert len(set(probes)) == 5


def test_long_distractors_differ_within_every_block_across_seeds():
    """make_long_distractor must be distinct block-by-block across seeds, not
    just as whole strings -- vLLM dedups by per-block content hash, so two
    long distractors sharing any 16-token-aligned window collapse onto the
    same stored block and stop building real CPU pressure (entry #57's
    lesson, generalized from the first block to the whole prompt). Approx a
    block as ~12 words and require every positional window to be unique
    across a real case's worth of seeds."""
    per_seed = [needle_workload.make_long_distractor(i).split() for i in range(12)]
    assert all(len(w) >= 180 for w in per_seed)
    for start in range(0, 180 - 12, 12):
        windows = {tuple(w[start : start + 12]) for w in per_seed}
        assert len(windows) == 12, f"block at word {start} collides across seeds"


def test_classify_needle_outcome_hit_when_recall_loaded_from_cpu_tier():
    """load_bytes > 0 means the needle's blocks were served straight from
    the CPU tier -- the policy preserved them under pressure."""
    assert (
        needle_workload.classify_needle_outcome(917504.0, 0.0)
        == needle_workload.NEEDLE_HIT
    )


def test_classify_needle_outcome_miss_when_recall_recomputed_and_restored():
    """load_bytes == 0 but store_bytes > 0 means the blocks were absent from
    both GPU and CPU tiers, so the recall had to recompute and re-store
    them -- the policy evicted them. This is the exact case the old
    `expected_code in recall_text` check could never see, since vLLM
    recomputes the same (correct) answer regardless (issues log #58)."""
    assert (
        needle_workload.classify_needle_outcome(0.0, 3670016.0)
        == needle_workload.NEEDLE_MISS
    )


def test_classify_needle_outcome_not_pressured_when_neither_counter_moved():
    """load == store == 0 means the blocks were still resident in the GPU
    prefix cache -- the CPU tier was never consulted, the run is not under
    capacity pressure, and the result is uninterpretable. The built-in
    validity check the old always-1.0 metric lacked."""
    assert (
        needle_workload.classify_needle_outcome(0.0, 0.0)
        == needle_workload.NEEDLE_NOT_PRESSURED
    )


def test_classify_needle_outcome_prefers_hit_even_if_store_also_moved():
    """A HIT can coincide with a small store (e.g. the recall's own trailing
    question blocks): a positive load is decisive -- the needle's blocks did
    come from the CPU tier."""
    assert (
        needle_workload.classify_needle_outcome(917504.0, 458752.0)
        == needle_workload.NEEDLE_HIT
    )


def test_run_needle_v2_case_classifies_from_recall_isolated_deltas():
    """run_needle_v2_case must snapshot metrics around the RECALL request
    alone, so the outcome reflects the recall's own CPU-tier interaction and
    not the distractors' store volume. Drives it with a fake HTTP client and
    a fake, monotonic counter that advances only on the recall to prove the
    isolation and the resulting classification."""
    from unittest.mock import patch

    calls = {"n": 0}

    def fake_complete(base_url, model, prompt, max_tokens, timeout_s):
        calls["n"] += 1
        return "code is 12345-Zephyr", 0.01

    # store rises during distractors, then load rises on the recall (a HIT):
    # snapshots are taken by the drain (>=2), then before, then after recall.
    load = {"v": 1000.0}

    def fake_snapshot():
        return {
            "vllm:kv_offload_load_bytes_total": load["v"],
            "vllm:kv_offload_store_bytes_total": 5000.0,
        }

    def bump_load_on_recall(base_url, model, prompt, max_tokens, timeout_s):
        # the recall is the request whose prompt restates the needle +
        # question; bump the load counter to simulate a CPU-tier hit.
        if "Question:" in prompt:
            load["v"] += 917504.0
        return "code is 12345-Zephyr", 0.01

    with patch.object(needle_workload, "_complete", side_effect=bump_load_on_recall):
        result = needle_workload.run_needle_v2_case(
            "http://x",
            reference_count=0,
            num_distractors=2,
            model="m",
            seed=0,
            snapshot_metrics=fake_snapshot,
            settle_s=0.0,
            max_settle_polls=1,
        )

    assert result["needle_outcome"] == needle_workload.NEEDLE_HIT
    assert result["recall_load_bytes"] == 917504.0


def test_run_needle_v2_case_outcome_is_none_without_metrics_access():
    """No snapshot callable -> no preservation signal; the case still runs
    and returns latencies but reports needle_outcome=None rather than
    guessing."""
    from unittest.mock import patch

    def fake_complete(base_url, model, prompt, max_tokens, timeout_s):
        return "code is 12345-Zephyr", 0.01

    with patch.object(needle_workload, "_complete", side_effect=fake_complete):
        result = needle_workload.run_needle_v2_case(
            "http://x",
            reference_count=1,
            num_distractors=2,
            model="m",
            seed=0,
        )

    assert result["needle_outcome"] is None
    assert result["recall_load_bytes"] is None


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


def test_mixed_case_produces_one_row_per_subworkload_plus_needle_rows():
    from unittest.mock import patch

    import benchmarks.run_latency_suite as rls

    def fake_bench_serve(
        base_url, model, workload, num_prompts, rate, scale, path, seed=None
    ):
        return {"duration_s": 1.0, "throughput_tok_s": 42.0}

    def fake_needle_case(base_url, *, reference_count, num_distractors, model, seed):
        return {"hit": True}

    with (
        patch.object(rls, "run_vllm_bench_serve", side_effect=fake_bench_serve),
        patch.object(
            rls.needle_workload, "run_needle_case", side_effect=fake_needle_case
        ),
    ):
        rows = rls.run_mixed_case(
            "http://x",
            "model",
            "/tmp",
            "semantic-minmax",
            10.0,
            600,
            1.0,
            None,
            [0, 1],
        )

    sub_workloads = {r["sub_workload"] for r in rows}
    assert sub_workloads == {"chat", "rag", "longdoc", "needle"}
    for r in rows:
        assert r["workload"] == "mixed"
        assert "error" not in r
    chat_row = next(r for r in rows if r["sub_workload"] == "chat")
    assert chat_row["num_prompts"] == round(600 * 0.40)
    assert chat_row["request_rate"] == pytest.approx(10.0 * 0.40)


def test_mixed_case_isolates_one_subworkload_failure():
    from unittest.mock import patch

    import benchmarks.run_latency_suite as rls

    def flaky_bench_serve(
        base_url, model, workload, num_prompts, rate, scale, path, seed=None
    ):
        if workload == "rag":
            raise RuntimeError("simulated stall (vLLM issue #45388)")
        return {"duration_s": 1.0, "throughput_tok_s": 42.0}

    def fake_needle_case(base_url, *, reference_count, num_distractors, model, seed):
        return {"hit": False}

    with (
        patch.object(rls, "run_vllm_bench_serve", side_effect=flaky_bench_serve),
        patch.object(
            rls.needle_workload, "run_needle_case", side_effect=fake_needle_case
        ),
    ):
        rows = rls.run_mixed_case(
            "http://x", "model", "/tmp", "lru", 10.0, 100, 1.0, None, [0]
        )

    by_sub = {r["sub_workload"]: r for r in rows}
    assert "error" in by_sub["rag"]
    assert "error" not in by_sub["chat"]
    assert "error" not in by_sub["longdoc"]
    assert by_sub["needle"]["needle_hit_rate"] == 0.0


def test_resolve_num_prompts_from_target_duration():
    from benchmarks.run_latency_suite import resolve_num_prompts

    assert resolve_num_prompts(2.0, None, 30.0) == 60
    assert resolve_num_prompts(2.0, None, 1.0) == 2


def test_resolve_num_prompts_explicit_num_prompts_wins_when_given():
    from benchmarks.run_latency_suite import resolve_num_prompts

    assert resolve_num_prompts(2.0, 15, None) == 15


def test_resolve_num_prompts_rejects_infinite_rate_with_target_duration():
    from benchmarks.run_latency_suite import resolve_num_prompts

    with pytest.raises(ValueError):
        resolve_num_prompts(float("inf"), None, 30.0)


def test_resolve_num_prompts_requires_one_of_the_two():
    from benchmarks.run_latency_suite import resolve_num_prompts

    with pytest.raises(ValueError):
        resolve_num_prompts(2.0, None, None)


def test_resolve_num_prompts_rejects_below_workload_minimum():
    from benchmarks.run_latency_suite import resolve_num_prompts

    # rate=0.15, duration=60s -> resolves to 9, below rag's num_prefixes floor
    with pytest.raises(ValueError, match="minimum"):
        resolve_num_prompts(0.15, None, 60.0, min_num_prompts=10)


def test_resolve_num_prompts_allows_at_or_above_workload_minimum():
    from benchmarks.run_latency_suite import resolve_num_prompts

    assert resolve_num_prompts(0.15, None, 600.0, min_num_prompts=10) == 90
    assert resolve_num_prompts(2.0, 10, None, min_num_prompts=10) == 10


def test_grid_sweep_cell_args_include_seed_and_duration():
    from benchmarks.run_grid_sweep import build_run_latency_suite_args

    args = build_run_latency_suite_args(
        model="m",
        policy="semantic-mean",
        workloads="chat,rag",
        request_rates="2.0,8.0",
        needle_reference_counts="0,1",
        target_duration_s=600.0,
        num_prompts=None,
        scale=1.0,
        cpu_bytes_to_use=1000,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        num_gpu_blocks_override=200,
        port=8199,
        seed=3,
        output_dir="/tmp/out",
    )
    assert (
        "--policies" in args and args[args.index("--policies") + 1] == "semantic-mean"
    )
    assert "--seed" in args and args[args.index("--seed") + 1] == "3"
    assert (
        "--target-duration-s" in args
        and args[args.index("--target-duration-s") + 1] == "600.0"
    )
    assert "--num-prompts" not in args  # mutually exclusive with target-duration-s


def test_grid_sweep_cell_args_use_num_prompts_when_no_duration_given():
    from benchmarks.run_grid_sweep import build_run_latency_suite_args

    args = build_run_latency_suite_args(
        model="m",
        policy="lru",
        workloads="chat",
        request_rates="2.0",
        needle_reference_counts="0",
        target_duration_s=None,
        num_prompts=20,
        scale=1.0,
        cpu_bytes_to_use=1000,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        num_gpu_blocks_override=None,
        port=8199,
        seed=1,
        output_dir="/tmp/out",
    )
    assert "--num-prompts" in args and args[args.index("--num-prompts") + 1] == "20"
    assert "--target-duration-s" not in args
    assert "--num-gpu-blocks-override" not in args


def test_build_slots_assigns_unique_port_and_output_dir_per_gpu():
    from benchmarks.run_grid_sweep import build_slots

    slots = build_slots(["0", "3", "7"], base_port=8199, output_dir="/tmp/out")

    assert [s["gpu_id"] for s in slots] == ["0", "3", "7"]
    assert [s["port"] for s in slots] == [8199, 8200, 8201]
    assert len({s["output_dir"] for s in slots}) == 3  # all unique
    assert all(s["output_dir"].startswith("/tmp/out/gpu") for s in slots)


def test_build_slots_single_gpu_reproduces_original_port_and_dir():
    from benchmarks.run_grid_sweep import build_slots

    slots = build_slots(["0"], base_port=8199, output_dir="/tmp/out")

    assert slots == [{"gpu_id": "0", "port": 8199, "output_dir": "/tmp/out/gpu0"}]


def test_merge_results_concatenates_all_slots_keeping_one_header(tmp_path):
    from benchmarks.run_grid_sweep import build_slots, merge_results

    slots = build_slots(["0", "1"], base_port=8199, output_dir=str(tmp_path))
    for slot in slots:
        os.makedirs(slot["output_dir"], exist_ok=True)

    with open(os.path.join(slots[0]["output_dir"], "results.csv"), "w") as f:
        f.write("policy,seed\nlru,1\n")
    with open(os.path.join(slots[1]["output_dir"], "results.csv"), "w") as f:
        f.write("policy,seed\nsemantic-minmax,2\n")

    merged_path = merge_results(str(tmp_path), slots)

    with open(merged_path) as f:
        content = f.read()
    assert content.count("policy,seed") == 1  # exactly one header
    assert "lru,1" in content
    assert "semantic-minmax,2" in content


def test_merge_results_skips_slots_with_no_results_file(tmp_path):
    from benchmarks.run_grid_sweep import build_slots, merge_results

    slots = build_slots(["0", "1"], base_port=8199, output_dir=str(tmp_path))
    os.makedirs(slots[0]["output_dir"], exist_ok=True)
    os.makedirs(slots[1]["output_dir"], exist_ok=True)
    with open(os.path.join(slots[0]["output_dir"], "results.csv"), "w") as f:
        f.write("policy,seed\nlru,1\n")
    # slots[1] never produced a results.csv (every cell dispatched to it failed).

    merged_path = merge_results(str(tmp_path), slots)

    with open(merged_path) as f:
        content = f.read()
    assert "lru,1" in content


def _write_lines(path, lines):
    with open(path, "a") as f:
        for line in lines:
            f.write(line + "\n")
            f.flush()


def _watch_in_background(log_path, tag, timeout_s=5.0):
    import threading

    holder = {}

    def target():
        holder["result"] = adaptive_splice_probe.watch_for_splice(
            str(log_path), tag, timeout_s=timeout_s, poll_interval_s=0.05
        )

    thread = threading.Thread(target=target)
    thread.start()
    time.sleep(0.2)  # let watch_for_splice open the file and seek to end first
    return thread, holder


def test_watch_for_splice_matches_partial_splice_with_spliced_ge_1(tmp_path):
    """Simulates real live tailing: the watch must already be running
    (seeked to the file's current end) before the matching line is
    appended -- writing the line first and watching after (as an earlier,
    buggy version of this test did) doesn't exercise real behavior at all,
    since seek(0, 2) skips content that already existed."""
    log_path = tmp_path / "server.log"
    log_path.write_text("")
    thread, holder = _watch_in_background(log_path, "TAG123")
    _write_lines(
        log_path,
        [
            "PREFETCH_EFFECT_DEBUG cmpl-other-req-0-abc: PARTIAL SPLICE spliced=2 reloaded=1 covered=0.67",
            "PREFETCH_EFFECT_DEBUG cmpl-TAG123-0-xyz: PARTIAL SPLICE spliced=1 reloaded=3 covered=0.25",
        ],
    )
    thread.join(timeout=5.0)
    result = holder["result"]
    assert result is not None
    assert result["req_id"] == "cmpl-TAG123-0-xyz"
    assert result["spliced"] == 1


def test_watch_for_splice_ignores_zero_spliced_key_mismatch_lines(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text("")
    _write_lines(
        log_path,
        [
            "PREFETCH_EFFECT_DEBUG cmpl-TAG123-0-xyz: KEY MISMATCH spliced=0 keys_to_load=5 prefetch.keys=5",
        ],
    )
    result = adaptive_splice_probe.watch_for_splice(
        str(log_path), "TAG123", timeout_s=1.0, poll_interval_s=0.05
    )
    assert result is None


def test_watch_for_splice_matches_legacy_unconditional_spliced_marker(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text("")
    thread, holder = _watch_in_background(log_path, "TAG123")
    _write_lines(
        log_path,
        ["PREFETCH_EFFECT_DEBUG cmpl-TAG123-0-xyz: SPLICED n_blocks=4"],
    )
    thread.join(timeout=5.0)
    result = holder["result"]
    assert result is not None
    assert result["req_id"] == "cmpl-TAG123-0-xyz"


def test_watch_for_splice_times_out_when_tag_never_appears(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text("")
    _write_lines(
        log_path,
        [
            "PREFETCH_EFFECT_DEBUG cmpl-someone-else-0-xyz: PARTIAL SPLICE spliced=1 reloaded=0 covered=1.00"
        ],
    )
    result = adaptive_splice_probe.watch_for_splice(
        str(log_path), "TAG123", timeout_s=0.5, poll_interval_s=0.05
    )
    assert result is None


def test_watch_for_splice_only_sees_lines_written_after_it_starts(tmp_path):
    """Confirms the tail starts from the CURRENT end of the file, not the
    beginning -- replaying old lines could false-positive on a stale event
    from a previous, unrelated request that happens to share a tag prefix
    substring by coincidence."""
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "PREFETCH_EFFECT_DEBUG cmpl-TAG123-0-old: PARTIAL SPLICE spliced=1 reloaded=0 covered=1.00\n"
    )
    result = adaptive_splice_probe.watch_for_splice(
        str(log_path), "TAG123", timeout_s=0.5, poll_interval_s=0.05
    )
    assert result is None
