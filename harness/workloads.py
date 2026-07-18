# SPDX-License-Identifier: Apache-2.0
"""Workload definitions -> `vllm bench serve` CLI args, per the plan's
Benchmarking section (.claude/docs/semantic-eviction-plan.md in the
vllm-semantic-cache repo). `chat`/`rag`/`longdoc` map directly onto
existing `vllm bench serve` datasets (`random` and `prefix_repetition`) --
per the plan's own explicit instruction to wrap, not reimplement.

Sizes below are the plan's PRODUCTION-SCALE numbers (chat 0.5-2k, rag
8-16k, longdoc 32-64k). Real Step 1.6 runs on the rented 7-8B-class
hardware use these as-is; validating the harness plumbing against the
small dev model needs a `scale` factor to fit the dev box's much smaller
max-model-len, matching this project's "validation, not signal" dev-
hardware discipline (see plan's hardware note, §1) -- get_args() takes a
`scale` in (0, 1] that shrinks all length parameters proportionally.
"""

WORKLOAD_NAMES = (
    "chat",
    "rag",
    "longdoc",
)  # "needle" and "mixed" are separate, see below


def get_args(
    workload: str,
    *,
    num_prompts: int,
    request_rate: float,
    scale: float = 1.0,
) -> list[str]:
    if not 0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")

    common = [
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(request_rate),
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
    ]

    if workload == "chat":
        # 0.5-2k context, short outputs, high arrival rate -- expected tie
        # with LRU/ARC (recency and relevance naturally correlate here).
        input_len = max(32, int(1200 * scale))
        output_len = max(8, int(150 * scale))
        return common + [
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            "--random-range-ratio",
            "0.5",
        ]

    if workload == "rag":
        # 8-16k context with shared-prefix structure -- `prefix_repetition`
        # IS this workload already (num_prefixes shared prompts, each
        # reused by prompts_per_prefix requests): a shared prefix reused
        # across independent requests is real re-referenced content, the
        # primary realistic venue for a genuine semantic win (plan's
        # Status 2026-07-16, entries #10-#12).
        prefix_len = max(64, int(12000 * scale))
        suffix_len = max(16, int(500 * scale))
        return common + [
            "--dataset-name",
            "prefix_repetition",
            "--prefix-repetition-num-prefixes",
            "10",
            "--prefix-repetition-prefix-len",
            str(prefix_len),
            "--prefix-repetition-suffix-len",
            str(suffix_len),
            "--prefix-repetition-output-len",
            "150",
        ]

    if workload == "longdoc":
        # 32-64k context, arrives mid-batch -- exercises preemption (and,
        # in Phase 2, tiering).
        input_len = max(128, int(48000 * scale))
        output_len = max(8, int(100 * scale))
        return common + [
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            "--random-range-ratio",
            "0.3",
        ]

    raise ValueError(f"unknown workload {workload!r}, expected one of {WORKLOAD_NAMES}")
