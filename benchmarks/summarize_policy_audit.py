# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize all result sets produced by dgx_policy_audit.sh."""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

NUMERIC_FIELDS = (
    "ttft_p50_ms",
    "ttft_p99_ms",
    "throughput_tok_s",
    "load_bytes_delta",
    "store_bytes_delta",
    "preemptions_delta",
)
TIMING_PATTERN = re.compile(
    r"SEMANTIC_TIMING bucket=(\S+) pid=(\d+) calls=(\d+) "
    r"total_s=([0-9.]+) mean_ms=([0-9.]+)"
)
PROCESS_PATTERN = re.compile(
    r"SEMANTIC_PROCESS bucket=(\S+) pid=(\d+) calls=(\d+) rss_mb=([-0-9.]+)"
)
SERVER_LOG_PATTERN = re.compile(r"server_(.+?)_\d+_\d+\.log$")
EXPECTED_VARIANTS = {
    "leaderboard_ref0",
    "leaderboard_ref1",
    "leaderboard_ref2",
    "serving",
    "signal_first_max",
    "signal_middle_max",
    "signal_middle_mean",
    "signal_middle_mean_alpha06",
    "session_decay8",
    "prefetch_001",
    "prefetch_005",
    "capture_stride4",
}
LEADERBOARD_POLICIES = {
    "lru",
    "arc",
    "semantic-minmax",
    "semantic-mean",
    "semantic-cuboid-mean",
}
SERVING_POLICIES = LEADERBOARD_POLICIES
ABLATION_VARIANTS = EXPECTED_VARIANTS - {
    "serving",
    "leaderboard_ref0",
    "leaderboard_ref1",
    "leaderboard_ref2",
}
MIXED_SUBWORKLOAD_RATES = {
    "2.0": {"chat": "0.8", "rag": "0.7", "longdoc": "0.5"},
    "8.0": {"chat": "3.2", "rag": "2.8", "longdoc": "2.0"},
}
ALPHA_BASELINE_VARIANT = "signal_middle_mean"
ALPHA_CANDIDATE_VARIANT = "signal_middle_mean_alpha06"
ALPHA_COMMON_CONFIG = {
    "probe_layer": "middle",
    "head_aggregation": "mean",
    "prefetch_budget_fraction": 0,
}
ALPHA_METRICS = {
    "ttft_p50_ms": "lower",
    "ttft_p99_ms": "lower",
    "throughput_tok_s": "higher",
    "load_bytes_delta": "context",
    "store_bytes_delta": "context",
    "preemptions_delta": "lower",
}


def _display(row: dict, field: str) -> str:
    value = row[field]
    return f"{value:.2f}" if value != "" else "-"


def _number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def load_results(root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    result_paths = sorted(root.glob("*/results.csv"))
    if not result_paths:
        return [], [f"no variant results.csv files found under {root}"]
    for path in result_paths:
        variant = path.parent.name
        with path.open(newline="") as handle:
            current = list(csv.DictReader(handle))
        if not current:
            errors.append(f"{variant}: results.csv has no rows")
        for index, row in enumerate(current, start=2):
            row["variant"] = variant
            error = (row.get("error") or "").strip()
            if error:
                errors.append(
                    f"{variant}:{index} {row.get('policy')} "
                    f"{row.get('workload')}: {error}"
                )
            rows.append(row)
    return rows, errors


def _read_variant_manifest(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    errors: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        return {}, [f"{path.parent.name}: cannot read variant manifest: {error}"]
    for line_number, line in enumerate(lines, start=1):
        if "=" not in line:
            errors.append(f"{path}:{line_number}: malformed manifest line")
            continue
        key, value = line.split("=", 1)
        if key in values:
            errors.append(f"{path}:{line_number}: duplicate manifest key {key}")
        values[key] = value
    return values, errors


def validate_variant_manifests(
    root: Path, expected_seeds: set[str] | None = None
) -> list[str]:
    """Bind directory labels to the exact alpha configs used by the runner."""
    errors: list[str] = []
    manifests: dict[str, dict[str, str]] = {}
    for variant in sorted(EXPECTED_VARIANTS):
        path = root / variant / "variant_manifest.txt"
        if not path.is_file():
            errors.append(f"{variant}: missing variant_manifest.txt")
            continue
        values, current_errors = _read_variant_manifest(path)
        errors.extend(current_errors)
        manifests[variant] = values
        if values.get("variant") != variant:
            errors.append(
                f"{variant}: manifest variant is {values.get('variant')!r}, expected {variant!r}"
            )

    manifest_seed_sets: dict[str, set[str]] = {}
    for variant, values in manifests.items():
        recorded_seeds = {
            seed.strip() for seed in values.get("seeds", "").split(",") if seed.strip()
        }
        if not recorded_seeds:
            errors.append(f"{variant}: manifest has no seeds")
            continue
        manifest_seed_sets[variant] = recorded_seeds
        if expected_seeds is not None and recorded_seeds != expected_seeds:
            errors.append(
                f"{variant}: manifest seeds {sorted(recorded_seeds)!r} do not match "
                f"expected seeds {sorted(expected_seeds)!r}"
            )
    if manifest_seed_sets:
        seed_contracts = {frozenset(seeds) for seeds in manifest_seed_sets.values()}
        if len(seed_contracts) > 1:
            errors.append(
                "variant manifests disagree on seeds: "
                + ", ".join(
                    f"{variant}={','.join(sorted(seeds))}"
                    for variant, seeds in sorted(manifest_seed_sets.items())
                )
            )

    expected_configs = {
        ALPHA_BASELINE_VARIANT: {**ALPHA_COMMON_CONFIG, "alpha": 0.5},
        ALPHA_CANDIDATE_VARIANT: {**ALPHA_COMMON_CONFIG, "alpha": 0.6},
    }
    for variant, expected_config in expected_configs.items():
        values = manifests.get(variant)
        if values is None:
            continue
        try:
            actual_config = json.loads(values.get("extra-config", ""))
        except json.JSONDecodeError as error:
            errors.append(f"{variant}: invalid extra-config JSON: {error}")
            continue
        if actual_config != expected_config:
            errors.append(
                f"{variant}: extra-config {actual_config!r} does not match "
                f"expected {expected_config!r}"
            )
        if values.get("policies") != "semantic-mean":
            errors.append(f"{variant}: policies must be semantic-mean")
        if values.get("workloads") != "needle-v2,chat,rag":
            errors.append(f"{variant}: workloads must be needle-v2,chat,rag")
        if values.get("request-rates") != "8.0":
            errors.append(f"{variant}: request-rates must be 8.0")
        if values.get("needle-reference-counts") != "1":
            errors.append(f"{variant}: needle-reference-counts must be 1")

    baseline = manifests.get(ALPHA_BASELINE_VARIANT)
    candidate = manifests.get(ALPHA_CANDIDATE_VARIANT)
    if baseline is not None and candidate is not None:
        paired_fields = (
            "policies",
            "workloads",
            "request-rates",
            "needle-reference-counts",
            "num-prompts",
            "target-duration-s",
            "seeds",
            "gpus",
        )
        for field in paired_fields:
            if baseline.get(field) != candidate.get(field):
                errors.append(
                    f"alpha variants differ in {field}: "
                    f"{baseline.get(field)!r} != {candidate.get(field)!r}"
                )
    return errors


def validate_repository_state(
    root: Path, *, require_summary_end: bool = True
) -> list[str]:
    """Reject results produced while either repository changed underneath them."""
    state_path = root / "repository_state.txt"
    if not state_path.is_file():
        return ["missing repository_state.txt"]
    root_state, errors = _read_variant_manifest(state_path)
    fields = (
        "semantic_revision",
        "semantic_state",
        "vllm_revision",
        "vllm_state",
    )
    for field in fields:
        start = root_state.get(f"{field}_start", "")
        end = root_state.get(f"{field}_end", "")
        if not start or not end:
            errors.append(f"repository_state.txt: missing {field} start/end")
        elif start != end:
            errors.append(f"repository state drift: {field} {start!r} != {end!r}")
        summary_end = root_state.get(f"{field}_summary_end")
        if summary_end is None:
            if require_summary_end:
                errors.append(f"repository_state.txt: missing {field}_summary_end")
        elif start != summary_end:
            errors.append(
                f"repository state drift during summary: {field} "
                f"{start!r} != {summary_end!r}"
            )

    for variant in sorted(EXPECTED_VARIANTS):
        path = root / variant / "variant_manifest.txt"
        if not path.is_file():
            continue
        values, current_errors = _read_variant_manifest(path)
        errors.extend(current_errors)
        for field in fields:
            start = values.get(f"{field}_start", "")
            end = values.get(f"{field}_end", "")
            expected = root_state.get(f"{field}_start", "")
            if not start or not end:
                errors.append(f"{variant}: missing {field} start/end")
            elif start != end:
                errors.append(
                    f"{variant}: repository state drift for {field}: "
                    f"{start!r} != {end!r}"
                )
            elif expected and start != expected:
                errors.append(
                    f"{variant}: {field} {start!r} does not match audit "
                    f"start {expected!r}"
                )
    return errors


def validate_matrix(rows: list[dict], expected_seeds: set[str]) -> list[str]:
    """Reject plausible-looking summaries with missing or duplicate cells."""
    errors: list[str] = []
    variants = {row["variant"] for row in rows}
    for variant in sorted(EXPECTED_VARIANTS - variants):
        errors.append(f"missing expected variant: {variant}")

    def cell(row: dict) -> tuple[str, ...]:
        reference_count = row.get("reference_count", "")
        if row.get("workload") == "mixed" and row.get("sub_workload") == "needle":
            reference_count = "*"
        return tuple(
            str(value)
            for value in (
                row.get("variant", ""),
                row.get("policy", ""),
                row.get("seed", ""),
                row.get("workload", ""),
                row.get("sub_workload", ""),
                row.get("request_rate", ""),
                row.get("parent_request_rate", ""),
                reference_count,
            )
        )

    expected: set[tuple[str, ...]] = set()
    for seed in expected_seeds:
        for ref_count in (0, 1, 2):
            for policy in LEADERBOARD_POLICIES:
                expected.add(
                    (
                        f"leaderboard_ref{ref_count}",
                        policy,
                        seed,
                        "needle-v2",
                        "",
                        "",
                        "",
                        str(ref_count),
                    )
                )
        for policy in SERVING_POLICIES:
            for parent_rate, sub_rates in MIXED_SUBWORKLOAD_RATES.items():
                for workload in ("chat", "rag"):
                    expected.add(
                        (
                            "serving",
                            policy,
                            seed,
                            workload,
                            "",
                            parent_rate,
                            "",
                            "",
                        )
                    )
                for sub_workload, sub_rate in sub_rates.items():
                    expected.add(
                        (
                            "serving",
                            policy,
                            seed,
                            "mixed",
                            sub_workload,
                            sub_rate,
                            parent_rate,
                            "",
                        )
                    )
                expected.add(
                    (
                        "serving",
                        policy,
                        seed,
                        "mixed",
                        "needle",
                        "",
                        parent_rate,
                        "*",
                    )
                )
        for variant in ABLATION_VARIANTS:
            expected.add(
                (
                    variant,
                    "semantic-mean",
                    seed,
                    "needle-v2",
                    "",
                    "",
                    "",
                    "1",
                )
            )
            for workload in ("chat", "rag"):
                expected.add(
                    (
                        variant,
                        "semantic-mean",
                        seed,
                        workload,
                        "",
                        "8.0",
                        "",
                        "",
                    )
                )

    successful_rows = [row for row in rows if not (row.get("error") or "").strip()]
    actual = {cell(row) for row in successful_rows}
    for identity in sorted(expected - actual):
        errors.append(f"missing result cell: {identity}")
    for identity in sorted(actual - expected):
        errors.append(f"unexpected result cell: {identity}")

    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for row in successful_rows:
        identity = cell(row)
        counts[identity] += 1
        is_repeatable_needle = (
            row.get("workload") == "mixed" and row.get("sub_workload") == "needle"
        )
        if counts[identity] > 1 and not is_repeatable_needle:
            errors.append(f"duplicate result cell: {identity}")

        needle = row.get("workload") == "needle-v2" or (
            row.get("workload") == "mixed" and row.get("sub_workload") == "needle"
        )
        required = (
            ("needle_outcome", "recall_load_bytes", "recall_store_bytes")
            if needle
            else (
                "duration_s",
                "ttft_p50_ms",
                "ttft_p99_ms",
                "throughput_tok_s",
                "load_bytes_delta",
                "store_bytes_delta",
                "preemptions_delta",
            )
        )
        for field in required:
            value = (row.get(field) or "").strip()
            if not value:
                errors.append(f"{identity}: missing required field {field}")
            elif field != "needle_outcome" and _number(value) is None:
                errors.append(f"{identity}: non-finite numeric field {field}")
        if needle and row.get("needle_outcome") not in {
            "hit",
            "partial",
            "miss",
            "not_pressured",
        }:
            errors.append(f"{identity}: invalid needle outcome")
        if row.get("workload") == "mixed" and row.get("sub_workload") == "needle":
            reference_count = (row.get("reference_count") or "").strip()
            try:
                parsed_reference_count = int(reference_count)
            except ValueError:
                parsed_reference_count = -1
            if parsed_reference_count not in {0, 1, 2}:
                errors.append(f"{identity}: invalid mixed needle reference_count")

    for ref_count in (0, 1, 2):
        variant = f"leaderboard_ref{ref_count}"
        for policy in LEADERBOARD_POLICIES:
            outcomes = {
                row.get("needle_outcome", "")
                for row in successful_rows
                if row.get("variant") == variant and row.get("policy") == policy
            }
            if outcomes and outcomes <= {"not_pressured", ""}:
                errors.append(f"{variant}/{policy}: no pressured needle outcome")
    return errors


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["variant"],
            row.get("policy", ""),
            row.get("workload", ""),
            row.get("sub_workload", ""),
            row.get("request_rate", ""),
            row.get("parent_request_rate", ""),
            row.get("reference_count", ""),
        )
        groups[key].append(row)

    output: list[dict] = []
    for key, group in sorted(groups.items()):
        outcomes = [(row.get("needle_outcome") or "").strip() for row in group]
        hits = outcomes.count("hit")
        misses = outcomes.count("miss")
        partials = outcomes.count("partial")
        pressured = hits + misses + partials
        summary = {
            "variant": key[0],
            "policy": key[1],
            "workload": key[2],
            "sub_workload": key[3],
            "request_rate": key[4],
            "parent_request_rate": key[5],
            "reference_count": key[6],
            "runs": len(group),
            "needle_hits": hits,
            "needle_misses": misses,
            "needle_partials": partials,
            "needle_not_pressured": outcomes.count("not_pressured"),
            "needle_pressured_hit_rate": hits / pressured if pressured else "",
            "needle_pressured_any_load_rate": (
                (hits + partials) / pressured if pressured else ""
            ),
        }
        for field in NUMERIC_FIELDS:
            values = [
                value for row in group if (value := _number(row.get(field))) is not None
            ]
            summary[f"mean_{field}"] = statistics.fmean(values) if values else ""
        output.append(summary)
    return output


def alpha_pair_rows(rows: list[dict]) -> list[dict]:
    """Return seed-level alpha=.5/.6 comparisons without hiding outcomes."""

    def identity(row: dict) -> tuple[str, ...]:
        return tuple(
            row.get(field, "")
            for field in (
                "seed",
                "workload",
                "sub_workload",
                "request_rate",
                "parent_request_rate",
                "reference_count",
            )
        )

    successful = [row for row in rows if not (row.get("error") or "").strip()]
    baseline = {
        identity(row): row
        for row in successful
        if row.get("variant") == ALPHA_BASELINE_VARIANT
    }
    candidate = {
        identity(row): row
        for row in successful
        if row.get("variant") == ALPHA_CANDIDATE_VARIANT
    }

    output: list[dict] = []

    def append(
        key: tuple[str, ...],
        metric: str,
        direction: str,
        baseline_value: str | float,
        candidate_value: str | float,
        delta: str | float,
    ) -> None:
        output.append(
            {
                "seed": key[0],
                "workload": key[1],
                "sub_workload": key[2],
                "request_rate": key[3],
                "parent_request_rate": key[4],
                "reference_count": key[5],
                "metric": metric,
                "preferred_direction": direction,
                "alpha_05": baseline_value,
                "alpha_06": candidate_value,
                "delta_alpha06_minus_alpha05": delta,
            }
        )

    for key in sorted(baseline.keys() & candidate.keys()):
        baseline_row = baseline[key]
        candidate_row = candidate[key]
        if key[1] == "needle-v2":
            baseline_outcome = baseline_row.get("needle_outcome", "")
            candidate_outcome = candidate_row.get("needle_outcome", "")
            append(
                key,
                "needle_outcome",
                "categorical",
                baseline_outcome,
                candidate_outcome,
                "",
            )
            pressured = {"hit", "partial", "miss"}
            if baseline_outcome in pressured and candidate_outcome in pressured:
                for metric, successful_outcomes in (
                    ("needle_full_hit", {"hit"}),
                    ("needle_any_load", {"hit", "partial"}),
                ):
                    baseline_value = float(baseline_outcome in successful_outcomes)
                    candidate_value = float(candidate_outcome in successful_outcomes)
                    append(
                        key,
                        metric,
                        "higher",
                        baseline_value,
                        candidate_value,
                        candidate_value - baseline_value,
                    )
            continue

        for metric, direction in ALPHA_METRICS.items():
            baseline_value = _number(baseline_row.get(metric))
            candidate_value = _number(candidate_row.get(metric))
            if baseline_value is None or candidate_value is None:
                continue
            append(
                key,
                metric,
                direction,
                baseline_value,
                candidate_value,
                candidate_value - baseline_value,
            )
    return output


def aggregate_timings(root: Path) -> list[dict]:
    """Use each process/log's last cumulative timing snapshot."""
    latest: dict[tuple, tuple[int, float]] = {}
    max_rss: dict[str, float] = {}
    for path in root.rglob("*.log"):
        relative = path.relative_to(root)
        variant = relative.parts[0] if len(relative.parts) > 1 else path.stem
        server_match = SERVER_LOG_PATTERN.match(path.name)
        if server_match:
            variant = f"{variant}:{server_match.group(1)}"
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            timing = TIMING_PATTERN.search(line)
            if timing:
                bucket, pid, calls, total_s, _ = timing.groups()
                key = (variant, str(path), pid, bucket)
                value = (int(calls), float(total_s))
                if value[0] >= latest.get(key, (0, 0.0))[0]:
                    latest[key] = value
            process = PROCESS_PATTERN.search(line)
            if process:
                rss = float(process.group(4))
                max_rss[variant] = max(max_rss.get(variant, 0.0), rss)

    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (variant, _path, _pid, bucket), value in latest.items():
        grouped[(variant, bucket)].append(value)
    output = []
    for (variant, bucket), values in sorted(grouped.items()):
        calls = sum(value[0] for value in values)
        total_s = sum(value[1] for value in values)
        output.append(
            {
                "variant": variant,
                "bucket": bucket,
                "processes": len(values),
                "calls": calls,
                "total_s": total_s,
                "mean_ms": 1000 * total_s / calls if calls else "",
                "max_rss_mb": max_rss.get(variant, ""),
            }
        )
    return output


def write_summary(
    root: Path,
    summaries: list[dict],
    timings: list[dict],
    alpha_pairs: list[dict],
    errors: list[str],
) -> None:
    csv_path = root / "audit_summary.csv"
    if summaries:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    if timings:
        with (root / "timing_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
            writer.writeheader()
            writer.writerows(timings)
    if alpha_pairs:
        with (root / "alpha_paired_seed_deltas.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(alpha_pairs[0]))
            writer.writeheader()
            writer.writerows(alpha_pairs)

    markdown = ["# DGX policy audit summary", ""]
    if errors:
        markdown += ["## Validation failures", ""]
        markdown += [f"- {error}" for error in errors]
        markdown.append("")
    markdown += [
        (
            "| Variant | Policy | Workload | Rate/ref | Runs | Needle H/P/M/NP | "
            "Full-hit LB | Any-load rate | TTFT p50 ms | TTFT p99 ms | Tok/s | "
            "Preemptions |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        rate_ref = (
            row["parent_request_rate"]
            or row["request_rate"]
            or row["reference_count"]
            or "-"
        )
        hit_rate = row["needle_pressured_hit_rate"]
        hit_rate_text = f"{100 * hit_rate:.1f}%" if hit_rate != "" else "-"
        any_load_rate = row["needle_pressured_any_load_rate"]
        any_load_text = f"{100 * any_load_rate:.1f}%" if any_load_rate != "" else "-"

        markdown.append(
            f"| {row['variant']} | {row['policy']} | {row['workload']} "
            f"| {rate_ref} | {row['runs']} | {row['needle_hits']}/"
            f"{row['needle_partials']}/{row['needle_misses']}/"
            f"{row['needle_not_pressured']} | "
            f"{hit_rate_text} | {any_load_text} | "
            f"{_display(row, 'mean_ttft_p50_ms')} | "
            f"{_display(row, 'mean_ttft_p99_ms')} | "
            f"{_display(row, 'mean_throughput_tok_s')} | "
            f"{_display(row, 'mean_preemptions_delta')} |"
        )
    alpha_outcomes = [row for row in alpha_pairs if row["metric"] == "needle_outcome"]
    if alpha_outcomes:
        markdown += [
            "",
            "## Paired alpha 0.5 vs 0.6 seed outcomes",
            "",
            (
                "Alpha 0.6 remains experimental. Empty numeric pairs indicate that "
                "at least one seed was not pressured; concurrent-grid performance "
                "deltas remain provisional pending isolated one-GPU confirmation."
            ),
            "",
            "| Seed | Workload | Rate/ref | Alpha 0.5 | Alpha 0.6 |",
            "|---:|---|---:|---|---|",
        ]
        for row in alpha_outcomes:
            rate_ref = (
                row["parent_request_rate"]
                or row["request_rate"]
                or row["reference_count"]
                or "-"
            )
            markdown.append(
                f"| {row['seed']} | {row['workload']} | {rate_ref} | "
                f"{row['alpha_05']} | {row['alpha_06']} |"
            )

    numeric_alpha_pairs = [
        row for row in alpha_pairs if row["delta_alpha06_minus_alpha05"] != ""
    ]
    if numeric_alpha_pairs:
        grouped_pairs: dict[tuple[str, ...], list[dict]] = defaultdict(list)
        for row in numeric_alpha_pairs:
            key = (
                row["workload"],
                row["sub_workload"],
                row["request_rate"],
                row["parent_request_rate"],
                row["reference_count"],
                row["metric"],
                row["preferred_direction"],
            )
            grouped_pairs[key].append(row)
        markdown += [
            "",
            "### Paired alpha aggregate deltas",
            "",
            (
                "Deltas are alpha 0.6 minus alpha 0.5. Positive is preferred only "
                "for `higher` metrics; negative is preferred only for `lower` "
                "metrics. Byte deltas are context, not wins."
            ),
            "",
            (
                "| Workload | Rate/ref | Metric | Direction | Pairs | Alpha 0.5 | "
                "Alpha 0.6 | Mean delta |"
            ),
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
        for key, group in sorted(grouped_pairs.items()):
            rate_ref = key[3] or key[2] or key[4] or "-"
            markdown.append(
                f"| {key[0]} | {rate_ref} | {key[5]} | {key[6]} | "
                f"{len(group)} | "
                f"{statistics.fmean(float(row['alpha_05']) for row in group):.4f} | "
                f"{statistics.fmean(float(row['alpha_06']) for row in group):.4f} | "
                f"{statistics.fmean(float(row['delta_alpha06_minus_alpha05']) for row in group):.4f} |"
            )
    if timings:
        markdown += [
            "",
            "## Semantic hot-path timings",
            "",
            "| Variant | Bucket | Processes | Calls | Mean ms | Max RSS MB |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in timings:
            markdown.append(
                f"| {row['variant']} | {row['bucket']} | {row['processes']} | "
                f"{row['calls']} | {_display(row, 'mean_ms')} | "
                f"{_display(row, 'max_rss_mb')} |"
            )
    (root / "audit_summary.md").write_text("\n".join(markdown) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument(
        "--pre-summary",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--expected-seeds", default="1,2,3")
    args = parser.parse_args()
    rows, errors = load_results(args.root)
    expected_seeds = set(args.expected_seeds.split(","))
    errors.extend(validate_matrix(rows, expected_seeds))
    errors.extend(validate_variant_manifests(args.root, expected_seeds))
    errors.extend(
        validate_repository_state(args.root, require_summary_end=not args.pre_summary)
    )
    summaries = aggregate(rows)
    timings = aggregate_timings(args.root)
    alpha_pairs = alpha_pair_rows(rows)
    write_summary(args.root, summaries, timings, alpha_pairs, errors)
    print(f"summarized {len(rows)} rows into {args.root / 'audit_summary.md'}")
    if errors and not args.allow_errors:
        raise SystemExit(f"audit validation failed with {len(errors)} error(s)")


if __name__ == "__main__":
    main()
