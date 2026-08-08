# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize all result sets produced by dgx_policy_audit.sh."""

import argparse
import csv
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
    root: Path, summaries: list[dict], timings: list[dict], errors: list[str]
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

    markdown = ["# DGX policy audit summary", ""]
    if errors:
        markdown += ["## Validation failures", ""]
        markdown += [f"- {error}" for error in errors]
        markdown.append("")
    markdown += [
        "| Variant | Policy | Workload | Rate/ref | Runs | Needle H/P/M/NP | "
        "Full-hit LB | Any-load rate | TTFT p50 ms | TTFT p99 ms | Tok/s | "
        "Preemptions |",
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
    parser.add_argument("--expected-seeds", default="1,2,3")
    args = parser.parse_args()
    rows, errors = load_results(args.root)
    errors.extend(validate_matrix(rows, set(args.expected_seeds.split(","))))
    summaries = aggregate(rows)
    timings = aggregate_timings(args.root)
    write_summary(args.root, summaries, timings, errors)
    print(f"summarized {len(rows)} rows into {args.root / 'audit_summary.md'}")
    if errors and not args.allow_errors:
        raise SystemExit(f"audit validation failed with {len(errors)} error(s)")


if __name__ == "__main__":
    main()
