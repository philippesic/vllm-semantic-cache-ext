# SPDX-License-Identifier: Apache-2.0
"""Snapshot/diff vLLM's real Prometheus /metrics endpoint for the
offload-subsystem counters this project cares about (CPU-cache hit rate,
prefetch hit rate, transfer volumes, preemption count) -- `vllm bench
serve` reports TTFT/ITL/throughput itself; this fills the gap the plan's
Benchmarking section calls out separately ("Offload-subsystem internals
via existing Prometheus metrics + our additions").

Deliberately a thin, dependency-free text-format parser (stdlib urllib +
regex) rather than pulling in prometheus_client -- the exposition format
is simple enough (`name{labels} value`) that a real parser dependency
would be more machinery than the problem needs."""

import re
import urllib.request

# Prefixes worth tracking for the latency suite's "index overhead" and
# "offload internals" metrics (plan's Benchmarking section). Summed across
# all label combinations for a given metric name -- per-run totals, not
# broken out per-label, since none of these currently vary by label in
# this project's single-model single-worker setup.
TRACKED_METRICS = (
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:num_preemptions_total",
)

_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(?P<value>[-\d.eE+]+)\s*$"
)


def fetch_metrics_text(metrics_url: str, timeout_s: float = 10.0) -> str:
    with urllib.request.urlopen(metrics_url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_metrics(text: str) -> dict[str, float]:
    """Sum every tracked metric's value across all its label combinations."""
    totals: dict[str, float] = dict.fromkeys(TRACKED_METRICS, 0.0)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name in totals:
            try:
                totals[name] += float(m.group("value"))
            except ValueError:
                continue
    return totals


def snapshot(metrics_url: str) -> dict[str, float]:
    return parse_metrics(fetch_metrics_text(metrics_url))


def diff(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in before}
