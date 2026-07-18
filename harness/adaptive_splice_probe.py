# SPDX-License-Identifier: Apache-2.0
"""Adaptive splice-correctness probe -- the fix flagged in issues log entries
#43/#45/#46 (`.claude/docs/semantic-eviction-issues-log.md` in the
vllm-semantic-cache repo) for why a fixed-schedule driver kept missing the
byte-exact splice proof: splice-target selection is close to random among
concurrently-preempted requests, so tagging a handful of requests and firing
the recall on a fixed timer has repeatedly missed by chance (0 hits across
16 tagged attempts, against 55 real splices that happened to OTHER,
untagged requests in the same runs).

Instead of guessing when to send the recall, this WATCHES the server's own
debug log live (tail -f style, requires SEMANTIC_OFFLOAD_DEBUG=1) for the
tagged request's own id to appear in a `PARTIAL SPLICE`/`SPLICED` line with
`spliced>=1`, and fires the recall the instant that happens -- turning a
lucky-timing problem into a deterministic one.
"""

import re
import time

import requests

_SPLICE_RE = re.compile(
    r"PREFETCH_EFFECT_DEBUG (?P<req_id>\S+): "
    r"(?:PARTIAL SPLICE spliced=(?P<spliced>\d+)|SPLICED)"
)


def watch_for_splice(
    log_path: str,
    request_id_tag: str,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
) -> dict | None:
    """Tail `log_path` from its current end, watching for a
    PREFETCH_EFFECT_DEBUG line whose request id contains `request_id_tag`
    and reports a real splice (`spliced>=1`, or the older unconditional
    `SPLICED` marker). Returns a dict with the matched line and parsed
    fields, or None on timeout. Does not modify the log file."""
    with open(log_path) as f:
        f.seek(0, 2)  # start from current end -- don't replay old lines
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = f.readline()
            if not line:
                time.sleep(poll_interval_s)
                continue
            if request_id_tag not in line:
                continue
            m = _SPLICE_RE.search(line)
            if not m:
                continue
            spliced = m.group("spliced")
            if spliced is not None and int(spliced) < 1:
                continue
            return {
                "line": line.strip(),
                "req_id": m.group("req_id"),
                "spliced": int(spliced) if spliced is not None else None,
            }
    return None


def run_adaptive_probe(
    base_url: str,
    model: str,
    log_path: str,
    request_id_tag: str,
    needle_prompt: str,
    expected_code: str,
    recall_suffix: str = (
        "\n\nQuestion: what is the secret verification code mentioned "
        "above? Answer with ONLY the code, nothing else."
    ),
    splice_timeout_s: float = 300.0,
    recall_timeout_s: float = 120.0,
) -> dict:
    """Send the needle as a tagged request, then block until the log shows
    it was spliced (or timeout), then immediately send the recall and check
    for an exact match. Returns a result dict; never raises on a timeout or
    content mismatch -- those are reported in the result, not exceptions,
    since a negative result here is informative, not a bug in the probe."""
    needle_id = f"{request_id_tag}-needle"
    requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": needle_prompt,
            "max_tokens": 8,
            "temperature": 0.0,
            "request_id": needle_id,
        },
        timeout=recall_timeout_s,
    )

    splice_event = watch_for_splice(log_path, needle_id, timeout_s=splice_timeout_s)
    if splice_event is None:
        return {"hit_splice": False, "splice_event": None, "correct": None}

    recall_id = f"{request_id_tag}-recall"
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": needle_prompt + recall_suffix,
            "max_tokens": 20,
            "temperature": 0.0,
            "request_id": recall_id,
        },
        timeout=recall_timeout_s,
    )
    resp.raise_for_status()
    recall_text = resp.json()["choices"][0]["text"]
    return {
        "hit_splice": True,
        "splice_event": splice_event,
        "recall_text": recall_text,
        "expected_code": expected_code,
        "correct": expected_code in recall_text,
    }
