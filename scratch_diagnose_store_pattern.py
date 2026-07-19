# SPDX-License-Identifier: Apache-2.0
"""One-off diagnostic (not part of the test suite): sends the real needle
workload's 25 distinct distractor prompts one at a time to an already-
running server and prints the kv_offload_store_bytes_total delta after
EVERY call, to see exactly which calls store something vs which don't --
issues log entry #56 follow-up, the aggregate case-level totals alone
couldn't distinguish "each call stores ~0.36 blocks on average" from "most
calls store 0, a few store several"."""

import sys

sys.path.insert(0, "harness")

import requests  # noqa: E402

import metrics as metrics_mod  # noqa: E402
import needle_workload as nw  # noqa: E402

BASE_URL = "http://localhost:8199"
BLOCK_BYTES = 917_504  # 7B model, this project's derived constant

before_total = metrics_mod.snapshot(f"{BASE_URL}/metrics")
print(f"baseline store_bytes_total={before_total['vllm:kv_offload_store_bytes_total']}")

cumulative_blocks = 0.0
for i in range(25):
    prompt = nw.make_distractor(i)
    before = metrics_mod.snapshot(f"{BASE_URL}/metrics")
    resp = requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": prompt,
            "max_tokens": 80,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    after = metrics_mod.snapshot(f"{BASE_URL}/metrics")
    delta = metrics_mod.diff(before, after)["vllm:kv_offload_store_bytes_total"]
    blocks = delta / BLOCK_BYTES
    cumulative_blocks += blocks
    print(
        f"i={i:2d} blocks_stored={blocks:5.2f} "
        f"cumulative={cumulative_blocks:6.2f}  prompt={prompt!r}"
    )
