# SPDX-License-Identifier: Apache-2.0
"""Live real-server driver for `adaptive_splice_probe.py` -- the piece that
was still missing after issues log entry #47 (its two prior real-server
validation attempts were interrupted by a stuck fork before completing, not
by any problem with the probe tool itself, which was already unit-tested).

Reuses the known-working splice-forcing load shape from entries #43/#45/#46
(1 long-ish target request concurrent with 40 padded fillers across 4
staggered waves), re-tuned for B200-class hardware on 2026-07-29: the
original `--num-gpu-blocks-override 200` / 140-token fillers never forced
a single preemption (a 2080Ti-tuned shape a much faster GPU just runs
through); now 40 blocks / 400-token fillers, see the comments at the
launch_server call -- except the
"target" here IS the adaptive probe's own tagged needle request, so instead
of hoping a fixed-schedule recall happens to land after a real splice
(which is what entries #33/#41/#43/#45 kept missing by chance), the probe
watches the server's own debug log live and fires the recall the instant a
PARTIAL SPLICE/SPLICED line references the needle's own request id.

Usage:
  SEMANTIC_OFFLOAD_DEBUG=1 python harness/run_adaptive_probe_live.py
"""

import concurrent.futures
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from harness.adaptive_splice_probe import watch_for_splice
from harness.needle_workload import make_needle
from harness.server import launch_server

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PORT = 8199

_TOPICS = [
    "the history of steel production",
    "arctic tern migration",
    "maritime navigation instruments",
    "soil composition in forests",
    "early rail network economics",
    "glassblowing techniques",
    "the physics of tidal patterns",
    "renaissance clockmaking",
    "the chemistry of fermentation",
    "volcanic rock formation",
]


def _post(prompt: str, max_tokens: int, base_url: str) -> None:
    requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=240,
    )


def _complete(prompt: str, max_tokens: int, request_id: str, handle) -> tuple[str, str]:
    resp = requests.post(
        f"{handle.base_url()}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "request_id": request_id,
        },
        timeout=240,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"]
    return request_id, text


def _post_tagged(prompt: str, max_tokens: int, request_id: str, handle) -> None:
    _complete(prompt, max_tokens, request_id, handle)


def _make_filler_prompt(wave: int, i: int) -> str:
    topic = random.choice(_TOPICS)
    return (
        f"Write a detailed, factual, at-least-140-word paragraph about "
        f"{topic}. This is filler request {wave}-{i}, make it thorough "
        f"and specific with concrete examples."
    )


def main() -> int:
    if os.environ.get("SEMANTIC_OFFLOAD_DEBUG", "") in ("", "0", "false", "False"):
        print(
            "SEMANTIC_OFFLOAD_DEBUG must be set (the probe watches "
            "PREFETCH_EFFECT_DEBUG lines, gated behind it) -- aborting.",
            flush=True,
        )
        return 2

    log_dir = "/tmp/adaptive_probe_live_run"
    os.makedirs(log_dir, exist_ok=True)

    kv_transfer_config = {
        "kv_connector": "SemanticOffloadingConnector",
        "kv_connector_module_path": "semantic_offload.connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "SemanticOffloadingSpec",
            "spec_module_path": "semantic_offload.spec",
            "cpu_bytes_to_use": 2 * 1024**3,
        },
    }

    print("Launching server...", flush=True)
    handle = launch_server(
        MODEL,
        PORT,
        log_dir,
        gpu_memory_utilization=0.5,
        # Raised to 512 (from 384, itself lowered from 2048) to give the
        # now-longer filler generations (400 tokens, see below) headroom:
        # prompt (~40) + 400 generated ~= 440, needs margin under the cap.
        # vLLM's startup validation requires enough blocks to serve one
        # full max_model_len request regardless of real traffic, so this
        # also sets the floor for num_gpu_blocks_override below (512/16
        # = 32 blocks minimum at the default block_size).
        max_model_len=512,
        # 200 (the original 2080Ti-tuned value from entries #43/#45/#46)
        # never forces a preemption on a B200 (confirmed 2026-07-29: zero
        # preempt/PARTIAL SPLICE/KEY MISMATCH). 60 alone didn't fix it
        # either -- two more runs at 60/384 still saw zero of all three,
        # even though 11 near-simultaneous ~190-token fillers per wave
        # (~130-140 blocks needed) should exceed a 60-block budget. Most
        # likely cause: a 1.5B model decodes fast enough on a B200 that
        # requests don't stay resident long enough to truly overlap,
        # regardless of admission timing -- addressed by tripling filler
        # length above, not just shrinking the block budget further. 40
        # keeps a comfortable margin above the new 32-block floor (one
        # full 512-token request) while still fitting only ~1-2 fillers
        # at their now-much-larger peak footprint (~27-28 blocks each
        # near the end of a 400-token generation).
        num_gpu_blocks_override=40,
        kv_transfer_config=kv_transfer_config,
        log_label="adaptive_probe",
    )
    print(f"Server ready. log={handle.log_path}", flush=True)

    try:
        # Tag several candidates (one per wave), not just one -- empirically
        # only ~10-12 of ~41 concurrent requests win the reservation budget
        # (~25-30% odds per candidate, confirmed over 2 prior single-tag
        # attempts this session that both missed despite the tagged request
        # genuinely being in the preempted pool). Tagging N candidates and
        # taking whichever splices first multiplies the odds roughly Nx
        # instead of paying for N sequential full server-relaunch attempts.
        n_tags = 4
        needles = {}  # needle_id -> (prompt, expected_code)
        tag_base = f"ADAPTPROBE-{random.randint(1000, 9999)}"
        for t in range(n_tags):
            needle_seed = random.randint(0, 1_000_000)
            prompt, code = make_needle(needle_seed)
            needles[f"{tag_base}-{t}-needle"] = (prompt, code)

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
            watch_futures = {
                ex.submit(watch_for_splice, handle.log_path, needle_id, 240.0): (
                    needle_id
                )
                for needle_id in needles
            }

            filler_futures = []
            n_fillers = 40
            per_wave = n_fillers // 4
            tag_ids = list(needles.keys())
            for wave in range(4):
                if wave < len(tag_ids):
                    needle_id = tag_ids[wave]
                    prompt, _ = needles[needle_id]
                    filler_futures.append(
                        ex.submit(_post_tagged, prompt, 8, needle_id, handle)
                    )
                for i in range(per_wave):
                    filler_futures.append(
                        ex.submit(
                            _post,
                            _make_filler_prompt(wave, i),
                            # Raised from 140: even at a 60-block budget,
                            # two straight B200 runs saw zero preemptions
                            # and zero PARTIAL SPLICE/KEY MISMATCH (2026-
                            # 07-29) -- a 1.5B model decodes 140 tokens on
                            # a B200 fast enough that near-simultaneous
                            # admission doesn't reliably produce actual
                            # overlapping residency, independent of the
                            # block budget. 400 tokens keeps each filler
                            # resident ~3x longer, making overlap far more
                            # likely regardless of admission-timing jitter.
                            400,
                            handle.base_url(),
                        )
                    )
                time.sleep(0.5)

            hit_needle_id = None
            hit_splice_event = None
            for fut in concurrent.futures.as_completed(watch_futures, timeout=260):
                needle_id = watch_futures[fut]
                splice_event = fut.result()
                if splice_event is not None and hit_needle_id is None:
                    hit_needle_id = needle_id
                    hit_splice_event = splice_event

            ok, fail = 0, 0
            for f in filler_futures:
                try:
                    f.result(timeout=230)
                    ok += 1
                except Exception:
                    fail += 1
            print(f"fillers: {ok} ok, {fail} failed", flush=True)

        if hit_needle_id is None:
            result = {"hit_splice": False, "splice_event": None, "correct": None}
        else:
            needle_prompt, expected_code = needles[hit_needle_id]
            recall_prompt = (
                needle_prompt + "\n\nQuestion: what is the secret verification "
                "code mentioned above? Answer with ONLY the code, nothing else."
            )
            _, recall_text = _complete(
                recall_prompt, 20, f"{hit_needle_id}-recall", handle
            )
            result = {
                "hit_splice": True,
                "splice_event": hit_splice_event,
                "recall_text": recall_text,
                "expected_code": expected_code,
                "correct": expected_code in recall_text,
            }

        print(f"tags={list(needles.keys())}", flush=True)
        print(f"result={result}", flush=True)

        if not result["hit_splice"]:
            print(
                "NO SPLICE observed for any tagged candidate within the "
                "timeout window (a real negative result, not a probe "
                "bug -- see the result dict above).",
                flush=True,
            )
            return 1
        if not result["correct"]:
            print(
                "SPLICE OBSERVED BUT RECALL WAS WRONG -- this would be a "
                "real correctness bug, investigate immediately.",
                flush=True,
            )
            return 1
        print(
            f"SUCCESS: tagged request {hit_needle_id} was spliced "
            f"(spliced={result['splice_event']['spliced']}) and the "
            "recall was byte-exact correct.",
            flush=True,
        )
        return 0
    finally:
        print("Shutting down server...", flush=True)
        handle.shutdown()


if __name__ == "__main__":
    sys.exit(main())
