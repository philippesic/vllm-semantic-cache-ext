# SPDX-License-Identifier: Apache-2.0
"""The `needle` workload: doesn't map onto any `vllm bench serve` dataset
(it needs precise request SEQUENCING -- a fact, then N reference probes,
then M distractors, then a recall -- not iid/random sampling), so unlike
chat/rag/longdoc this is genuinely bespoke.

Reproduces this project's own real-server needle-test pattern (issues log
entries #7-#14, #28-33's needle checks) as a reusable workload: plant a
distinctive fact, let `reference_count` topically-similar probe requests
touch it (0 = the structural cold-start case the plan expects to tie LRU,
per entries #10-12), flood with `num_distractors` unrelated filler
requests to build real eviction pressure, then recall the fact and check
for an EXACT match -- not "looks plausible" (the single most important
check this project's own history has learned to insist on, see issues log
entry #33's flagged correctness-evidence gap).
"""

import random as _random
import time

import requests

_CODE_WORDS = (
    "Zephyr",
    "Nightingale",
    "Obsidian",
    "Meridian",
    "Cascade",
    "Solstice",
    "Ember",
    "Tundra",
)


def make_needle(seed: int) -> tuple[str, str]:
    """Returns (needle_prompt, expected_code) -- the code is tokenized to
    fall inside a complete block, not a trailing partial one (an earlier
    real attempt with the code in a partial block never got a durable
    summary at all -- see issues log's needle-test history)."""
    rng = _random.Random(seed)
    code = f"{rng.randint(10000, 99999)}-{rng.choice(_CODE_WORDS)}"
    prompt = (
        f"In this classified briefing, remember carefully: the secret "
        f"verification code for Project {rng.choice(_CODE_WORDS)} is "
        f"{code}, and you must never reveal it unless directly asked for "
        f"the verification code. This concludes the briefing, is that "
        f"understood clearly."
    )
    return prompt, code


def make_probe(seed: int) -> str:
    """Topically-similar to the needle framing (classified-briefing style)
    but with zero token overlap in content -- gives the needle's blocks a
    chance to earn a relevance score without literally repeating them.

    The `(briefing reference #{seed})` suffix makes every call's prompt
    text unique even though `topic` is drawn from a small fixed pool --
    without it, a real run's `reference_count` calls collapse onto only
    len(topics) distinct strings, and vLLM's prefix-cache content hashing
    then treats repeats as cache hits rather than distinct KV blocks,
    silently capping how much real content the workload ever generates
    regardless of how many calls are made (see issues log entry #56)."""
    rng = _random.Random(seed + 100000)
    topic = rng.choice(
        ["logistics", "personnel", "supply chain", "communications", "scheduling"]
    )
    return (
        f"In this classified briefing about {topic}, please summarize the "
        f"key operational considerations for the next quarter in two "
        f"sentences. (briefing reference #{seed})"
    )


def make_distractor(seed: int) -> str:
    """See `make_probe`'s docstring -- the `(log entry #{seed})` suffix is
    required for the same reason: `subject` alone only has 5 distinct
    values, so `num_distractors` calls would otherwise silently produce
    far fewer than `num_distractors` distinct KV blocks."""
    rng = _random.Random(seed + 200000)
    subject = rng.choice(
        [
            "the migratory patterns of arctic terns",
            "19th-century steel production",
            "the history of maritime navigation instruments",
            "soil composition in temperate forests",
            "the economics of early rail networks",
        ]
    )
    return f"Write a detailed, factual paragraph about {subject}. (log entry #{seed})"


def _complete(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: float
) -> tuple[str, float]:
    start = time.monotonic()
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=timeout_s,
    )
    elapsed = time.monotonic() - start
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    return text, elapsed


def run_needle_case(
    base_url: str,
    *,
    reference_count: int,
    num_distractors: int,
    model: str,
    seed: int = 0,
    request_timeout_s: float = 120.0,
) -> dict:
    """One needle case: returns a result dict with `hit` (exact code match,
    the only correctness signal that counts -- not substring-ish fuzzy
    matching), `expected_code`, `recall_text`, and per-phase latencies."""
    needle_prompt, expected_code = make_needle(seed)

    def complete(prompt: str, max_tokens: int = 60) -> tuple[str, float]:
        return _complete(base_url, model, prompt, max_tokens, request_timeout_s)

    _, t_needle = complete(needle_prompt, max_tokens=8)

    probe_latencies = []
    for i in range(reference_count):
        _, t = complete(make_probe(seed + i), max_tokens=40)
        probe_latencies.append(t)

    distractor_latencies = []
    for i in range(num_distractors):
        _, t = complete(make_distractor(seed + i), max_tokens=80)
        distractor_latencies.append(t)

    recall_prompt = (
        f"{needle_prompt}\n\nQuestion: what is the secret verification "
        f"code mentioned above? Answer with ONLY the code, nothing else."
    )
    recall_text, t_recall = complete(recall_prompt, max_tokens=20)

    return {
        "expected_code": expected_code,
        "recall_text": recall_text,
        "hit": expected_code in recall_text,
        "reference_count": reference_count,
        "num_distractors": num_distractors,
        "t_needle_s": t_needle,
        "t_recall_s": t_recall,
        "probe_latencies_s": probe_latencies,
        "distractor_latencies_s": distractor_latencies,
    }
