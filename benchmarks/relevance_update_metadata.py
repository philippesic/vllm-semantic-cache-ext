# SPDX-License-Identifier: Apache-2.0
"""Benchmark raw versus compact relevance metadata in the production envelope.

The multiprocess executor sends ``ModelRunnerOutput`` through a vLLM
``MessageQueue`` whose payload is highest-protocol pickle. This benchmark uses
the actual response enum and response/model/KV object envelope. Byte counts are
for its tensor-free pickle payload and exclude constant queue framing.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vllm  # noqa: E402
from vllm.v1.executor.multiproc_executor import WorkerProc  # noqa: E402
from vllm.v1.kv_offload.base import make_offload_key  # noqa: E402
from vllm.v1.outputs import (  # noqa: E402
    KVConnectorOutput,
    ModelRunnerOutput,
)

from semantic_offload.connector import SemanticWorkerMetadata  # noqa: E402
from semantic_offload.manager import (  # noqa: E402
    SemanticOffloadingManager,
    compose_relevance_updates,
)

DEFAULT_REQUESTS = (1, 16, 56)
DEFAULT_CANDIDATES = (32, 512, 2048)
METHOD = "mean"
SCORE_MODULUS = 1_000_003
MIN_ENVELOPE_REDUCTION = 4.0
MIN_SCHEDULER_SPEEDUP = 4.0
MIN_PIPELINE_SPEEDUP = 1.0

Score = tuple[bytes, float]
RankedScores = dict[str, dict[str, list[Score]]]


def _parse_int_list(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def _key(index: int) -> bytes:
    return make_offload_key(f"cp003-key-{index:08d}".encode(), 0)


def _score(seed: int, request_index: int, candidate_index: int) -> float:
    value = (
        (seed + 17) * 1_000_003
        + (request_index + 3) * 65_537
        + (candidate_index + 11) * 2_759
        + request_index * candidate_index * 31
    ) % SCORE_MODULUS
    return (float(value) - SCORE_MODULUS / 2.0) / (SCORE_MODULUS / 2.0)


def _ranked_scores(requests: int, candidates: int, seed: int) -> RankedScores:
    keys = [_key(index) for index in range(candidates)]
    return {
        METHOD: {
            f"req-{request_index:04d}": sorted(
                (
                    (key, _score(seed, request_index, candidate_index))
                    for candidate_index, key in enumerate(keys)
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for request_index in range(requests)
        }
    }


def _raw_metadata(scores: RankedScores) -> SemanticWorkerMetadata:
    requests = scores[METHOD]
    return SemanticWorkerMetadata(
        pending_scores=scores,
        score_head_counts={METHOD: {request_id: 1 for request_id in requests}},
        score_worker_counts={METHOD: {request_id: 1 for request_id in requests}},
        score_reduction="mean",
        score_group_size=1,
        score_contributors=1,
    )


def _compact_metadata(updates: Any) -> SemanticWorkerMetadata:
    return SemanticWorkerMetadata(
        relevance_updates=updates,
        score_reduction="mean",
        score_group_size=1,
        score_contributors=1,
    )


def _response_envelope(
    metadata: SemanticWorkerMetadata, requests: int
) -> tuple[WorkerProc.ResponseStatus, ModelRunnerOutput]:
    request_ids = [f"req-{index:04d}" for index in range(requests)]
    output = ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={
            request_id: index for index, request_id in enumerate(request_ids)
        },
        sampled_token_ids=[[] for _ in request_ids],
        kv_connector_output=KVConnectorOutput(kv_connector_worker_meta=metadata),
    )
    return WorkerProc.ResponseStatus.SUCCESS, output


def _serialize(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _round_trip_metadata(value: Any, requests: int) -> SemanticWorkerMetadata:
    decoded = pickle.loads(_serialize(value))
    if decoded[0] is not WorkerProc.ResponseStatus.SUCCESS:
        raise RuntimeError("production envelope did not preserve response status")
    output = decoded[1]
    expected_request_ids = [f"req-{index:04d}" for index in range(requests)]
    if output.req_ids != expected_request_ids or output.req_id_to_index != {
        request_id: index for index, request_id in enumerate(expected_request_ids)
    }:
        raise RuntimeError("production envelope did not preserve request identity")
    metadata = output.kv_connector_output.kv_connector_worker_meta
    if not isinstance(metadata, SemanticWorkerMetadata):
        raise RuntimeError("production envelope did not preserve semantic metadata")
    return metadata


def _manager(initial: dict[bytes, float] | None = None) -> SemanticOffloadingManager:
    manager = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    manager.relevance_ema = {METHOD: dict(initial)} if initial is not None else {}
    return manager


def _seeded_state(scores: RankedScores) -> dict[bytes, float]:
    first = next(iter(scores[METHOD].values()))
    return {
        key: (index - len(first) / 2.0) / max(len(first), 1)
        for index, (key, _) in enumerate(first)
        if index % 3 == 0
    }


def _assert_equivalent(scores: RankedScores, updates: Any) -> None:
    for initial in ({}, _seeded_state(scores)):
        raw = _manager(initial)
        compact = _manager(initial)
        raw.update_relevance(scores)
        compact.apply_relevance_updates(updates)
        raw_values = raw.relevance_ema[METHOD]
        compact_values = compact.relevance_ema[METHOD]
        same_values = raw_values.keys() == compact_values.keys() and all(
            math.isclose(
                raw_values[key], compact_values[key], rel_tol=1e-13, abs_tol=1e-13
            )
            for key in raw_values
        )
        raw_order = sorted(raw_values, key=lambda key: (-raw_values[key], key))
        compact_order = sorted(
            compact_values, key=lambda key: (-compact_values[key], key)
        )
        if not same_values or raw_order != compact_order:
            raise RuntimeError("compact update diverged from the sequential oracle")


def _timings(
    fn: Callable[[], Any], warmups: int, repetitions: int
) -> tuple[float, float]:
    for _ in range(warmups):
        fn()
    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repetitions):
            start = time.thread_time_ns()
            fn()
            samples.append((time.thread_time_ns() - start) / 1_000.0)
    finally:
        if gc_was_enabled:
            gc.enable()
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return statistics.median(ordered), ordered[p95_index]


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _run_cell(
    requests: int,
    candidates: int,
    seed: int,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    scores = _ranked_scores(requests, candidates, seed)
    updates = compose_relevance_updates(scores)
    _assert_equivalent(scores, updates)
    raw_metadata = _raw_metadata(scores)
    compact_metadata = _compact_metadata(updates)
    raw_envelope = _response_envelope(raw_metadata, requests)
    compact_envelope = _response_envelope(compact_metadata, requests)
    if _round_trip_metadata(raw_envelope, requests).pending_scores != scores:
        raise RuntimeError("raw production-envelope round trip changed scores")
    if _round_trip_metadata(compact_envelope, requests).relevance_updates != updates:
        raise RuntimeError("compact production-envelope round trip changed updates")

    initial = _seeded_state(scores)
    raw_manager = _manager()
    compact_manager = _manager()

    raw_build = _timings(lambda: _raw_metadata(scores), warmups, repetitions)
    compact_compose = _timings(
        lambda: compose_relevance_updates(scores), warmups, repetitions
    )
    raw_serialize = _timings(lambda: _serialize(raw_envelope), warmups, repetitions)
    compact_serialize = _timings(
        lambda: _serialize(compact_envelope), warmups, repetitions
    )

    def raw_apply() -> None:
        raw_manager.relevance_ema = {METHOD: dict(initial)}
        raw_manager.update_relevance(scores)

    def compact_apply() -> None:
        compact_manager.relevance_ema = {METHOD: dict(initial)}
        compact_manager.apply_relevance_updates(updates)

    raw_fold = _timings(raw_apply, warmups, repetitions)
    compact_fold = _timings(compact_apply, warmups, repetitions)

    def raw_pipeline() -> None:
        raw_manager.relevance_ema = {METHOD: dict(initial)}
        metadata = _raw_metadata(scores)
        _serialize(_response_envelope(metadata, requests))
        raw_manager.update_relevance(scores)

    def compact_pipeline() -> None:
        compact_manager.relevance_ema = {METHOD: dict(initial)}
        cell_updates = compose_relevance_updates(scores)
        metadata = _compact_metadata(cell_updates)
        _serialize(_response_envelope(metadata, requests))
        compact_manager.apply_relevance_updates(cell_updates)

    raw_pipeline_time = _timings(raw_pipeline, warmups, repetitions)
    compact_pipeline_time = _timings(compact_pipeline, warmups, repetitions)

    raw_metadata_bytes = len(_serialize(raw_metadata))
    compact_metadata_bytes = len(_serialize(compact_metadata))
    raw_envelope_bytes = len(_serialize(raw_envelope))
    compact_envelope_bytes = len(_serialize(compact_envelope))
    metadata_reduction = raw_metadata_bytes / compact_metadata_bytes
    envelope_reduction = raw_envelope_bytes / compact_envelope_bytes
    scheduler_speedup = raw_fold[0] / compact_fold[0]
    pipeline_speedup = raw_pipeline_time[0] / compact_pipeline_time[0]

    if requests == 1:
        production_path = "legacy_fallback"
        gate = "pass"
    elif requests >= 16 and candidates >= 512:
        production_path = "compact"
        gate = (
            "pass"
            if envelope_reduction >= MIN_ENVELOPE_REDUCTION
            and scheduler_speedup >= MIN_SCHEDULER_SPEEDUP
            and pipeline_speedup >= MIN_PIPELINE_SPEEDUP
            else "fail"
        )
    else:
        production_path = "compact"
        gate = "diagnostic"

    return {
        "R": requests,
        "C": candidates,
        "transport": "highest_protocol_pickle_payload",
        "python": sys.version.split()[0],
        "vllm": getattr(vllm, "__version__", "unknown"),
        "production_path": production_path,
        "gate": gate,
        "raw_metadata_bytes": raw_metadata_bytes,
        "compact_metadata_bytes": compact_metadata_bytes,
        "metadata_reduction_x": _fmt(metadata_reduction),
        "raw_envelope_bytes": raw_envelope_bytes,
        "compact_envelope_bytes": compact_envelope_bytes,
        "envelope_reduction_x": _fmt(envelope_reduction),
        "raw_build_median_us": _fmt(raw_build[0]),
        "compact_compose_median_us": _fmt(compact_compose[0]),
        "raw_serialize_median_us": _fmt(raw_serialize[0]),
        "compact_serialize_median_us": _fmt(compact_serialize[0]),
        "raw_seeded_fold_median_us": _fmt(raw_fold[0]),
        "compact_seeded_fold_median_us": _fmt(compact_fold[0]),
        "scheduler_speedup_x": _fmt(scheduler_speedup),
        "raw_pipeline_median_us": _fmt(raw_pipeline_time[0]),
        "raw_pipeline_p95_us": _fmt(raw_pipeline_time[1]),
        "compact_pipeline_median_us": _fmt(compact_pipeline_time[0]),
        "compact_pipeline_p95_us": _fmt(compact_pipeline_time[1]),
        "pipeline_speedup_x": _fmt(pipeline_speedup),
    }


FIELDNAMES = [
    "R",
    "C",
    "transport",
    "python",
    "vllm",
    "production_path",
    "gate",
    "raw_metadata_bytes",
    "compact_metadata_bytes",
    "metadata_reduction_x",
    "raw_envelope_bytes",
    "compact_envelope_bytes",
    "envelope_reduction_x",
    "raw_build_median_us",
    "compact_compose_median_us",
    "raw_serialize_median_us",
    "compact_serialize_median_us",
    "raw_seeded_fold_median_us",
    "compact_seeded_fold_median_us",
    "scheduler_speedup_x",
    "raw_pipeline_median_us",
    "raw_pipeline_p95_us",
    "compact_pipeline_median_us",
    "compact_pipeline_p95_us",
    "pipeline_speedup_x",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", default=",".join(str(value) for value in DEFAULT_REQUESTS)
    )
    parser.add_argument(
        "--candidates", default=",".join(str(value) for value in DEFAULT_CANDIDATES)
    )
    parser.add_argument("--include-8192", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", default="-")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions <= 0:
        parser.error("--warmups must be non-negative and --repetitions positive")
    requests = _parse_int_list(args.requests, "--requests")
    candidates = list(_parse_int_list(args.candidates, "--candidates"))
    if args.include_8192 and 8192 not in candidates:
        candidates.append(8192)
    rows = [
        _run_cell(r, c, args.seed, args.warmups, args.repetitions)
        for r in requests
        for c in candidates
    ]
    output = sys.stdout
    handle = None
    if args.output != "-":
        handle = open(args.output, "w", newline="", encoding="utf-8")
        output = handle
    try:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    main()
