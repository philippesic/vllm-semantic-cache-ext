# SPDX-License-Identifier: Apache-2.0
"""Fail-closed Linux GPU smoke for query-capture process ownership.

CP-002 is specifically about class patches and callbacks that are global to one
Python process.  Two ``vllm serve`` processes cannot exercise that boundary.
This script therefore creates engine A, drives real semantic capture and score
metadata, explicitly shuts A down, and creates engine B in the same process.

The script is intentionally a standalone evidence producer, not a pytest GPU
test.  It refuses an existing output directory and records scalar/JSON-safe
observations only; no worker, handle, layout, or CUDA tensor is retained by the
test-only callback tap.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_CPU_BYTES = 64 << 20
_FULL_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_SUFFIXES = {".json", ".jinja", ".py", ".so", ".yaml", ".yml"}
_REQUIRED_ENV = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
}


@dataclass(frozen=True)
class SmokeConfig:
    model: str
    model_revision: str
    extension_revision: str
    vllm_revision: str
    cuda_device: str
    cpu_bytes_to_use: int
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    num_gpu_blocks_override: int
    gpu_memory_utilization: float


def _full_revision(value: str) -> str:
    value = value.lower()
    if not _FULL_REVISION.fullmatch(value):
        raise argparse.ArgumentTypeError("revision must be a full 40-character SHA")
    return value


def _sha256(value: str) -> str:
    value = value.lower()
    if not _SHA256.fullmatch(value):
        raise argparse.ArgumentTypeError("digest must be a full 64-character SHA-256")
    return value


def _claim_output_dir(path: Path) -> None:
    """Atomically claim a fresh evidence root; never relabel old output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing existing output directory: {path}") from exc


def _prepare_runtime_environment(cuda_device: str) -> dict[str, str]:
    required = {**_REQUIRED_ENV, "CUDA_VISIBLE_DEVICES": cuda_device}
    for name, value in required.items():
        existing = os.environ.get(name)
        if existing is not None and existing != value:
            raise RuntimeError(
                f"{name} is already {existing!r}; this smoke requires {value!r} "
                "before importing vLLM"
            )
    os.environ.update(required)
    return required


def _require_runtime_imports_not_loaded(
    loaded_modules: object | None = None,
) -> None:
    """Require the WSL pin-memory contract to be set before runtime imports."""
    module_names = loaded_modules if loaded_modules is not None else sys.modules
    imported = sorted(
        package
        for package in ("torch", "vllm")
        if package in module_names
        or any(name.startswith(f"{package}.") for name in module_names)
    )
    if imported:
        raise RuntimeError(
            "runtime modules were imported before the required environment was "
            f"established: {imported}"
        )


def _validate_runtime_environment(expected: dict[str, str]) -> dict[str, str]:
    actual = {name: os.environ.get(name) for name in expected}
    mismatches = {
        name: {"actual": actual[name], "expected": value}
        for name, value in expected.items()
        if actual[name] != value
    }
    if mismatches:
        raise RuntimeError(f"required runtime environment changed: {mismatches}")
    return dict(expected)


def _exercise_uva_memory(
    torch_module,
    *,
    current_platform,
    pin_memory_available: bool,
    uva_available: bool,
    get_accelerator_view,
    uva_buffer_type,
) -> dict[str, object]:
    """Exercise the exact pinned-memory/UVA operations required by V2."""
    platform_pin_memory_available = bool(current_platform.is_pin_memory_available())
    if not platform_pin_memory_available or not pin_memory_available:
        raise RuntimeError(
            "pin memory is unavailable: "
            f"platform={platform_pin_memory_available}, helper={pin_memory_available}"
        )
    if not uva_available:
        raise RuntimeError("UVA is unavailable")

    expected = torch_module.arange(16, dtype=torch_module.int32, device="cpu")
    pinned = torch_module.empty(
        16, dtype=torch_module.int32, device="cpu", pin_memory=True
    )
    pinned.copy_(expected)
    if not pinned.is_pinned():
        raise RuntimeError("torch pinned-memory allocation is not pinned")
    uva_view = get_accelerator_view(pinned)
    if not uva_view.is_cuda:
        raise RuntimeError(f"torch UVA view is not CUDA-backed: {uva_view.device}")
    torch_module.cuda.synchronize()
    roundtrip = uva_view.clone().cpu()
    torch_module.cuda.synchronize()
    if not torch_module.equal(roundtrip, expected):
        raise RuntimeError("torch pinned-memory/UVA roundtrip mismatch")

    vllm_buffer = uva_buffer_type(16, torch_module.int32)
    if not vllm_buffer.cpu.is_pinned():
        raise RuntimeError("vLLM UvaBuffer CPU storage is not pinned")
    if not vllm_buffer.uva.is_cuda:
        raise RuntimeError(
            f"vLLM UvaBuffer view is not CUDA-backed: {vllm_buffer.uva.device}"
        )
    vllm_buffer.cpu.copy_(expected)
    vllm_roundtrip = vllm_buffer.uva.clone().cpu()
    torch_module.cuda.synchronize()
    if not torch_module.equal(vllm_roundtrip, expected):
        raise RuntimeError("vLLM UvaBuffer roundtrip mismatch")

    result = {
        "platform_pin_memory_available": platform_pin_memory_available,
        "pin_memory_available": bool(pin_memory_available),
        "uva_available": bool(uva_available),
        "torch_probe": {
            "element_count": int(expected.numel()),
            "dtype": str(expected.dtype),
            "cpu_pinned": bool(pinned.is_pinned()),
            "uva_device": str(uva_view.device),
            "roundtrip_equal": True,
        },
        "vllm_uva_buffer_probe": {
            "element_count": int(vllm_buffer.cpu.numel()),
            "dtype": str(vllm_buffer.cpu.dtype),
            "cpu_pinned": bool(vllm_buffer.cpu.is_pinned()),
            "uva_device": str(vllm_buffer.uva.device),
            "roundtrip_equal": True,
        },
    }
    del expected, pinned, uva_view, roundtrip
    del vllm_buffer, vllm_roundtrip
    gc.collect()
    torch_module.cuda.empty_cache()
    return result


def _probe_uva_capability(
    source_manifest: dict[str, str], vllm_source_root: Path
) -> dict[str, object]:
    """Run a same-process capability gate before claiming evidence output."""
    import torch
    import vllm
    from vllm.platforms import current_platform
    from vllm.utils.platform_utils import (
        is_pin_memory_available,
        is_uva_available,
    )
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    from vllm.v1.worker.gpu.buffer_utils import UvaBuffer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the CP-002 smoke requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    if platform.python_version() != source_manifest["python"]:
        raise RuntimeError("Python version does not match the source manifest")
    if torch.__version__.split("+", 1)[0] != source_manifest["torch"]:
        raise RuntimeError("PyTorch version does not match the source manifest")
    if not torch.__version__.endswith(f"+{source_manifest['cuda_variant']}"):
        raise RuntimeError("PyTorch CUDA variant does not match the source manifest")
    if vllm.__version__ != source_manifest["vllm_version"]:
        raise RuntimeError("vLLM version does not match the source manifest")
    vllm_file = Path(vllm.__file__).resolve(strict=True)
    expected_vllm_package = (vllm_source_root / "vllm").resolve(strict=True)
    if not vllm_file.is_relative_to(expected_vllm_package):
        raise RuntimeError(
            f"loaded vLLM from {vllm_file}, outside {expected_vllm_package}"
        )

    properties = torch.cuda.get_device_properties(0)
    result = _exercise_uva_memory(
        torch,
        current_platform=current_platform,
        pin_memory_available=is_pin_memory_available(),
        uva_available=is_uva_available(),
        get_accelerator_view=get_accelerator_view_from_cpu_tensor,
        uva_buffer_type=UvaBuffer,
    )
    result.update(
        {
            "checked_at_ns": time.time_ns(),
            "pid": os.getpid(),
            "kernel": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "platform": type(current_platform).__name__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "vllm_source": str(vllm_file),
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "total_memory": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            },
        }
    )
    return result


def _validate_source_manifest(
    path: Path, config: SmokeConfig, expected_sha256: str
) -> dict[str, str]:
    manifest_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source manifest sha256={actual_sha256!r}, expected {expected_sha256!r}"
        )
    values = {}
    for line_number, line in enumerate(manifest_bytes.decode().splitlines(), start=1):
        if not line or "=" not in line:
            raise ValueError(f"malformed source manifest line {line_number}: {line!r}")
        key, value = line.split("=", 1)
        if not key or not value:
            raise ValueError(f"malformed source manifest line {line_number}: {line!r}")
        if key in values:
            raise ValueError(f"duplicate source manifest key: {key}")
        values[key] = value
    required_keys = {
        "extension_commit",
        "vllm_commit",
        "vllm_version",
        "python",
        "torch",
        "cuda_variant",
    }
    if values.keys() != required_keys:
        missing = sorted(required_keys - values.keys())
        unknown = sorted(values.keys() - required_keys)
        raise ValueError(
            f"source manifest fields do not match contract: missing={missing}, "
            f"unknown={unknown}"
        )
    expected = {
        "extension_commit": config.extension_revision,
        "vllm_commit": config.vllm_revision,
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(
                f"source manifest {key}={values.get(key)!r}, expected {value!r}"
            )
    values["sha256"] = actual_sha256
    values["path"] = str(path.resolve())
    return values


def _validate_source_identity_manifest(
    path: Path,
    expected_sha256: str,
    config: SmokeConfig,
    source_manifest: dict[str, str],
    evidence_producer: dict[str, str],
) -> dict[str, str]:
    manifest_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source identity manifest sha256={actual_sha256!r}, "
            f"expected {expected_sha256!r}"
        )
    try:

        def reject_duplicates(pairs):
            parsed = {}
            for key, value in pairs:
                if key in parsed:
                    raise ValueError(f"duplicate source identity manifest key: {key}")
                parsed[key] = value
            return parsed

        values = json.loads(manifest_bytes, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError("malformed source identity manifest JSON") from exc
    required_keys = {
        "schema",
        "extension_revision",
        "vllm_revision",
        "source_manifest_sha256",
        "evidence_producer_sha256",
        "extension_source_root",
        "extension_tree_sha256",
        "vllm_source_root",
        "vllm_tree_sha256",
        "extension_archive_path",
        "extension_archive_sha256",
        "vllm_archive_path",
        "vllm_archive_sha256",
        "model",
        "model_revision",
        "required_wsl_pin_memory",
    }
    if not isinstance(values, dict) or values.keys() != required_keys:
        actual_keys = set(values) if isinstance(values, dict) else set()
        raise ValueError(
            "source identity manifest fields do not match contract: "
            f"missing={sorted(required_keys - actual_keys)}, "
            f"unknown={sorted(actual_keys - required_keys)}"
        )
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ValueError("source identity manifest values must be nonempty strings")
    expected_values = {
        "schema": "cp002-source-identity-v2",
        "extension_revision": config.extension_revision,
        "vllm_revision": config.vllm_revision,
        "source_manifest_sha256": source_manifest["sha256"],
        "evidence_producer_sha256": evidence_producer["sha256"],
        "model": config.model,
        "model_revision": config.model_revision,
        "required_wsl_pin_memory": "1",
    }
    for key, expected in expected_values.items():
        if values[key] != expected:
            raise ValueError(
                f"source identity manifest {key}={values[key]!r}, expected {expected!r}"
            )
    for key in (
        "extension_tree_sha256",
        "vllm_tree_sha256",
        "extension_archive_sha256",
        "vllm_archive_sha256",
    ):
        if not _SHA256.fullmatch(values[key]):
            raise ValueError(f"source identity manifest {key} is not a SHA-256")
    for key in (
        "extension_source_root",
        "vllm_source_root",
        "extension_archive_path",
        "vllm_archive_path",
    ):
        if not Path(values[key]).is_absolute():
            raise ValueError(f"source identity manifest {key} must be absolute")
    values["sha256"] = actual_sha256
    values["path"] = str(path.resolve())
    return values


def _require_file_identity(path: Path, expected_sha256: str) -> dict[str, str]:
    identity = _file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(
            f"file {identity['path']} sha256={identity['sha256']!r}, "
            f"expected {expected_sha256!r}"
        )
    return identity


def _key_id(key: object) -> str:
    hex_method = getattr(key, "hex", None)
    return str(hex_method()) if callable(hex_method) else repr(key)


def _score_snapshot(worker: object, req_ids: list[str]) -> dict[str, Any]:
    pending = getattr(worker, "_pending_scores", {})
    snapshot: dict[str, dict[str, list[dict[str, object]]]] = {}
    all_request_ids: set[str] = set()
    for method, requests in pending.items():
        all_request_ids.update(str(req_id) for req_id in requests)
        selected = {}
        for req_id in req_ids:
            if req_id not in requests:
                continue
            scored_candidates = []
            for key, score in requests[req_id]:
                scalar_score = float(score)
                if not math.isfinite(scalar_score):
                    raise ValueError(
                        f"non-finite semantic score for request {req_id}: {scalar_score}"
                    )
                scored_candidates.append({"key": _key_id(key), "score": scalar_score})
            selected[req_id] = scored_candidates
        if selected:
            snapshot[str(method)] = selected
    return {
        "selected": snapshot,
        "all_request_ids": sorted(all_request_ids),
    }


class CallbackTap:
    """Temporarily record live callbacks without retaining runtime objects."""

    def __init__(self, worker_cls, query_capture_module):
        self._worker_cls = worker_cls
        self._query_capture = query_capture_module
        self._original = worker_cls._on_queries_captured
        self._labels: dict[int, str] = {}
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def install(self) -> None:
        original = self._original
        owner = self

        def tapped(worker, req_ids, queries):
            worker_id = id(worker)
            handle = getattr(worker, "_query_capture_mode", None)
            current_handle = owner._query_capture._current_handle.get()
            layout = owner._query_capture._current_layout.get()
            original(worker, req_ids, queries)
            durable_keys = sorted(
                _key_id(key) for key in getattr(worker, "durable_summaries", {})
            )
            event = {
                "phase": owner._labels.get(worker_id, "unassigned"),
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "worker_id": worker_id,
                "handle_id": id(handle) if handle is not None else None,
                "current_handle_matches": current_handle is handle,
                "req_ids": [str(req_id) for req_id in req_ids],
                "query_shape": [int(size) for size in queries.shape],
                "query_device": str(queries.device),
                "layout_req_ids": (
                    [str(req_id) for req_id in layout.req_ids]
                    if layout is not None
                    else None
                ),
                "layout_boundaries": (
                    [[int(start), int(end)] for start, end in layout.boundaries]
                    if layout is not None
                    else None
                ),
                "durable_keys": durable_keys,
                "scores": _score_snapshot(worker, [str(req_id) for req_id in req_ids]),
                "time_ns": time.time_ns(),
            }
            with owner._lock:
                owner._events.append(event)

        self._worker_cls._on_queries_captured = tapped

    def label(self, worker: object, label: str) -> None:
        self._labels[id(worker)] = label

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def restore(self) -> None:
        self._worker_cls._on_queries_captured = self._original


@contextmanager
def _installed_tap(tap: CallbackTap):
    tap.install()
    try:
        yield
    finally:
        tap.restore()


def _stage_write(path: Path, payload: bytes) -> Path:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        staged = temporary_path
        temporary_path = None
        return staged
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _commit_staged(staged: Path, path: Path) -> None:
    os.replace(staged, path)


def _atomic_write(path: Path, payload: bytes) -> None:
    staged = _stage_write(path, payload)
    try:
        _commit_staged(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _source_tree_identity(root: Path) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"source root is not a directory: {resolved}")
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(resolved.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix not in _SOURCE_SUFFIXES
        ):
            continue
        relative = path.relative_to(resolved).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        file_count += 1
    if not file_count:
        raise ValueError(f"source root contains no tracked source files: {resolved}")
    return {
        "root": str(resolved),
        "file_count": file_count,
        "tree_sha256": digest.hexdigest(),
    }


def _validate_source_roots(
    source_identity_manifest: dict[str, str],
) -> dict[str, dict[str, object]]:
    """Authenticate extracted live source before importing any runtime code."""
    identities = {
        "extension": _source_tree_identity(
            Path(source_identity_manifest["extension_source_root"])
        ),
        "vllm": _source_tree_identity(
            Path(source_identity_manifest["vllm_source_root"])
        ),
    }
    mismatches = {
        name: {
            "actual": identity["tree_sha256"],
            "expected": source_identity_manifest[f"{name}_tree_sha256"],
        }
        for name, identity in identities.items()
        if identity["tree_sha256"] != source_identity_manifest[f"{name}_tree_sha256"]
    }
    if mismatches:
        raise ValueError(f"live source tree identity mismatch: {mismatches}")
    return identities


def _loaded_module_identity(
    module, source_root: Path, package: str
) -> dict[str, object]:
    resolved_source_root = source_root.resolve(strict=True)
    package_root = (resolved_source_root / package).resolve(strict=True)
    loaded_file = Path(module.__file__).resolve(strict=True)
    if not loaded_file.is_relative_to(package_root):
        raise ValueError(
            f"loaded {package} from {loaded_file}, outside expected {package_root}"
        )
    return {
        "loaded_file": _file_identity(loaded_file),
        "tree": _source_tree_identity(resolved_source_root),
    }


def _validate_loaded_sources(
    modules: dict[str, object],
    source_roots: dict[str, Path],
    expected_hashes: dict[str, str],
) -> dict[str, dict[str, object]]:
    identities = {
        "semantic_offload": _loaded_module_identity(
            modules["semantic_offload"], source_roots["extension"], "semantic_offload"
        ),
        "harness": _loaded_module_identity(
            modules["harness"], source_roots["extension"], "harness"
        ),
        "vllm": _loaded_module_identity(modules["vllm"], source_roots["vllm"], "vllm"),
    }
    actual_hashes = {
        name: identity["tree"]["tree_sha256"] for name, identity in identities.items()
    }
    if actual_hashes != expected_hashes:
        raise ValueError(
            f"loaded source hashes {actual_hashes} do not match expected "
            f"{expected_hashes}"
        )
    return identities


def _require_assertions_enabled(enabled: bool = __debug__) -> None:
    if not enabled:
        raise RuntimeError(
            "Python optimization disables this smoke's acceptance assertions"
        )


def _children_of_this_process() -> list[dict[str, object]]:
    result = subprocess.run(
        ["ps", "-o", "pid=,ppid=,comm=", "--ppid", str(os.getpid())],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"child process inventory failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    children = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split(None, 2)
        if len(fields) != 3:
            raise RuntimeError(f"malformed child process inventory line: {line!r}")
        try:
            pid = int(fields[0])
            ppid = int(fields[1])
        except ValueError as exc:
            raise RuntimeError(
                f"malformed child process inventory line: {line!r}"
            ) from exc
        if Path(fields[2]).name != "ps":
            children.append({"pid": pid, "ppid": ppid, "command": fields[2]})
    return children


def _assert_clean_capture_state(query_capture, runner_cls, originals) -> None:
    original_prepare, original_execute = originals
    assert query_capture._active_handle is None
    assert query_capture._runner_patch is None
    assert query_capture._current_handle.get() is None
    assert query_capture._current_layout.get() is None
    assert not hasattr(runner_cls, "_semantic_query_capture_owner")
    assert runner_cls.prepare_inputs is original_prepare
    assert runner_cls.execute_model is original_execute


def _drain_request(engine, req_id: str, token_ids: list[int], max_tokens: int) -> int:
    from vllm import SamplingParams, TokensPrompt

    engine.add_request(
        req_id,
        TokensPrompt(prompt_token_ids=token_ids),
        SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True),
    )
    steps = 0
    finished = False
    while engine.has_unfinished_requests():
        steps += 1
        if steps > 1024:
            raise TimeoutError(f"request {req_id} exceeded 1024 engine steps")
        for output in engine.step():
            if output.request_id == req_id and output.finished:
                finished = True
        from semantic_offload import query_capture

        assert query_capture._current_handle.get() is None
        assert query_capture._current_layout.get() is None
    assert finished, f"request {req_id} drained without a finished output"
    return steps


def _probe_events(
    events: list[dict[str, Any]], label: str, probe_req_id: str
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event["phase"] == label and probe_req_id in event["req_ids"]
    ]


def _validate_probe_events(
    events: list[dict[str, Any]],
    *,
    label: str,
    probe_req_id: str,
    worker_id: int,
    handle_id: int,
    durable_keys: set[str],
) -> None:
    probe_events = _probe_events(events, label, probe_req_id)
    assert probe_events, f"{label} emitted no callback for {probe_req_id}"
    scored = False
    for event in probe_events:
        assert event["pid"] == os.getpid()
        assert event["worker_id"] == worker_id
        assert event["handle_id"] == handle_id
        assert event["current_handle_matches"]
        assert event["query_device"].startswith("cuda")
        assert len(event["query_shape"]) == 3
        assert event["query_shape"][0] == len(event["req_ids"])
        assert probe_req_id in event["layout_req_ids"]
        assert all(
            req_id.startswith(f"cp002-{label}-")
            for req_id in event["scores"]["all_request_ids"]
        )
        selected = event["scores"]["selected"]
        for method_requests in selected.values():
            if probe_req_id not in method_requests:
                continue
            candidates = {
                candidate["key"] for candidate in method_requests[probe_req_id]
            }
            assert all(
                math.isfinite(candidate["score"])
                for candidate in method_requests[probe_req_id]
            )
            assert candidates
            assert candidates <= durable_keys
            scored = True
    assert scored, f"{label} callback had no live score metadata for {probe_req_id}"


def _locate_runtime(engine):
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.engine.core_client import InprocClient

    from semantic_offload.connector import SemanticOffloadingConnector
    from semantic_offload.worker import SemanticOffloadingWorker

    assert isinstance(engine.engine_core, InprocClient)
    connector = get_kv_transfer_group()
    assert isinstance(connector, SemanticOffloadingConnector)
    assert connector.connector_worker is not None
    semantic_worker = connector.connector_worker.worker
    assert isinstance(semantic_worker, SemanticOffloadingWorker)
    driver_worker = engine.model_executor.driver_worker.worker
    model_runner = driver_worker.model_runner
    assert semantic_worker._query_capture_mode is not None
    return semantic_worker, model_runner


def _make_engine(config: SmokeConfig, seed: int):
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    from harness.policies import kv_transfer_config

    raw_kv_config = kv_transfer_config(
        "semantic-mean",
        config.cpu_bytes_to_use,
        extra_config={
            "probe_layer": "middle",
            "head_aggregation": "mean",
            "capture_stride": 1,
            "prefetch_budget_fraction": 0.0,
        },
    )
    engine_args = EngineArgs(
        model=config.model,
        revision=config.model_revision,
        tokenizer_revision=config.model_revision,
        dtype="bfloat16",
        seed=seed,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enforce_eager=True,
        max_model_len=config.max_model_len,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        num_gpu_blocks_override=config.num_gpu_blocks_override,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        disable_log_stats=True,
        kv_transfer_config=KVTransferConfig(**raw_kv_config),
    )
    return LLMEngine.from_engine_args(
        engine_args, enable_multiprocessing=False
    ), raw_kv_config


def _run_engine_lifetime(
    config: SmokeConfig,
    *,
    label: str,
    seed: int,
    tap: CallbackTap,
    runner_cls,
    originals,
) -> tuple[dict[str, Any], object]:
    from vllm.distributed.kv_transfer import has_kv_transfer_group

    from semantic_offload import query_capture

    _assert_clean_capture_state(query_capture, runner_cls, originals)
    assert not has_kv_transfer_group()
    engine = None
    shutdown = False
    worker = None
    handle = None
    model_runner = None
    result: dict[str, Any] = {"label": label, "seed": seed}
    try:
        engine, raw_kv_config = _make_engine(config, seed)
        worker, model_runner = _locate_runtime(engine)
        tap.label(worker, label)
        handle = worker._query_capture_mode
        assert query_capture._active_handle is handle
        assert runner_cls._semantic_query_capture_owner is handle
        assert runner_cls.prepare_inputs is not originals[0]
        assert runner_cls.execute_model is not originals[1]
        assert handle.vllm_config is engine.vllm_config
        assert engine.vllm_config.use_v2_model_runner is True
        mode = engine.vllm_config.compilation_config.cudagraph_mode
        assert getattr(mode, "name", str(mode)) == "NONE"
        try:
            query_capture.preflight_install(engine.vllm_config)
        except RuntimeError as exc:
            assert "already installed" in str(exc)
            duplicate_rejected = str(exc)
        else:
            raise AssertionError("a second live query-capture owner was accepted")

        primer_requests = []
        for attempt in range(1, 4):
            req_id = f"cp002-{label}-primer-{attempt}"
            tokens = [
                1000 + ((index + seed * 17 + attempt) % 127) for index in range(256)
            ]
            steps = _drain_request(engine, req_id, tokens, max_tokens=2)
            primer_requests.append({"request_id": req_id, "steps": steps})
            if worker.durable_summaries:
                break
        assert worker.durable_summaries, (
            f"{label} produced no durable summaries after three primer requests"
        )

        probe_req_id = f"cp002-{label}-probe"
        probe_tokens = [3000 + ((index + seed * 11) % 97) for index in range(48)]
        probe_steps = _drain_request(engine, probe_req_id, probe_tokens, max_tokens=1)
        assert handle._runner is model_runner
        durable_keys = {_key_id(key) for key in worker.durable_summaries}
        _validate_probe_events(
            tap.events(),
            label=label,
            probe_req_id=probe_req_id,
            worker_id=id(worker),
            handle_id=id(handle),
            durable_keys=durable_keys,
        )
        result.update(
            {
                "worker_id": id(worker),
                "handle_id": id(handle),
                "runner_id": id(model_runner),
                "primer_requests": primer_requests,
                "probe_request_id": probe_req_id,
                "probe_steps": probe_steps,
                "durable_key_count": len(durable_keys),
                "duplicate_owner_rejected": duplicate_rejected,
                "kv_transfer_config": raw_kv_config,
            }
        )
    finally:
        if engine is not None and not shutdown:
            engine.engine_core.shutdown()
            shutdown = True

    assert handle is not None and worker is not None
    assert handle._closed is True
    assert handle._runner is None
    assert worker._query_capture_mode is None
    assert not has_kv_transfer_group()
    _assert_clean_capture_state(query_capture, runner_cls, originals)
    result["shutdown_clean"] = True
    result["event_count_after_shutdown"] = len(tap.events())

    closed_refs = {
        "engine": weakref.ref(engine),
        "handle": weakref.ref(handle),
        "worker": weakref.ref(worker),
        "runner": weakref.ref(model_runner),
    }
    del engine, handle, worker, model_runner
    return result, closed_refs


def _require_collected(
    label: str, references: dict[str, weakref.ReferenceType]
) -> None:
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    alive = sorted(
        name for name, reference in references.items() if reference() is not None
    )
    assert not alive, (
        f"{label} objects remain strongly reachable after shutdown: {alive}"
    )


def _run_smoke(
    config: SmokeConfig,
    source_manifest: dict[str, str],
    extension_source_root: Path,
    vllm_source_root: Path,
    extension_source_sha256: str,
    vllm_source_sha256: str,
) -> dict[str, Any]:
    import torch
    import vllm
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    import harness.policies as harness_policies
    import semantic_offload
    from semantic_offload import query_capture
    from semantic_offload.worker import SemanticOffloadingWorker

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert platform.python_version() == source_manifest["python"]
    assert torch.__version__.split("+", 1)[0] == source_manifest["torch"]
    assert torch.__version__.endswith(f"+{source_manifest['cuda_variant']}")
    assert vllm.__version__ == source_manifest["vllm_version"]
    assert config.vllm_revision[:9] in vllm.__version__
    loaded_sources = _validate_loaded_sources(
        {
            "semantic_offload": semantic_offload,
            "harness": harness_policies,
            "vllm": vllm,
        },
        {"extension": extension_source_root, "vllm": vllm_source_root},
        {
            "semantic_offload": extension_source_sha256,
            "harness": extension_source_sha256,
            "vllm": vllm_source_sha256,
        },
    )
    originals = (GPUModelRunner.prepare_inputs, GPUModelRunner.execute_model)
    _assert_clean_capture_state(query_capture, GPUModelRunner, originals)
    tap = CallbackTap(SemanticOffloadingWorker, query_capture)
    try:
        with _installed_tap(tap):
            started_children = _children_of_this_process()
            assert not started_children, (
                f"child processes existed before smoke: {started_children}"
            )
            first, first_refs = _run_engine_lifetime(
                config,
                label="A",
                seed=101,
                tap=tap,
                runner_cls=GPUModelRunner,
                originals=originals,
            )
            _require_collected("engine A", first_refs)
            first["objects_collected_before_next_engine"] = True
            a_event_count = len(tap.events())
            second, second_refs = _run_engine_lifetime(
                config,
                label="B",
                seed=202,
                tap=tap,
                runner_cls=GPUModelRunner,
                originals=originals,
            )
            _require_collected("engine B", second_refs)
            second["objects_collected_before_next_engine"] = True
            events = tap.events()
            assert all(event["phase"] != "A" for event in events[a_event_count:])
            assert all(
                not any(
                    req_id.startswith("cp002-A-")
                    for req_id in [
                        *event["req_ids"],
                        *event["scores"]["all_request_ids"],
                    ]
                )
                for event in events[a_event_count:]
            )
            final_children = _children_of_this_process()
            assert not final_children, (
                f"child processes remain after smoke: {final_children}"
            )
            loaded_sources_end = _validate_loaded_sources(
                {
                    "semantic_offload": semantic_offload,
                    "harness": harness_policies,
                    "vllm": vllm,
                },
                {"extension": extension_source_root, "vllm": vllm_source_root},
                {
                    "semantic_offload": extension_source_sha256,
                    "harness": extension_source_sha256,
                    "vllm": vllm_source_sha256,
                },
            )
            assert loaded_sources_end == loaded_sources
            return {
                "status": "pass",
                "pid": os.getpid(),
                "python": platform.python_version(),
                "kernel": platform.release(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "vllm": vllm.__version__,
                "vllm_source": str(Path(vllm.__file__).resolve()),
                "loaded_sources_start": loaded_sources,
                "loaded_sources_end": loaded_sources_end,
                "started_children": started_children,
                "final_children": final_children,
                "engine_a": first,
                "engine_b": second,
                "events": events,
            }
    finally:
        _assert_clean_capture_state(query_capture, GPUModelRunner, originals)


def _serialize_events(events: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(event, sort_keys=True) + "\n").encode() for event in events
    )


def _publish_success(
    output_dir: Path,
    result: dict[str, Any],
    events: list[dict[str, Any]],
    completion_validator: Callable[[], dict[str, Any]],
) -> None:
    event_payload = _serialize_events(events)
    event_path = output_dir / "callback_events.jsonl"
    _atomic_write(event_path, event_payload)
    completed_result = dict(result)
    completed_result["callback_events"] = {
        "path": event_path.name,
        "event_count": len(events),
        "byte_count": len(event_payload),
        "sha256": hashlib.sha256(event_payload).hexdigest(),
    }
    completion_provenance = completion_validator()
    completed_result["completion_provenance"] = completion_provenance
    result_path = output_dir / "result.json"
    result_payload = (
        json.dumps(completed_result, indent=2, sort_keys=True) + "\n"
    ).encode()
    staged_result = _stage_write(result_path, result_payload)
    try:
        if completion_validator() != completion_provenance:
            raise RuntimeError("provenance changed while staging pass result")
        _commit_staged(staged_result, result_path)
    finally:
        staged_result.unlink(missing_ok=True)


def _completion_provenance(
    *,
    config: SmokeConfig,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    source_manifest_start: dict[str, str],
    source_identity_manifest_path: Path,
    source_identity_manifest_sha256: str,
    source_identity_manifest_start: dict[str, str],
    source_archives_start: dict[str, dict[str, str]],
    manifest_path: Path,
    manifest_sha256: str,
    required_environment_start: dict[str, str],
) -> dict[str, Any]:
    import vllm

    import harness.policies as harness_policies
    import semantic_offload

    evidence_producer = _file_identity(Path(__file__))
    source_manifest = _validate_source_manifest(
        source_manifest_path, config, source_manifest_sha256
    )
    if source_manifest != source_manifest_start:
        raise RuntimeError("source manifest changed during the smoke")
    source_identity_manifest = _validate_source_identity_manifest(
        source_identity_manifest_path,
        source_identity_manifest_sha256,
        config,
        source_manifest,
        evidence_producer,
    )
    if source_identity_manifest != source_identity_manifest_start:
        raise RuntimeError("source identity manifest changed during the smoke")
    source_archives = {
        "extension": _require_file_identity(
            Path(source_identity_manifest["extension_archive_path"]),
            source_identity_manifest["extension_archive_sha256"],
        ),
        "vllm": _require_file_identity(
            Path(source_identity_manifest["vllm_archive_path"]),
            source_identity_manifest["vllm_archive_sha256"],
        ),
    }
    if source_archives != source_archives_start:
        raise RuntimeError("source archives changed during the smoke")
    loaded_sources = _validate_loaded_sources(
        {
            "semantic_offload": semantic_offload,
            "harness": harness_policies,
            "vllm": vllm,
        },
        {
            "extension": Path(source_identity_manifest["extension_source_root"]),
            "vllm": Path(source_identity_manifest["vllm_source_root"]),
        },
        {
            "semantic_offload": source_identity_manifest["extension_tree_sha256"],
            "harness": source_identity_manifest["extension_tree_sha256"],
            "vllm": source_identity_manifest["vllm_tree_sha256"],
        },
    )
    return {
        "required_environment": _validate_runtime_environment(
            required_environment_start
        ),
        "evidence_producer": evidence_producer,
        "source_manifest": source_manifest,
        "source_identity_manifest": source_identity_manifest,
        "source_archives": source_archives,
        "loaded_sources": loaded_sources,
        "manifest_file": _require_file_identity(manifest_path, manifest_sha256),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--source-identity-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-identity-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument("--extension-revision", type=_full_revision, required=True)
    parser.add_argument("--vllm-revision", type=_full_revision, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision", type=_full_revision, default=DEFAULT_MODEL_REVISION
    )
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--cpu-bytes-to-use", type=int, default=DEFAULT_CPU_BYTES)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=40)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    _require_assertions_enabled()
    args = build_parser().parse_args(argv)
    config = SmokeConfig(
        model=args.model,
        model_revision=args.model_revision,
        extension_revision=args.extension_revision,
        vllm_revision=args.vllm_revision,
        cuda_device=args.cuda_device,
        cpu_bytes_to_use=args.cpu_bytes_to_use,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    _require_runtime_imports_not_loaded()
    environment = _prepare_runtime_environment(config.cuda_device)
    source_manifest = _validate_source_manifest(
        args.source_manifest, config, args.source_manifest_sha256
    )
    evidence_producer = _file_identity(Path(__file__))
    source_identity_manifest = _validate_source_identity_manifest(
        args.source_identity_manifest,
        args.source_identity_manifest_sha256,
        config,
        source_manifest,
        evidence_producer,
    )
    source_archives = {
        "extension": _require_file_identity(
            Path(source_identity_manifest["extension_archive_path"]),
            source_identity_manifest["extension_archive_sha256"],
        ),
        "vllm": _require_file_identity(
            Path(source_identity_manifest["vllm_archive_path"]),
            source_identity_manifest["vllm_archive_sha256"],
        ),
    }
    source_roots_preflight = _validate_source_roots(source_identity_manifest)
    runtime_capability = _probe_uva_capability(
        source_manifest, Path(source_identity_manifest["vllm_source_root"])
    )
    _claim_output_dir(args.output_dir)
    manifest = {
        "config": asdict(config),
        "required_environment": environment,
        "runtime_capability": runtime_capability,
        "evidence_producer": evidence_producer,
        "source_manifest": source_manifest,
        "source_identity_manifest": source_identity_manifest,
        "source_archives_start": source_archives,
        "source_roots_preflight": source_roots_preflight,
        "command": [sys.executable, *sys.argv],
        "started_at_ns": time.time_ns(),
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_file_start = _file_identity(manifest_path)
    try:
        result = _run_smoke(
            config,
            source_manifest,
            Path(source_identity_manifest["extension_source_root"]),
            Path(source_identity_manifest["vllm_source_root"]),
            source_identity_manifest["extension_tree_sha256"],
            source_identity_manifest["vllm_tree_sha256"],
        )
        result["finished_at_ns"] = time.time_ns()
        result["runtime_capability"] = runtime_capability
        events = result.pop("events")
        _publish_success(
            args.output_dir,
            result,
            events,
            lambda: _completion_provenance(
                config=config,
                source_manifest_path=args.source_manifest,
                source_manifest_sha256=args.source_manifest_sha256,
                source_manifest_start=source_manifest,
                source_identity_manifest_path=args.source_identity_manifest,
                source_identity_manifest_sha256=args.source_identity_manifest_sha256,
                source_identity_manifest_start=source_identity_manifest,
                source_archives_start=source_archives,
                manifest_path=manifest_path,
                manifest_sha256=manifest_file_start["sha256"],
                required_environment_start=environment,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - preserve any smoke failure evidence
        failure = {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at_ns": time.time_ns(),
        }
        _write_json(args.output_dir / "result.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    print(f"CP-002 lifecycle smoke PASS: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
