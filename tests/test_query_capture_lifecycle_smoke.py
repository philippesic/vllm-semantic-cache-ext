# SPDX-License-Identifier: Apache-2.0
"""Offline contract tests for the CP-002 Linux GPU evidence script."""

import gc
import hashlib
import json
import math
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run_query_capture_lifecycle_smoke as smoke


def test_full_revision_requires_exact_sha():
    revision = "a" * 40
    assert smoke._full_revision(revision.upper()) == revision
    for invalid in ("a" * 39, "a" * 41, "g" * 40, "main"):
        with pytest.raises(Exception, match="40-character SHA"):
            smoke._full_revision(invalid)


def test_output_directory_refuses_reuse(tmp_path):
    output = tmp_path / "evidence"
    smoke._claim_output_dir(output)
    with pytest.raises(FileExistsError, match="refusing existing"):
        smoke._claim_output_dir(output)


def test_output_directory_claim_is_atomic(tmp_path):
    output = tmp_path / "evidence"

    def claim():
        try:
            smoke._claim_output_dir(output)
        except FileExistsError:
            return "refused"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(lambda _: claim(), range(2)))
    assert outcomes == ["claimed", "refused"]


def test_runtime_environment_refuses_conflicting_preimport_state(monkeypatch):
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "0")
    with pytest.raises(RuntimeError, match="requires '1'"):
        smoke._prepare_runtime_environment("0")


def test_runtime_environment_sets_exact_inprocess_contract(monkeypatch):
    expected = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "CUDA_VISIBLE_DEVICES": "3",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    assert smoke._prepare_runtime_environment("3") == expected


def test_runtime_import_gate_rejects_torch_or_vllm_preimport():
    smoke._require_runtime_imports_not_loaded({"json": object()})
    for imported in ({"torch": object()}, {"vllm.platforms": object()}):
        with pytest.raises(RuntimeError, match="imported before"):
            smoke._require_runtime_imports_not_loaded(imported)


def test_runtime_environment_validation_rejects_drift(monkeypatch):
    expected = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    assert smoke._validate_runtime_environment(expected) == expected
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "0")
    with pytest.raises(RuntimeError, match="environment changed"):
        smoke._validate_runtime_environment(expected)


class _FakeTensor:
    def __init__(self, values, *, device="cpu", pinned=False, shared=False):
        self._values = values if shared else list(values)
        self.device = device
        self.dtype = "torch.int32"
        self._pinned = pinned

    @property
    def is_cuda(self):
        return str(self.device).startswith("cuda")

    def is_pinned(self):
        return self._pinned

    def copy_(self, other):
        self._values[:] = other._values
        return self

    def clone(self):
        return _FakeTensor(self._values, device=self.device)

    def cpu(self):
        return _FakeTensor(self._values)

    def numel(self):
        return len(self._values)


class _FakeCuda:
    def synchronize(self):
        return None

    def empty_cache(self):
        return None


class _FakeTorch:
    int32 = "torch.int32"
    cuda = _FakeCuda()

    @staticmethod
    def arange(size, *, dtype, device):
        return _FakeTensor(range(size), device=device)

    @staticmethod
    def empty(size, *, dtype, device, pin_memory):
        return _FakeTensor([0] * size, device=device, pinned=pin_memory)

    @staticmethod
    def equal(left, right):
        return left._values == right._values


class _FakePlatform:
    @staticmethod
    def is_pin_memory_available():
        return True


class _FakeUvaBuffer:
    def __init__(self, size, dtype):
        self.cpu = _FakeTensor([0] * size, pinned=True)
        self.uva = _FakeTensor(self.cpu._values, device="cuda:0", shared=True)


def test_uva_memory_gate_exercises_torch_and_vllm_roundtrips():
    result = smoke._exercise_uva_memory(
        _FakeTorch,
        current_platform=_FakePlatform(),
        pin_memory_available=True,
        uva_available=True,
        get_accelerator_view=lambda tensor: _FakeTensor(
            tensor._values, device="cuda:0", shared=True
        ),
        uva_buffer_type=_FakeUvaBuffer,
    )
    assert result["pin_memory_available"] is True
    assert result["uva_available"] is True
    assert result["torch_probe"]["roundtrip_equal"] is True
    assert result["vllm_uva_buffer_probe"]["roundtrip_equal"] is True


@pytest.mark.parametrize(
    ("pin_memory_available", "uva_available", "error"),
    [(False, True, "pin memory"), (True, False, "UVA is unavailable")],
)
def test_uva_memory_gate_rejects_missing_capability(
    pin_memory_available, uva_available, error
):
    with pytest.raises(RuntimeError, match=error):
        smoke._exercise_uva_memory(
            _FakeTorch,
            current_platform=_FakePlatform(),
            pin_memory_available=pin_memory_available,
            uva_available=uva_available,
            get_accelerator_view=lambda tensor: tensor,
            uva_buffer_type=_FakeUvaBuffer,
        )


def test_uva_memory_gate_rejects_roundtrip_mismatch():
    with pytest.raises(RuntimeError, match="roundtrip mismatch"):
        smoke._exercise_uva_memory(
            _FakeTorch,
            current_platform=_FakePlatform(),
            pin_memory_available=True,
            uva_available=True,
            get_accelerator_view=lambda _tensor: _FakeTensor(
                [99] * 16, device="cuda:0"
            ),
            uva_buffer_type=_FakeUvaBuffer,
        )


def test_source_manifest_binds_both_repository_revisions(tmp_path):
    source_manifest = tmp_path / "SOURCE_MANIFEST.txt"
    source_manifest.write_text(
        "extension_commit="
        + "1" * 40
        + "\n"
        + "vllm_commit="
        + "2" * 40
        + "\n"
        + "vllm_version=0.26+g"
        + "2" * 9
        + "\npython=3.12.13\ntorch=2.13.0\ncuda_variant=cu130\n"
    )
    config = smoke.SmokeConfig(
        model=smoke.DEFAULT_MODEL,
        model_revision=smoke.DEFAULT_MODEL_REVISION,
        extension_revision="1" * 40,
        vllm_revision="2" * 40,
        cuda_device="0",
        cpu_bytes_to_use=smoke.DEFAULT_CPU_BYTES,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        num_gpu_blocks_override=40,
        gpu_memory_utilization=0.5,
    )
    digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    values = smoke._validate_source_manifest(source_manifest, config, digest)
    assert values["extension_commit"] == "1" * 40
    assert values["vllm_commit"] == "2" * 40
    assert len(values["sha256"]) == 64

    source_manifest.write_text(
        source_manifest.read_text().replace(
            "extension_commit=" + "1" * 40, "extension_commit=" + "3" * 40
        )
    )
    with pytest.raises(ValueError, match="extension_commit"):
        smoke._validate_source_manifest(
            source_manifest,
            config,
            hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        )


def test_source_manifest_rejects_digest_duplicate_and_malformed_fields(tmp_path):
    source_manifest = tmp_path / "SOURCE_MANIFEST.txt"
    valid = (
        f"extension_commit={'1' * 40}\n"
        f"vllm_commit={'2' * 40}\n"
        f"vllm_version=0.26+g{'2' * 9}\n"
        "python=3.12.13\ntorch=2.13.0\ncuda_variant=cu130\n"
    )
    config = smoke.SmokeConfig(
        model=smoke.DEFAULT_MODEL,
        model_revision=smoke.DEFAULT_MODEL_REVISION,
        extension_revision="1" * 40,
        vllm_revision="2" * 40,
        cuda_device="0",
        cpu_bytes_to_use=smoke.DEFAULT_CPU_BYTES,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        num_gpu_blocks_override=40,
        gpu_memory_utilization=0.5,
    )
    source_manifest.write_text(valid)
    with pytest.raises(ValueError, match="sha256"):
        smoke._validate_source_manifest(source_manifest, config, "0" * 64)

    for invalid, error in (
        (valid + f"vllm_commit={'2' * 40}\n", "duplicate"),
        (valid + "broken\n", "malformed"),
        (valid.replace("torch=2.13.0\n", ""), "missing"),
        (valid + "unexpected=value\n", "unknown"),
    ):
        source_manifest.write_text(invalid)
        with pytest.raises(ValueError, match=error):
            smoke._validate_source_manifest(
                source_manifest,
                config,
                hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            )


def test_file_identity_binds_evidence_producer_bytes(tmp_path):
    producer = tmp_path / "producer.py"
    producer.write_bytes(b"print('first')\n")
    first = smoke._file_identity(producer)
    assert first["path"] == str(producer.resolve())
    assert len(first["sha256"]) == 64

    producer.write_bytes(b"print('second')\n")
    assert smoke._file_identity(producer)["sha256"] != first["sha256"]


def test_loaded_module_identity_binds_expected_source_tree(tmp_path):
    package = tmp_path / "semantic_offload"
    package.mkdir()
    module_file = package / "__init__.py"
    module_file.write_text("VALUE = 1\n")
    identity = smoke._loaded_module_identity(
        SimpleNamespace(__file__=str(module_file)), tmp_path, "semantic_offload"
    )
    assert identity["loaded_file"]["path"] == str(module_file.resolve())
    assert identity["tree"]["file_count"] == 1
    first_hash = identity["tree"]["tree_sha256"]
    module_file.write_text("VALUE = 2\n")
    assert smoke._source_tree_identity(tmp_path)["tree_sha256"] != first_hash

    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="outside expected"):
        smoke._loaded_module_identity(
            SimpleNamespace(__file__=str(outside)), tmp_path, "semantic_offload"
        )


def test_loaded_source_validation_rejects_hash_mismatch(tmp_path):
    extension_root = tmp_path / "extension"
    vllm_root = tmp_path / "vllm-source"
    semantic_file = extension_root / "semantic_offload" / "__init__.py"
    harness_file = extension_root / "harness" / "policies.py"
    vllm_file = vllm_root / "vllm" / "__init__.py"
    for path in (semantic_file, harness_file, vllm_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"NAME = {path.parent.name!r}\n")
    modules = {
        "semantic_offload": SimpleNamespace(__file__=str(semantic_file)),
        "harness": SimpleNamespace(__file__=str(harness_file)),
        "vllm": SimpleNamespace(__file__=str(vllm_file)),
    }
    roots = {"extension": extension_root, "vllm": vllm_root}
    extension_hash = smoke._source_tree_identity(extension_root)["tree_sha256"]
    expected = {
        "semantic_offload": extension_hash,
        "harness": extension_hash,
        "vllm": smoke._source_tree_identity(vllm_root)["tree_sha256"],
    }
    identities = smoke._validate_loaded_sources(modules, roots, expected)
    assert set(identities) == {"semantic_offload", "harness", "vllm"}

    with pytest.raises(ValueError, match="do not match"):
        smoke._validate_loaded_sources(modules, roots, {**expected, "vllm": "0" * 64})

    harness_file.write_text("CHANGED = True\n")
    with pytest.raises(ValueError, match="do not match"):
        smoke._validate_loaded_sources(modules, roots, expected)


def test_loaded_source_validation_rejects_harness_outside_extension(tmp_path):
    extension_root = tmp_path / "extension"
    vllm_root = tmp_path / "vllm-source"
    semantic_file = extension_root / "semantic_offload" / "__init__.py"
    expected_harness = extension_root / "harness" / "__init__.py"
    vllm_file = vllm_root / "vllm" / "__init__.py"
    outside_harness = tmp_path / "outside" / "harness" / "policies.py"
    for path in (semantic_file, expected_harness, vllm_file, outside_harness):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="outside expected"):
        smoke._validate_loaded_sources(
            {
                "semantic_offload": SimpleNamespace(__file__=str(semantic_file)),
                "harness": SimpleNamespace(__file__=str(outside_harness)),
                "vllm": SimpleNamespace(__file__=str(vllm_file)),
            },
            {"extension": extension_root, "vllm": vllm_root},
            {
                "semantic_offload": "0" * 64,
                "harness": "0" * 64,
                "vllm": "0" * 64,
            },
        )


def test_smoke_refuses_when_python_optimization_disables_assertions():
    smoke._require_assertions_enabled(True)
    with pytest.raises(RuntimeError, match="optimization disables"):
        smoke._require_assertions_enabled(False)


def test_installed_tap_restores_callback_on_early_failure():
    class Worker:
        def _on_queries_captured(self):
            return None

    original = Worker._on_queries_captured
    tap = smoke.CallbackTap(Worker, SimpleNamespace())
    with pytest.raises(RuntimeError, match="inventory"), smoke._installed_tap(tap):
        assert Worker._on_queries_captured is not original
        raise RuntimeError("inventory failed")
    assert Worker._on_queries_captured is original


def test_child_inventory_fails_closed_on_command_error(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="ps failed"
        ),
    )
    with pytest.raises(RuntimeError, match="ps failed"):
        smoke._children_of_this_process()


def test_child_inventory_rejects_malformed_output(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="123 malformed\n", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="malformed"):
        smoke._children_of_this_process()


def test_child_inventory_reports_existing_child(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="123 456 python\n", stderr=""
        ),
    )
    assert smoke._children_of_this_process() == [
        {"pid": 123, "ppid": 456, "command": "python"}
    ]


def test_score_snapshot_selects_only_current_callback_requests():
    key_a = b"a"
    key_b = b"b"
    worker = SimpleNamespace(
        _pending_scores={
            "mean": {
                "probe": [(key_a, 1.0), (key_b, 0.5)],
                "old": [(key_b, 1.0)],
            }
        }
    )
    assert smoke._score_snapshot(worker, ["probe"]) == {
        "selected": {
            "mean": {
                "probe": [
                    {"key": "61", "score": 1.0},
                    {"key": "62", "score": 0.5},
                ]
            }
        },
        "all_request_ids": ["old", "probe"],
    }


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_score_snapshot_rejects_nonfinite_values(score):
    worker = SimpleNamespace(_pending_scores={"mean": {"probe": [(b"a", score)]}})
    with pytest.raises(ValueError, match="non-finite"):
        smoke._score_snapshot(worker, ["probe"])


def test_probe_validation_requires_live_scores_from_current_durable_keys():
    event = {
        "phase": "B",
        "pid": smoke.os.getpid(),
        "worker_id": 2,
        "handle_id": 3,
        "current_handle_matches": True,
        "req_ids": ["cp002-B-probe"],
        "query_shape": [1, 2, 4],
        "query_device": "cuda:0",
        "layout_req_ids": ["cp002-B-probe"],
        "scores": {
            "selected": {
                "mean": {"cp002-B-probe": [{"key": "candidate", "score": 0.75}]}
            },
            "all_request_ids": ["cp002-B-probe"],
        },
    }
    smoke._validate_probe_events(
        [event],
        label="B",
        probe_req_id="cp002-B-probe",
        worker_id=2,
        handle_id=3,
        durable_keys={"candidate"},
    )


def test_probe_validation_rejects_cross_lifetime_candidate():
    event = {
        "phase": "B",
        "pid": smoke.os.getpid(),
        "worker_id": 2,
        "handle_id": 3,
        "current_handle_matches": True,
        "req_ids": ["cp002-B-probe"],
        "query_shape": [1, 2, 4],
        "query_device": "cuda:0",
        "layout_req_ids": ["cp002-B-probe"],
        "scores": {
            "selected": {
                "mean": {"cp002-B-probe": [{"key": "engine-a-key", "score": 0.75}]}
            },
            "all_request_ids": ["cp002-B-probe"],
        },
    }
    with pytest.raises(AssertionError):
        smoke._validate_probe_events(
            [event],
            label="B",
            probe_req_id="cp002-B-probe",
            worker_id=2,
            handle_id=3,
            durable_keys={"engine-b-key"},
        )


def test_probe_validation_rejects_empty_score_candidates():
    event = {
        "phase": "B",
        "pid": smoke.os.getpid(),
        "worker_id": 2,
        "handle_id": 3,
        "current_handle_matches": True,
        "req_ids": ["cp002-B-probe"],
        "query_shape": [1, 2, 4],
        "query_device": "cuda:0",
        "layout_req_ids": ["cp002-B-probe"],
        "scores": {
            "selected": {"mean": {"cp002-B-probe": []}},
            "all_request_ids": ["cp002-B-probe"],
        },
    }
    with pytest.raises(AssertionError):
        smoke._validate_probe_events(
            [event],
            label="B",
            probe_req_id="cp002-B-probe",
            worker_id=2,
            handle_id=3,
            durable_keys={"candidate"},
        )


def test_probe_validation_rejects_cross_lifetime_pending_score_id():
    event = {
        "phase": "B",
        "pid": smoke.os.getpid(),
        "worker_id": 2,
        "handle_id": 3,
        "current_handle_matches": True,
        "req_ids": ["cp002-B-probe"],
        "query_shape": [1, 2, 4],
        "query_device": "cuda:0",
        "layout_req_ids": ["cp002-B-probe"],
        "scores": {
            "selected": {
                "mean": {"cp002-B-probe": [{"key": "candidate", "score": 0.75}]}
            },
            "all_request_ids": ["cp002-A-stale", "cp002-B-probe"],
        },
    }
    with pytest.raises(AssertionError):
        smoke._validate_probe_events(
            [event],
            label="B",
            probe_req_id="cp002-B-probe",
            worker_id=2,
            handle_id=3,
            durable_keys={"candidate"},
        )


def test_collection_gate_rejects_retained_closed_owner():
    class Owner:
        pass

    owner = Owner()
    owner_ref = weakref.ref(owner)
    closed_handle = SimpleNamespace(mode_factory=lambda retained=owner: retained)
    del owner
    gc.collect()
    with pytest.raises(AssertionError, match="strongly reachable"):
        smoke._require_collected("engine A", {"worker": owner_ref})
    del closed_handle
    smoke._require_collected("engine A", {"worker": owner_ref})


def test_success_publication_writes_verified_events_before_pass(tmp_path):
    events = [{"phase": "A"}, {"phase": "B"}]
    smoke._publish_success(
        tmp_path, {"status": "pass"}, events, lambda: {"stable": True}
    )
    payload = (tmp_path / "callback_events.jsonl").read_bytes()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "pass"
    assert result["completion_provenance"] == {"stable": True}
    assert result["callback_events"] == {
        "path": "callback_events.jsonl",
        "event_count": 2,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_event_write_failure_cannot_publish_pass(tmp_path, monkeypatch):
    original = smoke._atomic_write

    def fail_events(path, payload):
        if path.name == "callback_events.jsonl":
            raise OSError("injected event failure")
        original(path, payload)

    monkeypatch.setattr(smoke, "_atomic_write", fail_events)
    with pytest.raises(OSError, match="injected"):
        smoke._publish_success(
            tmp_path,
            {"status": "pass"},
            [{"phase": "A"}],
            lambda: {"stable": True},
        )
    assert not (tmp_path / "result.json").exists()


def test_pass_result_write_failure_leaves_no_pass(tmp_path, monkeypatch):
    original = smoke._stage_write

    def fail_result(path, payload):
        if path.name == "result.json":
            raise OSError("injected result failure")
        return original(path, payload)

    monkeypatch.setattr(smoke, "_stage_write", fail_result)
    with pytest.raises(OSError, match="injected"):
        smoke._publish_success(
            tmp_path,
            {"status": "pass"},
            [{"phase": "A"}],
            lambda: {"stable": True},
        )
    assert not (tmp_path / "result.json").exists()


def test_post_event_provenance_mutation_cannot_publish_pass(tmp_path, monkeypatch):
    protected = tmp_path / "protected.py"
    protected.write_text("stable\n")
    expected = hashlib.sha256(protected.read_bytes()).hexdigest()
    original = smoke._atomic_write

    def write_then_mutate(path, payload):
        original(path, payload)
        if path.name == "callback_events.jsonl":
            protected.write_text("changed\n")

    def validate():
        identity = smoke._file_identity(protected)
        if identity["sha256"] != expected:
            raise ValueError("provenance changed")
        return identity

    monkeypatch.setattr(smoke, "_atomic_write", write_then_mutate)
    with pytest.raises(ValueError, match="changed"):
        smoke._publish_success(tmp_path, {"status": "pass"}, [{"phase": "A"}], validate)
    assert not (tmp_path / "result.json").exists()


def test_staged_pass_revalidates_provenance_before_commit(tmp_path, monkeypatch):
    protected = tmp_path / "manifest.json"
    protected.write_text("stable\n")
    expected = hashlib.sha256(protected.read_bytes()).hexdigest()
    original = smoke._stage_write

    def stage_then_mutate(path, payload):
        staged = original(path, payload)
        if path.name == "result.json":
            protected.write_text("changed\n")
        return staged

    def validate():
        identity = smoke._file_identity(protected)
        if identity["sha256"] != expected:
            raise ValueError("provenance changed")
        return identity

    monkeypatch.setattr(smoke, "_stage_write", stage_then_mutate)
    with pytest.raises(ValueError, match="changed"):
        smoke._publish_success(tmp_path, {"status": "pass"}, [{"phase": "A"}], validate)
    assert not (tmp_path / "result.json").exists()


def _main_fixture(tmp_path):
    source_manifest = tmp_path / "SOURCE_MANIFEST.txt"
    source_manifest.write_text(
        f"extension_commit={'1' * 40}\n"
        f"vllm_commit={'2' * 40}\n"
        f"vllm_version=0.26+g{'2' * 9}\n"
        "python=3.12.13\ntorch=2.13.0\ncuda_variant=cu130\n"
    )
    extension_archive = tmp_path / "extension.tar.gz"
    vllm_archive = tmp_path / "vllm.tar.gz"
    extension_archive.write_bytes(b"extension archive")
    vllm_archive.write_bytes(b"vllm archive")
    extension_root = tmp_path / "extension"
    vllm_root = tmp_path / "vllm"
    for path in (
        extension_root / "semantic_offload" / "__init__.py",
        extension_root / "harness" / "policies.py",
        vllm_root / "vllm" / "__init__.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n")
    extension_tree_sha256 = smoke._source_tree_identity(extension_root)["tree_sha256"]
    vllm_tree_sha256 = smoke._source_tree_identity(vllm_root)["tree_sha256"]
    source_identity_manifest = tmp_path / "SOURCE_IDENTITY.json"
    source_identity_manifest.write_text(
        json.dumps(
            {
                "schema": "cp002-source-identity-v2",
                "extension_revision": "1" * 40,
                "vllm_revision": "2" * 40,
                "source_manifest_sha256": hashlib.sha256(
                    source_manifest.read_bytes()
                ).hexdigest(),
                "evidence_producer_sha256": smoke._file_identity(Path(smoke.__file__))[
                    "sha256"
                ],
                "extension_source_root": str(extension_root.resolve()),
                "extension_tree_sha256": extension_tree_sha256,
                "vllm_source_root": str(vllm_root.resolve()),
                "vllm_tree_sha256": vllm_tree_sha256,
                "extension_archive_path": str(extension_archive.resolve()),
                "extension_archive_sha256": hashlib.sha256(
                    extension_archive.read_bytes()
                ).hexdigest(),
                "vllm_archive_path": str(vllm_archive.resolve()),
                "vllm_archive_sha256": hashlib.sha256(
                    vllm_archive.read_bytes()
                ).hexdigest(),
                "model": smoke.DEFAULT_MODEL,
                "model_revision": smoke.DEFAULT_MODEL_REVISION,
                "required_wsl_pin_memory": "1",
            },
            sort_keys=True,
        )
        + "\n"
    )
    output_dir = tmp_path / "output"
    argv = [
        "--output-dir",
        str(output_dir),
        "--source-manifest",
        str(source_manifest),
        "--source-manifest-sha256",
        hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "--source-identity-manifest",
        str(source_identity_manifest),
        "--source-identity-manifest-sha256",
        hashlib.sha256(source_identity_manifest.read_bytes()).hexdigest(),
        "--extension-revision",
        "1" * 40,
        "--vllm-revision",
        "2" * 40,
    ]
    return output_dir, argv


def test_source_identity_manifest_authenticates_runner_and_source_tuple(tmp_path):
    _output_dir, argv = _main_fixture(tmp_path)
    args = smoke.build_parser().parse_args(argv)
    config = smoke.SmokeConfig(
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
    source_manifest = smoke._validate_source_manifest(
        args.source_manifest, config, args.source_manifest_sha256
    )
    producer = smoke._file_identity(Path(smoke.__file__))
    identity = smoke._validate_source_identity_manifest(
        args.source_identity_manifest,
        args.source_identity_manifest_sha256,
        config,
        source_manifest,
        producer,
    )
    assert identity["evidence_producer_sha256"] == producer["sha256"]
    assert (
        identity["extension_tree_sha256"]
        == smoke._source_tree_identity(Path(identity["extension_source_root"]))[
            "tree_sha256"
        ]
    )

    values = json.loads(args.source_identity_manifest.read_text())
    values["evidence_producer_sha256"] = "0" * 64
    args.source_identity_manifest.write_text(json.dumps(values, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="evidence_producer_sha256"):
        smoke._validate_source_identity_manifest(
            args.source_identity_manifest,
            hashlib.sha256(args.source_identity_manifest.read_bytes()).hexdigest(),
            config,
            source_manifest,
            producer,
        )


def test_main_event_write_failure_returns_nonzero_without_pass(tmp_path, monkeypatch):
    output_dir, argv = _main_fixture(tmp_path)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(smoke, "_require_runtime_imports_not_loaded", lambda: None)
    monkeypatch.setattr(
        smoke, "_probe_uva_capability", lambda *_args: {"status": "pass"}
    )
    monkeypatch.setattr(
        smoke,
        "_run_smoke",
        lambda *_args: {"status": "pass", "events": [{"phase": "A"}]},
    )
    monkeypatch.setattr(
        smoke, "_completion_provenance", lambda **_kwargs: {"stable": True}
    )
    original = smoke._atomic_write

    def fail_events(path, payload):
        if path.name == "callback_events.jsonl":
            raise OSError("injected event failure")
        original(path, payload)

    monkeypatch.setattr(smoke, "_atomic_write", fail_events)
    assert smoke.main(argv) == 1
    result = json.loads((output_dir / "result.json").read_text())
    assert result["status"] == "fail"
    assert not (output_dir / "callback_events.jsonl").exists()


def test_main_pass_write_failure_returns_nonzero_without_pass(tmp_path, monkeypatch):
    output_dir, argv = _main_fixture(tmp_path)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(smoke, "_require_runtime_imports_not_loaded", lambda: None)
    monkeypatch.setattr(
        smoke, "_probe_uva_capability", lambda *_args: {"status": "pass"}
    )
    monkeypatch.setattr(
        smoke,
        "_run_smoke",
        lambda *_args: {"status": "pass", "events": [{"phase": "A"}]},
    )
    original = smoke._stage_write
    result_writes = 0

    def fail_first_result(path, payload):
        nonlocal result_writes
        if path.name == "result.json":
            result_writes += 1
            if result_writes == 1:
                raise OSError("injected pass result failure")
        return original(path, payload)

    monkeypatch.setattr(
        smoke, "_completion_provenance", lambda **_kwargs: {"stable": True}
    )
    monkeypatch.setattr(smoke, "_stage_write", fail_first_result)
    assert smoke.main(argv) == 1
    assert json.loads((output_dir / "result.json").read_text())["status"] == "fail"


def test_capability_failure_does_not_consume_output_root(tmp_path, monkeypatch):
    output_dir, argv = _main_fixture(tmp_path)
    for name, value in {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(smoke, "_require_runtime_imports_not_loaded", lambda: None)

    def fail_before_claim(*_args):
        assert not output_dir.exists()
        raise RuntimeError("injected UVA capability failure")

    monkeypatch.setattr(smoke, "_probe_uva_capability", fail_before_claim)
    with pytest.raises(RuntimeError, match="capability failure"):
        smoke.main(argv)
    assert not output_dir.exists()


@pytest.mark.parametrize("source_name", ["extension", "vllm"])
def test_live_source_drift_rejected_before_capability_or_root_claim(
    tmp_path, monkeypatch, source_name
):
    output_dir, argv = _main_fixture(tmp_path)
    args = smoke.build_parser().parse_args(argv)
    identity = json.loads(args.source_identity_manifest.read_text())
    source_root = Path(identity[f"{source_name}_source_root"])
    (source_root / "drift.py").write_text("DRIFT = True\n")
    for name, value in {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(smoke, "_require_runtime_imports_not_loaded", lambda: None)
    capability_called = False

    def capability_must_not_run(*_args):
        nonlocal capability_called
        capability_called = True
        return {"status": "pass"}

    monkeypatch.setattr(smoke, "_probe_uva_capability", capability_must_not_run)
    with pytest.raises(ValueError, match="live source tree identity mismatch"):
        smoke.main(argv)
    assert capability_called is False
    assert not output_dir.exists()


def test_stable_live_source_roots_accept_authenticated_identity(tmp_path):
    _output_dir, argv = _main_fixture(tmp_path)
    args = smoke.build_parser().parse_args(argv)
    identity = json.loads(args.source_identity_manifest.read_text())
    roots = smoke._validate_source_roots(identity)
    assert roots["extension"]["tree_sha256"] == identity["extension_tree_sha256"]
    assert roots["vllm"]["tree_sha256"] == identity["vllm_tree_sha256"]


def test_parser_defaults_pin_established_small_model(tmp_path: Path):
    args = smoke.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--source-manifest",
            str(tmp_path / "SOURCE_MANIFEST.txt"),
            "--source-manifest-sha256",
            "0" * 64,
            "--source-identity-manifest",
            str(tmp_path / "SOURCE_IDENTITY.json"),
            "--source-identity-manifest-sha256",
            "5" * 64,
            "--extension-revision",
            "1" * 40,
            "--vllm-revision",
            "2" * 40,
        ]
    )
    assert args.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.model_revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert args.num_gpu_blocks_override == 40
    assert args.gpu_memory_utilization == 0.5
