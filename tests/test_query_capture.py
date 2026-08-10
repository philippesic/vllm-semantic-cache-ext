# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for query-capture throttling and hook lifecycle ownership."""

from types import SimpleNamespace
from threading import Event, Thread

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from semantic_offload import query_capture
from semantic_offload.query_capture import _should_sample_step, install


def _worker_config(mixed_cudagraph_mode: str):
    class CUDAGraphMode:
        def mixed_mode(self):
            return SimpleNamespace(name=mixed_cudagraph_mode)

    attention = SimpleNamespace(num_kv_heads=1, head_size=1)
    return SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode(),
            static_forward_context={"layer": attention},
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            world_size=1,
        ),
        model_config=SimpleNamespace(
            get_total_num_kv_heads=lambda: 1,
            get_num_attention_heads=lambda parallel_config: 1,
            use_mla=False,
        ),
        cache_config=SimpleNamespace(block_size=16),
    )


def test_stride_one_samples_every_step():
    """Default behavior (capture_stride=1) must be unchanged from before
    this knob existed: every eligible step fires."""
    assert all(_should_sample_step(i, 1) for i in range(10))


def test_stride_zero_or_negative_treated_as_no_throttle():
    """Defensive: a misconfigured stride (<=1) must not silently disable
    capture entirely -- falls back to "every step", the safe default."""
    assert all(_should_sample_step(i, 0) for i in range(5))
    assert all(_should_sample_step(i, -3) for i in range(5))


def test_stride_n_samples_every_nth_step_only():
    stride = 4
    sampled = [i for i in range(20) if _should_sample_step(i, stride)]
    assert sampled == [0, 4, 8, 12, 16]


def test_install_close_and_reinstall_own_prepare_inputs_patch(monkeypatch):
    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        if scheduler_output == "raise":
            raise RuntimeError("prepare failed")
        return SimpleNamespace(req_ids=["req"], num_scheduled_tokens=[2])

    def prior_execute_model(self_runner, scheduler_output=None):
        return GPUModelRunner.prepare_inputs(self_runner, scheduler_output, None)

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    config = SimpleNamespace(use_v2_model_runner=True)

    first = install(config, "layer", lambda req_ids, query: None)
    assert GPUModelRunner.prepare_inputs is not prior_prepare_inputs
    assert GPUModelRunner.execute_model is not prior_execute_model

    with pytest.raises(RuntimeError, match="already installed"):
        install(config, "layer", lambda req_ids, query: None)

    owner = SimpleNamespace(vllm_config=config)
    result = GPUModelRunner.execute_model(owner, None)
    assert result.req_ids == ["req"]
    assert query_capture._current_layout.get() is None

    with pytest.raises(RuntimeError, match="unexpected runner"):
        GPUModelRunner.execute_model(SimpleNamespace(vllm_config=config), None)

    other = SimpleNamespace(vllm_config=SimpleNamespace())
    assert GPUModelRunner.execute_model(other, None).req_ids == ["req"]
    assert query_capture._current_layout.get() is None

    with pytest.raises(RuntimeError, match="prepare failed"):
        GPUModelRunner.execute_model(owner, "raise")
    assert query_capture._current_layout.get() is None
    assert query_capture._current_handle.get() is None

    first.close()
    assert GPUModelRunner.prepare_inputs is prior_prepare_inputs
    assert GPUModelRunner.execute_model is prior_execute_model
    first.close()

    second = install(config, "other-layer", lambda req_ids, query: None)
    second.close()
    assert GPUModelRunner.prepare_inputs is prior_prepare_inputs


def test_worker_shutdown_closes_query_capture_before_base_shutdown(monkeypatch):
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

    from semantic_offload.worker import SemanticOffloadingWorker

    calls = []

    class Handle:
        def close(self):
            calls.append("capture")

    worker = object.__new__(SemanticOffloadingWorker)
    worker._query_capture_mode = Handle()
    monkeypatch.setattr(
        CPUOffloadingWorker, "shutdown", lambda self: calls.append("base")
    )

    SemanticOffloadingWorker.shutdown(worker)

    assert calls == ["capture", "base"]
    assert worker._query_capture_mode is None


def test_worker_shutdown_preserves_resources_when_capture_close_fails(monkeypatch):
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

    from semantic_offload.worker import SemanticOffloadingWorker

    calls = []

    class Handle:
        def close(self):
            calls.append("capture")
            raise RuntimeError("capture still active")

    handle = Handle()
    worker = object.__new__(SemanticOffloadingWorker)
    worker._query_capture_mode = handle
    monkeypatch.setattr(
        CPUOffloadingWorker, "shutdown", lambda self: calls.append("base")
    )

    with pytest.raises(RuntimeError, match="still active"):
        SemanticOffloadingWorker.shutdown(worker)

    assert calls == ["capture"]
    assert worker._query_capture_mode is handle


def test_worker_rejects_full_cudagraph_before_base_allocation(monkeypatch):
    from semantic_offload import worker as worker_module

    base_allocations = []
    monkeypatch.setattr(
        worker_module,
        "init_cpu_offloading_worker_base",
        lambda *args, **kwargs: base_allocations.append(kwargs),
    )

    with pytest.raises(ValueError, match="outside FULL CUDA graphs"):
        worker_module.SemanticOffloadingWorker(
            kv_caches=None,
            blocks_per_chunk=1,
            num_cpu_blocks=1,
            vllm_config=_worker_config("FULL"),
        )

    assert base_allocations == []


def test_worker_rejects_duplicate_owner_before_base_allocation(monkeypatch):
    from semantic_offload import worker as worker_module

    owner_config = SimpleNamespace(use_v2_model_runner=True)
    owner = install(owner_config, "layer", lambda req_ids, query: None)
    base_allocations = []
    monkeypatch.setattr(
        worker_module,
        "init_cpu_offloading_worker_base",
        lambda *args, **kwargs: base_allocations.append(kwargs),
    )
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            worker_module.SemanticOffloadingWorker(
                kv_caches=None,
                blocks_per_chunk=1,
                num_cpu_blocks=1,
                vllm_config=_worker_config("PIECEWISE"),
            )
    finally:
        owner.close()

    assert base_allocations == []


def test_callback_exception_consumes_layout_and_cannot_replay(monkeypatch):
    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=["req"], num_scheduled_tokens=[2])

    class FakeAttention:
        def __str__(self):
            return "vllm.unified_attention_with_output.default"

        def __call__(self, *args, **kwargs):
            return "attention-result"

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    config = SimpleNamespace(use_v2_model_runner=True)
    callbacks = []

    def fail_callback(req_ids, query):
        callbacks.append(req_ids)
        raise RuntimeError("callback failed")

    handle = install(config, "layer", fail_callback)
    try:
        query_capture._current_layout.set(query_capture._BatchLayout(["req"], [(0, 2)]))
        args = (torch.ones(2, 2, 1), None, None, None, "layer")
        mode = handle._mode_factory()

        other_args = (torch.ones(2, 2, 1), None, None, None, "other-layer")
        assert (
            mode.__torch_dispatch__(FakeAttention(), (), other_args)
            == "attention-result"
        )
        assert query_capture._current_layout.get() is not None

        with pytest.raises(RuntimeError, match="callback failed"):
            mode.__torch_dispatch__(FakeAttention(), (), args)

        assert query_capture._current_layout.get() is None
        assert mode.__torch_dispatch__(FakeAttention(), (), args) == "attention-result"
        assert callbacks == [["req"]]
    finally:
        handle.close()


def test_mode_entry_failure_clears_execution_context(monkeypatch):
    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def prior_execute_model(self_runner):
        return None

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    original_enter = TorchDispatchMode.__enter__

    def fail_enter(self):
        raise RuntimeError("mode enter failed")

    monkeypatch.setattr(TorchDispatchMode, "__enter__", fail_enter)
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda req_ids, query: None)
    with pytest.raises(RuntimeError, match="mode enter failed"):
        GPUModelRunner.execute_model(SimpleNamespace(vllm_config=config))

    assert query_capture._current_layout.get() is None
    assert query_capture._current_handle.get() is None
    handle.close()

    monkeypatch.setattr(TorchDispatchMode, "__enter__", original_enter)
    handle = install(config, "layer", lambda req_ids, query: None)
    handle.close()


def test_full_cudagraph_mode_is_rejected_before_patching(monkeypatch):
    class FullMode:
        name = "FULL"

        def mixed_mode(self):
            return self

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=FullMode()),
    )

    with pytest.raises(ValueError, match="outside FULL CUDA graphs"):
        install(config, "layer", lambda req_ids, query: None)

    assert GPUModelRunner.prepare_inputs is prior_prepare_inputs


def test_sequential_installations_do_not_reuse_callbacks(monkeypatch):
    class FakeAttention:
        def __str__(self):
            return "vllm.unified_attention_with_output.default"

        def __call__(self, *args, **kwargs):
            return None

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    config = SimpleNamespace(use_v2_model_runner=True)
    callbacks = []
    args = (torch.ones(1, 1, 1), None, None, None, "layer")

    first = install(
        config, "layer", lambda req_ids, query: callbacks.append(("first", req_ids))
    )
    query_capture._current_layout.set(query_capture._BatchLayout(["a"], [(0, 1)]))
    first._mode_factory().__torch_dispatch__(FakeAttention(), (), args)
    first.close()

    second = install(
        config, "layer", lambda req_ids, query: callbacks.append(("second", req_ids))
    )
    query_capture._current_layout.set(query_capture._BatchLayout(["b"], [(0, 1)]))
    second._mode_factory().__torch_dispatch__(FakeAttention(), (), args)
    second.close()

    assert callbacks == [("first", ["a"]), ("second", ["b"])]


def test_external_patch_conflict_is_not_overwritten(monkeypatch):
    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def external_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=["external"], num_scheduled_tokens=[1])

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda req_ids, query: None)
    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", external_prepare_inputs)
    with pytest.raises(RuntimeError, match="methods changed"):
        handle.close()

    assert GPUModelRunner.prepare_inputs is external_prepare_inputs
    replacement = install(config, "layer", lambda req_ids, query: None)
    replacement.close()
    assert GPUModelRunner.prepare_inputs is external_prepare_inputs


def test_close_waits_for_execution_on_another_thread(monkeypatch):
    execution_started = Event()
    release_execution = Event()
    errors = []

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def prior_execute_model(self_runner):
        execution_started.set()
        if not release_execution.wait(timeout=2):
            raise RuntimeError("test execution was not released")

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda req_ids, query: None)
    owner = SimpleNamespace(vllm_config=config)

    def execute():
        try:
            GPUModelRunner.execute_model(owner)
        except BaseException as exc:
            errors.append(exc)

    def close():
        try:
            handle.close()
        except BaseException as exc:
            errors.append(exc)

    execute_thread = Thread(target=execute)
    execute_thread.start()
    assert execution_started.wait(timeout=2)
    close_thread = Thread(target=close)
    close_thread.start()
    with query_capture._lifecycle_condition:
        assert handle._closing
        assert handle._active_executions == 1
    assert close_thread.is_alive()

    release_execution.set()
    execute_thread.join(timeout=2)
    close_thread.join(timeout=2)

    assert not execute_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert GPUModelRunner.prepare_inputs is prior_prepare_inputs
    assert GPUModelRunner.execute_model is prior_execute_model


def test_close_from_active_execution_fails_without_deadlock(monkeypatch):
    handle = None

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def prior_execute_model(self_runner):
        handle.close()

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda req_ids, query: None)

    with pytest.raises(RuntimeError, match="active execution"):
        GPUModelRunner.execute_model(SimpleNamespace(vllm_config=config))

    assert handle._active_executions == 0
    handle.close()
    assert GPUModelRunner.execute_model is prior_execute_model
