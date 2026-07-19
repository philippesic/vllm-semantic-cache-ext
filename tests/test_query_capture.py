# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for query_capture.py's `_should_sample_step` -- the pure
throttle-decision logic behind the `capture_stride` knob (TTFT-tax follow-up
investigation, see semantic-eviction-plan.md). The rest of query_capture.py
(the TorchDispatchMode/prepare_inputs monkeypatches) is deliberately not
unit-tested here, matching the module's own docstring: those only work under
a real compiled/CUDA-graph server and are validated end-to-end there instead.
"""

from semantic_offload.query_capture import _should_sample_step


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
