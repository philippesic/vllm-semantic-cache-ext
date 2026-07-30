# Step 1.5 — Partial-Splice: First Real-Server Byte-Exact Correctness Proof

**Date:** 2026-07-29. Hardware: B200 (DGX box), replacing the 2080Ti this
project was previously validated against.

## Result

`harness/run_adaptive_probe_live.py`, run 1 of 3 after the fixes below
landed, produced:

```
result={'hit_splice': True, 'splice_event': {'line': '(EngineCore
pid=2358412) PREFETCH_EFFECT_DEBUG cmpl-ADAPTPROBE-4999-3-needle-0-afc5f68e:
PARTIAL SPLICE spliced=1 reloaded=2 covered=0.33', 'req_id':
'cmpl-ADAPTPROBE-4999-3-needle-0-afc5f68e', 'spliced': 1},
'recall_text': ' 98921-Nightingale\n\nThe secret verification code for
Project Meridian is ', 'expected_code': '98921-Nightingale', 'correct': True}
SUCCESS: tagged request ADAPTPROBE-4999-3-needle was spliced (spliced=1)
and the recall was byte-exact correct.
```

A tagged request was re-admitted after preemption, spliced 1 GPU block by
content identity (`OffloadKey`, not position) and reloaded 2 more via
normal CPU→GPU DMA in the same re-admission (coverage 0.33). The
subsequent recall of the request's secret verification code came back
byte-exact (`98921-Nightingale`), proving the spliced block held the
*correct* content — the specific silent-corruption failure mode
(Q3-1, wrong-content splice from a positional mismatch) that
`step-1.5-partial-splice-plan.md`'s acceptance check was designed to
catch. This is a full pass of that plan's acceptance check: `spliced>=1`
and `reloaded>=1` in one re-admission, plus a correct spot-checked output.

Two subsequent runs came back `NO SPLICE` for the tagged candidate (an
expected, documented outcome — only ~4 of ~44 requests per run are
tagged, and splicing had already been observed firing for *other*,
untagged requests in the same runs even before this proof).

## What was actually broken and fixed to get here

The prior ~4 attempts at this proof (all in this session, all on this
B200) failed for a real code reason, not workload tuning: **vLLM removed
the scheduler-side `on_request_preempted` hook** this connector's
prefetch pipeline was entirely gated behind (confirmed absent from the
whole `vllm/` tree at the pinned commit `dc1be79031`, replaced by
`SchedulerOutput.preempted_req_ids` read from inside
`build_connector_meta`). `_preempted_pending` was therefore never
populated, so `_retry_pending_prefetches` always no-op'd regardless of
real GPU contention — every prior workload-tuning attempt (block budget,
`max_model_len`, filler decode length) was fighting a dead code path, not
the actual bottleneck.

Fixed in `semantic_offload/connector.py`: added `_queue_preempted`,
called first in `build_connector_meta`, seeding `_preempted_pending` from
`scheduler_output.preempted_req_ids` directly (same queue-only contract
the old hook had). `on_request_preempted` itself was left in place,
unmodified — it's independently unit-tested and still correct in
isolation, just no longer vLLM's actual trigger.

Also needed for this to even launch on a B200 at all (unrelated
environment issues, not project bugs): `torchaudio`/`torch` CUDA-build
mismatch (uninstalled torchaudio, not needed for text-only serving),
`flashinfer-python`/`flashinfer-cubin` version mismatch (pinned
flashinfer-python down to match the only published cubin build), and
`harness/run_adaptive_probe_live.py`'s original 2080Ti-tuned load shape
(`num_gpu_blocks_override=200`, 140-token fillers) never forcing a single
preemption on much-faster B200 hardware — retuned to 40 blocks /
`max_model_len=512` / 400-token fillers.

## Open item, not yet explained

Filler request failure counts crept up across the session's later runs
(0 → 5 → 7 → 11 failed out of 44) with no error surfaced server-side or
client-side (the harness's `except Exception: fail += 1` doesn't log
what actually failed). Plausible cause: the tight 40-block budget causing
some fillers to queue long enough under real contention to hit the
harness's 230s client-side result timeout. Not yet investigated further
since it didn't prevent the proof above; worth a look if it starts
suppressing tagged-candidate hits specifically or climbs further.

## Next steps (per the original plan's acceptance check)

- Capture the **aggregate** graded metric (Σ spliced / Σ (spliced +
  reloaded)) across a longer run, ideally against a
  splice-disabled control, per the plan's "stronger pass" — this proof
  is one data point (0.33 coverage on one request), not yet an aggregate.
- Investigate the climbing filler-failure count if it recurs.
- Run `benchmarks/run_grid_sweep.py` for the still-owed Step 1.6
  rigorous multi-seed benchmark.
