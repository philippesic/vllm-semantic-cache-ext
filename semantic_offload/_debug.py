# SPDX-License-Identifier: Apache-2.0
"""Debug-print gating for the hot-path prints scattered through
worker.py/connector.py/query_capture.py (every query-capture event, every
splice attempt) -- these were unconditional `print(..., flush=True)` since
Step 1.3, fine for the small-scale dev testing they were built for, but at
Step 1.6's benchmark scale (~1000+ runs) the per-call I/O flush and
unstructured stdout volume would both cost real time and pollute logs
(issues log entry #34). Default off; set SEMANTIC_OFFLOAD_DEBUG=1 to
restore the old always-on behavior for interactive debugging."""

import os

ENABLED = os.environ.get("SEMANTIC_OFFLOAD_DEBUG", "") not in (
    "",
    "0",
    "false",
    "False",
)


def debug_print(*args, **kwargs) -> None:
    if ENABLED:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)
