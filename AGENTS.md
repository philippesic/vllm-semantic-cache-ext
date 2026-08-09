# Agent Instructions for the Semantic Cache Audit

## Batch diagnosis before DGX handoff

When diagnosing an audit failure, inspect the runner, harness, validator, and
all available summaries/logs together. Group independent failure modes and
make the supported fixes in one pass. Do not ask the operator to rerun until
the local code path has been checked, relevant tests or static checks have
passed, and one consolidated rerun command is ready.

If a run shows completed metrics followed by a cell timeout, inspect process
shutdown and child-process cleanup before describing it as a stalled request.
Use the failed-cell and validation-error counts to distinguish missing output
from an actual model or cache failure.
