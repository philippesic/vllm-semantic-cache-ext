# SPDX-License-Identifier: Apache-2.0
"""Policy -> --kv-transfer-config JSON, for Step 1.6's benchmark grid:
{LRU, ARC, semantic-minmax, semantic-mean, semantic-cuboid-mean}.

LRU/ARC use vLLM's stock `CPUOffloadingSpec` (`eviction_policy` extra_config
key -- confirmed in vllm/v1/kv_offload/cpu/spec.py, NOT `cache_policy`, a
real gotcha caught by reading the source rather than guessing). The three
semantic variants use `SemanticOffloadingSpec`/`SemanticOffloadingConnector`
with `method` selecting the scoring method (issues log entry #34 -- this
selection knob didn't exist until tonight).
"""

POLICY_NAMES = (
    "lru",
    "arc",
    "semantic-minmax",
    "semantic-mean",
    "semantic-cuboid-mean",
)


def kv_transfer_config(policy: str, cpu_bytes_to_use: int) -> dict:
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown policy {policy!r}, expected one of {POLICY_NAMES}")

    if policy in ("lru", "arc"):
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": cpu_bytes_to_use,
                "eviction_policy": policy,
            },
        }

    method = policy.removeprefix("semantic-").replace("-", "_")
    return {
        "kv_connector": "SemanticOffloadingConnector",
        "kv_connector_module_path": "semantic_offload.connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "SemanticOffloadingSpec",
            "spec_module_path": "semantic_offload.spec",
            "cpu_bytes_to_use": cpu_bytes_to_use,
            "method": method,
        },
    }
