# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingSpec: CPUOffloadingSpec with a SemanticOffloadingManager.

Register via kv_connector_extra_config={"spec_name": "SemanticOffloadingSpec",
"spec_module_path": "semantic_offload.spec", "cpu_bytes_to_use": ...}. See
.claude/docs/semantic-eviction-plan.md, Step 1.1.
"""

from typing_extensions import override

from semantic_offload.manager import SemanticOffloadingManager
from semantic_offload.worker import SemanticOffloadingWorker
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


class SemanticOffloadingSpec(CPUOffloadingSpec):
    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> SemanticOffloadingWorker:
        return SemanticOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            vllm_config=self.vllm_config,
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            grace_window_blocks = int(self.extra_config.get("grace_window_blocks", 0))
            eviction_mode = str(self.extra_config.get("eviction_mode", "blend"))
            self._manager = SemanticOffloadingManager(
                num_blocks=self.num_blocks,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                grace_window_blocks=grace_window_blocks,
                eviction_mode=eviction_mode,
            )
        return self._manager
