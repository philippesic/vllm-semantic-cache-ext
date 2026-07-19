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
        # Worker-side scoring only ever needs the one configured method
        # (SemanticPolicy only consults its own method's relevance EMA) --
        # same extra_config key get_manager() already reads below.
        method = str(self.extra_config.get("method", "minmax"))
        # Only every capture_stride-th eligible query-capture step actually
        # scores/updates relevance (default 1: unchanged, every step) -- a
        # TTFT-tax follow-up knob, see query_capture.py's
        # `_should_sample_step` and semantic-eviction-plan.md's post-fix
        # optimization pass. Relies on the manager's EMA staleness-tolerance
        # to carry signal across skipped steps.
        capture_stride = int(self.extra_config.get("capture_stride", 1))
        return SemanticOffloadingWorker(
            kv_caches=kv_caches,
            # OffloadingSpec's own attribute was renamed block_size_factor ->
            # blocks_per_chunk upstream (vLLM #48150).
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
            vllm_config=self.vllm_config,
            method=method,
            capture_stride=capture_stride,
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            grace_window_blocks = int(self.extra_config.get("grace_window_blocks", 0))
            eviction_mode = str(self.extra_config.get("eviction_mode", "blend"))
            # Structural reuse bonuses (SemanticPolicy's chain/session logic).
            # Read here so the benchmark grid can actually exercise them --
            # previously they defaulted off with no launch-time knob, so every
            # Step 1.6 cell ran with both fully disabled (issues log entry #66
            # follow-up).
            chain_aware = bool(self.extra_config.get("chain_aware", False))
            session_aware = bool(self.extra_config.get("session_aware", False))
            session_bonus_half_life = int(
                self.extra_config.get("session_bonus_half_life", 0)
            )
            # Step 1.6's benchmark grid needs {minmax, mean, cuboid_mean} as
            # separate, selectable configs -- previously hardcoded to
            # manager.py's _DEFAULT_METHOD with no way to pick another at
            # launch time (issues log entry #34).
            method = str(self.extra_config.get("method", "minmax"))
            self._manager = SemanticOffloadingManager(
                num_blocks=self.num_blocks,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                grace_window_blocks=grace_window_blocks,
                eviction_mode=eviction_mode,
                chain_aware=chain_aware,
                session_aware=session_aware,
                session_bonus_half_life=session_bonus_half_life,
                method=method,
            )
        return self._manager
