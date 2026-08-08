# SPDX-License-Identifier: Apache-2.0
"""Pure store-transfer identity reconstruction."""

from typing import TypeVar

KeyT = TypeVar("KeyT")


def build_store_job_layout(
    key_cpu_blocks: dict[KeyT, int],
    dst_block_ids,
    src_block_ids,
    blocks_per_chunk: int,
) -> list[tuple[KeyT, list[int]]] | None:
    """Recover exact key order from ordered CPU and GPU transfer specs."""
    key_by_cpu_block = {block_id: key for key, block_id in key_cpu_blocks.items()}
    try:
        ordered_keys = [key_by_cpu_block[int(block_id)] for block_id in dst_block_ids]
    except KeyError:
        return None
    src_ids = [int(block_id) for block_id in src_block_ids]
    if len(src_ids) != len(ordered_keys) * blocks_per_chunk:
        return None
    return [
        (
            key,
            src_ids[index * blocks_per_chunk : (index + 1) * blocks_per_chunk],
        )
        for index, key in enumerate(ordered_keys)
    ]
