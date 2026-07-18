# SPDX-License-Identifier: Apache-2.0
"""Pluggable block-summary build + scoring for semantic KV-cache eviction.

Three scoring methods share one summary payload (min, max, mean, and MAD --
mean absolute deviation from the block centroid, used by cuboid-mean's radius
term). Nothing in Step 1.1 calls these yet; the interface is stubbed here so
Step 1.2 (worker-side summary computation) and Step 1.3 (scoring bridge) don't
require a rewrite. See .claude/docs/semantic-eviction-plan.md, Status
(2026-07-15), for why all three methods are carried forward instead of one.
"""

from dataclasses import dataclass

import torch


@dataclass
class BlockSummary:
    """Per-block key summary, one instance per (block, KV head)."""

    min: torch.Tensor
    max: torch.Tensor
    mean: torch.Tensor
    mad: torch.Tensor


def build_summary(keys: torch.Tensor) -> BlockSummary:
    """Build a BlockSummary from one block's Key vectors, one KV head.

    Args:
        keys: Key vectors for the block. Shape [block_tokens, head_dim].

    Returns:
        The block's min/max/mean/MAD summary.
    """
    mean = keys.mean(dim=0)
    return BlockSummary(
        min=keys.amin(dim=0),
        max=keys.amax(dim=0),
        mean=mean,
        mad=(keys - mean).abs().mean(dim=0),
    )


def score_minmax(query: torch.Tensor, summary: BlockSummary) -> float:
    """Quest-style bounding-box upper bound on any key's dot product with query."""
    return torch.maximum(query * summary.max, query * summary.min).sum().item()


def score_mean(query: torch.Tensor, summary: BlockSummary) -> float:
    """Dot product against the block centroid."""
    return torch.dot(query, summary.mean).item()


def score_cuboid_mean(query: torch.Tensor, summary: BlockSummary) -> float:
    """ArkVale-style cuboid-mean: min/max box with a mean-derived radius.

    Box bounds are the centroid +/- the mean absolute deviation (the radius),
    rather than the block's true min/max. See ArkVale (NeurIPS 2024) eq. 4.
    """
    box_min = summary.mean - summary.mad
    box_max = summary.mean + summary.mad
    return torch.maximum(query * box_max, query * box_min).sum().item()


_SCORERS = {
    "minmax": score_minmax,
    "mean": score_mean,
    "cuboid_mean": score_cuboid_mean,
}


def score_minmax_batch(
    query: torch.Tensor, maxs: torch.Tensor, mins: torch.Tensor
) -> torch.Tensor:
    """Batched score_minmax -- mathematically identical to score_minmax()
    applied per-candidate, but vectorized so it costs one GPU op (and one
    CPU/GPU sync when the caller reads the result) instead of one op-plus-
    sync per candidate. `query`: [num_kv_heads, head_dim]. `maxs`/`mins`:
    [n_candidates, num_kv_heads, head_dim]. Returns [n_candidates,
    num_kv_heads] (unreduced across heads -- caller combines, e.g. max())."""
    return torch.maximum(query * maxs, query * mins).sum(dim=-1)


def score_mean_batch(query: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Batched score_mean -- see score_minmax_batch's docstring for shapes."""
    return (query * means).sum(dim=-1)


def score_cuboid_mean_batch(
    query: torch.Tensor, means: torch.Tensor, mads: torch.Tensor
) -> torch.Tensor:
    """Batched score_cuboid_mean -- see score_minmax_batch's docstring for
    shapes."""
    box_min = means - mads
    box_max = means + mads
    return torch.maximum(query * box_max, query * box_min).sum(dim=-1)


def average_summaries(summaries: list[BlockSummary]) -> BlockSummary:
    """Collapse a block's per-KV-head summaries into one `[head_dim]`
    representation, dimensionally matching a query representation averaged
    over query heads (GQA head-count mismatch doesn't matter once both sides
    are collapsed to a single vector) -- see Step 1.3."""
    return BlockSummary(
        min=torch.stack([s.min for s in summaries]).mean(dim=0),
        max=torch.stack([s.max for s in summaries]).mean(dim=0),
        mean=torch.stack([s.mean for s in summaries]).mean(dim=0),
        mad=torch.stack([s.mad for s in summaries]).mean(dim=0),
    )


def score(method: str, query: torch.Tensor, summary: BlockSummary) -> float:
    """Score a block summary against a query using the named method."""
    scorer = _SCORERS.get(method)
    if scorer is None:
        raise ValueError(
            f"Unknown scoring method: {method!r}. Supported: {list(_SCORERS)}"
        )
    return scorer(query, summary)
