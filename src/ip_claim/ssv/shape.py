"""Community and subgraph energies of the existing overlay pair table.

Phi already consumes this square table as kept pair mass. Destination leftover
unpaid stays on Covering. Refuse mass is zero here and does not pay the
quality function. Column count of the soft assignment is a host width, not
occupy support size.
"""

from __future__ import annotations

from math import sqrt
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class ShapeEnergy(NamedTuple):
    """Unsupervised overlay geometry on kept pair mass.

    Attributes:
        total: Community addend plus subgraph addend.
        community: Negative Newman modularity plus weighted collapse.
        subgraph: Local-global mutual information of induced neighborhoods.
        modularity: Soft Newman Q of the symmetrised pair table.
        collapse: DMoN column-balance penalty of the soft assignment.
    """

    total: Tensor
    community: Tensor
    subgraph: Tensor
    modularity: Tensor
    collapse: Tensor


def shape_family(
    pair_mass: Tensor,
    assignment: Tensor,
    occupied: Tensor,
    *,
    collapse_weight: float = 1.0,
) -> ShapeEnergy:
    """Shape energy of kept pair mass against a soft community assignment.

    The pair table is the square addend Phi already reads. The assignment
    lives on the same slot axis. Occupied rows only enter modularity and
    collapse. Self-pairs and unoccupied pairs are zero. Subgraph contrast
    is local-global mutual information of induced neighborhoods of that
    same table.

    Args:
        pair_mass: Kept typed mass ``(..., K, K)``.
        assignment: Soft community weights ``(..., K, r)`` with ``r >= 2``.
        occupied: Occupy support ``(..., K)``.
        collapse_weight: Weight on the DMoN column-balance penalty.

    Returns:
        Community addend, subgraph addend, and the two diagnostics.
    """
    slots = pair_mass.size(-1)
    if pair_mass.size(-2) != slots:
        msg = 'kept pair mass must be square on the slot axis'
        raise ValueError(msg)
    if assignment.size(-2) != slots:
        msg = 'community assignment must align with the pair-table slot axis'
        raise ValueError(msg)
    if assignment.size(-1) < 2:
        msg = 'community assignment needs at least two columns'
        raise ValueError(msg)
    if occupied.shape != pair_mass.shape[:-1]:
        msg = 'occupied support must match the pair-table slot axis'
        raise ValueError(msg)
    live = occupied.to(dtype=torch.bool)
    gate = live.unsqueeze(-1) & live.unsqueeze(-2)
    identity = torch.eye(slots, dtype=torch.bool, device=pair_mass.device)
    kept = pair_mass.masked_fill(~gate | identity, 0)
    support = live.to(dtype=assignment.dtype)
    clustered = assignment * support.unsqueeze(-1)
    columns = assignment.size(-1)
    adjacency = kept + kept.transpose(-1, -2)
    degree = adjacency.sum(dim=-1)
    twice_mass = degree.sum(dim=-1)
    associated = clustered.transpose(-1, -2) @ adjacency @ clustered
    modularity_trace = associated.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    projected = (clustered.transpose(-1, -2) @ degree.unsqueeze(-1)).squeeze(-1)
    safe_mass = twice_mass.clamp_min(1)
    null = projected.square().sum(dim=-1) / safe_mass
    modularity = torch.where(
        twice_mass > 0,
        (modularity_trace - null) / safe_mass,
        twice_mass.new_zeros(twice_mass.shape),
    )
    support_size = support.sum(dim=-1).clamp_min(1)
    collapse = clustered.sum(dim=-2).norm(dim=-1) / support_size * sqrt(columns) - 1
    community = -modularity + clustered.new_tensor(collapse_weight) * collapse
    patch = torch.cat((adjacency, adjacency @ adjacency), dim=-1)
    live_f = live.to(dtype=patch.dtype)
    live_weight = occupied.to(dtype=patch.dtype) * live_f
    denom = live_weight.sum(dim=-1, keepdim=True).clamp_min(1)
    summary = (patch * (live_weight / denom).unsqueeze(-1)).sum(dim=-2)
    aligned = summary.unsqueeze(-2).expand_as(patch)
    positive = F.cosine_similarity(patch, aligned, dim=-1)
    distant = F.cosine_similarity(patch.roll(1, dims=-2), aligned, dim=-1)
    nll = F.softplus(-positive) + F.softplus(distant)
    subgraph = (nll * live_f).sum(dim=-1) / live_f.sum(dim=-1).clamp_min(1)
    return ShapeEnergy(
        total=community + subgraph,
        community=community,
        subgraph=subgraph,
        modularity=modularity,
        collapse=collapse,
    )


__all__ = ['ShapeEnergy', 'shape_family']
