"""Cheng Lagrange mix rescale: divide the primal step by one plus the dual sum."""

from __future__ import annotations

from torch import Tensor


def cheng_rescale(
    task: Tensor,
    constraint: Tensor,
    lambdas: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``((task + constraint) / Z, Z)`` with ``Z = 1 + sum(λ)``."""
    scale = lambdas.sum() + lambdas.new_ones(())
    return (task + constraint) / scale, scale
