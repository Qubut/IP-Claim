"""Overlay intensity as occupy mass plus incidence of kept pair addends.

Leftover unpaid stays on Covering. This module owns the off-diagonal
identity that moves with the module, and scatters kept directed pair
mass into a square addend table. Consumed pairs do not enter that table.
A living typed pair table is composed into that square addend when
supplied.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PhiIntensity(nn.Module):
    """Slot intensity of a kept overlay.

    Occupy mass is zero off the occupied set. A pair addend pays only when
    both endpoints are occupied, the pair is off-diagonal, and the slot is
    one of those endpoints. Living typed pair tables are composed into that
    addend when supplied. Self-pairs and refused (zero) pairs add nothing.
    The identity buffer is not a parameter; leftover unpaid stays on Covering.

    Args:
        slots: Slot count ``K``. Registers a ``K`` by ``K`` boolean identity.
    """

    identity: Tensor

    def __init__(self, slots: int) -> None:
        super().__init__()
        if slots < 1:
            msg = 'overlay intensity needs a positive slot count'
            raise ValueError(msg)
        self.identity = nn.Buffer(
            torch.eye(slots, dtype=torch.bool),
            persistent=False,
        )

    def kept_pair_addends(
        self,
        sources: Tensor,
        destinations: Tensor,
        mass: Tensor,
        keep: Tensor,
    ) -> Tensor:
        """Square slot table of kept directed pair mass.

        Consumed pairs contribute nothing. The table is the addend
        ``forward`` reads; leftover unpaid stays on Covering. The zeros
        table and pair indices follow this module's device.
        """
        if sources.shape != destinations.shape or sources.shape != mass.shape:
            msg = 'kept pair sources, destinations, and mass must share one pair axis'
            raise ValueError(msg)
        if keep.shape != mass.shape:
            msg = 'kept mask must match pair mass'
            raise ValueError(msg)
        device = self.identity.device
        slots = self.identity.size(0)
        table = torch.zeros(slots, slots, dtype=mass.dtype, device=device)
        gated = mass.to(device=device).masked_fill(
            ~keep.to(device=device, dtype=torch.bool),
            0,
        )
        return table.index_put(
            (
                sources.to(device=device, dtype=torch.long),
                destinations.to(device=device, dtype=torch.long),
            ),
            gated,
            accumulate=True,
        )

    def forward(
        self,
        occupy: Tensor,
        pair_mass: Tensor,
        occupied: Tensor,
        living: Tensor | None = None,
    ) -> Tensor:
        """Slot intensity of a kept overlay.

        Args:
            occupy: Nonnegative occupy mass ``(..., K)``.
            pair_mass: Kept-edge addend ``(..., K, K)``. Entry ``(i, j)`` is the
                mass of directed pair ``i`` to ``j``.
            occupied: Occupied-set mask ``(..., K)``, true on live slots.
            living: Optional living typed pair table ``(..., K, K)``. Other-filing
                kept addends. ``None`` is an empty living table.

        Returns:
            Intensity ``(..., K)``. Each slot is occupy mass plus outgoing and
            incoming kept-pair mass, including living links that touch live ends.
        """
        slots = occupy.size(-1)
        if slots != self.identity.size(0):
            msg = 'occupy slot axis must match the registered identity'
            raise ValueError(msg)
        if pair_mass.size(-1) != slots or pair_mass.size(-2) != slots:
            msg = 'kept pair addends must be square on the occupy slot axis'
            raise ValueError(msg)
        if occupied.shape != occupy.shape:
            msg = 'occupied support must match occupy mass'
            raise ValueError(msg)
        if living is not None and (living.size(-1) != slots or living.size(-2) != slots):
            msg = 'living pair addends must be square on the occupy slot axis'
            raise ValueError(msg)
        composed = pair_mass
        if living is not None:
            composed = pair_mass + living.to(device=pair_mass.device, dtype=pair_mass.dtype)
        live = occupied.to(dtype=torch.bool)
        mass = occupy * live.to(dtype=occupy.dtype)
        gate = live.unsqueeze(-1) & live.unsqueeze(-2)
        identity = self.identity.to(device=composed.device)
        kept = composed.masked_fill(~gate | identity, 0)
        return mass + kept.sum(dim=-1) + kept.sum(dim=-2)


__all__ = ['PhiIntensity']
