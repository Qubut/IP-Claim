"""Phi is occupy mass plus kept-edge incidence. One rolled kept edge moves n and U.

These tests lock the overlay intensity map against leftover unpaid. They do
not train a trunk and they do not call encode.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ip_claim.collision.cover import Covering
from ip_claim.ssv.phi import PhiIntensity

pytestmark = pytest.mark.experiment
_PHI = PhiIntensity(4)

_DEMAND = torch.tensor([2.0, 1.0, 0.0, 0.0], dtype=torch.float64)
_OCCUPY = torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float64)
_OCCUPIED = torch.tensor([True, True, True, True])
_KEPT_ON_SUPPORT = (0, 1)
_KEPT_OFF_SUPPORT = (2, 3)
_EDGE_MASS = 0.5


def _pair_table(*kept: tuple[int, int, float], slots: int = 4) -> Tensor:
    table = torch.zeros(slots, slots, dtype=torch.float64)
    rows = torch.tensor(kept, dtype=torch.float64)
    table[rows[:, 0].long(), rows[:, 1].long()] = rows[:, 2]
    return table


class TestPhiIntensity:
    """Host-free overlay intensity against ``CollisionContainer.covering``."""

    @staticmethod
    def leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
        """Demand-weighted unpaid fraction of one query against one supply."""
        return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()

    def test_intensity_matches_occupy_plus_incident_pair_sum(self) -> None:
        """Closed-form n equals occupy on S plus pair mass on incident kept ends."""

        def cube_intensity(occupy: Tensor, pair_mass: Tensor, occupied: Tensor) -> Tensor:
            slots = occupy.size(-1)
            index = torch.arange(slots, device=occupy.device)
            incident = (index.view(1, 1, slots) == index.view(slots, 1, 1)) | (
                index.view(1, 1, slots) == index.view(1, slots, 1)
            )
            off_diag = index.view(slots, 1, 1) != index.view(1, slots, 1)
            live = occupied.to(dtype=torch.bool)
            omega = (
                pair_mass.unsqueeze(-1)
                * incident.to(dtype=pair_mass.dtype)
                * off_diag.to(dtype=pair_mass.dtype)
                * live.unsqueeze(-1).unsqueeze(-1).to(dtype=pair_mass.dtype)
                * live.unsqueeze(-2).unsqueeze(-1).to(dtype=pair_mass.dtype)
            )
            return occupy * live.to(dtype=occupy.dtype) + omega.sum(dim=(-3, -2))

        pairs = _pair_table((*_KEPT_ON_SUPPORT, _EDGE_MASS), (*_KEPT_OFF_SUPPORT, 0.25))
        intensity = _PHI(_OCCUPY, pairs, _OCCUPIED)
        expected = cube_intensity(_OCCUPY, pairs, _OCCUPIED)
        hand = torch.tensor([1.5, 1.5, 0.75, 0.75], dtype=torch.float64)
        assert torch.allclose(intensity, expected)
        assert torch.allclose(intensity, hand)

    def test_empty_kept_edges_leave_occupy_on_live_slots_only(self) -> None:
        """With no kept pairs, intensity is occupy mass and zero off the occupied set."""
        leaked = torch.tensor([1.0, 1.0, 0.5, 9.0], dtype=torch.float64)
        live = torch.tensor([True, True, True, False])
        empty = torch.zeros(4, 4, dtype=torch.float64)
        intensity = _PHI(leaked, empty, live)
        assert torch.allclose(
            intensity,
            torch.tensor([1.0, 1.0, 0.5, 0.0], dtype=torch.float64),
        )

    def test_self_pair_and_unoccupied_pair_do_not_add(self) -> None:
        """Diagonal mass and a pair touching an empty slot do not change intensity."""
        live = torch.tensor([True, True, False, False])
        occupy = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        pairs = _pair_table((0, 0, 3.0), (0, 2, 3.0), (2, 3, 3.0))
        intensity = _PHI(occupy, pairs, live)
        assert torch.allclose(intensity, occupy)

    def test_rolling_kept_edge_on_demand_support_moves_intensity_and_unpaid(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Zeroing one kept edge on supp(L) changes Phi on those slots and leftover unpaid."""
        pairs = _pair_table((*_KEPT_ON_SUPPORT, _EDGE_MASS), (*_KEPT_OFF_SUPPORT, 0.25))
        rolled = pairs.clone()
        rolled[_KEPT_ON_SUPPORT] = 0.0
        paid = _PHI(_OCCUPY, pairs, _OCCUPIED)
        dropped = _PHI(_OCCUPY, rolled, _OCCUPIED)
        unpaid_paid = self.leftover(covering, _DEMAND, paid)
        unpaid_dropped = self.leftover(covering, _DEMAND, dropped)
        request.node.user_properties.extend((
            ('n_paid_0', paid[0].item()),
            ('n_dropped_0', dropped[0].item()),
            ('unpaid_paid', unpaid_paid.item()),
            ('unpaid_dropped', unpaid_dropped.item()),
        ))
        assert paid[0].item() != dropped[0].item()
        assert paid[1].item() != dropped[1].item()
        assert paid[2].item() == dropped[2].item()
        assert paid[3].item() == dropped[3].item()
        assert unpaid_paid.item() != unpaid_dropped.item()
        assert unpaid_paid.item() < unpaid_dropped.item()

    def test_rolling_kept_edge_off_demand_support_leaves_unpaid_put(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """A kept edge whose ends miss supp(L) moves n off support and leaves leftover unpaid."""
        pairs = _pair_table((*_KEPT_ON_SUPPORT, _EDGE_MASS), (*_KEPT_OFF_SUPPORT, 0.25))
        rolled = pairs.clone()
        rolled[_KEPT_OFF_SUPPORT] = 0.0
        paid = _PHI(_OCCUPY, pairs, _OCCUPIED)
        dropped = _PHI(_OCCUPY, rolled, _OCCUPIED)
        unpaid_paid = self.leftover(covering, _DEMAND, paid)
        unpaid_dropped = self.leftover(covering, _DEMAND, dropped)
        request.node.user_properties.extend((
            ('n_paid_2', paid[2].item()),
            ('n_dropped_2', dropped[2].item()),
            ('unpaid_paid', unpaid_paid.item()),
            ('unpaid_dropped', unpaid_dropped.item()),
        ))
        assert paid[2].item() != dropped[2].item()
        assert paid[3].item() != dropped[3].item()
        assert torch.allclose(paid[:2], dropped[:2])
        assert torch.allclose(unpaid_paid, unpaid_dropped)
