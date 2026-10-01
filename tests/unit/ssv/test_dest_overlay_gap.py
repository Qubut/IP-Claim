"""Dest gap is leftover unpaid of claim demand against Phi of a kept overlay.

Rolling a kept edge that pays demand support changes leftover unpaid and dest
loss. A refuse-only pair or an off-support roll leaves leftover unpaid put.
Edge unpaid of pair tables is not that residual. Covering comes from the
collision composition root. These tests do not train a trunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from tests._ssv_fixtures import ssv_fixture_batch, ssv_smoke_module
from torch import Tensor

from ip_claim.app.container.collision import CollisionContainer
from ip_claim.collision.cover import Covering
from ip_claim.ssv.inventory import CoveringInventory
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.phi import PhiIntensity

_DEMAND = torch.tensor([2.0, 1.0, 0.0, 0.0], dtype=torch.float64)
_EDGE_MASS = 1.0
_ON_SUPPORT = (0, 1)
_OFF_SUPPORT = (3, 4)
_REFUSE_PAIR = (0, 2)
_DEST_SHIFT = 1
_OFF_Q_SHIFT = 2
_OFF_SLOTS = 6


@pytest.fixture
def covering() -> Covering:
    return CollisionContainer().covering()


def _leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


def _named_delta(covering: Covering, alpha: float, beta: float) -> Tensor:
    scale = covering.sigma.to(dtype=torch.float64)
    return scale * (beta - alpha) / ((beta + scale) * (alpha + scale))


def _intensity(
    phi: PhiIntensity,
    occupy: Tensor,
    sources: Tensor,
    destinations: Tensor,
    mass: Tensor,
    keep: Tensor,
    *,
    source_shift: int = 0,
    dest_shift: int = 0,
) -> Tensor:
    addends = phi.kept_pair_addends(sources, destinations, mass, keep)
    rolled = addends.roll(source_shift, dims=-2).roll(dest_shift, dims=-1)
    live = torch.ones_like(occupy, dtype=torch.bool)
    return phi(occupy, rolled, live)


class TestDestGapOnKeptOverlay:
    """Leftover unpaid dest gap of Phi after a kept-overlay roll."""

    def test_on_support_roll_off_demand_meets_named_delta(
        self,
        covering: Covering,
    ) -> None:
        """Moving a kept pair off supp(L) drops leftover unpaid by at least delta."""
        phi = PhiIntensity(4)
        occupy = torch.zeros(4, dtype=torch.float64)
        sources = torch.tensor([_ON_SUPPORT[0]])
        destinations = torch.tensor([_ON_SUPPORT[1]])
        mass = torch.tensor([_EDGE_MASS], dtype=torch.float64)
        keep = torch.tensor([True])
        paid = _intensity(phi, occupy, sources, destinations, mass, keep)
        moved = _intensity(
            phi,
            occupy,
            sources,
            destinations,
            mass,
            keep,
            source_shift=_OFF_Q_SHIFT,
            dest_shift=_OFF_Q_SHIFT,
        )
        unpaid_paid = _leftover(covering, _DEMAND, paid)
        unpaid_moved = _leftover(covering, _DEMAND, moved)
        delta = _named_delta(covering, 0.0, _EDGE_MASS)
        assert torch.equal(paid[:2], paid.new_tensor((_EDGE_MASS, _EDGE_MASS)))
        assert torch.equal(moved[:2], moved.new_zeros(2))
        assert unpaid_paid.item() <= unpaid_moved.item() - float(delta.item()) + 1e-12
        assert float(covering.sigma.item()) == pytest.approx(1.0)
        assert unpaid_paid.item() == pytest.approx(0.5, abs=1e-12)
        assert unpaid_moved.item() == pytest.approx(1.0, abs=1e-12)
        assert float(delta.item()) == pytest.approx(0.5, abs=1e-12)

    def test_dest_axis_roll_on_support_moves_leftover(
        self,
        covering: Covering,
    ) -> None:
        """Dest-axis roll of a kept edge on supp(L) moves leftover unpaid."""
        phi = PhiIntensity(4)
        occupy = torch.zeros(4, dtype=torch.float64)
        sources = torch.tensor([_ON_SUPPORT[0]])
        destinations = torch.tensor([_ON_SUPPORT[1]])
        mass = torch.tensor([_EDGE_MASS], dtype=torch.float64)
        keep = torch.tensor([True])
        paid = _intensity(phi, occupy, sources, destinations, mass, keep)
        moved = _intensity(
            phi,
            occupy,
            sources,
            destinations,
            mass,
            keep,
            dest_shift=_DEST_SHIFT,
        )
        unpaid_paid = _leftover(covering, _DEMAND, paid)
        unpaid_moved = _leftover(covering, _DEMAND, moved)
        assert paid[1].item() != moved[1].item()
        assert unpaid_paid.item() < unpaid_moved.item()

    def test_refuse_roll_leaves_leftover_put(self, covering: Covering) -> None:
        """A consumed pair adds no addend; dest-rolling that zero table leaves U."""
        phi = PhiIntensity(4)
        occupy = torch.ones(4, dtype=torch.float64)
        sources = torch.tensor([_REFUSE_PAIR[0]])
        destinations = torch.tensor([_REFUSE_PAIR[1]])
        mass = torch.tensor([3.0], dtype=torch.float64)
        keep = torch.tensor([False])
        refused = _intensity(phi, occupy, sources, destinations, mass, keep)
        dest_rolled = _intensity(
            phi,
            occupy,
            sources,
            destinations,
            mass,
            keep,
            dest_shift=_DEST_SHIFT,
        )
        assert torch.allclose(refused, occupy)
        assert torch.allclose(dest_rolled, occupy)
        assert torch.allclose(
            _leftover(covering, _DEMAND, refused),
            _leftover(covering, _DEMAND, dest_rolled),
        )

    def test_off_support_roll_leaves_leftover_put(self, covering: Covering) -> None:
        """Dest-roll of a kept edge whose ends miss supp(L) leaves leftover unpaid."""
        phi = PhiIntensity(_OFF_SLOTS)
        occupy = torch.ones(_OFF_SLOTS, dtype=torch.float64)
        demand = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        sources = torch.tensor([_OFF_SUPPORT[0]])
        destinations = torch.tensor([_OFF_SUPPORT[1]])
        mass = torch.tensor([0.25], dtype=torch.float64)
        keep = torch.tensor([True])
        paid = _intensity(phi, occupy, sources, destinations, mass, keep)
        moved = _intensity(
            phi,
            occupy,
            sources,
            destinations,
            mass,
            keep,
            dest_shift=_DEST_SHIFT,
        )
        assert paid[4].item() != moved[4].item()
        assert paid[5].item() != moved[5].item()
        assert torch.allclose(paid[:2], moved[:2])
        assert torch.allclose(_leftover(covering, demand, paid), _leftover(covering, demand, moved))

    def test_dest_gap_is_leftover_unpaid_not_edge_unpaid(
        self,
        covering: Covering,
    ) -> None:
        """Dest gap is leftover unpaid of L against Phi, not pair-table edge unpaid."""
        phi = PhiIntensity(4)
        occupy = torch.zeros(4, dtype=torch.float64)
        sources = torch.tensor([_ON_SUPPORT[0]])
        destinations = torch.tensor([_ON_SUPPORT[1]])
        mass = torch.tensor([_EDGE_MASS], dtype=torch.float64)
        keep = torch.tensor([True])
        addends = phi.kept_pair_addends(sources, destinations, mass, keep)
        rolled = addends.roll(_DEST_SHIFT, dims=-1)
        paid = phi(occupy, addends, torch.ones(4, dtype=torch.bool))
        moved = phi(occupy, rolled, torch.ones(4, dtype=torch.bool))
        leftover_gap = _leftover(covering, _DEMAND, moved) - _leftover(covering, _DEMAND, paid)
        edge_paid = covering.edge_unpaid_fraction(addends, addends)
        edge_rolled = covering.edge_unpaid_fraction(addends, rolled)
        edge_gap = edge_rolled - edge_paid
        assert leftover_gap.item() > 0.0
        assert not torch.allclose(leftover_gap, edge_gap.to(dtype=leftover_gap.dtype))


class TestDestTrainOnKeptOverlayPhi:
    """Dest train leftover unpaid of claim-mask demand against Phi of a kept overlay."""

    @staticmethod
    def planted(demand: Tensor, supply: Tensor) -> CoveringInventory:
        slots = int(demand.size(-1))
        return CoveringInventory(
            n_entity_claim=demand,
            n_entity_full=supply,
            n_relation_claim=torch.zeros_like(demand),
            n_relation_full=torch.zeros_like(supply),
            mean_row_entropy=torch.zeros(()),
            batch_usage=torch.zeros(slots),
            relation_row_entropy=0.0,
        )

    @staticmethod
    def logged_dest(
        module: SsvLightningModule,
        payload: CoveringInventory,
    ) -> dict[str, Tensor]:
        batch = ssv_fixture_batch(module)
        logged: dict[str, Tensor] = {}

        def capture_log(name: str, value: object = None, **_kwargs: object) -> None:
            if torch.is_tensor(value):
                logged[name] = cast(Tensor, value)

        module.train()
        with (
            patch.object(module.inventory, 'forward', return_value=payload),
            patch.object(module, 'log', side_effect=capture_log),
            patch.object(module, 'log_dict'),
        ):
            _ = module.training_step(batch, 0)
        return logged

    @staticmethod
    def bank_row(values: Tensor, slots: int) -> Tensor:
        row = values.to(dtype=torch.float32).new_zeros(slots)
        row[: values.size(-1)] = values.to(dtype=torch.float32)
        return row.unsqueeze(0)

    def test_dest_unpaid_gap_matches_leftover_of_overlay_phi(
        self,
        tmp_path: Path,
        covering: Covering,
    ) -> None:
        """Logged dest gap is leftover unpaid of L against Phi versus slot-rolled Phi."""
        module = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
        slots = int(module.config.arch.entity_bank_size)
        phi = PhiIntensity(4)
        occupy = torch.zeros(4, dtype=torch.float64)
        supply = _intensity(
            phi,
            occupy,
            torch.tensor([_ON_SUPPORT[0]]),
            torch.tensor([_ON_SUPPORT[1]]),
            torch.tensor([_EDGE_MASS], dtype=torch.float64),
            torch.tensor([True]),
        )
        demand = self.bank_row(_DEMAND, slots)
        planted_supply = self.bank_row(supply, slots)
        logged = self.logged_dest(module, self.planted(demand, planted_supply))
        dest_shift = int(module.config.dest_comparison.dest_shift)
        match_u = covering.unpaid_fraction(covering.pair_table(demand, planted_supply))
        dest_u = covering.unpaid_fraction(
            covering.pair_table(demand, planted_supply.roll(dest_shift, dims=-1)),
        )
        expected = torch.nanmean(dest_u - match_u)
        assert 'dest_unpaid_gap' in logged
        assert torch.allclose(logged['dest_unpaid_gap'].to(dtype=expected.dtype), expected)
        assert not torch.allclose(match_u, dest_u)

    def test_dest_loss_moves_when_on_support_kept_overlay_rolls(
        self,
        tmp_path: Path,
        covering: Covering,
    ) -> None:
        """Dest loss moves when dest-axis roll of a kept edge changes Phi on supp(L)."""
        module = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
        slots = int(module.config.arch.entity_bank_size)
        phi = PhiIntensity(4)
        occupy = torch.zeros(4, dtype=torch.float64)
        sources = torch.tensor([_ON_SUPPORT[0]])
        destinations = torch.tensor([_ON_SUPPORT[1]])
        mass = torch.tensor([_EDGE_MASS], dtype=torch.float64)
        keep = torch.tensor([True])
        paid = _intensity(phi, occupy, sources, destinations, mass, keep)
        moved = _intensity(
            phi,
            occupy,
            sources,
            destinations,
            mass,
            keep,
            dest_shift=_DEST_SHIFT,
        )
        demand = self.bank_row(_DEMAND, slots)
        paid_loss = self.logged_dest(module, self.planted(demand, self.bank_row(paid, slots)))
        moved_loss = self.logged_dest(module, self.planted(demand, self.bank_row(moved, slots)))
        unpaid_paid = _leftover(covering, _DEMAND, paid)
        unpaid_moved = _leftover(covering, _DEMAND, moved)
        assert unpaid_paid.item() != unpaid_moved.item()
        assert not torch.allclose(paid_loss['dest_loss'], moved_loss['dest_loss'])
        assert not torch.allclose(paid_loss['dest_unpaid_gap'], moved_loss['dest_unpaid_gap'])

    def test_dest_loss_stays_when_refuse_overlay_rolls(
        self,
        tmp_path: Path,
        covering: Covering,
    ) -> None:
        """Dest loss stays when the rolled pair is consumed refuse."""
        module = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
        slots = int(module.config.arch.entity_bank_size)
        phi = PhiIntensity(4)
        occupy = torch.ones(4, dtype=torch.float64)
        sources = torch.tensor([_REFUSE_PAIR[0]])
        destinations = torch.tensor([_REFUSE_PAIR[1]])
        mass = torch.tensor([3.0], dtype=torch.float64)
        keep = torch.tensor([False])
        refused = _intensity(phi, occupy, sources, destinations, mass, keep)
        dest_rolled = _intensity(
            phi,
            occupy,
            sources,
            destinations,
            mass,
            keep,
            dest_shift=_DEST_SHIFT,
        )
        demand = self.bank_row(_DEMAND, slots)
        refused_log = self.logged_dest(
            module,
            self.planted(demand, self.bank_row(refused, slots)),
        )
        rolled_log = self.logged_dest(
            module,
            self.planted(demand, self.bank_row(dest_rolled, slots)),
        )
        assert torch.allclose(refused, dest_rolled)
        assert torch.allclose(
            _leftover(covering, _DEMAND, refused),
            _leftover(covering, _DEMAND, dest_rolled),
        )
        assert torch.allclose(refused_log['dest_loss'], rolled_log['dest_loss'])
        assert torch.allclose(refused_log['dest_unpaid_gap'], rolled_log['dest_unpaid_gap'])
