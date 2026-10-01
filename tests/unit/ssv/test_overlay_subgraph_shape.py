"""Induced neighborhoods of the kept pair table move a local-global energy.

The encoder reads the same square pair mass PhiIntensity already consumes.
A Phi-preserving rewiring of those neighborhoods can leave leftover unpaid
put. A refused pair stays zero in that table and does not pay a patch.
Citation identifiers do not index the table. Destination leftover unpaid
stays Covering unpaid_fraction of pair_table of numbered-claim demand
against Phi. These tests do not wire a train mix.
"""

from __future__ import annotations

import inspect
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.shape import shape_family

_SLOTS = 4
_EDGE_MASS = 1.0
_OCCUPY_MASS = 1.0
_CLAIM_MASS = 2.0
_REFUSE_MASS = 9.0


class PlantedOverlay(NamedTuple):
    """Occupy support, kept pair table, and numbered-claim demand."""

    occupy: Tensor
    pair_mass: Tensor
    ingress: Tensor
    demand: Tensor


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def overlay() -> PhiIntensity:
    return PhiIntensity(_SLOTS)


def _leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


def _occupy() -> Tensor:
    return torch.full((_SLOTS,), _OCCUPY_MASS, dtype=torch.float64)


def _live() -> Tensor:
    return torch.ones(_SLOTS, dtype=torch.bool)


def _demand() -> Tensor:
    return torch.full((_SLOTS,), _CLAIM_MASS, dtype=torch.float64)


def _kept_pairs(
    overlay: PhiIntensity,
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
    mass: Tensor,
    keep: Tensor,
) -> Tensor:
    return overlay.kept_pair_addends(
        torch.tensor(sources),
        torch.tensor(destinations),
        mass,
        keep,
    )


def _two_block_pairs(overlay: PhiIntensity) -> Tensor:
    mass = torch.full((4,), _EDGE_MASS, dtype=torch.float64)
    keep = torch.ones(4, dtype=torch.bool)
    return _kept_pairs(overlay, (0, 1, 2, 3), (1, 0, 3, 2), mass, keep)


def _cycle_pairs(overlay: PhiIntensity) -> Tensor:
    mass = torch.full((4,), _EDGE_MASS, dtype=torch.float64)
    keep = torch.ones(4, dtype=torch.bool)
    return _kept_pairs(overlay, (0, 1, 2, 3), (1, 2, 3, 0), mass, keep)


def _two_block_with_refused_cut(overlay: PhiIntensity) -> Tensor:
    mass = torch.tensor(
        (_EDGE_MASS, _EDGE_MASS, _EDGE_MASS, _EDGE_MASS, _REFUSE_MASS),
        dtype=torch.float64,
    )
    keep = torch.tensor((True, True, True, True, False))
    return _kept_pairs(overlay, (0, 1, 2, 3, 0), (1, 0, 3, 2, 2), mass, keep)


def _two_block_with_paid_cut(overlay: PhiIntensity) -> Tensor:
    mass = torch.tensor(
        (_EDGE_MASS, _EDGE_MASS, _EDGE_MASS, _EDGE_MASS, _REFUSE_MASS),
        dtype=torch.float64,
    )
    keep = torch.ones(5, dtype=torch.bool)
    return _kept_pairs(overlay, (0, 1, 2, 3, 0), (1, 0, 3, 2, 2), mass, keep)


def _occupied_clique_window(occupy: Tensor) -> Tensor:
    live = occupy > 0
    clique = live.unsqueeze(-1) & live.unsqueeze(-2)
    identity = torch.eye(occupy.size(-1), dtype=torch.bool, device=occupy.device)
    return clique.masked_fill(identity, False).to(dtype=occupy.dtype)


def _subgraph_energy(pair_mass: Tensor, occupy: Tensor, occupied: Tensor) -> Tensor:
    """Local-global MI of induced neighborhoods of the kept pair table."""
    columns = 2
    slots = pair_mass.size(-1)
    labels = (torch.arange(slots, device=pair_mass.device) * columns) // slots
    assignment = F.one_hot(labels, columns).to(dtype=pair_mass.dtype)
    support = occupy * occupied.to(dtype=occupy.dtype)
    return shape_family(pair_mass, assignment, support).subgraph


def _scene(overlay: PhiIntensity, pair_mass: Tensor) -> PlantedOverlay:
    occupy = _occupy()
    return PlantedOverlay(
        occupy=occupy,
        pair_mass=pair_mass,
        ingress=_live(),
        demand=_demand(),
    )


class TestInducedNeighborhoodShape:
    """Equal-Phi neighborhood rewiring moves subgraph energy. Dest stays leftover unpaid."""

    def test_energy_reads_the_pair_table_phi_consumes(self, overlay: PhiIntensity) -> None:
        """The fixture table is kept_pair_addends, the same square PhiIntensity.forward reads."""
        pair_mass = _two_block_pairs(overlay)
        occupy = _occupy()
        live = _live()
        intensity = overlay(occupy, pair_mass, live)
        row = pair_mass.sum(dim=-1)
        col = pair_mass.sum(dim=-2)
        assert pair_mass.shape == (_SLOTS, _SLOTS)
        assert torch.equal(pair_mass.diag(), pair_mass.new_zeros((_SLOTS,)))
        assert torch.allclose(intensity, occupy + row + col)
        assert inspect.signature(_subgraph_energy).parameters.keys() == {
            'pair_mass',
            'occupy',
            'occupied',
        }

    def test_equal_phi_neighborhood_swap_moves_subgraph_energy(
        self,
        overlay: PhiIntensity,
    ) -> None:
        """Two induced neighborhoods of equal Phi mass, different wiring, move L-sub."""
        blocks = _scene(overlay, _two_block_pairs(overlay))
        cycle = _scene(overlay, _cycle_pairs(overlay))
        phi_blocks = overlay(blocks.occupy, blocks.pair_mass, blocks.ingress)
        phi_cycle = overlay(cycle.occupy, cycle.pair_mass, cycle.ingress)
        energy_blocks = _subgraph_energy(blocks.pair_mass, blocks.occupy, blocks.ingress)
        energy_cycle = _subgraph_energy(cycle.pair_mass, cycle.occupy, cycle.ingress)
        assert torch.allclose(blocks.occupy, cycle.occupy)
        assert torch.allclose(blocks.pair_mass.sum(dim=-1), cycle.pair_mass.sum(dim=-1))
        assert torch.allclose(blocks.pair_mass.sum(dim=-2), cycle.pair_mass.sum(dim=-2))
        assert torch.allclose(phi_blocks, phi_cycle)
        assert not torch.equal(blocks.pair_mass, cycle.pair_mass)
        assert not torch.allclose(energy_blocks, energy_cycle)

    def test_neighborhood_isolation_leaves_leftover_unpaid_put(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Phi-preserving neighborhood rewiring leaves leftover unpaid put while shape moves."""
        blocks = _scene(overlay, _two_block_pairs(overlay))
        cycle = _scene(overlay, _cycle_pairs(overlay))
        unpaid_blocks = _leftover(
            covering,
            blocks.demand,
            overlay(blocks.occupy, blocks.pair_mass, blocks.ingress),
        )
        unpaid_cycle = _leftover(
            covering,
            cycle.demand,
            overlay(cycle.occupy, cycle.pair_mass, cycle.ingress),
        )
        energy_blocks = _subgraph_energy(blocks.pair_mass, blocks.occupy, blocks.ingress)
        energy_cycle = _subgraph_energy(cycle.pair_mass, cycle.occupy, cycle.ingress)
        assert torch.allclose(unpaid_blocks, unpaid_cycle)
        assert torch.isfinite(unpaid_blocks)
        assert not torch.allclose(energy_blocks, energy_cycle)

    def test_refused_pair_does_not_pay_a_patch(self, overlay: PhiIntensity) -> None:
        """A consumed refuse cell stays zero and does not change the neighborhood energy."""
        blocks = _scene(overlay, _two_block_pairs(overlay))
        refused = _scene(overlay, _two_block_with_refused_cut(overlay))
        paid = _scene(overlay, _two_block_with_paid_cut(overlay))
        energy_blocks = _subgraph_energy(blocks.pair_mass, blocks.occupy, blocks.ingress)
        energy_refused = _subgraph_energy(refused.pair_mass, refused.occupy, refused.ingress)
        energy_paid = _subgraph_energy(paid.pair_mass, paid.occupy, paid.ingress)
        assert torch.equal(refused.pair_mass[0, 2], refused.pair_mass.new_zeros(()))
        assert torch.allclose(paid.pair_mass[0, 2], paid.pair_mass.new_tensor(_REFUSE_MASS))
        assert torch.equal(refused.pair_mass, blocks.pair_mass)
        assert torch.allclose(energy_refused, energy_blocks)
        assert not torch.allclose(energy_paid, energy_blocks)

    def test_one_hop_window_that_ignores_refuse_is_a_named_fail(
        self,
        overlay: PhiIntensity,
    ) -> None:
        """A complete occupied window that ignores typed refuse is not the pair table."""
        refused = _scene(overlay, _two_block_with_refused_cut(overlay))
        clique = _occupied_clique_window(refused.occupy)
        energy_refused = _subgraph_energy(refused.pair_mass, refused.occupy, refused.ingress)
        energy_clique = _subgraph_energy(clique, refused.occupy, refused.ingress)
        assert not torch.equal(clique, refused.pair_mass)
        assert not torch.equal(clique[0, 2], clique.new_zeros(()))
        assert torch.equal(refused.pair_mass[0, 2], refused.pair_mass.new_zeros(()))
        assert not torch.allclose(energy_clique, energy_refused)

    def test_destination_type_is_covering_leftover_unpaid_of_phi(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """The covering head reads unpaid_fraction of pair_table(L, Phi), not the shape scalar."""
        blocks = _scene(overlay, _two_block_pairs(overlay))
        supply = overlay(blocks.occupy, blocks.pair_mass, blocks.ingress)
        leftover = _leftover(covering, blocks.demand, supply)
        energy = _subgraph_energy(blocks.pair_mass, blocks.occupy, blocks.ingress)
        assert leftover.ndim == 0
        assert energy.ndim == 0
        assert torch.isfinite(leftover)
        assert torch.isfinite(energy)
        assert not torch.allclose(leftover, energy)
        assert inspect.signature(covering.unpaid_fraction).parameters.keys() >= {'table'}

    def test_cite_fields_unused(self, overlay: PhiIntensity) -> None:
        """Citation identifiers do not index the pair table or the subgraph energy."""
        blocks = _scene(overlay, _two_block_pairs(overlay))
        energy = _subgraph_energy(blocks.pair_mass, blocks.occupy, blocks.ingress)
        assert torch.isfinite(energy)
        assert 'cited' not in PlantedOverlay._fields
        assert 'category' not in PlantedOverlay._fields
        assert 'hupd' not in PlantedOverlay._fields
        assert 'cited' not in inspect.signature(_subgraph_energy).parameters
        assert 'category' not in inspect.signature(_subgraph_energy).parameters
