"""Community energy of kept pair mass moves when a Phi-preserving rewiring fills the cut.

Two typed cliques on disjoint occupy subsets share in- and out-strengths
with a cut-filled table that transfers intra-clique mass onto the cut.
Leftover unpaid of numbered-claim demand against Phi stays put because
Phi reads those strengths. Naive extra cut mass is not that isolation:
it raises Phi and can move leftover unpaid. One-column assignment moves
the collapse term. Dest leftover unpaid stays on Covering. Citation
fields are not edges. These tests do not wire the train mix.
"""

from __future__ import annotations

from math import sqrt
from typing import NamedTuple

import pytest
import torch
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.shape import shape_family

_SLOTS = 6
_COMMUNITY = torch.tensor([0, 0, 0, 1, 1, 1])
_INTRA_MASS = 1.0
_OCCUPY_MASS = 1.0
_CLAIM_MASS = 2.0
_COLLAPSE_WEIGHT = 1.0
_COMMUNITY_COLUMNS = 2
_NAIVE_CUT = 0.5


class PlantedOverlay(NamedTuple):
    """Occupy, kept pair table, planted assignment, and numbered-claim demand."""

    occupy: Tensor
    pair_mass: Tensor
    assignment: Tensor
    occupied: Tensor
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


def _occupied() -> Tensor:
    return torch.ones(_SLOTS, dtype=torch.bool)


def _demand() -> Tensor:
    return torch.full((_SLOTS,), _CLAIM_MASS, dtype=torch.float64)


def _intra_mask() -> Tensor:
    same = _COMMUNITY.unsqueeze(-1) == _COMMUNITY.unsqueeze(-2)
    return same & ~torch.eye(_SLOTS, dtype=torch.bool)


def _cut_mask() -> Tensor:
    return _COMMUNITY.unsqueeze(-1) != _COMMUNITY.unsqueeze(-2)


def _clique_table() -> Tensor:
    return _intra_mask().to(dtype=torch.float64) * _INTRA_MASS


def _strength_matched_cut() -> Tensor:
    neighbors = _INTRA_MASS * (_SLOTS // _COMMUNITY_COLUMNS - 1)
    other = float(_SLOTS // _COMMUNITY_COLUMNS)
    return _cut_mask().to(dtype=torch.float64) * (neighbors / other)


def _naive_cut_fill() -> Tensor:
    return _clique_table() + _cut_mask().to(dtype=torch.float64) * _NAIVE_CUT


def _planted_assignment() -> Tensor:
    labels = torch.nn.functional.one_hot(_COMMUNITY, _COMMUNITY_COLUMNS)
    return labels.to(dtype=torch.float64)


def _collapsed_assignment() -> Tensor:
    assignment = torch.zeros(_SLOTS, _COMMUNITY_COLUMNS, dtype=torch.float64)
    return assignment.index_fill(-1, assignment.new_zeros((), dtype=torch.long), 1)


def _plant(pair_mass: Tensor) -> PlantedOverlay:
    return PlantedOverlay(
        occupy=_occupy(),
        pair_mass=pair_mass,
        assignment=_planted_assignment(),
        occupied=_occupied(),
        demand=_demand(),
    )


def _energy(scene: PlantedOverlay, assignment: Tensor | None = None):
    return shape_family(
        scene.pair_mass,
        scene.assignment if assignment is None else assignment,
        scene.occupied,
        collapse_weight=_COLLAPSE_WEIGHT,
    )


def _strengths(pair_mass: Tensor) -> tuple[Tensor, Tensor]:
    return pair_mass.sum(dim=-1), pair_mass.sum(dim=-2)


def _from_kept_pairs(overlay: PhiIntensity, pair_mass: Tensor) -> Tensor:
    sources, destinations = torch.nonzero(pair_mass, as_tuple=True)
    mass = pair_mass[sources, destinations]
    return overlay.kept_pair_addends(
        sources,
        destinations,
        mass,
        torch.ones(sources.shape, dtype=torch.bool),
    )


class TestCommunityShapeOnPairTable:
    """Newman/DMoN community addend of Phi pair mass, isolated from leftover unpaid."""

    def test_phi_preserving_cut_fill_moves_community_and_holds_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Strength-matched cut fill moves community energy; leftover unpaid stays put."""
        cliques = _plant(_clique_table())
        filled = _plant(_strength_matched_cut())
        out_clique, in_clique = _strengths(cliques.pair_mass)
        out_filled, in_filled = _strengths(filled.pair_mass)
        phi_clique = overlay(cliques.occupy, cliques.pair_mass, cliques.occupied)
        phi_filled = overlay(filled.occupy, filled.pair_mass, filled.occupied)
        unpaid_clique = _leftover(covering, cliques.demand, phi_clique)
        unpaid_filled = _leftover(covering, filled.demand, phi_filled)
        energy_clique = _energy(cliques)
        energy_filled = _energy(filled)
        assert torch.allclose(out_clique, out_filled)
        assert torch.allclose(in_clique, in_filled)
        assert torch.allclose(phi_clique, phi_filled)
        assert torch.allclose(unpaid_clique, unpaid_filled)
        assert energy_clique.community.item() != energy_filled.community.item()
        assert energy_clique.modularity.item() > energy_filled.modularity.item()
        assert torch.allclose(energy_clique.collapse, energy_filled.collapse)
        assert torch.isfinite(energy_clique.subgraph)
        assert torch.allclose(
            energy_clique.total,
            energy_clique.community + energy_clique.subgraph,
        )

    def test_naive_cut_mass_moves_phi_and_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Adding cut mass without transferring intra mass is not isolation."""
        cliques = _plant(_clique_table())
        naive = _plant(_naive_cut_fill())
        out_clique, in_clique = _strengths(cliques.pair_mass)
        out_naive, in_naive = _strengths(naive.pair_mass)
        phi_clique = overlay(cliques.occupy, cliques.pair_mass, cliques.occupied)
        phi_naive = overlay(naive.occupy, naive.pair_mass, naive.occupied)
        unpaid_clique = _leftover(covering, cliques.demand, phi_clique)
        unpaid_naive = _leftover(covering, naive.demand, phi_naive)
        assert not torch.allclose(out_clique, out_naive)
        assert not torch.allclose(in_clique, in_naive)
        assert not torch.allclose(phi_clique, phi_naive)
        assert unpaid_naive.item() < unpaid_clique.item()

    def test_one_column_assignment_moves_collapse(self) -> None:
        """Collapse rises when every occupied row sits in one community column."""
        scene = _plant(_clique_table())
        planted = _energy(scene)
        collapsed = _energy(scene, _collapsed_assignment())
        expected = scene.pair_mass.new_tensor(sqrt(_COMMUNITY_COLUMNS) - 1)
        assert torch.allclose(planted.collapse, planted.collapse.new_zeros(()))
        assert torch.allclose(collapsed.collapse, expected)
        assert collapsed.collapse.item() != planted.collapse.item()
        assert collapsed.community.item() != planted.community.item()
        assert _SLOTS != _COMMUNITY_COLUMNS

    def test_pair_table_is_phi_kept_addends_not_covering_table(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Community energy reads Phi pair mass, not leftover unpaid of demand."""
        planted = _clique_table()
        rebuilt = _from_kept_pairs(overlay, planted)
        scene = _plant(rebuilt)
        supply = overlay(scene.occupy, scene.pair_mass, scene.occupied)
        leftover_table = covering.pair_table(scene.demand, supply)
        energy = _energy(scene)
        refuse = overlay.kept_pair_addends(
            torch.tensor((0,)),
            torch.tensor((3,)),
            torch.tensor((_INTRA_MASS,), dtype=torch.float64),
            torch.tensor((False,)),
        )
        refused = _plant(rebuilt + refuse)
        assert torch.equal(rebuilt, planted)
        assert leftover_table.unpaid_mass.shape != scene.pair_mass.shape
        assert energy.modularity.item() == pytest.approx(0.5)
        assert torch.equal(refused.pair_mass, scene.pair_mass)
        assert torch.allclose(_energy(refused).community, energy.community)
        assert 'cited_hupd' not in PlantedOverlay.__annotations__
        assert 'categories' not in PlantedOverlay.__annotations__
        assert not hasattr(scene, 'cited_hupd')
