"""Covering-head dest stays leftover unpaid of pair_table of demand against Phi.

Shape energies of the existing pair table may move under a Phi-preserving
rewiring while dest stays put. Dest is not edge unpaid of pair masses and
not a shape scalar. Zeroing the shape mix weight leaves dest identity
put. Copying a scored cosine adjacency into pair mass is a named fail.
Refuse stays zero. Citation fields are not edges. Dest identity stays
leftover unpaid after the overlay-shape addend is imported.
"""

from __future__ import annotations

import inspect
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from tests.unit.ssv.test_community_shape_loss import (
    _SLOTS as _COMMUNITY_SLOTS,
)
from tests.unit.ssv.test_community_shape_loss import (
    _clique_table,
    _energy,
    _leftover,
    _naive_cut_fill,
    _plant,
    _strength_matched_cut,
)
from tests.unit.ssv.test_overlay_subgraph_shape import (
    _SLOTS as _SUBGRAPH_SLOTS,
)
from tests.unit.ssv.test_overlay_subgraph_shape import (
    _cycle_pairs,
    _scene,
    _subgraph_energy,
    _two_block_pairs,
    _two_block_with_refused_cut,
)
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.shape import shape_family


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def overlay() -> PhiIntensity:
    return PhiIntensity(_COMMUNITY_SLOTS)


@pytest.fixture
def subgraph_overlay() -> PhiIntensity:
    return PhiIntensity(_SUBGRAPH_SLOTS)


class IsolationOverlay(NamedTuple):
    """Occupy, kept pair table, and numbered-claim demand for dest isolation."""

    occupy: Tensor
    pair_mass: Tensor
    ingress: Tensor
    demand: Tensor


class CosineUnwrap(NamedTuple):
    """Named fail: cosine adjacency scored, then copied into pair mass."""

    scores: Tensor
    copied: Tensor


def _dest(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


def _mix(dest: Tensor, shape: Tensor, dest_weight: float, shape_weight: float) -> Tensor:
    return dest * dest.new_tensor(dest_weight) + shape * shape.new_tensor(shape_weight)


def _cosine_scores(slots: int) -> Tensor:
    angles = torch.linspace(0, 2.4, slots, dtype=torch.float64)
    features = torch.stack((angles.cos(), angles.sin()), dim=-1)
    scores = F.normalize(features, dim=-1) @ F.normalize(features, dim=-1).T
    return scores.masked_fill(torch.eye(slots, dtype=torch.bool), 0)


class TestCoveringHeadIsolation:
    """Dest is leftover unpaid of pair_table. Shape is a separate energy of B."""

    def test_dest_is_leftover_unpaid_of_pair_table(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """The covering head scores unpaid_fraction of pair_table(L, Phi)."""
        scene = _plant(_clique_table())
        isolation = IsolationOverlay(
            occupy=scene.occupy,
            pair_mass=scene.pair_mass,
            ingress=scene.occupied,
            demand=scene.demand,
        )
        supply = overlay(isolation.occupy, isolation.pair_mass, isolation.ingress)
        dest = _dest(covering, isolation.demand, supply)
        table = covering.pair_table(isolation.demand, supply)
        assert dest.ndim == 0
        assert torch.isfinite(dest)
        assert dest > 0
        assert torch.allclose(_leftover(covering, isolation.demand, supply), dest)
        assert inspect.signature(covering.unpaid_fraction).parameters.keys() >= {'table'}
        assert inspect.signature(covering.pair_table).parameters.keys() >= {
            'n_query',
            'n_document',
        }
        assert table.unpaid_mass.shape != scene.pair_mass.shape

    def test_phi_preserving_isolation_moves_shape_and_holds_dest(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        subgraph_overlay: PhiIntensity,
    ) -> None:
        """Community and subgraph energies move; leftover unpaid of Phi stays put."""
        cliques = _plant(_clique_table())
        filled = _plant(_strength_matched_cut())
        phi_clique = overlay(cliques.occupy, cliques.pair_mass, cliques.occupied)
        phi_filled = overlay(filled.occupy, filled.pair_mass, filled.occupied)
        dest_clique = _dest(covering, cliques.demand, phi_clique)
        dest_filled = _dest(covering, filled.demand, phi_filled)
        community_clique = _energy(cliques)
        community_filled = _energy(filled)
        blocks = _scene(subgraph_overlay, _two_block_pairs(subgraph_overlay))
        cycle = _scene(subgraph_overlay, _cycle_pairs(subgraph_overlay))
        dest_blocks = _dest(
            covering,
            blocks.demand,
            subgraph_overlay(blocks.occupy, blocks.pair_mass, blocks.ingress),
        )
        dest_cycle = _dest(
            covering,
            cycle.demand,
            subgraph_overlay(cycle.occupy, cycle.pair_mass, cycle.ingress),
        )
        subgraph_blocks = _subgraph_energy(blocks.pair_mass, blocks.occupy, blocks.ingress)
        subgraph_cycle = _subgraph_energy(cycle.pair_mass, cycle.occupy, cycle.ingress)
        assert torch.allclose(phi_clique, phi_filled)
        assert torch.allclose(dest_clique, dest_filled)
        assert community_clique.community.item() != community_filled.community.item()
        assert torch.allclose(dest_blocks, dest_cycle)
        assert not torch.allclose(subgraph_blocks, subgraph_cycle)
        assert not torch.allclose(dest_clique, community_clique.total.to(dtype=dest_clique.dtype))
        assert not torch.allclose(dest_blocks, subgraph_blocks.to(dtype=dest_blocks.dtype))

    def test_dest_is_not_edge_unpaid_of_pair_masses(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Dest is leftover unpaid of pair_table, not edge unpaid of pair masses."""
        scene = _plant(_clique_table())
        supply = overlay(scene.occupy, scene.pair_mass, scene.occupied)
        dest = _dest(covering, scene.demand, supply)
        edge = covering.edge_unpaid_fraction(scene.pair_mass, scene.pair_mass)
        naive = _plant(_naive_cut_fill())
        naive_supply = overlay(naive.occupy, naive.pair_mass, naive.occupied)
        dest_naive = _dest(covering, naive.demand, naive_supply)
        assert dest.ndim == 0
        assert edge.ndim == 0
        assert torch.isfinite(dest)
        assert torch.isfinite(edge)
        assert not torch.allclose(dest, edge.to(dtype=dest.dtype))
        assert dest_naive.item() < dest.item()
        assert not torch.allclose(dest, dest_naive)

    def test_zero_shape_weight_does_not_rewrite_dest(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Ablating the shape mix weight leaves dest leftover unpaid put."""
        cliques = _plant(_clique_table())
        filled = _plant(_strength_matched_cut())
        dest_clique = _dest(
            covering,
            cliques.demand,
            overlay(cliques.occupy, cliques.pair_mass, cliques.occupied),
        )
        dest_filled = _dest(
            covering,
            filled.demand,
            overlay(filled.occupy, filled.pair_mass, filled.occupied),
        )
        shape_clique = _energy(cliques).total
        shape_filled = _energy(filled).total
        dest_weight = 1.0
        shape_on = 1.0
        shape_off = 0.0
        mix_clique_on = _mix(dest_clique, shape_clique, dest_weight, shape_on)
        mix_filled_on = _mix(dest_filled, shape_filled, dest_weight, shape_on)
        mix_clique_off = _mix(dest_clique, shape_clique, dest_weight, shape_off)
        mix_filled_off = _mix(dest_filled, shape_filled, dest_weight, shape_off)
        mix_shape_only = _mix(dest_clique, shape_clique, 0.0, shape_on)
        assert torch.allclose(dest_clique, dest_filled)
        assert not torch.allclose(shape_clique, shape_filled)
        assert torch.allclose(mix_clique_off, dest_clique)
        assert torch.allclose(mix_filled_off, dest_filled)
        assert torch.allclose(mix_clique_off, mix_filled_off)
        assert not torch.allclose(mix_clique_on, mix_filled_on)
        assert torch.allclose(mix_shape_only, shape_clique)
        assert not torch.allclose(mix_shape_only, dest_clique)

    def test_cosine_adjacency_copied_into_pair_mass_is_named_fail(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Scoring cosine then copying it into pair mass is not dest and not B."""
        typed = _plant(_clique_table())
        scores = _cosine_scores(typed.pair_mass.size(-1))
        unwrap = CosineUnwrap(scores=scores, copied=scores)
        typed_supply = overlay(typed.occupy, typed.pair_mass, typed.occupied)
        cosine_supply = overlay(typed.occupy, unwrap.copied, typed.occupied)
        dest = _dest(covering, typed.demand, typed_supply)
        cosine_unpaid = _dest(covering, typed.demand, cosine_supply)
        cosine_edge = covering.edge_unpaid_fraction(unwrap.copied, unwrap.copied)
        cosine_shape = shape_family(
            unwrap.copied,
            typed.assignment,
            typed.occupied,
        ).total
        assert not torch.equal(unwrap.copied, typed.pair_mass)
        assert not torch.allclose(typed_supply, cosine_supply)
        assert torch.isfinite(dest)
        assert not torch.allclose(dest, cosine_unpaid)
        assert not torch.allclose(dest, cosine_edge.to(dtype=dest.dtype))
        assert not torch.allclose(dest, cosine_shape.to(dtype=dest.dtype))
        assert not isinstance(dest, CosineUnwrap)
        assert type(dest) is not type(unwrap)

    def test_refused_pair_stays_zero_and_does_not_rewrite_dest(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        subgraph_overlay: PhiIntensity,
    ) -> None:
        """A consumed refuse cell stays zero and leaves dest leftover unpaid put."""
        planted = _clique_table()
        refuse = overlay.kept_pair_addends(
            torch.tensor((0,)),
            torch.tensor((3,)),
            torch.tensor((1.0,), dtype=torch.float64),
            torch.tensor((False,)),
        )
        typed = _plant(planted)
        refused = _plant(planted + refuse)
        dest_typed = _dest(
            covering,
            typed.demand,
            overlay(typed.occupy, typed.pair_mass, typed.occupied),
        )
        dest_refused = _dest(
            covering,
            refused.demand,
            overlay(refused.occupy, refused.pair_mass, refused.occupied),
        )
        blocks = _scene(subgraph_overlay, _two_block_pairs(subgraph_overlay))
        refused_cut = _scene(
            subgraph_overlay,
            _two_block_with_refused_cut(subgraph_overlay),
        )
        dest_blocks = _dest(
            covering,
            blocks.demand,
            subgraph_overlay(blocks.occupy, blocks.pair_mass, blocks.ingress),
        )
        dest_cut = _dest(
            covering,
            refused_cut.demand,
            subgraph_overlay(
                refused_cut.occupy,
                refused_cut.pair_mass,
                refused_cut.ingress,
            ),
        )
        assert torch.equal(refuse, refuse.new_zeros(refuse.shape))
        assert torch.equal(refused.pair_mass, typed.pair_mass)
        assert torch.allclose(dest_typed, dest_refused)
        assert torch.equal(refused_cut.pair_mass[0, 2], refused_cut.pair_mass.new_zeros(()))
        assert torch.equal(refused_cut.pair_mass, blocks.pair_mass)
        assert torch.allclose(dest_blocks, dest_cut)

    def test_cite_fields_unused_and_dest_stays_leftover_unpaid(self) -> None:
        """Citation identifiers do not index dest. Dest stays leftover unpaid of pair_table."""
        step = inspect.getsource(SsvLightningModule.training_step)
        assert 'cited' not in IsolationOverlay._fields
        assert 'category' not in IsolationOverlay._fields
        assert 'hupd' not in IsolationOverlay._fields
        assert 'cited' not in CosineUnwrap._fields
        assert 'category' not in CosineUnwrap._fields
        assert 'pair_table' in step
        assert 'unpaid_fraction' in step
        assert 'shape_family' in step
        assert 'edge_unpaid_fraction' not in step
        assert inspect.signature(_dest).parameters.keys() == {
            'covering',
            'demand',
            'supply',
        }
        assert inspect.signature(_mix).parameters.keys() == {
            'dest',
            'shape',
            'dest_weight',
            'shape_weight',
        }
