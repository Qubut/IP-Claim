"""Embeddings and covering share leftover unpaid of Phi.

Slot codes and relation codes are the entity and relation banks. Dest
after a living-graph read is leftover unpaid of numbered-claim demand
against Phi. Wrapping that residual in a second graph type, then
unwrapping it, is a named fail. Edge unpaid of pair masses is not dest.
Citation fields are not edges. These tests do not rewire occupy select
or production Phi living-link load.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
from tests._ssv_fixtures import ssv_tiny_vocab_config
from tests.unit.ssv.test_leftover_unpaid_sufficient_occupy import (
    PlantedFiling,
    _leftover,
)
from tests.unit.ssv.test_living_overlay_read import (
    LivingRead,
    _read_living,
    _scene,
)
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.soft_vocab import SoftVocabModule

_SLOTS = 4
_SOFT_DIM = 8
_RELATION_BANK = 2


class SecondGraphPlane(NamedTuple):
    """Named fail: leftover unpaid lifted into a second graph container."""

    leftover: Tensor
    adjacency: Tensor


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def overlay() -> PhiIntensity:
    return PhiIntensity(_SLOTS)


@pytest.fixture
def vocab() -> SoftVocabModule:
    return SoftVocabModule(
        ssv_tiny_vocab_config(
            entity_bank_size=_SLOTS,
            relation_bank_size=_RELATION_BANK,
            soft_dim=_SOFT_DIM,
        )
    )


def _living_read(overlay: PhiIntensity) -> PlantedFiling:
    scene = _scene(overlay)
    return _read_living(scene.reader, scene.living)


class TestEmbeddingsShareLeftoverUnpaid:
    """Bank codes are embeddings. Dest stays leftover unpaid of pair_table."""

    def test_dest_stays_leftover_unpaid_of_pair_table_after_living_read(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """The covering head still scores unpaid_fraction of pair_table(L, Phi)."""
        scene = _scene(overlay)
        read = _read_living(scene.reader, scene.living)
        supply = overlay(read.occupy, read.pair_mass, read.ingress)
        leftover = covering.unpaid_fraction(covering.pair_table(read.demands[0], supply)).squeeze()
        assert leftover.ndim == 0
        assert torch.isfinite(leftover)
        assert leftover > 0
        assert torch.allclose(_leftover(covering, read.demands[0], supply), leftover)
        assert scene.written_by != 'this-filing'

    def test_slot_and_relation_codes_are_the_embeddings(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
    ) -> None:
        """Entity-bank and relation-bank rows are the node and edge embeddings."""
        read = _living_read(overlay)
        supply = overlay(read.occupy, read.pair_mass, read.ingress)
        leftover = _leftover(covering, read.demands[0], supply)
        slot = torch.zeros(vocab.entity_bank_size, dtype=vocab.entity_bank.dtype)
        slot[0] = 1.0
        relation = torch.zeros(vocab.relation_bank_size, dtype=vocab.relation_bank.dtype)
        relation[0] = 1.0
        assert vocab.entity_bank.shape == (_SLOTS, _SOFT_DIM)
        assert vocab.relation_bank.shape == (_RELATION_BANK, _SOFT_DIM)
        assert torch.allclose(vocab.soft_entities(slot), vocab.entity_bank[0])
        assert torch.allclose(vocab.soft_relations(relation), vocab.relation_bank[0])
        assert leftover.shape != vocab.entity_bank.shape
        assert leftover.shape != vocab.relation_bank.shape
        assert read.pair_mass.shape != vocab.entity_bank.shape
        assert read.pair_mass.shape != vocab.relation_bank.shape

    def test_wrapping_leftover_unpaid_in_a_second_graph_type_is_named_fail(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Lift leftover unpaid into a graph container, unwrap, and dest is unchanged."""
        read = _living_read(overlay)
        supply = overlay(read.occupy, read.pair_mass, read.ingress)
        dest = _leftover(covering, read.demands[0], supply)
        plane = SecondGraphPlane(leftover=dest, adjacency=read.pair_mass)
        edge = covering.edge_unpaid_fraction(plane.adjacency, plane.adjacency)
        assert torch.allclose(plane.leftover, dest)
        assert dest.ndim == 0
        assert not isinstance(dest, SecondGraphPlane)
        assert type(dest) is not type(plane)
        assert not torch.allclose(dest, edge.to(dtype=dest.dtype))

    def test_edge_unpaid_of_pair_masses_is_not_dest(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Dest is leftover unpaid of pair_table, not edge unpaid of pair masses."""
        read = _living_read(overlay)
        supply = overlay(read.occupy, read.pair_mass, read.ingress)
        dest = _leftover(covering, read.demands[0], supply)
        edge = covering.edge_unpaid_fraction(read.pair_mass, read.pair_mass)
        assert dest.ndim == 0
        assert edge.ndim == 0
        assert torch.isfinite(dest)
        assert torch.isfinite(edge)
        assert not torch.allclose(dest, edge.to(dtype=dest.dtype))

    def test_cite_fields_are_absent_from_the_covering_object(
        self,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
    ) -> None:
        """Living read, planted occupy, and the wrap fail carry no cite fields."""
        scene = _scene(overlay)
        assert vocab.entity_bank.ndim == 2
        assert vocab.relation_bank.ndim == 2
        assert 'cited' not in LivingRead._fields
        assert 'category' not in LivingRead._fields
        assert 'cited' not in PlantedFiling._fields
        assert 'category' not in PlantedFiling._fields
        assert 'cited' not in SecondGraphPlane._fields
        assert 'category' not in SecondGraphPlane._fields
        assert scene.written_by == 'other-filing'
