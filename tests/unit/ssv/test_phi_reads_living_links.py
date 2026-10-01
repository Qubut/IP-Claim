"""Production Phi reads typed living slot links into leftover unpaid.

An other-filing kept pair is typed by the relation bank, scattered into
the living pair table, and composed by PhiIntensity and inventory
overlay intensity. Leftover unpaid of numbered-claim demand against
that Phi moves when the living link is dest-rolled and stays put on an
unrelated roll. Consumed refuse adds nothing. Dest stays leftover unpaid
of pair_table. Host occupy ceiling is not leftover-unpaid-sufficient
width. Citation fields are not edges. These tests do not rewire occupy
select.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from tests._ssv_fixtures import ssv_tiny_vocab_config, type_aligned_pairs
from tests.unit.ssv.test_embeddings_share_leftover_unpaid import SecondGraphPlane
from tests.unit.ssv.test_leftover_unpaid_sufficient_occupy import (
    PlantedFiling,
    _leftover,
    _leftover_on_claims,
    _smallest_leftover_unpaid_sufficient,
)
from tests.unit.ssv.test_living_overlay_read import (
    LivingRead,
    _reader_own,
    _unrelated_other_filing,
)
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.soft_vocab import RelationAssignment, SoftVocabModule

_SLOTS = 4
_DEMANDED = 0
_LIVE_SRC = 2
_SILENT = 3
_UNRELATED = 1
_HOST_CEILING = 8
_DEST_SHIFT = 1
_WRITER = 'other-filing'
_RELATION_BANK = 2
_SOFT_DIM = 8


class LivingLoad(NamedTuple):
    """Reader occupy plus a relation-bank typed living pair table."""

    reader: PlantedFiling
    living: Tensor
    assignment: RelationAssignment
    written_by: str


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
            relation_temperature=0.07,
        )
    )


@pytest.fixture
def inventory() -> Inventory:
    return Inventory(occupied_floor=0.0)


def _typed_living(
    overlay: PhiIntensity,
    vocab: SoftVocabModule,
    source: int,
    destination: int,
) -> tuple[Tensor, RelationAssignment]:
    src = torch.zeros(vocab.entity_bank_size, dtype=vocab.entity_bank.dtype)
    dst = torch.zeros(vocab.entity_bank_size, dtype=vocab.entity_bank.dtype)
    src[source] = 1.0
    dst[destination] = 1.0
    with type_aligned_pairs(vocab):
        feats = vocab.pair_features(
            vocab.soft_entities(src).unsqueeze(0),
            vocab.soft_entities(dst).unsqueeze(0),
        )
    assignment = vocab.soft_assign_relations(feats)
    living = overlay.kept_pair_addends(
        torch.tensor((source,)),
        torch.tensor((destination,)),
        assignment.kept_pair_mass(),
        ~assignment.is_consumed(),
    )
    return living, assignment


def _refused_living(
    overlay: PhiIntensity,
    vocab: SoftVocabModule,
) -> tuple[Tensor, RelationAssignment]:
    refuse = F.normalize(vocab.refuse_probe.detach(), dim=-1).unsqueeze(0)
    assignment = vocab.soft_assign_relations(refuse)
    living = overlay.kept_pair_addends(
        torch.tensor((_LIVE_SRC,)),
        torch.tensor((_DEMANDED,)),
        assignment.kept_pair_mass(),
        ~assignment.is_consumed(),
    )
    return living, assignment


def _scene(overlay: PhiIntensity, vocab: SoftVocabModule) -> LivingLoad:
    living, assignment = _typed_living(overlay, vocab, _LIVE_SRC, _DEMANDED)
    return LivingLoad(
        reader=_reader_own(),
        living=living,
        assignment=assignment,
        written_by=_WRITER,
    )


def _own_labeled(occupy: Tensor, relations: int) -> Tensor:
    return occupy.new_zeros((*occupy.shape, occupy.size(-1), relations))


class TestPhiReadsLivingLinks:
    """Production Phi composes relation-bank typed living pairs. Dest stays leftover unpaid."""

    def test_inventory_overlay_reads_typed_living_link(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
        inventory: Inventory,
    ) -> None:
        """Overlay intensity with a typed other-filing pair lowers leftover unpaid."""
        scene = _scene(overlay, vocab)
        occupy = scene.reader.occupy.unsqueeze(0)
        labeled = _own_labeled(occupy, vocab.relation_bank_size)
        living = scene.living.unsqueeze(0)
        own = inventory.overlay_intensity(occupy, labeled)
        read = inventory.overlay_intensity(occupy, labeled, living)
        leftover_own = _leftover(covering, scene.reader.demands[0], own.squeeze(0))
        leftover_read = _leftover(covering, scene.reader.demands[0], read.squeeze(0))
        assert not scene.assignment.is_consumed().item()
        assert int(scene.assignment.typed.argmax(dim=-1).item()) == 0
        assert scene.assignment.kept_pair_mass().item() > 0
        assert scene.written_by == _WRITER
        assert not torch.allclose(leftover_own, leftover_read)
        assert leftover_read < leftover_own

    def test_phi_forward_composes_living_table(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
    ) -> None:
        """Phi.forward with living matches occupy plus the composed pair table."""
        scene = _scene(overlay, vocab)
        own = overlay(scene.reader.occupy, scene.reader.pair_mass, scene.reader.ingress)
        read = overlay(
            scene.reader.occupy,
            scene.reader.pair_mass,
            scene.reader.ingress,
            scene.living,
        )
        composed = overlay(
            scene.reader.occupy,
            scene.reader.pair_mass + scene.living,
            scene.reader.ingress,
        )
        leftover_own = _leftover(covering, scene.reader.demands[0], own)
        leftover_read = _leftover(covering, scene.reader.demands[0], read)
        assert torch.allclose(read, composed)
        assert not torch.allclose(own, read)
        assert leftover_read < leftover_own

    def test_dest_roll_of_living_link_moves_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
        inventory: Inventory,
    ) -> None:
        """Dest-axis roll of the living pair moves leftover unpaid on demanded support."""
        scene = _scene(overlay, vocab)
        occupy = scene.reader.occupy.unsqueeze(0)
        labeled = _own_labeled(occupy, vocab.relation_bank_size)
        paid = inventory.overlay_intensity(occupy, labeled, scene.living.unsqueeze(0))
        rolled = inventory.overlay_intensity(
            occupy,
            labeled,
            scene.living.roll(_DEST_SHIFT, dims=-1).unsqueeze(0),
        )
        leftover_paid = _leftover(covering, scene.reader.demands[0], paid.squeeze(0))
        leftover_rolled = _leftover(covering, scene.reader.demands[0], rolled.squeeze(0))
        assert not torch.allclose(leftover_paid, leftover_rolled)

    def test_unrelated_living_roll_leaves_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
        inventory: Inventory,
    ) -> None:
        """Dest-axis roll of a living pair that does not pay demand leaves U put."""
        scene = _scene(overlay, vocab)
        extra = _unrelated_other_filing(overlay)
        occupy = scene.reader.occupy.unsqueeze(0)
        labeled = _own_labeled(occupy, vocab.relation_bank_size)
        living = scene.living + extra.pair_mass
        paid = inventory.overlay_intensity(occupy, labeled, living.unsqueeze(0))
        rolled = inventory.overlay_intensity(
            occupy,
            labeled,
            (scene.living + extra.pair_mass.roll(_DEST_SHIFT, dims=-1)).unsqueeze(0),
        )
        leftover_paid = _leftover(covering, scene.reader.demands[0], paid.squeeze(0))
        leftover_rolled = _leftover(covering, scene.reader.demands[0], rolled.squeeze(0))
        assert extra.pair_mass[_SILENT, _UNRELATED].item() > 0
        assert torch.allclose(leftover_paid, leftover_rolled)

    def test_consumed_refuse_living_pair_adds_nothing(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
        inventory: Inventory,
    ) -> None:
        """A living pair the relation bank refuses does not move leftover unpaid."""
        reader = _reader_own()
        living, assignment = _refused_living(overlay, vocab)
        occupy = reader.occupy.unsqueeze(0)
        labeled = _own_labeled(occupy, vocab.relation_bank_size)
        own = inventory.overlay_intensity(occupy, labeled)
        refused = inventory.overlay_intensity(occupy, labeled, living.unsqueeze(0))
        leftover_own = _leftover(covering, reader.demands[0], own.squeeze(0))
        leftover_refused = _leftover(covering, reader.demands[0], refused.squeeze(0))
        assert assignment.is_consumed().item()
        assert torch.equal(living, living.new_zeros(living.shape))
        assert torch.allclose(leftover_own, leftover_refused)

    def test_destination_type_is_covering_leftover_unpaid_of_phi(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
        inventory: Inventory,
    ) -> None:
        """Dest is unpaid_fraction of pair_table, not a second graph plane."""
        scene = _scene(overlay, vocab)
        occupy = scene.reader.occupy.unsqueeze(0)
        labeled = _own_labeled(occupy, vocab.relation_bank_size)
        supply = inventory.overlay_intensity(occupy, labeled, scene.living.unsqueeze(0)).squeeze(0)
        leftover = covering.unpaid_fraction(
            covering.pair_table(scene.reader.demands[0], supply)
        ).squeeze()
        edge = covering.edge_unpaid_fraction(scene.living, scene.living)
        plane = SecondGraphPlane(leftover=leftover, adjacency=scene.living)
        assert leftover.ndim == 0
        assert torch.isfinite(leftover)
        assert leftover > 0
        assert torch.allclose(_leftover(covering, scene.reader.demands[0], supply), leftover)
        assert not isinstance(leftover, SecondGraphPlane)
        assert not torch.allclose(leftover, edge.to(dtype=leftover.dtype))
        assert torch.allclose(plane.leftover, leftover)

    def test_host_ceiling_is_not_leftover_unpaid_sufficient_width(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
        inventory: Inventory,
    ) -> None:
        """Leftover-unpaid-sufficient width is per filing. It is not a host integer."""
        scene = _scene(overlay, vocab)
        read = PlantedFiling(
            occupy=scene.reader.occupy,
            pair_mass=scene.reader.pair_mass + scene.living,
            ingress=scene.reader.ingress,
            demands=scene.reader.demands,
        )
        star = _smallest_leftover_unpaid_sufficient(covering, overlay, read)
        assert star == frozenset((_DEMANDED, _LIVE_SRC))
        assert 'occupied_max' not in dict(inventory.named_buffers())
        assert len(read.demands) != len(star)

    def test_cite_fields_are_absent_from_the_living_load(
        self,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
    ) -> None:
        """Living load, planted occupy, and the wrap fail carry no cite fields."""
        scene = _scene(overlay, vocab)
        assert scene.written_by == _WRITER
        assert 'cited' not in LivingLoad._fields
        assert 'category' not in LivingLoad._fields
        assert 'cited' not in LivingRead._fields
        assert 'category' not in LivingRead._fields
        assert 'cited' not in PlantedFiling._fields
        assert 'category' not in PlantedFiling._fields
        assert 'cited' not in SecondGraphPlane._fields
        assert 'category' not in SecondGraphPlane._fields
        assert scene.living.shape == (_SLOTS, _SLOTS)

    def test_empty_living_leaves_own_overlay_put(
        self,
        covering: Covering,
        overlay: PhiIntensity,
        vocab: SoftVocabModule,
        inventory: Inventory,
    ) -> None:
        """None or a zero living table leaves leftover unpaid of own pairs put."""
        scene = _scene(overlay, vocab)
        occupy = scene.reader.occupy.unsqueeze(0)
        labeled = _own_labeled(occupy, vocab.relation_bank_size)
        own = inventory.overlay_intensity(occupy, labeled)
        none = inventory.overlay_intensity(occupy, labeled, None)
        zeros = inventory.overlay_intensity(
            occupy,
            labeled,
            scene.living.new_zeros(1, _SLOTS, _SLOTS),
        )
        leftover_own = _leftover_on_claims(covering, scene.reader.demands, own.squeeze(0))
        leftover_none = _leftover_on_claims(covering, scene.reader.demands, none.squeeze(0))
        leftover_zeros = _leftover_on_claims(covering, scene.reader.demands, zeros.squeeze(0))
        assert torch.allclose(own, none)
        assert torch.allclose(own, zeros)
        assert torch.allclose(leftover_own, leftover_none)
        assert torch.allclose(leftover_own, leftover_zeros)
        assert tuple(inventory.children()) == ()
