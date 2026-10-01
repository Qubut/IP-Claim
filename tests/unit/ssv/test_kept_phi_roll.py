"""Dest-axis roll of a kept overlay moves Phi and leftover unpaid.

Refuse pairs are consumed non-edges: they do not enter the addend table, so
rolling or consuming them does not change intensity. Leftover unpaid stays
on Covering. Covering inventory n is occupy plus those kept addends. These
tests do not train a trunk and do not call encode.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from tests._ssv_fixtures import ssv_tiny_vocab_config, type_aligned_pairs
from tests.unit.ssv.test_consumed_refuse import (
    _KEPT_EDGE,
    _PAIR_DST,
    _PAIR_SRC,
    _REFUSE_EDGE,
    _SLOTS,
    _planted_bundle,
    _planted_vocab,
    _scored_tournament,
    _three_slot_assignment,
)
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.explain import Explain
from ip_claim.ssv import module as dest_mod
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.soft_vocab import SoftVocabModule

_PHI = PhiIntensity(4)
_PHI_REFUSE = PhiIntensity(_SLOTS)
_DEMAND = torch.tensor([2.0, 1.0, 0.0, 0.0], dtype=torch.float64)
_OCCUPY = torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float64)
_OCCUPIED = torch.tensor([True, True, True, True])
_ON_SUPPORT = (0, 1)
_OFF_SUPPORT = (2, 3)
_REFUSE_PAIR = (0, 2)
_EDGE_MASS = 0.5
_OFF_EDGE_MASS = 0.25
_DEST_SHIFT = 1


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


def _leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


class TestKeptOverlayRollsPhi:
    """Refuse-aware kept addends against occupy-plus-incident intensity and leftover unpaid."""

    def test_kept_addends_match_phi_intensity(self) -> None:
        """Scatter of kept directed pairs matches occupy plus incident pair sums."""
        sources = torch.tensor([_ON_SUPPORT[0], _OFF_SUPPORT[0], _REFUSE_PAIR[0]])
        destinations = torch.tensor([_ON_SUPPORT[1], _OFF_SUPPORT[1], _REFUSE_PAIR[1]])
        mass = torch.tensor([_EDGE_MASS, _OFF_EDGE_MASS, 9.0], dtype=torch.float64)
        keep = torch.tensor([True, True, False])
        addends = _PHI.kept_pair_addends(sources, destinations, mass, keep)
        assert torch.equal(addends[_REFUSE_PAIR], addends.new_zeros(()))
        assert torch.allclose(addends[_ON_SUPPORT], addends.new_tensor(_EDGE_MASS))
        intensity = _PHI(_OCCUPY, addends, _OCCUPIED)
        hand = torch.tensor([1.5, 1.5, 0.75, 0.75], dtype=torch.float64)
        assert torch.allclose(intensity, hand)

    def test_dest_roll_kept_edge_on_support_moves_intensity_and_unpaid(
        self,
        covering: Covering,
    ) -> None:
        """Dest-axis roll of a kept edge on supp(L) moves Phi on those slots and leftover unpaid."""
        sources = torch.tensor([_ON_SUPPORT[0]])
        destinations = torch.tensor([_ON_SUPPORT[1]])
        mass = torch.tensor([_EDGE_MASS], dtype=torch.float64)
        keep = torch.tensor([True])
        addends = _PHI.kept_pair_addends(sources, destinations, mass, keep)
        rolled = addends.roll(_DEST_SHIFT, dims=-1)
        paid = _PHI(_OCCUPY, addends, _OCCUPIED)
        moved = _PHI(_OCCUPY, rolled, _OCCUPIED)
        unpaid_paid = _leftover(covering, _DEMAND, paid)
        unpaid_moved = _leftover(covering, _DEMAND, moved)
        assert paid[1].item() != moved[1].item()
        assert paid[2].item() != moved[2].item()
        assert torch.allclose(paid[0], moved[0])
        assert unpaid_paid.item() != unpaid_moved.item()
        assert unpaid_paid.item() < unpaid_moved.item()

    def test_dest_roll_or_consume_refuse_does_not_add_to_phi(
        self,
        covering: Covering,
    ) -> None:
        """A consumed pair adds no addend; dest-rolling that zero table leaves Phi and U put."""
        occupy = torch.ones(_SLOTS, dtype=torch.float64)
        live = torch.ones(_SLOTS, dtype=torch.bool)
        demand = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
        sources = torch.tensor([_REFUSE_EDGE[0]])
        destinations = torch.tensor([_REFUSE_EDGE[1]])
        mass = torch.tensor([3.0], dtype=torch.float64)
        keep = torch.tensor([False])
        addends = _PHI_REFUSE.kept_pair_addends(sources, destinations, mass, keep)
        rolled = addends.roll(_DEST_SHIFT, dims=-1)
        baseline = _PHI_REFUSE(occupy, torch.zeros(_SLOTS, _SLOTS, dtype=torch.float64), live)
        refused = _PHI_REFUSE(occupy, addends, live)
        dest_rolled = _PHI_REFUSE(occupy, rolled, live)
        assert torch.allclose(addends, torch.zeros_like(addends))
        assert torch.allclose(refused, baseline)
        assert torch.allclose(dest_rolled, baseline)
        assert torch.allclose(
            _leftover(covering, demand, refused),
            _leftover(covering, demand, dest_rolled),
        )

    def test_dest_roll_off_support_kept_edge_leaves_unpaid_put(
        self,
        covering: Covering,
    ) -> None:
        """Dest-roll of a kept edge whose ends miss supp(L) moves n off support and leaves U."""
        slots = 6
        occupy = torch.ones(slots, dtype=torch.float64)
        live = torch.ones(slots, dtype=torch.bool)
        demand = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        sources = torch.tensor([3])
        destinations = torch.tensor([4])
        mass = torch.tensor([_OFF_EDGE_MASS], dtype=torch.float64)
        keep = torch.tensor([True])
        phi = PhiIntensity(slots)
        addends = phi.kept_pair_addends(sources, destinations, mass, keep)
        rolled = addends.roll(_DEST_SHIFT, dims=-1)
        paid = phi(occupy, addends, live)
        moved = phi(occupy, rolled, live)
        assert paid[4].item() != moved[4].item()
        assert paid[5].item() != moved[5].item()
        assert torch.allclose(paid[:2], moved[:2])
        assert torch.allclose(_leftover(covering, demand, paid), _leftover(covering, demand, moved))

    def test_assignment_and_bundle_kept_tables_agree_and_dest_roll_moves_unpaid(
        self,
        covering: Covering,
    ) -> None:
        """RelationAssignment keep and the dropped bundle scatter the same addends."""
        vocab = _planted_vocab()
        scored = _scored_tournament(vocab)
        keep = ~scored.is_consumed()
        from_assignment = _PHI_REFUSE.kept_pair_addends(
            _PAIR_SRC,
            _PAIR_DST,
            scored.kept_pair_mass(),
            keep,
        )
        assignment, mask = _three_slot_assignment()
        bundle = _planted_bundle(vocab)
        labeled = Explain().graphs_from_bundle(
            assignment,
            mask,
            bundle,
            keep_relation=False,
        )
        from_bundle = labeled[0]
        assert torch.allclose(from_assignment, from_bundle[:_SLOTS, :_SLOTS], atol=1e-5)
        assert torch.equal(from_assignment[_REFUSE_EDGE], from_assignment.new_zeros(()))
        assert torch.equal(from_bundle[_REFUSE_EDGE], from_bundle.new_zeros(()))
        assert from_assignment[_KEPT_EDGE].item() > 0.9

        bank = int(from_bundle.size(-1))
        occupy = from_bundle.new_zeros(bank)
        occupy[:_SLOTS] = 1.0
        live = occupy > 0
        demand = from_bundle.new_zeros(bank)
        demand[0] = 2.0
        demand[2] = 1.0
        phi = PhiIntensity(bank)
        paid = phi(occupy, from_bundle, live)
        moved = phi(occupy, from_bundle.roll(_DEST_SHIFT, dims=-1), live)
        unpaid_paid = _leftover(covering, demand, paid)
        unpaid_moved = _leftover(covering, demand, moved)
        assert not torch.allclose(paid, moved)
        assert unpaid_paid.item() != unpaid_moved.item()


class TestPhiIntensityModule:
    """Identity is a non-parameter buffer; addends scatter on the module device."""

    def test_identity_is_buffer_not_parameter(self) -> None:
        """Registered eye is a buffer, omitted from state_dict, and unused by Adam."""
        phi = PhiIntensity(4)
        buffers = dict(phi.named_buffers())
        assert 'identity' in buffers
        assert buffers['identity'].shape == (4, 4)
        assert buffers['identity'].dtype == torch.bool
        assert torch.equal(buffers['identity'], torch.eye(4, dtype=torch.bool))
        assert list(phi.parameters()) == []
        assert 'identity' not in phi.state_dict()

    def test_kept_addends_table_follows_module_device(self) -> None:
        """index_put accumulate writes the square table on the identity device."""
        phi = PhiIntensity(3)
        sources = torch.tensor([0])
        destinations = torch.tensor([1])
        mass = torch.tensor([0.5], dtype=torch.float64)
        keep = torch.tensor([True])
        table = phi.kept_pair_addends(sources, destinations, mass, keep)
        assert table.device == phi.identity.device
        assert table.shape == (3, 3)
        assert torch.allclose(table[0, 1], table.new_tensor(0.5))
        assert torch.equal(table[1, 0], table.new_zeros(()))


class TestCoveringInventoryReadsPhi:
    """Live covering n is occupy plus kept-edge addends."""

    def test_inventory_n_matches_occupy_plus_kept_addend(self) -> None:
        """Inventory covering n equals Phi of occupy mass and the kept pair table."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config())
        last_layer = torch.randn(1, 6, 32)
        attention = torch.ones(1, 6)
        claim = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        model = SimpleNamespace(soft_vocab=vocab)
        with type_aligned_pairs(vocab):
            payload = Inventory(occupied_floor=0.0)(
                last_layer,
                attention,
                claim,
                model=model,
                claim_texts=('1. A photodiode.',),
            )
        late, _ = vocab.soft_assign(last_layer)
        occupy = vocab.masked_intensity(late, attention)
        assert payload.full_labeled is not None
        assert payload.claim_demand is not None
        pairs = payload.full_labeled
        inventory = Inventory(occupied_floor=0.0)
        intensity = inventory.overlay_intensity(
            occupy,
            pairs,
            demand=payload.claim_demand,
        )
        assert torch.allclose(payload.n_entity_full, intensity)
        assert pairs.abs().sum().item() > 0
        assert not torch.allclose(payload.n_entity_full, occupy)

    def test_kept_edge_roll_moves_covering_n(self) -> None:
        """Dest-axis roll of the kept pair table inventory used moves covering n."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config())
        last_layer = torch.randn(1, 6, 32)
        attention = torch.ones(1, 6)
        claim = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        model = SimpleNamespace(soft_vocab=vocab)
        with type_aligned_pairs(vocab):
            payload = Inventory(occupied_floor=0.0)(
                last_layer,
                attention,
                claim,
                model=model,
                claim_texts=('1. A photodiode.',),
            )
        late, _ = vocab.soft_assign(last_layer)
        occupy = vocab.masked_intensity(late, attention)
        assert payload.full_labeled is not None
        assert payload.claim_demand is not None
        pairs = payload.full_labeled
        inventory = Inventory(occupied_floor=0.0)
        paid = inventory.overlay_intensity(occupy, pairs, demand=payload.claim_demand)
        moved = inventory.overlay_intensity(
            occupy,
            pairs.roll(_DEST_SHIFT, dims=-1),
            demand=payload.claim_demand,
        )
        assert torch.allclose(payload.n_entity_full, paid)
        assert not torch.allclose(paid, moved)

    def test_consumed_refuse_roll_leaves_covering_n(self) -> None:
        """Consumed refuse writes no addend; dest-rolling that zero table leaves n."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config())
        last_layer = torch.randn(1, 6, 32)
        attention = torch.ones(1, 6)
        claim = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        refuse = F.normalize(vocab.refuse_probe.detach(), dim=-1)

        def pair_features(heads: Tensor, tails: Tensor) -> Tensor:
            _ = tails
            aligned = refuse.to(device=heads.device, dtype=heads.dtype)
            return aligned.expand(heads.size(0), -1).clone()

        model = SimpleNamespace(soft_vocab=vocab)
        with patch.object(vocab, 'pair_features', side_effect=pair_features):
            payload = Inventory(occupied_floor=0.0)(
                last_layer,
                attention,
                claim,
                model=model,
                claim_texts=('1. A photodiode.',),
            )
        late, _ = vocab.soft_assign(last_layer)
        occupy = vocab.masked_intensity(late, attention)
        assert payload.full_labeled is not None
        assert payload.claim_demand is not None
        pairs = payload.full_labeled
        inventory = Inventory(occupied_floor=0.0)
        overlay = PhiIntensity(int(occupy.size(-1)))
        live = occupy > 0
        paid = inventory.overlay_intensity(occupy, pairs, demand=payload.claim_demand)
        assert torch.allclose(pairs, torch.zeros_like(pairs))
        assert torch.allclose(payload.n_entity_full, paid)
        assert torch.allclose(
            payload.n_entity_full,
            inventory.overlay_intensity(
                occupy,
                pairs.roll(_DEST_SHIFT, dims=-1),
                demand=payload.claim_demand,
            ),
        )
        refused = overlay.kept_pair_addends(
            torch.tensor([0]),
            torch.tensor([1]),
            occupy.new_tensor([3.0]),
            torch.tensor([False]),
        )
        assert torch.allclose(overlay(occupy, refused, live), occupy)

    def test_dest_comparison_reads_leftover_unpaid_of_inventory(self) -> None:
        """Dest train scores leftover unpaid of claim demand against covering n."""
        assert 'PhiIntensity' not in dest_mod.__dict__
        assert 'ip_claim.ssv.phi' not in inspect.getsource(dest_mod)
        step = inspect.getsource(dest_mod.SsvLightningModule.training_step)
        assert 'unpaid_fraction' in step
        assert 'pair_table' in step
        assert 'n_entity_claim' in step
        assert 'n_entity_full' in step
        assert 'edge_unpaid_fraction' not in step
        assert 'PhiIntensity' not in step
