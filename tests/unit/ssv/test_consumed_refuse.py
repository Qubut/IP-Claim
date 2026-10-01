"""Relation assignment exposes consumed refuse; overlay consumers drop those pairs.

Candidate gather stays every ordered pair of occupied codes. Kept overlays,
merged HeteroData, and dest slot tables omit consumed refuse.
"""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F
from tests._ssv_fixtures import ssv_tiny_vocab_config
from torch_geometric.data import HeteroData

from ip_claim.collision.explain import Explain
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.soft_graph import (
    SOFT_RELATED,
    build_soft_relation_bundle,
    complete_directed_pairs,
    merge_soft_overlay,
    select_leftover_unpaid_occupy,
)
from ip_claim.ssv.soft_vocab import RelationAssignment, SoftVocabModule

_SLOTS = 3
_PAIR_SRC = torch.tensor([0, 0, 1, 1, 2, 2])
_PAIR_DST = torch.tensor([1, 2, 0, 2, 0, 1])
_REFUSE_PAIR = 0
_REFUSE_EDGE = (0, 1)
_KEPT_EDGE = (0, 2)


def _planted_vocab() -> SoftVocabModule:
    vocab = SoftVocabModule(
        ssv_tiny_vocab_config(
            entity_bank_size=8,
            relation_bank_size=2,
            soft_dim=8,
            relation_temperature=0.07,
        )
    )
    basis = F.normalize(torch.eye(3, 8), dim=-1)
    _ = vocab.entity_bank.data.copy_(F.normalize(torch.eye(8), dim=-1))
    _ = vocab.relation_bank.data.copy_(basis[:2])
    _ = vocab.refuse_probe.data.copy_(basis[2])
    return vocab


def _scored_tournament(vocab: SoftVocabModule) -> RelationAssignment:
    basis = F.normalize(torch.eye(3, 8), dim=-1)
    feats = basis[0].expand(6, 8).clone()
    feats[_REFUSE_PAIR] = basis[2]
    return vocab.soft_assign_relations(feats)


def _pair_table(mass: torch.Tensor) -> torch.Tensor:
    table = torch.zeros(_SLOTS, _SLOTS)
    table[_PAIR_SRC, _PAIR_DST] = mass
    return table


def _three_slot_assignment() -> tuple[torch.Tensor, torch.Tensor]:
    assignment = torch.zeros(1, 3, 8)
    assignment[0, torch.arange(3), torch.arange(3)] = 1.0
    return assignment, torch.ones(1, 3)


def _refuse_on_zero_to_one(vocab: SoftVocabModule):
    basis = F.normalize(torch.eye(3, 8), dim=-1)
    entity = F.normalize(vocab.entity_bank.detach(), dim=-1)

    def pair_features(heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        head_code = torch.einsum('pd,kd->pk', F.normalize(heads, dim=-1), entity).argmax(-1)
        tail_code = torch.einsum('pd,kd->pk', F.normalize(tails, dim=-1), entity).argmax(-1)
        typed = basis[0].expand(heads.size(0), 8)
        refuse = ((head_code == 0) & (tail_code == 1)).unsqueeze(-1)
        return torch.where(refuse, basis[2], typed)

    return pair_features


def _occupied_three(vocab: SoftVocabModule):
    occupancy = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    demand = (occupancy > 0.5).to(dtype=occupancy.dtype)
    pairs = occupancy.new_zeros(1, occupancy.size(-1), occupancy.size(-1))
    return select_leftover_unpaid_occupy(
        vocab.entity_bank,
        occupancy,
        demand=demand,
        pair_mass=pairs,
        mass_floor=0.5,
    ).occupied


def _overlay_bank_pairs(overlay) -> set[tuple[int, int]]:
    return set(
        zip(
            overlay.code_ids[overlay.edge_index[0]].tolist(),
            overlay.code_ids[overlay.edge_index[1]].tolist(),
            strict=True,
        )
    )


def _planted_bundle(vocab: SoftVocabModule):
    assignment, mask = _three_slot_assignment()
    with patch.object(vocab, 'pair_features', side_effect=_refuse_on_zero_to_one(vocab)):
        return build_soft_relation_bundle(
            vocab,
            assignment,
            mask,
            mass_floor=0.5,
            demand=vocab.masked_intensity(assignment, mask),
        )


class TestConsumedRefuseAssignment:
    """Joint type-or-refuse softmax marks ⊥ as consumed leftover, not a bank row."""

    def test_joint_mass_is_typed_plus_consumed(self) -> None:
        vocab = _planted_vocab()
        scored = _scored_tournament(vocab)
        assert scored.typed.shape == (6, vocab.relation_bank_size)
        assert scored.consumed.shape == (6,)
        assert torch.allclose(
            scored.typed.sum(dim=-1) + scored.consumed,
            torch.ones(6),
            atol=1e-5,
        )
        assert torch.allclose(scored.kept_pair_mass(), scored.typed.sum(dim=-1))

    def test_refuse_aligned_pair_is_consumed_non_edge(self) -> None:
        vocab = _planted_vocab()
        scored = _scored_tournament(vocab)
        consumed = scored.is_consumed()
        assert bool(consumed[_REFUSE_PAIR].item())
        assert not bool(consumed[1].item())
        assert scored.kept_pair_mass()[_REFUSE_PAIR].item() < 0.05
        assert scored.kept_pair_mass()[1].item() > 0.9
        assert scored.consumed[_REFUSE_PAIR].item() > scored.typed[_REFUSE_PAIR].amax().item()

    def test_refuse_pair_mass_does_not_add_to_phi(self) -> None:
        vocab = _planted_vocab()
        scored = _scored_tournament(vocab)
        occupy = torch.ones(_SLOTS)
        live = torch.ones(_SLOTS, dtype=torch.bool)
        kept = _pair_table(scored.kept_pair_mass())
        clique = torch.ones(_SLOTS, _SLOTS) - torch.eye(_SLOTS)
        phi = PhiIntensity(_SLOTS)
        paid = phi(occupy, clique, live)
        leftover = phi(occupy, kept, live)
        assert leftover[0].item() < paid[0].item()
        assert leftover[1].item() < paid[1].item()
        assert torch.allclose(leftover[2], paid[2], atol=0.05)
        zero_refuse = clique.clone()
        zero_refuse[0, 1] = 0.0
        assert torch.allclose(
            phi(occupy, zero_refuse, live),
            leftover,
            atol=0.05,
        )

    def test_candidate_gather_stays_complete_overlay_drops_refuse(self) -> None:
        vocab = _planted_vocab()
        occupied = _occupied_three(vocab)
        pairs = complete_directed_pairs(occupied)
        bundle = _planted_bundle(vocab)
        overlay_pairs = _overlay_bank_pairs(bundle.overlays[0])
        assert pairs.pair_counts == (6,)
        assert pairs.heads.size(0) == 6
        assert bundle.pair_count == 5
        assert bundle.overlays[0].edge_index.size(1) == 5
        assert bundle.relation_assignment is not None
        assert bundle.relation_assignment.shape == (5, vocab.relation_bank_size)
        assert _REFUSE_EDGE not in overlay_pairs
        assert _KEPT_EDGE in overlay_pairs


class TestOverlayConsumersDropRefuse:
    """Merge, dest pair tables, and inspect-width overlays omit consumed pairs."""

    def test_merged_hetero_omits_consumed_pair(self) -> None:
        vocab = _planted_vocab()
        bundle = _planted_bundle(vocab)
        graph = HeteroData()
        graph['cpc'].x = torch.zeros((1, 1))
        graph['claim'].x = torch.zeros((1, 1))
        merged = merge_soft_overlay(graph, bundle.overlays[0], bank_size=8)
        merged_pairs = set(
            zip(
                merged[SOFT_RELATED].edge_index[0].tolist(),
                merged[SOFT_RELATED].edge_index[1].tolist(),
                strict=True,
            )
        )
        assert _REFUSE_EDGE not in merged_pairs
        assert _KEPT_EDGE in merged_pairs
        assert merged[SOFT_RELATED].edge_index.size(1) == 5

    def test_dest_slot_table_omits_consumed_pair(self) -> None:
        vocab = _planted_vocab()
        assignment, mask = _three_slot_assignment()
        bundle = _planted_bundle(vocab)
        labeled = Explain().graphs_from_bundle(
            assignment,
            mask,
            bundle,
            keep_relation=True,
            relation_bank=int(vocab.relation_bank_size),
        )
        assert labeled.shape == (1, 8, 8, vocab.relation_bank_size)
        assert torch.allclose(labeled[0, 0, 1], labeled.new_zeros(vocab.relation_bank_size))
        assert labeled[0, 0, 2].sum().item() > 0.0
