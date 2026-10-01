"""Occupied-code pair contracts: long-range edges, scored R, same slots as n."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F
from tests._ssv_fixtures import ssv_tiny_vocab_config, type_aligned_pairs
from torch_geometric.data import HeteroData

from ip_claim.collision.explain import Explain
from ip_claim.ssv.config import ArchSpec
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.soft_graph import (
    SOFT_RELATED,
    OccupiedCodes,
    build_soft_relation_bundle,
    complete_directed_pairs,
    merge_soft_overlay,
    select_leftover_unpaid_occupy,
)
from ip_claim.ssv.soft_vocab import SoftVocabModule


def _tiny_vocab(
    *,
    entity_bank_size: int = 8,
    relation_bank_size: int = 4,
    soft_dim: int = 16,
) -> SoftVocabModule:
    return SoftVocabModule(
        ssv_tiny_vocab_config(
            entity_bank_size=entity_bank_size,
            relation_bank_size=relation_bank_size,
            soft_dim=soft_dim,
            relation_temperature=0.07,
        )
    )


def _distant_occupancy(*, tokens: int = 12, bank: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot codes on the first and last tokens only; twelve-token span."""
    assignment = torch.zeros(1, tokens, bank)
    assignment[0, 0, 2] = 1.0
    assignment[0, tokens - 1, 5] = 1.0
    mask = torch.zeros(1, tokens)
    mask[0, 0] = 1.0
    mask[0, tokens - 1] = 1.0
    return assignment, mask


def _occupy_ingress(
    bank: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    mass_floor: float = 0.0,
) -> OccupiedCodes:
    demand = (occupancy > mass_floor).to(dtype=occupancy.dtype)
    pairs = occupancy.new_zeros(occupancy.size(0), occupancy.size(-1), occupancy.size(-1))
    return select_leftover_unpaid_occupy(
        bank,
        occupancy,
        demand=demand,
        pair_mass=pairs,
        mass_floor=mass_floor,
    ).occupied


def test_graph_knobs_are_floor_not_a_token_window_or_occupy_integer() -> None:
    assert 'soft_pair_window' not in ArchSpec.model_fields
    assert 'soft_pair_max_per_doc' not in ArchSpec.model_fields
    assert 'soft_occupied_max' not in ArchSpec.model_fields
    assert 'soft_occupied_floor' in ArchSpec.model_fields
    assert 'mask_query_window' in ArchSpec.model_fields


def test_leftover_unpaid_occupy_keeps_ingress_and_empty_demand_is_empty() -> None:
    bank = torch.eye(6)
    occupancy = torch.tensor([[0.9, 0.4, 0.05, 0.8, 0.0, 0.2]])
    kept = _occupy_ingress(bank, occupancy, mass_floor=0.1)
    assert set(kept.code_ids[0][kept.keep[0]].tolist()) == {0, 1, 3, 5}
    empty = select_leftover_unpaid_occupy(
        bank,
        occupancy,
        demand=occupancy.new_zeros(occupancy.shape),
        pair_mass=occupancy.new_zeros(1, 6, 6),
        mass_floor=0.0,
    )
    assert int(empty.occupied.lengths.item()) == 0


def test_complete_directed_pairs_are_n_times_n_minus_one() -> None:
    bank = torch.randn(5, 4)
    occupancy = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    occupied = _occupy_ingress(bank, occupancy, mass_floor=0.5)
    pairs = complete_directed_pairs(occupied)
    assert pairs.pair_counts == (6,)
    assert pairs.heads.size(0) == 6
    singleton = _occupy_ingress(
        bank,
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        mass_floor=0.0,
    )
    assert complete_directed_pairs(singleton).pair_counts == (0,)


def test_distant_occupied_codes_form_both_directed_edges() -> None:
    vocab = _tiny_vocab()
    assignment, mask = _distant_occupancy()
    with type_aligned_pairs(vocab):
        bundle = build_soft_relation_bundle(
            vocab,
            assignment,
            mask,
            mass_floor=0.0,
            demand=vocab.masked_intensity(assignment, mask),
        )
    assert bundle.pair_count == 2
    overlay = bundle.overlays[0]
    src = overlay.code_ids[overlay.edge_index[0]].tolist()
    dst = overlay.code_ids[overlay.edge_index[1]].tolist()
    assert set(zip(src, dst, strict=True)) == {(2, 5), (5, 2)}
    assert bundle.relation_assignment is not None
    assert bundle.relation_assignment.shape == (2, vocab.relation_bank_size)


def test_merge_addresses_pair_edges_at_bank_slots() -> None:
    vocab = _tiny_vocab()
    assignment, mask = _distant_occupancy()
    with type_aligned_pairs(vocab):
        bundle = build_soft_relation_bundle(
            vocab,
            assignment,
            mask,
            mass_floor=0.0,
            demand=vocab.masked_intensity(assignment, mask),
        )
    overlay = bundle.overlays[0]
    graph = HeteroData()
    graph['cpc'].x = torch.zeros((1, 1))
    graph['claim'].x = torch.zeros((1, 1))
    bank = int(assignment.size(-1))
    merged = merge_soft_overlay(graph, overlay, bank_size=bank)
    assert int(merged['soft_entity'].num_nodes) == bank
    keep = merged['soft_entity'].occupied.tolist()
    assert keep == [False, False, True, False, False, True, False, False]
    pairs = set(
        zip(
            merged[SOFT_RELATED].edge_index[0].tolist(),
            merged[SOFT_RELATED].edge_index[1].tolist(),
            strict=True,
        )
    )
    assert pairs == {(2, 5), (5, 2)}


def test_relation_scores_can_peak_or_stay_flat_without_dropping_edges() -> None:
    vocab = _tiny_vocab(relation_bank_size=2, soft_dim=8)
    orthonormal = F.normalize(torch.eye(3, 8), dim=-1)
    _ = vocab.relation_bank.data.copy_(orthonormal[:2])
    _ = vocab.refuse_probe.data.copy_(orthonormal[2])
    aligned = orthonormal[0]
    flat = F.normalize(torch.ones(8), dim=-1)
    peaked = vocab.soft_assign_relations(aligned.unsqueeze(0))
    smeared = vocab.soft_assign_relations(flat.unsqueeze(0))
    assert peaked.typed.max().detach() > 0.9
    assert smeared.typed.max().detach() < peaked.typed.max().detach()
    assert torch.allclose(
        peaked.typed.sum(-1) + peaked.consumed,
        torch.ones(1),
        atol=1e-5,
    )
    assert torch.allclose(
        smeared.typed.sum(-1) + smeared.consumed,
        torch.ones(1),
        atol=1e-5,
    )
    assignment, mask = _distant_occupancy()
    with type_aligned_pairs(vocab):
        bundle = build_soft_relation_bundle(
            vocab,
            assignment,
            mask,
            mass_floor=0.0,
            demand=vocab.masked_intensity(assignment, mask),
        )
    assert bundle.pair_count == 2
    assert bundle.overlays[0].edge_index.size(1) == 2


def test_relation_mass_and_slot_adjacency_live_on_occupied_codes() -> None:
    vocab = _tiny_vocab()
    assignment, mask = _distant_occupancy()
    inventory = Inventory(occupied_floor=0.0)
    with type_aligned_pairs(vocab):
        bundle = inventory.relation_bundle(
            assignment,
            mask,
            vocab,
            demand=vocab.masked_intensity(assignment, mask),
        )
    occupancy = vocab.masked_intensity(assignment, mask)
    assert torch.allclose(occupancy[0, 2], occupancy.new_tensor(1.0))
    assert torch.allclose(occupancy[0, 5], occupancy.new_tensor(1.0))
    assert torch.allclose(occupancy[0, 0], occupancy.new_tensor(0.0))
    assert bundle.relation_assignment is not None
    rel_mass = bundle.relation_assignment.sum(dim=0)
    assert rel_mass.shape == (vocab.relation_bank_size,)
    overlay = bundle.overlays[0]
    adj = Explain().slot_adjacency(
        overlay.edge_index,
        bundle.relation_assignment.sum(dim=-1),
        overlay.code_ids,
        bank=int(assignment.size(-1)),
        device=assignment.device,
        dtype=assignment.dtype,
    )
    zeros = adj.new_zeros(())
    assert adj.shape == (8, 8)
    assert adj[2, 5] > 0
    assert adj[5, 2] > 0
    assert torch.allclose(adj[2, 2], zeros)
    assert rel_mass.sum().item() <= 2.0 + 1e-5
    assert rel_mass.sum().item() > 0.0


def test_relation_seed_uses_all_distinct_entity_pairs() -> None:
    vocab = _tiny_vocab(entity_bank_size=5, relation_bank_size=2, soft_dim=8)
    projected = torch.randn(4, 16, 8)
    with patch.object(vocab, 'pair_features', wraps=vocab.pair_features) as spy:
        vocab.seed_banks_from_projected(projected)
    heads = spy.call_args[0][0]
    assert heads.size(0) == 5 * 4
