"""Edge-unpaid identities of labeled pair tables, plus dest leftover unpaid on smoke."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch
import torch.nn.functional as F
from tests._ssv_fixtures import (
    ssv_fixture_batch,
    ssv_smoke_module,
    ssv_tiny_vocab_config,
    type_aligned_pairs,
)
from torch_geometric.data import HeteroData

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.explain import Explain
from ip_claim.ssv.covering_trace import shift_overlay_edges
from ip_claim.ssv.model import apply_overlay_edge_shift
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.soft_graph import (
    SOFT_RELATED,
    build_soft_relation_bundle,
    merge_soft_overlay,
)
from ip_claim.ssv.soft_vocab import SoftVocabModule


def _tiny_vocab(*, entity_bank_size: int = 8) -> SoftVocabModule:
    return SoftVocabModule(
        ssv_tiny_vocab_config(
            entity_bank_size=entity_bank_size,
            relation_bank_size=4,
            soft_dim=16,
            relation_temperature=0.07,
        )
    )


def _two_slot_assignment(*, tokens: int = 6, bank: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    assignment = torch.zeros(1, tokens, bank)
    assignment[0, 0, 2] = 1.0
    assignment[0, 1, 5] = 1.0
    mask = torch.zeros(1, tokens)
    mask[0, 0] = 1.0
    mask[0, 1] = 1.0
    return assignment, mask


def test_softplus_zero_is_log_two() -> None:
    assert torch.allclose(F.softplus(torch.zeros(())), torch.tensor(math.log(2.0)))


def test_empty_pair_tables_pay_zero_not_nan() -> None:
    covering = Covering(CoveringKnobs())
    empty = torch.zeros(3, 4, 4, 2)
    unpaid = covering.edge_unpaid_fraction(empty, empty)
    assert torch.equal(unpaid, empty.new_zeros(3, 4))
    assert torch.isfinite(unpaid).all()


def test_identical_labeled_tables_have_zero_unpaid_gap() -> None:
    covering = Covering(CoveringKnobs())
    table = torch.rand(2, 5, 5, 3)
    matching = covering.edge_unpaid_fraction(table, table)
    shuffled = covering.edge_unpaid_fraction(table, table)
    gap = torch.nanmean(shuffled - matching)
    dest_loss = torch.nanmean(F.softplus(matching - shuffled))
    assert torch.allclose(gap, gap.new_zeros(()))
    assert torch.allclose(dest_loss, dest_loss.new_tensor(math.log(2.0)))


def test_complete_tournament_unpaid_is_dest_support_invariant() -> None:
    covering = Covering(CoveringKnobs(sigma_edge=1.0))
    bank = 6
    matching = torch.zeros(1, bank, bank)
    shuffled = torch.zeros(1, bank, bank)
    src, dst = torch.tensor([0, 1, 2, 1, 2, 0]), torch.tensor([1, 2, 0, 0, 1, 2])
    matching[0, src, dst] = 1.0
    shuffled[0, src + 1, dst + 1] = 1.0
    match_u = covering.edge_unpaid_fraction(matching, matching)
    shuf_u = covering.edge_unpaid_fraction(shuffled, shuffled)
    assert not torch.allclose(matching, shuffled)
    assert torch.allclose(match_u, shuf_u)
    assert torch.allclose(match_u, match_u.new_tensor(0.5))


def test_matching_w_dest_permuted_unpaid_differs() -> None:
    covering = Covering(CoveringKnobs(sigma_edge=1.0))
    bank = 6
    matching = torch.zeros(1, bank, bank, 1)
    src, dst = torch.tensor([0, 1, 2, 1, 2, 0]), torch.tensor([1, 2, 0, 0, 1, 2])
    matching[0, src, dst, 0] = 1.0
    dest_shift = 1
    dest_permuted = matching.roll(dest_shift, dims=-2)

    def pair_unpaid(query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        return covering.edge_unpaid_fraction(
            query.reshape(query.size(0), -1, query.size(-1)),
            document.reshape(document.size(0), -1, document.size(-1)),
        )

    match_u = pair_unpaid(matching, matching)
    complete_vs_complete = pair_unpaid(dest_permuted, dest_permuted)
    dest_u = pair_unpaid(matching, dest_permuted)
    assert torch.allclose(match_u, complete_vs_complete)
    assert torch.allclose(match_u, match_u.new_tensor(0.5))
    assert not torch.allclose(match_u, dest_u)
    assert float(torch.nanmean(dest_u - match_u)) > 1e-3


def test_graphs_from_bundle_ignore_assignment_values() -> None:
    vocab = _tiny_vocab()
    assignment, mask = _two_slot_assignment()
    with type_aligned_pairs(vocab):
        bundle = build_soft_relation_bundle(
            vocab,
            assignment,
            mask,
            mass_floor=0.0,
            demand=vocab.masked_intensity(assignment, mask),
        )
    explain = Explain()
    noise = torch.randn_like(assignment)
    peaked_w = explain.graphs_from_bundle(
        assignment,
        mask,
        bundle,
        keep_relation=True,
        relation_bank=int(vocab.relation_bank_size),
    )
    noise_w = explain.graphs_from_bundle(
        noise,
        mask,
        bundle,
        keep_relation=True,
        relation_bank=int(vocab.relation_bank_size),
    )
    assert torch.allclose(peaked_w, noise_w)


def test_dest_shift_empties_occupied_to_occupied_on_sparse_codes() -> None:
    vocab = _tiny_vocab()
    assignment, mask = _two_slot_assignment()
    with type_aligned_pairs(vocab):
        bundle = build_soft_relation_bundle(
            vocab,
            assignment,
            mask,
            mass_floor=0.0,
            demand=vocab.masked_intensity(assignment, mask),
        )
    graph = HeteroData()
    graph['cpc'].x = torch.zeros((1, 1))
    graph['claim'].x = torch.zeros((1, 1))
    bank = int(assignment.size(-1))
    merged = merge_soft_overlay(graph, bundle.overlays[0], bank_size=bank)
    occupied = set(merged['soft_entity'].occupied.nonzero(as_tuple=True)[0].tolist())
    assert occupied == {2, 5}
    with shift_overlay_edges(dest=1):
        shifted = apply_overlay_edge_shift(merged, bank_size=bank)
    dests = set(shifted[SOFT_RELATED].edge_index[1].tolist())
    assert dests == {3, 6}
    assert dests.isdisjoint(occupied)


def test_labeled_w_is_invariant_to_assignment_values_when_occupancy_set_holds() -> None:
    vocab = _tiny_vocab()
    peaked, mask = _two_slot_assignment()
    smeared = peaked.clone()
    smeared[0, 0, 2] = 0.7
    smeared[0, 0, 0] = 0.3
    smeared[0, 1, 5] = 0.6
    smeared[0, 1, 1] = 0.4
    explain = Explain()
    with type_aligned_pairs(vocab):
        peaked_bundle = build_soft_relation_bundle(
            vocab,
            peaked,
            mask,
            mass_floor=0.0,
            demand=vocab.masked_intensity(peaked, mask),
        )
        smeared_bundle = build_soft_relation_bundle(
            vocab,
            smeared,
            mask,
            mass_floor=0.0,
            demand=vocab.masked_intensity(peaked, mask),
        )
    peaked_codes = set(peaked_bundle.overlays[0].code_ids.tolist())
    smeared_codes = set(smeared_bundle.overlays[0].code_ids.tolist())
    assert peaked_codes == smeared_codes == {2, 5}
    peaked_w = explain.graphs_from_bundle(
        peaked,
        mask,
        peaked_bundle,
        keep_relation=True,
        relation_bank=int(vocab.relation_bank_size),
    )
    smeared_w = explain.graphs_from_bundle(
        smeared,
        mask,
        smeared_bundle,
        keep_relation=True,
        relation_bank=int(vocab.relation_bank_size),
    )
    assert torch.allclose(peaked_w, smeared_w)
    covering = Covering(CoveringKnobs())
    assert torch.allclose(
        covering.edge_unpaid_fraction(peaked_w, peaked_w),
        covering.edge_unpaid_fraction(smeared_w, smeared_w),
    )


def test_smoke_dest_gap_uses_leftover_unpaid_of_inventory(
    tmp_path: Path,
) -> None:
    module = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
    batch = ssv_fixture_batch(module)
    captured: list[torch.Tensor] = []
    real = SsvLightningModule.forward

    def spy(
        self: SsvLightningModule,
        step_batch: object,
        living: torch.Tensor | None = None,
    ) -> object:
        out = real(self, step_batch, living=living)
        captured.append(out.text_hidden.detach().clone())
        return out

    logged: dict[str, torch.Tensor] = {}

    def capture_log(name: str, value: object = None, **_kwargs: object) -> None:
        if torch.is_tensor(value):
            logged[name] = cast(torch.Tensor, value)

    living = module.model.living_snapshot()
    module.train()
    module.forward = spy.__get__(module, SsvLightningModule)  # type: ignore[method-assign]
    with patch.object(module, 'log', side_effect=capture_log), patch.object(module, 'log_dict'):
        _ = module.training_step(batch, 0)
    assert 'dest_unpaid_gap' in logged
    assert 'dest_loss' in logged
    gap = float(logged['dest_unpaid_gap'].detach())
    dest_loss = float(logged['dest_loss'].detach())
    assert abs(gap) > 1e-3
    assert abs(dest_loss - math.log(2.0)) > 1e-3
    assert len(captured) == 1
    claim = module.inventory.claim_mask(
        module.model.host_tokenizer(),
        batch.texts,
        batch.claim_texts,
        batch.attention_mask,
        max_length=int(module.config.arch.max_length),
    )
    matching = module.inventory(
        captured[0],
        batch.attention_mask,
        claim,
        model=module.model,
        texts=batch.texts,
        input_ids=batch.unmasked_input_ids,
        living=living,
        claim_texts=batch.claim_texts,
    )
    covering = module.covering
    dest_shift = int(module.config.dest_comparison.dest_shift)
    demand = matching.n_entity_claim.detach()
    supply = matching.n_entity_full.detach()
    match_u = covering.unpaid_fraction(covering.pair_table(demand, supply))
    dest_u = covering.unpaid_fraction(covering.pair_table(demand, supply.roll(dest_shift, dims=-1)))
    unpaid_gap = float(torch.nanmean(dest_u - match_u))
    assert not torch.allclose(match_u, dest_u)
    assert abs(unpaid_gap) > 1e-3
    assert abs(unpaid_gap - gap) < 0.02
