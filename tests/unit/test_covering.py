"""Saturation covering identities: dilution, verbosity, masks, and contour sums."""

from __future__ import annotations

import pytest
import torch

from ip_claim.collision.cover import Covering, CoveringKnobs


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


def test_sigma_buffers_are_not_parameters(covering: Covering) -> None:
    assert list(covering.parameters()) == []
    names = dict(covering.named_buffers())
    assert set(names) == {
        'sigma',
        'sigma_edge',
        'lambda_relation',
        'row_top_k',
        'row_mass_keep',
        'slot_top_k',
        'slot_mass_keep',
    }
    assert torch.allclose(names['sigma'], torch.tensor(1.0))
    assert int(names['row_top_k'].item()) == -1
    assert torch.allclose(names['slot_mass_keep'], torch.tensor(0.0))
    assert 'sigma' in covering.state_dict()


def test_presence_is_zero_at_empty_and_rises_with_mass(covering: Covering) -> None:
    empty = covering.presence(torch.tensor([0.0]))
    one = covering.presence(torch.tensor([1.0]))
    huge = covering.presence(torch.tensor([1.0e6]))
    assert torch.allclose(empty, torch.zeros(1))
    assert torch.allclose(one, torch.tensor([0.5]))
    assert huge.item() > 0.999
    assert one.item() < 0.7


def test_completeness_covering_near_one_only_when_demanded_slots_saturate(
    covering: Covering,
) -> None:
    n_query = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
    missing = covering(n_query, torch.tensor([0.0, 20.0, 0.0], dtype=torch.float64))
    paid = covering(n_query, torch.tensor([8.0, 8.0, 0.0], dtype=torch.float64))
    assert missing.covering.item() < 0.4
    assert paid.covering.item() > 0.85
    assert paid.covering.item() < 1.0
    hard = covering.hard_presence(torch.tensor([2.0, 2.0, 0.0]), epsilon=1.0)
    assert torch.equal(hard, torch.tensor([1.0, 1.0, 0.0]))


def test_extra_unused_slots_do_not_change_covering(covering: Covering) -> None:
    n_query = torch.tensor([3.0, 1.0, 0.0, 0.0])
    n_document = torch.tensor([2.0, 4.0, 0.0, 0.0])
    n_broader = torch.tensor([2.0, 4.0, 8.0, 5.0])
    narrow = covering(n_query, n_document)
    broad = covering(n_query, n_broader)
    assert torch.allclose(narrow.residual, broad.residual)
    assert torch.allclose(narrow.covering, broad.covering)
    simplex_narrow = n_document[:2] / n_document[:2].sum()
    simplex_broad = n_broader / n_broader.sum()
    assert simplex_broad[0] < simplex_narrow[0]


def test_verbosity_drops_unpaid_versus_raw_difference_not_to_zero(
    covering: Covering,
) -> None:
    n_query = torch.tensor([10.0, 0.0])
    n_document = torch.tensor([1.0, 0.0])
    scored = covering(n_query, n_document)
    raw_gap = (n_query - n_document).clamp(min=0.0).sum()
    assert scored.unpaid_mass.item() < raw_gap.item()
    assert scored.unpaid_mass.item() > 0.0
    assert torch.allclose(scored.residual[0], torch.tensor(5.0))


def test_claim_only_document_raises_unpaid_when_slot_lives_outside_claims(
    covering: Covering,
) -> None:
    assignment = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    claim_mask = torch.tensor([1.0, 0.0, 0.0])
    full_mask = torch.tensor([1.0, 1.0, 1.0])
    n_query = torch.tensor([0.0, 2.0])
    n_full = covering.masked_intensity(assignment, full_mask)
    n_claim_doc = covering.masked_intensity(assignment, claim_mask)
    full = covering(n_query, n_full)
    claim_only = covering(n_query, n_claim_doc)
    assert n_full[1].item() > 0.0
    assert torch.allclose(n_claim_doc[1], torch.zeros(()))
    assert claim_only.unpaid_mass.item() > full.unpaid_mass.item()


def test_query_contour_integrates_to_unpaid_demand(covering: Covering) -> None:
    assignment = torch.tensor([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.5, 0.5],
    ])
    claim_mask = torch.tensor([1.0, 1.0, 0.0])
    n_query = covering.masked_intensity(assignment, claim_mask)
    n_document = torch.tensor([0.5, 0.0])
    scored = covering(n_query, n_document)
    field = covering.query_unpaid_field(assignment, n_document, n_query)
    integral = (claim_mask * field).sum()
    assert torch.allclose(integral, scored.unpaid_mass)


def test_paying_contour_integrates_to_paid_demand_and_is_zero_on_empty_slots(
    covering: Covering,
) -> None:
    assignment = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])
    full_mask = torch.ones(3)
    n_document = covering.masked_intensity(assignment, full_mask)
    n_query = torch.tensor([2.0, 0.0])
    scored = covering(n_query, n_document)
    field = covering.document_paying_field(assignment, n_query, n_document)
    integral = (full_mask * field).sum()
    assert torch.allclose(integral, scored.paid.sum())
    empty_doc = torch.zeros(2)
    empty_field = covering.document_paying_field(assignment, n_query, empty_doc)
    assert torch.allclose(empty_field, torch.zeros(3))
    assert torch.isfinite(empty_field).all()


def test_union_covering_is_at_least_each_single_document(covering: Covering) -> None:
    n_query = torch.tensor([2.0, 2.0])
    first = torch.tensor([4.0, 0.0])
    second = torch.tensor([0.0, 4.0])
    one = covering(n_query, first)
    two = covering(n_query, second)
    both = covering.union_covering(n_query, first, second)
    assert both.covering.item() >= one.covering.item()
    assert both.covering.item() >= two.covering.item()


def test_relation_covering_uses_edge_scale() -> None:
    covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=2.0))
    n_query = torch.tensor([1.0])
    n_document = torch.tensor([1.0])
    entity = covering(n_query, n_document)
    relation = covering.relation_covering(n_query, n_document)
    assert relation.unpaid_mass.item() > entity.unpaid_mass.item()


def test_alignment_is_paid_mass_on_matching_slots(covering: Covering) -> None:
    assignment_query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assignment_document = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    paid = torch.tensor([2.0, 0.0])
    paired = covering.alignment(assignment_query, assignment_document, paid)
    assert paired.shape == (2, 2)
    assert torch.allclose(paired[0], torch.tensor([2.0, 0.4]))
    assert torch.allclose(paired[1], torch.zeros(2))


def test_keep_prefix_is_identity_when_knobs_off(covering: Covering) -> None:
    mass = torch.tensor([4.0, 2.0, 1.0, 0.5])
    assert torch.equal(covering.keep_slots(mass), mass)
    assignment = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    assert torch.equal(covering.keep_rows(assignment), assignment)


def test_keep_prefix_mass_fraction_keeps_shortest_prefix() -> None:
    covering = Covering(CoveringKnobs(slot_mass_keep=0.7))
    mass = torch.tensor([20.0, 0.4, 0.4, 0.4, 0.4, 0.4])
    kept = covering.keep_slots(mass)
    assert torch.allclose(kept[0], mass[0])
    assert torch.allclose(kept[1:], torch.zeros(5))
    empty = covering.keep_slots(torch.zeros(4))
    assert torch.allclose(empty, torch.zeros(4))


def test_keep_prefix_top_k_caps_slots() -> None:
    covering = Covering(CoveringKnobs(slot_top_k=2, slot_mass_keep=1.0))
    mass = torch.tensor([5.0, 4.0, 3.0, 2.0])
    kept = covering.keep_slots(mass)
    assert torch.allclose(kept, torch.tensor([5.0, 4.0, 0.0, 0.0]))


def test_row_top_k_then_sum_drops_token_tail() -> None:
    covering = Covering(CoveringKnobs(row_top_k=1))
    assignment = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.8, 0.15, 0.05],
        [0.1, 0.1, 0.8],
    ])
    mask = torch.ones(3)
    dense = Covering(CoveringKnobs()).masked_intensity(assignment, mask)
    sparse = covering.masked_intensity(assignment, mask)
    assert dense[1] > 0.0
    assert torch.allclose(sparse, torch.tensor([1.7, 0.0, 0.8]))


def test_sparse_n_unused_slots_do_not_dilute() -> None:
    keeper = Covering(CoveringKnobs(slot_mass_keep=0.7))
    n_query = keeper.keep_slots(torch.tensor([20.0, 0.4, 0.4, 0.4, 0.4, 0.4]))
    n_document = keeper.keep_slots(torch.tensor([8.0, 0.4, 0.4, 0.4, 0.4, 0.4]))
    n_broader = n_document.clone()
    n_broader[-1] = 12.0
    covering = Covering(CoveringKnobs())
    narrow = covering(n_query, n_document)
    broad = covering(n_query, n_broader)
    assert torch.allclose(narrow.residual, broad.residual)
    assert torch.allclose(narrow.covering, broad.covering)
    assert n_query[0].item() > 0.0
    assert torch.allclose(n_query[1:], torch.zeros(5))


def test_pair_table_matches_broadcast_score(covering: Covering) -> None:
    n_query = torch.tensor([[2.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
    n_document = torch.tensor([[8.0, 0.0, 0.0], [0.0, 4.0, 1.0], [1.0, 1.0, 1.0]])
    table = covering.pair_table(n_query, n_document)
    scored = covering(n_query.unsqueeze(1), n_document.unsqueeze(0))
    assert table.unpaid_mass.shape == (2, 3)
    assert torch.allclose(table.unpaid_mass, scored.unpaid_mass)
    assert torch.allclose(table.covering, scored.covering)
    assert torch.allclose(table.demand_l1, scored.demand_l1[:, 0])


def test_pair_tables_shards_match_full_table(covering: Covering) -> None:
    n_query = torch.tensor([[2.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
    n_document = torch.tensor([[8.0, 0.0, 0.0], [0.0, 4.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 3.0]])
    full = covering.pair_table(n_query, n_document)
    sharded = covering.pair_tables(n_query, n_document.tensor_split(2, dim=0))
    assert torch.allclose(full.unpaid_mass, sharded.unpaid_mass)
    assert torch.allclose(full.covering, sharded.covering)
    assert torch.allclose(full.demand_l1, sharded.demand_l1)


def test_unpaid_fraction_is_nan_on_empty_demand(covering: Covering) -> None:
    table = covering.pair_table(torch.zeros(2, dtype=torch.float64), torch.ones(2))
    assert torch.isnan(covering.unpaid_fraction(table)).all()


def test_unpaid_fraction_is_one_on_empty_disclosure(covering: Covering) -> None:
    table = covering.pair_table(torch.tensor([2.0, 0.0]), torch.zeros(2))
    assert torch.allclose(covering.unpaid_fraction(table), torch.ones(1, 1))
    assert torch.allclose(table.covering, torch.zeros(1, 1))


def test_compose_windows_is_not_max_or_sum(covering: Covering) -> None:
    """Residual gaps multiply. Max and sum of n are different supplies."""
    query = torch.tensor([[1.0, 1.0]])
    windows = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    owners = torch.tensor([0, 0], dtype=torch.long)
    composed = covering.compose_windows(windows, owners, 1)
    assert torch.allclose(composed, torch.tensor([[3.0, 1.0]]))
    unpaid = covering.pair_table(query, composed).unpaid_mass
    summed = covering.pair_table(query, windows.sum(dim=0, keepdim=True)).unpaid_mass
    maxed = covering.pair_table(query, windows.max(dim=0, keepdim=True).values).unpaid_mass
    assert unpaid.item() == pytest.approx(0.75)
    assert summed.item() == pytest.approx(5.0 / 6.0)
    assert maxed.item() == pytest.approx(1.0)
    assert unpaid.item() < covering.pair_table(query, windows[:1]).unpaid_mass.item()


def test_compose_windows_one_window_is_identity(covering: Covering) -> None:
    windows = torch.tensor([[2.0, 0.5, 4.0]])
    owners = torch.tensor([0], dtype=torch.long)
    assert torch.allclose(covering.compose_windows(windows, owners, 1), windows)


def test_reciprocal_score_sees_x_when_forward_saturates(covering: Covering) -> None:
    """When both documents pay the claim, extra document mass stays unpaid."""
    query = torch.tensor([8.0, 0.0])
    cited = torch.tensor([1.0e6, 0.0])
    background = torch.tensor([1.0e6, 1.0e6])
    one_way_x = covering(query, cited)
    one_way_a = covering(query, background)
    reciprocal_x = covering.reciprocal_score(query, cited)
    reciprocal_a = covering.reciprocal_score(query, background)
    assert one_way_x.unpaid_mass.item() == pytest.approx(one_way_a.unpaid_mass.item(), abs=1e-6)
    assert reciprocal_x.unpaid_mass.item() < reciprocal_a.unpaid_mass.item()


def test_compose_windows_empty_owner_stays_unpaid(covering: Covering) -> None:
    windows = torch.tensor([[4.0, 4.0]])
    owners = torch.tensor([0], dtype=torch.long)
    composed = covering.compose_windows(windows, owners, 2)
    assert torch.allclose(composed[0], windows[0])
    assert torch.allclose(composed[1], torch.zeros(2))


def test_normalized_unpaid_gap_positive_when_matching_pays(covering: Covering) -> None:
    query = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    matched = torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    gap = covering.normalized_unpaid_gap(covering.pair_table(query, matched))
    assert gap.item() == pytest.approx(0.8)
    swapped = covering.normalized_unpaid_gap(covering.pair_table(query, matched.flip(0)))
    assert swapped.item() < 0.0


def test_edge_unpaid_uses_saturation_not_count_match(covering: Covering) -> None:
    query_edges = torch.tensor([[0.0, 4.0], [0.0, 0.0]])
    document_edges = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    unpaid = covering.edge_unpaid_fraction(query_edges, document_edges)
    raw = (query_edges - document_edges).clamp(min=0.0).sum() / query_edges.sum()
    assert unpaid.item() < raw.item()
    assert unpaid.item() > 0.0
