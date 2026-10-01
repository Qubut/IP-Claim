"""Contour pullback, harmonic lift, conductance, and nested tau filtration."""

from __future__ import annotations

import torch

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.explain import Explain
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.soft_vocab import SoftVocabModule
from tests._ssv_fixtures import ssv_tiny_vocab_config


def _explain() -> Explain:
    return Explain(Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0)), lift_lambda=1.0)


def test_lift_lambda_is_a_buffer() -> None:
    explain = _explain()
    assert list(explain.parameters()) == []
    names = dict(explain.named_buffers())
    assert 'lift_lambda' in names
    assert torch.allclose(names['lift_lambda'], torch.tensor(1.0))


def test_harmonic_lift_solves_the_energy_stationarity() -> None:
    explain = _explain()
    pooled = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    lifted = explain.harmonic_lift(pooled, edge_index)
    degree = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    adjacency = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    laplacian = torch.diag(degree) - adjacency
    eye = torch.eye(3, dtype=torch.float64)
    system = eye + laplacian
    assert torch.allclose(system @ lifted, pooled, atol=1e-6)

    def energy(field: torch.Tensor) -> torch.Tensor:
        residual = field - pooled
        return residual.square().sum() + field @ laplacian @ field

    noise = torch.tensor([0.2, -0.1, 0.15], dtype=torch.float64)
    assert energy(lifted).item() < energy(lifted + noise).item()
    assert energy(lifted).item() <= energy(pooled).item() + 1e-8


def test_zero_conductance_on_an_isolated_paid_block() -> None:
    explain = _explain()
    paid_edges = torch.tensor([
        [0.0, 2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 3.0, 0.0],
    ])
    isolated = torch.tensor([True, True, False, False])
    assert torch.allclose(explain.conductance(paid_edges, isolated), torch.tensor(0.0))
    leaking = paid_edges.clone()
    leaking[1, 2] = 1.0
    leaking[2, 1] = 1.0
    assert explain.conductance(leaking, isolated).item() > 0.0
    empty = torch.zeros(4, dtype=torch.bool)
    full = torch.ones(4, dtype=torch.bool)
    assert torch.allclose(explain.conductance(paid_edges, empty), torch.tensor(0.0))
    assert torch.allclose(explain.conductance(paid_edges, full), torch.tensor(0.0))


def test_paid_filtration_is_nested() -> None:
    explain = _explain()
    n_query = torch.tensor([4.0, 2.0, 0.5, 0.0])
    n_document = torch.tensor([3.0, 1.0, 0.2, 0.0])
    query_edges = torch.zeros(4, 4)
    document_edges = torch.zeros(4, 4)
    tight = explain.community(n_query, n_document, query_edges, document_edges, tau=1.0)
    loose = explain.community(n_query, n_document, query_edges, document_edges, tau=0.2)
    assert bool((~tight.paid_mask | loose.paid_mask).all())
    assert tight.paid_mask.sum().item() <= loose.paid_mask.sum().item()
    scored = explain.covering(n_query, n_document)
    assert tight.paid_mask.tolist() == (scored.paid >= 1.0).tolist()


def test_slot_keep_contour_still_integrates_to_unpaid() -> None:
    explain = Explain(Covering(CoveringKnobs(slot_mass_keep=0.35)), lift_lambda=1.0)
    assignment_query = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.2, 0.4, 0.4],
    ])
    assignment_document = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.2, 0.8],
    ])
    query_mask = torch.tensor([1.0, 1.0, 0.0])
    n_query = explain.covering.masked_intensity(assignment_query, query_mask)
    n_document = explain.covering.masked_intensity(assignment_document, torch.ones(3))
    before = assignment_query.clone()
    contour = explain.contour(
        assignment_query,
        assignment_document,
        n_query,
        n_document,
        query_mask,
        tau=0.3,
    )
    scored = explain.covering(n_query, n_document)
    ungated = explain.covering.query_unpaid_field(assignment_query, n_document)
    document_mask = torch.ones(3)
    assert torch.equal(assignment_query, before)
    assert torch.allclose((query_mask * contour.query_field).sum(), scored.unpaid_mass)
    assert torch.allclose((document_mask * contour.document_field).sum(), scored.paid.sum())
    assert not torch.allclose((query_mask * ungated).sum(), scored.unpaid_mass)
    assert int((explain.covering.keep_slots(n_query) > 0).sum().item()) == 1
    assert torch.allclose(
        contour.alignment,
        explain.covering.alignment(
            explain.covering.keep_rows(assignment_query),
            explain.covering.keep_rows(assignment_document),
            scored.paid,
        ),
    )


def test_community_vertex_mass_matches_contour_paid_slots() -> None:
    explain = _explain()
    assignment_query = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])
    assignment_document = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    query_mask = torch.tensor([1.0, 1.0, 0.0])
    document_mask = torch.ones(3)
    n_query = explain.covering.masked_intensity(assignment_query, query_mask)
    n_document = explain.covering.masked_intensity(assignment_document, document_mask)
    query_edges = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
    document_edges = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    contour, community = explain(
        assignment_query,
        assignment_document,
        n_query,
        n_document,
        query_edges,
        document_edges,
        query_mask,
        tau=0.0,
    )
    scored = explain.covering(n_query, n_document)
    assert torch.allclose((query_mask * contour.query_field).sum(), scored.unpaid_mass)
    assert torch.allclose((document_mask * contour.document_field).sum(), scored.paid.sum())
    assert torch.allclose(community.paid_demand, scored.paid.sum())
    assert torch.allclose(
        community.paid_demand,
        scored.paid[community.paid_mask].sum(),
    )
    assert torch.allclose(
        contour.document_field,
        explain.covering.document_paying_field(assignment_document, n_query, n_document),
    )
    assert torch.allclose(
        community.unpaid_edge,
        explain.covering.edge_unpaid_fraction(query_edges, document_edges),
    )


def test_slot_adjacency_scatters_pair_mass_onto_occupied_codes() -> None:
    explain = _explain()
    pair_index = torch.tensor([[0], [1]])
    pair_mass = torch.tensor([3.0])
    adjacency = explain.slot_adjacency(
        pair_index,
        pair_mass,
        torch.tensor([0, 1]),
        bank=2,
        device=pair_mass.device,
        dtype=pair_mass.dtype,
    )
    assert torch.allclose(adjacency, torch.tensor([[0.0, 3.0], [0.0, 0.0]]))


def test_graphs_from_bundle_rebuilds_w_from_late_pairs() -> None:
    vocab = SoftVocabModule(ssv_tiny_vocab_config())
    last_layer = torch.randn(1, 6, 32)
    attention = torch.ones(1, 6)
    inventory = Inventory(occupied_floor=0.0)
    assignment, _ = vocab.soft_assign(last_layer)
    bundle = inventory.relation_bundle(assignment, attention, vocab)
    explain = Explain(Covering(CoveringKnobs()))
    graphs = explain.graphs_from_bundle(assignment, attention, bundle)
    labeled = explain.graphs_from_bundle(
        assignment,
        attention,
        bundle,
        keep_relation=True,
        relation_bank=4,
    )
    assert graphs.shape == (1, 8, 8)
    assert labeled.shape == (1, 8, 8, 4)
    assert graphs.sum().item() >= 0.0
    torch.testing.assert_close(graphs, labeled.sum(dim=-1))


def test_level_sets_are_maximal_runs() -> None:
    explain = _explain()
    field = torch.tensor([0.1, 0.8, 0.9, 0.2, 0.7, 0.0])
    assert explain.level_sets(field, 0.5) == ((1, 3), (4, 5))
    assert explain.level_sets(field, 0.95) == ()
