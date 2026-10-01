"""Unit tests for SSV soft-graph encode and host-side projection."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch

import ip_claim.ssv.encode as encode_mod
import ip_claim.ssv.project as project_mod
from ip_claim.ssv.config import ArchSpec, HostSpec, SsvTrainConfig
from ip_claim.ssv.encode import (
    FoundationHeteroSchema,
    SoftEncodeOutput,
    SoftGraphEncoder,
    apply_compose_convs,
    apply_hgt_convs,
    attend_soft_tokens,
    coalesce_hetero_batch,
    mean_structural_ke_loss,
    project_compose_inputs,
    project_hgt_inputs,
    project_node_x_dict,
    relation_param_key,
    score_compose_pairs,
)
from ip_claim.ssv.graph_batch import graph_batch_from_hupd_dict
from ip_claim.ssv.project import SoftProjectOutput, SoftTokenProjector
from ip_claim.ssv.soft_graph import SOFT_RELATED, SoftGraphOverlay, merge_soft_overlay

_FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'hupd'


def _tiny_config(
    *,
    n_soft: int = 4,
    gnn_hidden: int = 32,
    gnn_heads: int = 4,
    gnn_layers: int = 1,
    d_model: int = 24,
    soft_dim: int = 16,
) -> SsvTrainConfig:
    return SsvTrainConfig(
        host=HostSpec(d_model=d_model),
        arch=ArchSpec(
            n_soft_tokens=n_soft,
            gnn_hidden=gnn_hidden,
            gnn_heads=gnn_heads,
            gnn_layers=gnn_layers,
            soft_dim=soft_dim,
        ),
    )


def _fixture_graphs(*names: str):
    return tuple(
        graph_batch_from_hupd_dict(json.loads((_FIXTURES / name).read_text(encoding='utf-8'))).data
        for name in names
    )


def test_encode_single_graph_soft_token_shape() -> None:
    config = _tiny_config()
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    (graph,) = _fixture_graphs('13817165.json')
    out = encoder(graph)
    assert isinstance(out, SoftEncodeOutput)
    assert out.soft_tokens.shape == (1, config.arch.n_soft_tokens, config.arch.gnn_hidden)
    assert torch.isfinite(out.soft_tokens).all()


def test_encode_batch_graphs_soft_token_shape() -> None:
    config = _tiny_config(n_soft=6, gnn_hidden=16, gnn_heads=2)
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    graphs = _fixture_graphs('13817165.json', '14111139.json')
    out = encoder(graphs)
    assert out.soft_tokens.shape == (2, config.arch.n_soft_tokens, config.arch.gnn_hidden)
    assert torch.isfinite(out.soft_tokens).all()


def test_project_maps_to_host_d_model() -> None:
    config = _tiny_config(d_model=40, gnn_hidden=32, gnn_heads=4)
    encoder = SoftGraphEncoder(config)
    projector = SoftTokenProjector(config)
    encoder.eval()
    projector.eval()
    (graph,) = _fixture_graphs('13817165.json')
    encoded = encoder(graph)
    projected = projector(encoded.soft_tokens)
    assert isinstance(projected, SoftProjectOutput)
    assert projected.tokens.shape == (1, config.arch.n_soft_tokens, config.host.d_model)
    assert torch.isfinite(projected.tokens).all()


def test_encode_project_backprop() -> None:
    config = _tiny_config()
    encoder = SoftGraphEncoder(config)
    projector = SoftTokenProjector(config)
    encoder.train()
    projector.train()
    (graph,) = _fixture_graphs('14112715.json')
    tokens = projector(encoder(graph).soft_tokens).tokens
    loss = tokens.square().mean()
    loss.backward()
    assert encoder.queries.grad is not None
    assert torch.isfinite(encoder.queries.grad).all()
    assert projector.fc2.weight.grad is not None
    assert torch.isfinite(projector.fc2.weight.grad).all()


def test_encoder_rejects_indivisible_hidden_heads() -> None:
    config = _tiny_config(gnn_hidden=30, gnn_heads=4)
    with pytest.raises(ValueError, match='divisible'):
        SoftGraphEncoder(config)


def test_encode_batch_matches_single_graph_slices() -> None:
    """Batched readout must match per-graph solo encode (to_dense_batch ordering)."""
    config = _tiny_config(n_soft=4, gnn_hidden=32, gnn_heads=4)
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    graphs = _fixture_graphs('13817165.json', '14111139.json')
    with torch.no_grad():
        batched = encoder(graphs).soft_tokens
        solo = tuple(encoder(graph).soft_tokens[0] for graph in graphs)
    for idx, expected in enumerate(solo):
        assert torch.allclose(batched[idx], expected, atol=1e-5)


def test_encode_batch_isolates_graphs_under_neighbor_perturbation() -> None:
    """Changing graph 1 must not alter graph 0 soft tokens in the same batch."""
    config = _tiny_config(n_soft=4, gnn_hidden=32, gnn_heads=4)
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    g0, g1 = _fixture_graphs('13817165.json', '14111139.json')
    g1_perturbed = g1.clone()
    if 'cpc' in g1_perturbed.node_types and g1_perturbed['cpc'].num_nodes:
        g1_perturbed['cpc'].cpc_id = g1_perturbed['cpc'].cpc_id.clone() + 17
    with torch.no_grad():
        baseline = encoder([g0, g1]).soft_tokens
        perturbed = encoder([g0, g1_perturbed]).soft_tokens
    assert torch.allclose(baseline[0], perturbed[0], atol=1e-5)
    assert not torch.allclose(baseline[1], perturbed[1], atol=1e-5)


def test_no_lora_surface_in_encode_project_modules() -> None:
    sources = (inspect.getsource(encode_mod), inspect.getsource(project_mod))
    assert all('peft' not in source for source in sources)
    assert all('LoraConfig' not in source for source in sources)
    assert all('get_peft_model' not in source for source in sources)


def test_encoder_schema_keeps_soft_out_of_hgt() -> None:
    config = _tiny_config()
    encoder = SoftGraphEncoder(config)
    nodes, edges = encoder.schema.hgt_metadata()
    assert 'soft_entity' not in nodes
    assert ('soft_entity', 'soft_related', 'soft_entity') not in edges
    assert encoder.schema.compose_edge_type() == ('soft_entity', 'soft_related', 'soft_entity')
    assert 'soft_entity' in encoder.lin_dict
    assert 'soft_entity__soft_related__soft_entity' not in encoder.relation
    assert 'cpc__parent_of__cpc' in encoder.relation
    assert 'claim__depends_on__claim' in encoder.relation
    assert len(encoder.compose_convs) == config.arch.gnn_layers


def _filing_with_soft(
    *,
    soft_dim: int,
    n_soft: int = 4,
    n_edges: int = 3,
    rel: torch.Tensor | None = None,
    bank_size: int | None = None,
) -> HeteroData:
    graph = HeteroData()
    graph['cpc'].x = torch.zeros((2, 1))
    graph['cpc', 'parent_of', 'cpc'].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    graph['claim'].x = torch.zeros((2, 1))
    graph['claim', 'depends_on', 'claim'].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    overlay = SoftGraphOverlay(
        soft_x=torch.randn(n_soft, soft_dim),
        edge_index=torch.stack(
            (
                torch.arange(n_edges, dtype=torch.long),
                torch.arange(1, n_edges + 1, dtype=torch.long),
            ),
        ),
        edge_attr=torch.randn(n_edges, soft_dim) if rel is None else rel,
        code_ids=torch.arange(n_soft, dtype=torch.long),
    )
    return merge_soft_overlay(graph, overlay, bank_size=n_soft if bank_size is None else bank_size)


def test_overlay_edge_attr_matches_edge_index() -> None:
    rel = torch.arange(6, dtype=torch.float).reshape(3, 2).repeat(1, 8)[:, :16]
    graph = _filing_with_soft(soft_dim=16, rel=rel)
    assert graph[SOFT_RELATED].edge_attr.shape == (3, 16)
    assert graph[SOFT_RELATED].edge_index.shape == (2, 3)
    assert torch.equal(graph[SOFT_RELATED].edge_attr, rel)


def test_compose_relation_vectors_change_soft_not_filing() -> None:
    config = _tiny_config(soft_dim=16, gnn_hidden=32, gnn_heads=4, gnn_layers=2)
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    base_rel = torch.zeros(3, 16)
    alt_rel = torch.ones(3, 16)
    base = _filing_with_soft(soft_dim=16, rel=base_rel)
    alt = _filing_with_soft(soft_dim=16, rel=alt_rel)
    alt['cpc'].x = base['cpc'].x.clone()
    alt['claim'].x = base['claim'].x.clone()
    alt['soft_entity'].x = base['soft_entity'].x.clone()
    alt[SOFT_RELATED].edge_index = base[SOFT_RELATED].edge_index.clone()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        x_base = project_node_x_dict(base, encoder.schema, encoder.lin_dict, device)
        x_alt = project_node_x_dict(alt, encoder.schema, encoder.lin_dict, device)
        hgt_base = apply_hgt_convs(
            project_hgt_inputs(base, encoder.schema, x_base, device),
            encoder.convs,
        )
        hgt_alt = apply_hgt_convs(
            project_hgt_inputs(alt, encoder.schema, x_alt, device),
            encoder.convs,
        )
        soft_base = apply_compose_convs(
            project_compose_inputs(
                base,
                encoder.schema,
                x_base['soft_entity'],
                encoder.relation_in,
                device,
            ),
            encoder.compose_convs,
        )
        soft_alt = apply_compose_convs(
            project_compose_inputs(
                alt,
                encoder.schema,
                x_alt['soft_entity'],
                encoder.relation_in,
                device,
            ),
            encoder.compose_convs,
        )
    for node_type in encoder.schema.hgt_node_types():
        assert torch.allclose(hgt_base[node_type], hgt_alt[node_type], atol=1e-5)
    assert not torch.allclose(soft_base.x, soft_alt.x, atol=1e-5)


def test_empty_soft_pairs_still_emit_tokens() -> None:
    config = _tiny_config()
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    graph = HeteroData()
    graph['cpc'].x = torch.zeros((1, 1))
    graph['claim'].x = torch.zeros((1, 1))
    overlay = SoftGraphOverlay(
        soft_x=torch.randn(3, config.arch.soft_dim),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, config.arch.soft_dim)),
        code_ids=torch.arange(3, dtype=torch.long),
    )
    merged = merge_soft_overlay(graph, overlay, bank_size=int(overlay.soft_x.size(0)))
    out = encoder(merged)
    assert out.soft_tokens.shape == (1, config.arch.n_soft_tokens, config.arch.gnn_hidden)
    assert torch.isfinite(out.soft_tokens).all()
    assert float(out.compose_scores.compose_mrr) == pytest.approx(0.0)
    assert float(out.compose_scores.compose_transe) == pytest.approx(0.0)


def test_compose_stack_uses_gnn_layers() -> None:
    config_one = _tiny_config(gnn_layers=1)
    config_two = _tiny_config(gnn_layers=2)
    one = SoftGraphEncoder(config_one)
    two = SoftGraphEncoder(config_two)
    assert len(one.compose_convs) == 1
    assert len(two.compose_convs) == 2
    graph = _filing_with_soft(soft_dim=config_one.arch.soft_dim)
    one.eval()
    two.eval()
    with torch.no_grad():
        tokens_one = one(graph).soft_tokens
        tokens_two = two(graph).soft_tokens
    assert tokens_one.shape == tokens_two.shape
    assert torch.isfinite(tokens_one).all()
    assert torch.isfinite(tokens_two).all()


def test_score_compose_pairs_perfect_transe_is_hits_at_1() -> None:
    states = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    relations = torch.tensor([[1.0, 0.0]])
    batch_index = torch.zeros(3, dtype=torch.long)
    report = score_compose_pairs(states, edge_index, relations, batch_index)
    assert float(report.compose_transe) == pytest.approx(0.0)
    assert float(report.compose_mrr) == pytest.approx(1.0)
    assert float(report.compose_mr) == pytest.approx(1.0)
    assert float(report.compose_hits_at_1) == pytest.approx(1.0)
    assert float(report.compose_hits_at_3) == pytest.approx(1.0)
    assert float(report.compose_hits_at_10) == pytest.approx(1.0)


def test_score_compose_pairs_filters_other_graphs() -> None:
    states = torch.tensor([[0.0, 0.0], [10.0, 0.0], [1.0, 0.0]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    relations = torch.tensor([[1.0, 0.0]])
    batch_index = torch.tensor([0, 0, 1], dtype=torch.long)
    report = score_compose_pairs(states, edge_index, relations, batch_index)
    assert float(report.compose_mr) == pytest.approx(2.0)
    leaked = score_compose_pairs(states, edge_index, relations, torch.zeros(3, dtype=torch.long))
    assert float(leaked.compose_mr) == pytest.approx(3.0)
    home_ids = torch.zeros(2, dtype=torch.long)
    only_home = score_compose_pairs(states[:2], edge_index, relations, home_ids)
    assert float(report.compose_mr) == pytest.approx(float(only_home.compose_mr))


def test_encode_reports_finite_compose_scores() -> None:
    config = _tiny_config()
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    graph = _filing_with_soft(soft_dim=config.arch.soft_dim)
    with torch.no_grad():
        out = encoder(graph)
    scores = out.compose_scores
    assert torch.isfinite(scores.compose_transe)
    assert torch.isfinite(scores.compose_mrr)
    assert 1.0 <= float(scores.compose_mr) <= 4.0
    assert 0.0 <= float(scores.compose_hits_at_1) <= 1.0


def test_text_query_conditions_soft_tokens() -> None:
    config = _tiny_config()
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    (graph,) = _fixture_graphs('13817165.json')
    d_model = int(config.host.d_model)
    with torch.no_grad():
        baseline = encoder(graph).soft_tokens
        omitted = encoder(graph, text_query=None).soft_tokens
        zero_init = encoder(graph, text_query=torch.zeros(1, d_model)).soft_tokens
    assert torch.allclose(baseline, omitted, atol=1e-5)
    assert torch.allclose(baseline, zero_init, atol=1e-5)
    with torch.no_grad():
        encoder.film_gamma.weight.fill_(0.1)
        encoder.film_beta.weight.fill_(0.1)
        first = encoder(graph, text_query=torch.zeros(1, d_model)).soft_tokens
        second = encoder(graph, text_query=torch.ones(1, d_model)).soft_tokens
    assert not torch.allclose(first, second, atol=1e-5)


def test_encode_exposes_dense_compose_node_states() -> None:
    """Dense node states are the RMS-matched CPC mix; readout keys those mixed rows."""
    config = _tiny_config(soft_dim=16, gnn_hidden=32, gnn_heads=4)
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    n_nodes = (4, 3)
    graphs = tuple(
        _filing_with_soft(soft_dim=config.arch.soft_dim, n_soft=count, n_edges=max(count - 1, 0))
        for count in n_nodes
    )
    device = next(encoder.parameters()).device
    with torch.no_grad():
        out = encoder(graphs)
        batch = coalesce_hetero_batch(graphs, seed_type=encoder.schema.node_types[0]).to(device)
        encoder.fill_filing_features(batch, device)
        x_all = project_node_x_dict(batch, encoder.schema, encoder.lin_dict, device)
        inputs = project_hgt_inputs(batch, encoder.schema, x_all, device)
        filing_states = apply_hgt_convs(inputs, encoder.convs)
        compose = apply_compose_convs(
            project_compose_inputs(
                batch,
                encoder.schema,
                x_all[encoder.schema.compose_node_type()],
                encoder.relation_in,
                device,
            ),
            encoder.compose_convs,
        )
        store = batch[encoder.schema.compose_node_type()]
        graph_ids = getattr(store, 'batch', None)
        batch_index = (
            torch.zeros(compose.x.size(0), dtype=torch.long, device=device)
            if graph_ids is None
            else graph_ids.to(device)
        )
        graph_count = int(batch.num_graphs) if hasattr(batch, 'num_graphs') else 1

        def rms(values: torch.Tensor) -> torch.Tensor:
            return (
                values
                .square()
                .mean(dim=-1, keepdim=True)
                .sqrt()
                .clamp_min(torch.finfo(values.dtype).eps)
            )

        cpc_states = filing_states['cpc']
        raw_cpc_batch = getattr(batch['cpc'], 'batch', None)
        cpc_batch = (
            torch.zeros(cpc_states.size(0), dtype=torch.long, device=device)
            if raw_cpc_batch is None
            else raw_cpc_batch.to(device)
        )
        dense, cpc_mask = to_dense_batch(cpc_states, cpc_batch, batch_size=graph_count)
        present = cpc_mask.to(dtype=dense.dtype).unsqueeze(-1)
        pooled = (dense * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)
        identity = encoder.cpc_to_compose(pooled)[batch_index]
        has_cpc = present.sum(dim=1).squeeze(-1)[batch_index].gt(0).unsqueeze(-1)
        aligned = identity * (rms(compose.x) / rms(identity))
        mixed = compose.x + torch.where(has_cpc, aligned, aligned.new_zeros(()))
        expected_states, expected_mask = to_dense_batch(
            mixed,
            batch_index,
            batch_size=graph_count,
        )
        compose_scores = score_compose_pairs(
            compose.x,
            compose.edge_index,
            compose.edge_attr,
            batch_index,
        )
        ke_states = {**filing_states, encoder.schema.compose_node_type(): compose.x}
        attend_states = {**filing_states, encoder.schema.compose_node_type(): mixed}
        soft_tokens = attend_soft_tokens(
            attend_states,
            batch,
            encoder.schema,
            encoder.queries,
            encoder.readout,
            device,
            None,
        )
        ke_loss = mean_structural_ke_loss(
            ke_states,
            inputs.edge_index_dict,
            encoder.relation,
            encoder.schema,
            device,
        )
    assert out.node_states.shape == (2, max(n_nodes), config.arch.gnn_hidden)
    assert out.node_mask.shape == (2, max(n_nodes))
    assert out.node_mask.dtype == torch.bool
    assert out.batch_index.shape == (sum(n_nodes),)
    assert int(out.node_mask.sum()) == sum(n_nodes)
    assert torch.equal(out.node_states, expected_states)
    assert torch.equal(out.node_mask, expected_mask)
    assert torch.equal(out.batch_index, batch_index)
    assert torch.equal(out.soft_tokens, soft_tokens)
    assert torch.equal(out.ke_loss, ke_loss)
    assert torch.equal(out.compose_scores.compose_transe, compose_scores.compose_transe)
    assert torch.equal(out.compose_scores.compose_mrr, compose_scores.compose_mrr)
    assert torch.equal(out.compose_scores.compose_mr, compose_scores.compose_mr)
    assert torch.equal(out.compose_scores.compose_hits_at_1, compose_scores.compose_hits_at_1)
    assert torch.equal(out.compose_scores.compose_hits_at_3, compose_scores.compose_hits_at_3)
    assert torch.equal(out.compose_scores.compose_hits_at_10, compose_scores.compose_hits_at_10)


def test_encode_empty_soft_entities_emit_empty_node_states() -> None:
    config = _tiny_config()
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    graphs = _fixture_graphs('13817165.json', '14111139.json')
    with torch.no_grad():
        out = encoder(graphs)
    hidden = int(config.arch.gnn_hidden)
    assert out.node_states.shape == (2, 0, hidden)
    assert out.node_mask.shape == (2, 0)
    assert out.batch_index.shape == (0,)
    assert out.soft_tokens.shape == (2, config.arch.n_soft_tokens, hidden)


def test_mean_structural_ke_loss_skips_empty_and_unknown_edges() -> None:
    schema = FoundationHeteroSchema()
    hidden = 4
    cpc_key = relation_param_key(('cpc', 'parent_of', 'cpc'))
    claim_key = relation_param_key(('claim', 'depends_on', 'claim'))
    relation = nn.ParameterDict({
        cpc_key: nn.Parameter(torch.ones(hidden)),
        claim_key: nn.Parameter(torch.ones(hidden)),
    })
    h_dict = {
        'cpc': torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        'claim': torch.zeros(2, hidden),
    }
    live = torch.tensor([[0], [1]])
    empty = torch.zeros((2, 0), dtype=torch.long)
    device = torch.device('cpu')
    skipped = mean_structural_ke_loss(h_dict, {}, relation, schema, device)
    assert skipped.ndim == 0
    assert torch.equal(skipped, skipped.new_zeros(()))
    one_live = mean_structural_ke_loss(
        h_dict,
        {('cpc', 'parent_of', 'cpc'): live, ('claim', 'depends_on', 'claim'): empty},
        relation,
        schema,
        device,
    )
    src = h_dict['cpc'][0]
    dst = h_dict['cpc'][1]
    rel = relation[cpc_key]
    scale = hidden**-0.5
    expected = torch.mean(torch.sum(((src + rel - dst) * scale) ** 2, dim=-1))
    assert torch.allclose(one_live, expected)
    missing_claim_states = mean_structural_ke_loss(
        {'cpc': h_dict['cpc']},
        {('cpc', 'parent_of', 'cpc'): live, ('claim', 'depends_on', 'claim'): live},
        relation,
        schema,
        device,
    )
    assert torch.allclose(missing_claim_states, expected)


def test_dest_rewire_off_occupied_slots_moves_prefix() -> None:
    """Prefix changes when pair destinations land outside the occupied set."""
    config = _tiny_config(soft_dim=16, gnn_hidden=32, gnn_heads=4)
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    bank_size = 8
    occupied_ids = torch.tensor([1, 3, 5], dtype=torch.long)
    overlay = SoftGraphOverlay(
        soft_x=torch.randn(3, 16),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.randn(3, 16),
        code_ids=occupied_ids,
    )
    graph = HeteroData()
    graph['cpc'].x = torch.zeros((2, 1))
    graph['cpc', 'parent_of', 'cpc'].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    graph['claim'].x = torch.zeros((2, 1))
    graph['claim', 'depends_on', 'claim'].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    matching = merge_soft_overlay(graph, overlay, bank_size=bank_size)
    assert int(matching['soft_entity'].num_nodes) == bank_size
    assert int(matching['soft_entity'].occupied.sum().item()) == 3
    assert matching[SOFT_RELATED].edge_index.tolist() == [[1, 3, 5], [3, 5, 1]]
    rewired = matching.clone()
    off_occupied = torch.tensor([0, 2, 4], dtype=torch.long)
    rewired[SOFT_RELATED].edge_index = torch.stack(
        (matching[SOFT_RELATED].edge_index[0], off_occupied),
    )
    with torch.no_grad():
        base = encoder(matching)
        moved = encoder(rewired)
    assert base.node_states.shape == (1, bank_size, config.arch.gnn_hidden)
    assert int(base.node_mask.sum().item()) == 3
    assert torch.isfinite(base.soft_tokens).all()
    assert torch.isfinite(moved.soft_tokens).all()
    assert not torch.allclose(base.soft_tokens, moved.soft_tokens, atol=1e-5)
