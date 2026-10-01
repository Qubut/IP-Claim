"""CPC-prefix identity features: fill, probe, and in-batch permutation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from tests._ssv_fixtures import SSV_HUPD_FIXTURES, ssv_fixture_batch, ssv_smoke_module
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ip_claim.ingestion.adapters.hupd_json.paths import patent_from_hupd_path
from ip_claim.ingestion.models import Patent
from ip_claim.ssv.collate import SoftMlmBatch
from ip_claim.ssv.encode import (
    SoftGraphEncoder,
    apply_compose_convs,
    apply_hgt_convs,
    coalesce_hetero_batch,
    project_compose_inputs,
    project_hgt_inputs,
    project_node_x_dict,
    store_tensor,
)
from ip_claim.ssv.graph_batch import (
    cpc_identity_index,
    cpc_prefix_labels,
    patent_claim_blob,
)
from ip_claim.ssv.graph_prefix import measure_graph_inject_delta, measure_graph_prefix_delta
from ip_claim.ssv.host_tokenizer import batch_encoding_tensor, load_host_tokenizer
from ip_claim.ssv.model import SoftTrunkModel, mask_local_query, masked_mean_pool
from ip_claim.ssv.soft_graph import build_soft_relation_bundle, merge_soft_overlay

_FIXTURE_NAMES = ('13817165.json', '14111139.json', '14112715.json')


def _permute_cpc_ids(graphs: Sequence[HeteroData]) -> tuple[HeteroData, ...]:
    clones = tuple(graph.clone() for graph in graphs)
    stacked = torch.cat(tuple(graph['cpc'].cpc_id for graph in clones))
    rolled = stacked.roll(1)
    cursor = 0
    for graph in clones:
        count = int(graph['cpc'].num_nodes)
        graph['cpc'].cpc_id = rolled[cursor : cursor + count].clone()
        cursor += count
    return clones


def _batch_with_graphs(batch: SoftMlmBatch, graphs: tuple[HeteroData, ...]) -> SoftMlmBatch:
    return SoftMlmBatch(
        input_ids=batch.input_ids,
        unmasked_input_ids=batch.unmasked_input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        graphs=graphs,
        rho=batch.rho,
        require_supervised=batch.require_supervised,
    )


def _forward_terms(model: SoftTrunkModel, batch: SoftMlmBatch) -> tuple[float, float, float]:
    with torch.no_grad():
        out = model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
        )
    inject = measure_graph_inject_delta(model, batch)
    prefix = measure_graph_prefix_delta(model, batch)
    return float(out.dea_gap), float(inject.host_only_nll), float(prefix.host_only_nll)


def _pooled_claim_features(
    embed: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    claim_blobs: Sequence[str],
) -> Tensor:
    encoded = tokenizer(
        list(claim_blobs),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=48,
    )
    with torch.no_grad():
        ids = batch_encoding_tensor(encoded, 'input_ids')
        attention = batch_encoding_tensor(encoded, 'attention_mask')
        embeds = embed(ids)
        weights = attention.unsqueeze(-1).to(dtype=embeds.dtype)
        return embeds.mul(weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _linear_probe_accuracy(features: Tensor, labels: Tensor) -> float:
    n_classes = int(labels.max().item()) + 1
    target = F.one_hot(labels, n_classes).to(dtype=features.dtype)
    weights = torch.linalg.lstsq(features, target).solution
    preds = features.matmul(weights).argmax(dim=-1)
    return float(preds.eq(labels).to(dtype=torch.float).mean())


def _majority_accuracy(labels: Tensor) -> float:
    counts = torch.bincount(labels)
    return float(counts.max().to(dtype=torch.float) / labels.numel())


def test_encoder_schema_in_channels_follows_identity_width(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    encoder = module.model.encoder
    assert isinstance(encoder, SoftGraphEncoder)
    assert encoder.schema.in_channels == module.config.arch.cpc_identity_dim
    assert encoder.cpc_identity.num_embeddings > 1
    assert encoder.cpc_identity.embedding_dim == encoder.schema.in_channels


def test_identity_fill_writes_distinct_rows_for_distinct_prefixes(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    encoder = module.model.encoder
    encoder.eval()
    batch = ssv_fixture_batch(module)
    with torch.no_grad():
        _ = encoder(batch.graphs)
    table = encoder.cpc_identity
    ids = torch.cat(tuple(graph['cpc'].cpc_id for graph in batch.graphs))
    rows = table(ids)
    assert ids.unique().numel() >= 2
    assert not torch.allclose(rows[0], rows[1])


def test_cpc_identity_perturbation_moves_node_states_dea_and_inject(tmp_path: Path) -> None:
    """A large CPC-table blow-up must reach compose consumers; claim HGT must not."""
    module = ssv_smoke_module(tmp_path)
    model = module.model
    model.eval()
    encoder = model.encoder
    batch = ssv_fixture_batch(module)
    embed = model.host.get_input_embeddings()
    clean = embed(batch.unmasked_input_ids)
    text_query = masked_mean_pool(clean, batch.attention_mask) + mask_local_query(
        clean,
        batch.labels,
        batch.attention_mask,
        window=int(module.config.arch.mask_query_window),
    )
    model.soft_vocab.ensure_banks_seeded(clean)
    early_assign, _ = model.soft_vocab.soft_assign(clean)
    bundle = build_soft_relation_bundle(
        model.soft_vocab,
        early_assign,
        batch.attention_mask,
        mass_floor=model.soft_occupied_floor,
        demand=model.soft_vocab.masked_intensity(early_assign, batch.attention_mask),
    )
    merged = tuple(
        merge_soft_overlay(
            graph,
            overlay,
            bank_size=int(model.soft_vocab.entity_bank.size(0)),
        )
        for graph, overlay in zip(batch.graphs, bundle.overlays, strict=True)
    )

    def encode_consumers() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = next(encoder.parameters()).device
        hetero = coalesce_hetero_batch(merged, seed_type=encoder.schema.node_types[0]).to(device)
        encoder.fill_filing_features(hetero, device)
        x_all = project_node_x_dict(hetero, encoder.schema, encoder.lin_dict, device)
        hgt_inputs = project_hgt_inputs(hetero, encoder.schema, x_all, device)
        filing = apply_hgt_convs(hgt_inputs, encoder.convs)
        encoded = encoder(merged, text_query=text_query)
        dea_logits = model.dea_head(encoded.node_states, model.soft_vocab.entity_bank)
        injected = model.slot_projector(encoded.node_states).tokens
        return (
            encoded.node_states.detach().clone(),
            dea_logits.detach().clone(),
            injected.detach().clone(),
            encoded.soft_tokens.detach().clone(),
            filing['claim'].detach().clone(),
        )

    with torch.no_grad():
        baseline = encode_consumers()
        exploded = encoder.cpc_identity.weight.detach().clone()
        exploded.mul_(0.0).add_(1.0e3)
        encoder.cpc_identity.weight.copy_(exploded)
        perturbed = encode_consumers()

    node_delta = float((baseline[0] - perturbed[0]).abs().max())
    dea_delta = float((baseline[1] - perturbed[1]).abs().max())
    inject_delta = float((baseline[2] - perturbed[2]).abs().max())
    prefix_delta = float((baseline[3] - perturbed[3]).abs().max())
    claim_delta = float((baseline[4] - perturbed[4]).abs().max())
    assert node_delta > 1e-5
    assert dea_delta > 1e-5
    assert inject_delta > 1e-5
    assert prefix_delta > 1e-5
    assert claim_delta < 1e-8
    assert baseline[0].numel() > 0


def test_filing_identity_residual_rms_matches_compose(tmp_path: Path) -> None:
    """The CPC residual is RMS-matched to each compose row, including after a 1e3 blow-up."""
    module = ssv_smoke_module(tmp_path)
    model = module.model
    model.eval()
    encoder = model.encoder
    batch = ssv_fixture_batch(module)
    embed = model.host.get_input_embeddings()
    clean = embed(batch.unmasked_input_ids)
    model.soft_vocab.ensure_banks_seeded(clean)
    early_assign, _ = model.soft_vocab.soft_assign(clean)
    bundle = build_soft_relation_bundle(
        model.soft_vocab,
        early_assign,
        batch.attention_mask,
        mass_floor=model.soft_occupied_floor,
        demand=model.soft_vocab.masked_intensity(early_assign, batch.attention_mask),
    )
    merged = tuple(
        merge_soft_overlay(
            graph,
            overlay,
            bank_size=int(model.soft_vocab.entity_bank.size(0)),
        )
        for graph, overlay in zip(batch.graphs, bundle.overlays, strict=True)
    )

    def packed_residual() -> tuple[Tensor, Tensor]:
        device = next(encoder.parameters()).device
        hetero = coalesce_hetero_batch(merged, seed_type=encoder.schema.node_types[0]).to(device)
        encoder.fill_filing_features(hetero, device)
        x_all = project_node_x_dict(hetero, encoder.schema, encoder.lin_dict, device)
        compose = apply_compose_convs(
            project_compose_inputs(
                hetero,
                encoder.schema,
                x_all[encoder.schema.compose_node_type()],
                encoder.relation_in,
                device,
            ),
            encoder.compose_convs,
        )
        encoded = encoder(merged)
        occupied = store_tensor(
            hetero[encoder.schema.compose_node_type()],
            'occupied',
            missing=torch.ones(compose.x.size(0), dtype=torch.bool, device=device),
        )
        mixed = encoded.node_states[encoded.node_mask]
        residual = mixed - compose.x[occupied]
        return compose.x[occupied], residual

    def row_rms(values: Tensor) -> Tensor:
        return values.square().mean(dim=-1).sqrt()

    with torch.no_grad():
        compose_x, residual = packed_residual()
        assert compose_x.size(0) > 0
        assert torch.allclose(row_rms(residual), row_rms(compose_x), rtol=1e-4, atol=1e-5)
        exploded = encoder.cpc_identity.weight.detach().clone()
        exploded.mul_(0.0).add_(1.0e3)
        encoder.cpc_identity.weight.copy_(exploded)
        blown_compose, blown_residual = packed_residual()
        assert blown_compose.size(0) == compose_x.size(0)
        assert not torch.allclose(blown_residual, residual, atol=1e-5)
        assert torch.allclose(row_rms(blown_residual), row_rms(blown_compose), rtol=1e-4, atol=1e-5)


def test_in_batch_cpc_permutation_moves_soft_tokens(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    encoder = module.model.encoder
    encoder.eval()
    batch = ssv_fixture_batch(module)
    permuted = _permute_cpc_ids(batch.graphs)
    assert not torch.equal(
        torch.cat(tuple(graph['cpc'].cpc_id for graph in batch.graphs)),
        torch.cat(tuple(graph['cpc'].cpc_id for graph in permuted)),
    )
    with torch.no_grad():
        baseline = encoder(batch.graphs).soft_tokens
        moved = encoder(permuted).soft_tokens
    assert not torch.allclose(baseline, moved, atol=1e-5)


def test_cpc_identity_probe_and_permutation_numbers(tmp_path: Path) -> None:
    """Record probe and permutation numbers. A null effect is a valid result."""
    module = ssv_smoke_module(tmp_path)
    model = module.model
    model.eval()
    model.set_inject_scale(0.5)
    batch = ssv_fixture_batch(module)
    patents = tuple(patent_from_hupd_path(SSV_HUPD_FIXTURES / name) for name in _FIXTURE_NAMES)
    claim_blobs = tuple(patent_claim_blob(patent) for patent in patents)

    def main_subclass(patent: Patent) -> str:
        prefixes = cpc_prefix_labels(patent.classification.main_cpc or '')
        return prefixes[-1] if prefixes else '_UNC'

    subclass_names = tuple(main_subclass(patent) for patent in patents)
    name_to_row = {name: i for i, name in enumerate(dict.fromkeys(subclass_names))}
    gold = torch.tensor([name_to_row[name] for name in subclass_names], dtype=torch.long)
    tokenizer = load_host_tokenizer(module.config)
    features = _pooled_claim_features(model.host.get_input_embeddings(), tokenizer, claim_blobs)
    probe_acc = _linear_probe_accuracy(features, gold)
    majority_acc = _majority_accuracy(gold)
    shuffled = gold[torch.randperm(gold.numel(), generator=torch.Generator().manual_seed(0))]
    shuffle_acc = _linear_probe_accuracy(features, shuffled)

    identity_gap, identity_inject_nll, identity_prefix_nll = _forward_terms(model, batch)
    model.encoder.cpc_feature_source = 'depth'
    depth_gap, depth_inject_nll, depth_prefix_nll = _forward_terms(model, batch)
    model.encoder.cpc_feature_source = 'identity'
    permuted_batch = _batch_with_graphs(batch, _permute_cpc_ids(batch.graphs))
    perm_gap, perm_inject_nll, perm_prefix_nll = _forward_terms(model, permuted_batch)

    report = pl.DataFrame({
        'metric': [
            'probe_acc',
            'majority_acc',
            'shuffle_acc',
            'n_probe',
            'n_subclass',
            'identity_dea_gap',
            'depth_dea_gap',
            'perm_dea_gap',
            'identity_inject_nll',
            'depth_inject_nll',
            'perm_inject_nll',
            'identity_prefix_nll',
            'depth_prefix_nll',
            'perm_prefix_nll',
            'dea_gap_perm_minus_identity',
            'dea_gap_identity_minus_depth',
            'inject_nll_perm_minus_identity',
            'inject_nll_identity_minus_depth',
            'prefix_nll_perm_minus_identity',
            'prefix_nll_identity_minus_depth',
        ],
        'value': [
            probe_acc,
            majority_acc,
            shuffle_acc,
            float(gold.numel()),
            float(gold.unique().numel()),
            identity_gap,
            depth_gap,
            perm_gap,
            identity_inject_nll,
            depth_inject_nll,
            perm_inject_nll,
            identity_prefix_nll,
            depth_prefix_nll,
            perm_prefix_nll,
            perm_gap - identity_gap,
            identity_gap - depth_gap,
            perm_inject_nll - identity_inject_nll,
            identity_inject_nll - depth_inject_nll,
            perm_prefix_nll - identity_prefix_nll,
            identity_prefix_nll - depth_prefix_nll,
        ],
    })
    with pl.Config(tbl_rows=-1, fmt_str_lengths=48):
        print(report)
    assert gold.numel() == len(_FIXTURE_NAMES)
    for value in report['value'].to_list():
        assert math.isfinite(value)

    subclass_ids = tuple(cpc_identity_index(name) for name in subclass_names)
    assert len(set(subclass_ids)) == gold.unique().numel()
