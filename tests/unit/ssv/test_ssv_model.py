"""Unit tests for the ``text_query`` pooling that FiLM-conditions the soft prefix.

Covers ``masked_mean_pool`` (the shared weighted-mean primitive) and
``mask_local_query`` (the per-draw, mask-position-aware pooling that fixes
``text_query``'s prior invariance to which positions a given MLM draw masked).
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from ip_claim.ssv.config import ArchSpec, HostSpec, SsvTrainConfig
from ip_claim.ssv.encode import FoundationHeteroSchema, SoftGraphEncoder
from ip_claim.ssv.model import mask_local_query, masked_mean_pool


def _empty_graph(schema: FoundationHeteroSchema) -> HeteroData:
    """A graph with zero nodes of every declared type, node counts explicit."""
    graph = HeteroData()
    for node_type in schema.node_types:
        graph[node_type].num_nodes = 0
    return graph


def test_masked_mean_pool_averages_active_positions() -> None:
    values = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [99.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = masked_mean_pool(values, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 0.0]]))


def test_masked_mean_pool_empty_mask_is_zero_not_nan() -> None:
    values = torch.randn(2, 4, 3)
    mask = torch.zeros(2, 4)
    pooled = masked_mean_pool(values, mask)
    assert torch.equal(pooled, torch.zeros(2, 3))


def test_mask_local_query_varies_with_the_masking_draw() -> None:
    """The prior ``text_query`` formula was a pure function of ``(unmasked_input_ids,
    attention_mask)`` -- identical for any two masking draws over the same document.
    ``mask_local_query`` must break that invariance.
    """
    length = 12
    clean_embeds = torch.randn(1, length, 4)
    attention_mask = torch.ones(1, length, dtype=torch.long)

    labels_a = torch.full((1, length), -100, dtype=torch.long)
    labels_a[0, 2] = 7
    labels_b = torch.full((1, length), -100, dtype=torch.long)
    labels_b[0, 9] = 7

    query_a = mask_local_query(clean_embeds, labels_a, attention_mask, window=2)
    query_b = mask_local_query(clean_embeds, labels_b, attention_mask, window=2)
    assert not torch.allclose(query_a, query_b)

    # The whole-document pool this replaces stays a function of the padding mask
    # only, so it is unchanged by which positions the draw masked -- the exact
    # defect this fix targets.
    doc_a = masked_mean_pool(clean_embeds, attention_mask)
    doc_b = masked_mean_pool(clean_embeds, attention_mask)
    assert torch.equal(doc_a, doc_b)


def test_mask_local_query_ignores_content_at_masked_positions() -> None:
    """Changing the embedding stored at a masked position must not change the
    query built for that draw: only unmasked, in-window context may feed it.
    """
    length = 10
    attention_mask = torch.ones(1, length, dtype=torch.long)
    labels = torch.full((1, length), -100, dtype=torch.long)
    labels[0, 5] = 3

    base = torch.randn(1, length, 4)
    perturbed = base.clone()
    perturbed[0, 5] = 1000.0

    query_base = mask_local_query(base, labels, attention_mask, window=2)
    query_perturbed = mask_local_query(perturbed, labels, attention_mask, window=2)
    assert torch.allclose(query_base, query_perturbed)


def test_mask_local_query_respects_the_window_radius() -> None:
    length = 8
    attention_mask = torch.ones(1, length, dtype=torch.long)
    labels = torch.full((1, length), -100, dtype=torch.long)
    labels[0, 5] = 3

    attention_mask[0, 6] = 0  # padding: outside the real sequence, must not count
    clean_embeds = torch.zeros(1, length, 2)
    clean_embeds[0, 4] = torch.tensor([1.0, 0.0])  # inside a radius-1 window
    clean_embeds[0, 0] = torch.tensor([0.0, 100.0])  # far outside any small window

    query = mask_local_query(clean_embeds, labels, attention_mask, window=1)
    assert torch.allclose(query, torch.tensor([[1.0, 0.0]]))


def test_mask_local_query_no_masked_tokens_is_zero() -> None:
    length = 6
    clean_embeds = torch.randn(1, length, 3)
    attention_mask = torch.ones(1, length, dtype=torch.long)
    labels = torch.full((1, length), -100, dtype=torch.long)
    query = mask_local_query(clean_embeds, labels, attention_mask, window=2)
    assert torch.equal(query, torch.zeros(1, 3))


def _tiny_encoder_config(d_model: int) -> SsvTrainConfig:
    return SsvTrainConfig(
        host=HostSpec(d_model=d_model),
        arch=ArchSpec(
            n_soft_tokens=4,
            gnn_hidden=16,
            gnn_heads=2,
            gnn_layers=1,
            soft_dim=8,
        ),
    )


def test_different_masking_draws_move_the_soft_prefix_once_film_is_live() -> None:
    """End-to-end: two draws over one document must yield different soft tokens
    once FiLM has learned away from its zero-init identity, closing the loop
    between ``mask_local_query`` and ``SoftGraphEncoder``'s existing FiLM path.
    """
    d_model = 6
    length = 12
    config = _tiny_encoder_config(d_model)
    encoder = SoftGraphEncoder(config)
    encoder.eval()
    graph = _empty_graph(encoder.schema)

    clean_embeds = torch.randn(1, length, d_model)
    attention_mask = torch.ones(1, length, dtype=torch.long)
    labels_a = torch.full((1, length), -100, dtype=torch.long)
    labels_a[0, 1] = 5
    labels_b = torch.full((1, length), -100, dtype=torch.long)
    labels_b[0, 10] = 5

    def text_query_for(labels: torch.Tensor) -> torch.Tensor:
        return masked_mean_pool(clean_embeds, attention_mask) + mask_local_query(
            clean_embeds, labels, attention_mask, window=2
        )

    with torch.no_grad():
        encoder.film_gamma.weight.fill_(0.1)
        encoder.film_beta.weight.fill_(0.1)
        soft_a = encoder(graph, text_query=text_query_for(labels_a)).soft_tokens
        soft_b = encoder(graph, text_query=text_query_for(labels_b)).soft_tokens
    assert not torch.allclose(soft_a, soft_b, atol=1e-6)
