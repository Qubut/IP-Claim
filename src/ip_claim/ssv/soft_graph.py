"""Occupied-code overlay: tables, then HeteroData at the merge boundary.

Stages, in order:

1. Leftover-unpaid-sufficient occupy from numbered-claim demand. Empty
   demand is empty occupy. There is no top-M integer rewrite.
2. ``SoftPairBatch`` as the complete directed candidate graph over those rows.
3. Relation-bank scores on that pair table, including consumed refuse.
4. ``SoftGraphOverlay`` rows as the per-document projection of kept typed pairs.

Compose states pull back onto tokens by assignment mass on each kept code.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from returns.maybe import Maybe
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from ip_claim.ssv.soft_vocab import SoftVocabModule


class OccupiedCodes(BaseModel):
    """Padded occupied entity-bank rows for one batch."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    features: Tensor
    """Bank rows ``(B, M, soft_dim)``, zeroed where ``keep`` is false."""

    code_ids: Tensor
    """Entity-bank indices ``(B, M)``."""

    keep: Tensor
    """Live occupied columns ``(B, M)``."""

    lengths: Tensor
    """Occupied count per document ``(B,)``."""


class OccupySelect(BaseModel):
    """Leftover-unpaid-sufficient occupy. Width is per-filing |S*|."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    occupied: OccupiedCodes
    scientific_width: Tensor
    """Leftover-unpaid-sufficient width per document ``(B,)``."""


class SoftPairBatch(BaseModel):
    """Flattened directed pairs across a document batch."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    heads: Tensor
    """Head occupied-code features ``(P, soft_dim)``."""

    tails: Tensor
    """Tail occupied-code features ``(P, soft_dim)``."""

    pair_index: Tensor
    """Global ``[2, P]`` indices into the concatenated occupied rows."""

    pair_counts: tuple[int, ...]
    """Pairs contributed by each document."""

    node_counts: tuple[int, ...]
    """Occupied-code nodes contributed by each document."""


class SoftGraphOverlay(BaseModel):
    """One document's occupied-code nodes, pair edges, and relation vectors."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    soft_x: Tensor
    """Occupied entity-bank rows ``(N, soft_dim)``."""

    edge_index: Tensor
    """Soft-related edges ``[2, E]`` over local occupied-code indices."""

    edge_attr: Tensor
    """Relation vectors ``(E, soft_dim)`` aligned with ``edge_index`` columns."""

    code_ids: Tensor
    """Entity-bank row ids ``(N,)`` aligned with ``soft_x``."""


class SoftRelationBundle(BaseModel):
    """Soft relation assignment outputs for one training step."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    overlays: tuple[SoftGraphOverlay, ...]
    relation_assignment: Tensor | None = None
    soft_relations: Tensor | None = None
    relation_div_loss: Tensor | None = None
    mean_relation_entropy: Tensor | None = None
    relation_batch_usage: Tensor | None = None
    soft_ke_loss: Tensor
    pair_count: int = Field(ge=0)
    n_relation_restarts: Tensor | None = None
    """Relation bank rows replaced this step; null when no pairs were scored."""

    relation_occupancy: Tensor | None = None
    """Pre-restart fraction of relation slots at or above ``utilization_eps``."""

    n_relation_dead_before_restart: Tensor | None = None
    """Relation rows below ``utilization_eps`` before this step's restart."""

    scientific_width: Tensor | None = None
    """Leftover-unpaid-sufficient occupy width for this bundle."""


class PairScores(BaseModel):
    """Relation-bank scores on one pair table."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    relation_vectors: Tensor
    assignment: Tensor | None = None
    diversity: Tensor | None = None
    entropy: Tensor | None = None
    usage: Tensor
    ke_loss: Tensor
    pair_features: Tensor


SOFT_ENTITY: NodeType = 'soft_entity'
SOFT_RELATED: EdgeType = (SOFT_ENTITY, 'soft_related', SOFT_ENTITY)


def leftover_unpaid_occupy_mask(
    occupancy: Tensor,
    demand: Tensor,
    pair_mass: Tensor,
    mass_floor: float,
    living: Tensor | None = None,
) -> Tensor:
    """Boolean leftover-unpaid-sufficient occupy support.

    Ingress is occupancy above the mass floor. A slot stays when it is
    demanded or is an endpoint of a kept or living pair that addend-pays
    a demanded slot. Identity pairs add nothing. Leftover unpaid stays
    on Covering.
    """

    def demanded_slots() -> Tensor:
        batch, slots = occupancy.shape
        match demand.ndim:
            case 1:
                return (demand > 0).unsqueeze(0).expand(batch, slots)
            case 2:
                if demand.shape == occupancy.shape:
                    return demand > 0
                return demand.gt(0).any(dim=0).unsqueeze(0).expand(batch, slots)
            case 3:
                return demand.gt(0).any(dim=-2)
            case _:
                msg = 'numbered-claim demand must be slots, claims-by-slots, or batched'
                raise ValueError(msg)

    ingress = occupancy > mass_floor
    demanded = demanded_slots()
    composed = pair_mass if living is None else pair_mass + living
    if composed.ndim == 2:
        composed = composed.expand(occupancy.size(0), -1, -1)
    identity = torch.eye(
        occupancy.size(-1),
        dtype=torch.bool,
        device=occupancy.device,
    )
    paying = (composed > 0) & ~identity
    pays_demand = paying & (demanded.unsqueeze(-1) | demanded.unsqueeze(-2))
    endpoint = pays_demand.any(dim=-1) | pays_demand.any(dim=-2)
    return ingress & (demanded | endpoint)


def pack_occupied_support(
    bank: Tensor,
    occupancy: Tensor,
    live: Tensor,
) -> OccupiedCodes:
    """Pack every leftover-unpaid-sufficient slot. No integer rewrite."""
    batch = occupancy.size(0)
    width = int(live.sum(dim=-1).max().item()) if batch > 0 and bool(live.any()) else 0
    if batch == 0 or width < 1:
        device = occupancy.device
        dim = int(bank.size(-1))
        return OccupiedCodes(
            features=bank.new_zeros((batch, 0, dim)),
            code_ids=torch.zeros((batch, 0), dtype=torch.long, device=device),
            keep=torch.zeros((batch, 0), dtype=torch.bool, device=device),
            lengths=torch.zeros(batch, dtype=torch.long, device=device),
        )
    scores = occupancy.masked_fill(~live, occupancy.new_full((), float('-inf')))
    top_mass, code_ids = scores.topk(width, dim=-1)
    keep = torch.isfinite(top_mass) & live.gather(-1, code_ids)
    features = bank[code_ids] * keep.unsqueeze(-1).to(dtype=bank.dtype)
    return OccupiedCodes(
        features=features,
        code_ids=code_ids,
        keep=keep,
        lengths=keep.sum(dim=-1),
    )


def select_leftover_unpaid_occupy(
    bank: Tensor,
    occupancy: Tensor,
    *,
    demand: Tensor,
    pair_mass: Tensor,
    mass_floor: float,
    living: Tensor | None = None,
) -> OccupySelect:
    """Occupy leftover-unpaid-sufficient slots. Width is |S*| per filing."""
    star = leftover_unpaid_occupy_mask(
        occupancy,
        demand,
        pair_mass,
        mass_floor,
        living,
    )
    return OccupySelect(
        occupied=pack_occupied_support(bank, occupancy, star),
        scientific_width=star.sum(dim=-1),
    )


def complete_directed_pairs(occupied: OccupiedCodes) -> SoftPairBatch:
    """Every ordered pair of distinct occupied codes in each document."""
    features = occupied.features
    lengths = occupied.lengths
    batch, width, dim = features.shape
    device = features.device
    node_counts = tuple(int(n) for n in lengths.tolist())
    empty = SoftPairBatch(
        heads=features.new_zeros((0, dim)),
        tails=features.new_zeros((0, dim)),
        pair_index=torch.zeros((2, 0), dtype=torch.long, device=device),
        pair_counts=tuple(0 for _ in node_counts),
        node_counts=node_counts,
    )
    if batch == 0 or width < 2:
        return empty
    idx = torch.arange(width, device=device)
    src = idx.view(1, width, 1).expand(batch, width, width)
    dst = idx.view(1, 1, width).expand(batch, width, width)
    span = lengths.view(batch, 1, 1)
    valid = (src < span) & (dst < span) & (src != dst)
    batch_idx, src_i, dst_i = valid.nonzero(as_tuple=True)
    offsets = lengths.cumsum(dim=0) - lengths
    return SoftPairBatch(
        heads=features[batch_idx, src_i],
        tails=features[batch_idx, dst_i],
        pair_index=torch.stack((src_i + offsets[batch_idx], dst_i + offsets[batch_idx]), dim=0),
        pair_counts=tuple(int(n) for n in valid.flatten(1).sum(dim=-1).tolist()),
        node_counts=node_counts,
    )


def project_overlays(
    occupied: OccupiedCodes,
    pairs: SoftPairBatch,
    relation_vectors: Tensor,
) -> tuple[SoftGraphOverlay, ...]:
    """Project kept typed pairs onto per-document overlay rows."""
    if occupied.features.size(0) == 0:
        return ()
    pair_counts = list(pairs.pair_counts)
    node_splits = occupied.lengths.tolist()
    device = pairs.pair_index.device
    node_counts = torch.as_tensor(pairs.node_counts, dtype=torch.long, device=device)
    starts = node_counts.cumsum(dim=0) - node_counts
    pair_doc = torch.repeat_interleave(
        torch.arange(node_counts.size(0), device=device),
        torch.as_tensor(pair_counts, dtype=torch.long, device=device),
    )
    local_index = (
        pairs.pair_index if pairs.pair_index.size(1) == 0 else pairs.pair_index - starts[pair_doc]
    )
    return tuple(
        SoftGraphOverlay(soft_x=features, edge_index=edges, edge_attr=rels, code_ids=codes)
        for features, codes, edges, rels in zip(
            torch.split(occupied.features[occupied.keep], node_splits, dim=0),
            torch.split(occupied.code_ids[occupied.keep], node_splits, dim=0),
            torch.split(local_index, pair_counts, dim=1),
            torch.split(relation_vectors, pair_counts, dim=0),
            strict=True,
        )
    )


def pullback_compose_to_tokens(
    node_states: Tensor,
    node_mask: Tensor,
    overlays: Sequence[SoftGraphOverlay],
    assignment: Tensor,
) -> Tensor:
    """Weight compose states onto tokens by assignment mass on each occupied code."""
    batch, max_nodes, hidden = node_states.shape
    tokens = int(assignment.size(1))
    if max_nodes == 0 or not overlays:
        return node_states.new_zeros((batch, tokens, hidden))
    bank = int(assignment.size(-1))
    if max_nodes == bank:
        weights = assignment * node_mask.unsqueeze(1).to(dtype=assignment.dtype)
        return torch.einsum('btk,bkh->bth', weights, node_states)
    ids = pad_sequence([overlay.code_ids for overlay in overlays], batch_first=True)
    width = int(ids.size(1))
    padded = F.pad(ids, (0, max(max_nodes - width, 0)))[:, :max_nodes]
    gathered = assignment.gather(
        -1,
        padded.unsqueeze(1).expand(batch, tokens, max_nodes).clamp(0, max(bank - 1, 0)),
    )
    weights = gathered * node_mask.unsqueeze(1).to(dtype=assignment.dtype)
    return torch.einsum('btm,bmh->bth', weights, node_states)


def merge_soft_overlay(
    gifted: HeteroData,
    overlay: SoftGraphOverlay,
    *,
    bank_size: int,
) -> HeteroData:
    """Clone filing structure and attach a bank-sized soft-entity table.

    Occupied rows sit at entity-bank indices so kept typed pair edges address
    destination slots. Consumed refuse is not an overlay row.
    """
    merged = gifted.clone()
    dim = int(overlay.soft_x.size(-1))
    table = overlay.soft_x.new_zeros((bank_size, dim))
    occupied = torch.zeros(bank_size, dtype=torch.bool, device=overlay.soft_x.device)
    codes = overlay.code_ids.to(dtype=torch.long)
    if codes.numel() > 0:
        table = table.index_copy(0, codes, overlay.soft_x)
        occupied[codes] = True
        edges = overlay.edge_index
        remapped = codes[edges] if edges.numel() else edges
    else:
        remapped = overlay.edge_index
    merged[SOFT_ENTITY].x = table
    merged[SOFT_ENTITY].num_nodes = int(bank_size)
    merged[SOFT_ENTITY].occupied = occupied
    merged[SOFT_RELATED].edge_index = remapped
    merged[SOFT_RELATED].edge_attr = overlay.edge_attr
    return merged


def build_soft_relation_bundle(
    soft_vocab: SoftVocabModule,
    assignment: Tensor,
    token_mask: Tensor,
    *,
    mass_floor: float = 0.0,
    demand: Tensor | None = None,
    pair_mass: Tensor | None = None,
    living: Tensor | None = None,
) -> SoftRelationBundle:
    """Score occupied-code candidates and emit overlays of kept typed pairs.

    Numbered-claim demand selects leftover-unpaid-sufficient occupy. Empty
    demand is empty occupy. There is no top-M integer rewrite.
    """
    occupancy = soft_vocab.masked_intensity(assignment, token_mask)
    empty_pairs = occupancy.new_zeros(
        occupancy.size(0),
        occupancy.size(-1),
        occupancy.size(-1),
    )
    selected = select_leftover_unpaid_occupy(
        soft_vocab.entity_bank,
        occupancy,
        demand=occupancy.new_zeros(occupancy.shape) if demand is None else demand,
        pair_mass=empty_pairs if pair_mass is None else pair_mass,
        mass_floor=mass_floor,
        living=living,
    )
    occupied = selected.occupied
    candidates = complete_directed_pairs(occupied)
    dim = int(soft_vocab.entity_bank.size(-1))

    def empty_scores(pair_features: Tensor) -> PairScores:
        return PairScores(
            relation_vectors=occupancy.new_zeros((0, dim)),
            usage=occupancy.new_zeros(soft_vocab.relation_bank_size),
            ke_loss=soft_vocab.bank_anchor(),
            pair_features=pair_features,
        )

    def score() -> tuple[SoftPairBatch, PairScores]:
        if candidates.heads.size(0) == 0:
            return candidates, empty_scores(occupancy.new_zeros((0, dim)))
        pair_feat = soft_vocab.pair_features(candidates.heads, candidates.tails)
        scored_pairs = soft_vocab.soft_assign_relations(pair_feat)
        keep = ~scored_pairs.is_consumed()
        device = candidates.pair_index.device
        n_docs = len(candidates.pair_counts)
        pair_doc = torch.repeat_interleave(
            torch.arange(n_docs, device=device),
            torch.as_tensor(candidates.pair_counts, dtype=torch.long, device=device),
        )
        pairs = SoftPairBatch(
            heads=candidates.heads[keep],
            tails=candidates.tails[keep],
            pair_index=candidates.pair_index[:, keep],
            pair_counts=tuple(
                int(n) for n in torch.bincount(pair_doc[keep], minlength=n_docs).tolist()
            ),
            node_counts=candidates.node_counts,
        )
        if pairs.heads.size(0) == 0:
            return pairs, empty_scores(pair_feat)
        rel_assignment = scored_pairs.typed[keep]
        soft_rels = soft_vocab.soft_relations(rel_assignment)
        rel_div, mean_ent, rel_usage = soft_vocab.relation_diversity_loss(rel_assignment)
        soft_ke = soft_vocab.soft_transe_loss(pairs.heads, pairs.tails, soft_rels)
        return pairs, PairScores(
            relation_vectors=soft_rels,
            assignment=rel_assignment,
            diversity=rel_div,
            entropy=mean_ent,
            usage=rel_usage,
            ke_loss=soft_ke + soft_vocab.bank_anchor(),
            pair_features=pair_feat,
        )

    pairs, scored = score()
    n_restarts, rel_occ, n_dead = soft_vocab.update_relation_usage(
        scored.usage,
        scored.pair_features,
    )
    return SoftRelationBundle(
        overlays=project_overlays(occupied, pairs, scored.relation_vectors),
        relation_assignment=scored.assignment,
        soft_relations=(
            Maybe
            .from_optional(scored.assignment)
            .bind_optional(lambda _: scored.relation_vectors)
            .value_or(None)
        ),
        relation_div_loss=scored.diversity,
        mean_relation_entropy=scored.entropy,
        relation_batch_usage=(
            Maybe
            .from_optional(scored.assignment)
            .bind_optional(lambda _: scored.usage)
            .value_or(None)
        ),
        soft_ke_loss=scored.ke_loss,
        pair_count=int(pairs.heads.size(0)),
        n_relation_restarts=n_restarts,
        relation_occupancy=rel_occ,
        n_relation_dead_before_restart=n_dead,
        scientific_width=selected.scientific_width,
    )


__all__ = [
    'SOFT_ENTITY',
    'SOFT_RELATED',
    'OccupiedCodes',
    'OccupySelect',
    'PairScores',
    'SoftGraphOverlay',
    'SoftPairBatch',
    'SoftRelationBundle',
    'build_soft_relation_bundle',
    'complete_directed_pairs',
    'leftover_unpaid_occupy_mask',
    'merge_soft_overlay',
    'pack_occupied_support',
    'project_overlays',
    'pullback_compose_to_tokens',
    'select_leftover_unpaid_occupy',
]
