"""Five relational covering arms, endpoint shuffles, and seam retention.

Vertex, relation-type, collapsed-endpoint, labeled-endpoint, and joint scores
share product ``Covering`` kernels. Endpoint shuffles are deterministic rolls
of the destination-slot or relation-code axis. Retention names which seam
tensors survive for later offline probes, with a per-condition byte budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch_geometric.utils import scatter

from ip_claim.collision.artefacts import write_json
from ip_claim.collision.cover import Covering, CoveringScore
from ip_claim.collision.explain import Explain
from ip_claim.ssv.covering_trace import CoveringTrace
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.soft_vocab import SoftVocabModule

CLAIM_MASK = 0
FULL_MASK = 1
RELATIONAL_CONDITIONS = (
    'matching',
    'rolled_filing',
    'endpoint_shuffle',
    'relation_label_shuffle',
    'prefix_off',
    'residual_off',
    'both_off',
    'adapter_off',
)
FUSION_PAIR = ('matching', 'endpoint_shuffle')
DECLARED_EDGE_SWEEP = (0.00312, 0.00740, 0.01752)
ENTITY_BANK = 256
RELATION_BANK = 64
_FLOAT32 = 4
_LABELED_ROW = ENTITY_BANK * ENTITY_BANK * RELATION_BANK * _FLOAT32
_COLLAPSED_ROW = ENTITY_BANK * ENTITY_BANK * _FLOAT32
_EXPLAIN = Explain()


class RelationalArms(BaseModel):
    """One condition's five covering arms plus window provenance."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    vertex: CoveringScore
    relation_type: CoveringScore
    collapsed_unpaid: Tensor
    labeled_unpaid: Tensor
    joint_covering: Tensor
    collapsed_paid: Tensor
    collapsed_residual: Tensor
    labeled_paid: Tensor
    labeled_residual: Tensor
    strongest_vertex_window: Tensor | None = None
    strongest_edge_window: Tensor | None = None
    passage_ids: tuple[str, ...] = ()
    window_owners: tuple[int, ...] = ()


class EndpointMarginals(BaseModel):
    """Declared endpoint-tensor totals used by the shuffle unit test."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    source_mass: Tensor
    dest_mass: Tensor
    relation_totals: Tensor
    edge_count: Tensor
    total_mass: Tensor
    collapsed: Tensor


# fmt: off
SEAM_RETENTION = pl.DataFrame(
    {
        'name': (
            'labeled_endpoint', 'collapsed_endpoint', 'disclosure_intensity',
            'claim_intensity', 'projected_prefix', 'token_residual',
            'adapter_delta', 'adapter_delta', 'host_text_state', 'late_assignment',
            'termhood_weights', 'overlay_state', 'compose_state', 'mixed_compose',
        ),
        'conditions': (
            list(RELATIONAL_CONDITIONS), list(RELATIONAL_CONDITIONS),
            list(RELATIONAL_CONDITIONS), list(RELATIONAL_CONDITIONS),
            list(RELATIONAL_CONDITIONS), list(RELATIONAL_CONDITIONS),
            list(FUSION_PAIR), list(RELATIONAL_CONDITIONS), list(FUSION_PAIR),
            ['matching'],
            list(RELATIONAL_CONDITIONS), list(RELATIONAL_CONDITIONS),
            list(RELATIONAL_CONDITIONS), list(RELATIONAL_CONDITIONS),
        ),
        'scope': (
            'claim_and_composed', 'claim_and_composed', 'composed', 'composed',
            'composed', 'token_mean', 'fusion_pair_token_mean', 'norms',
            'fusion_pair_token_mean', 'claim_and_composed',
            'composed', 'composed', 'composed', 'composed',
        ),
        'bytes_per_row': (
            2 * _LABELED_ROW, 2 * _COLLAPSED_ROW, ENTITY_BANK * _FLOAT32,
            ENTITY_BANK * _FLOAT32, 16 * 1024 * _FLOAT32, 1024 * _FLOAT32,
            64 * 3072 * _FLOAT32, 64 * _FLOAT32, 1024 * _FLOAT32,
            512 * ENTITY_BANK * _FLOAT32,
            512 * _FLOAT32, 64 * 1024 * _FLOAT32,
            256 * 1024 * _FLOAT32, 256 * 1024 * _FLOAT32,
        ),
        'reason': (
            'Claim demand and one document row. Per-window labeled tables are dropped.',
            'Weaker label-collapsed table kept beside the labeled tensor.',
            'Composed vertex supply for the vertex arm.',
            'Claim vertex demand.',
            'Soft-prefix table is small and shared by every counterfactual.',
            'Token-mean RMS-matched residual on every condition.',
            'Token-mean per LoRA module on matching versus endpoint shuffle only.',
            'Per-module L2 norms on every condition. Full sequences are not kept.',
            'Token-mean last-layer host state on the fusion pair only.',
            'Claim late assignment on matching only, for later linear probes.',
            'Per-token occupy weights. Dropping this seam reports empty termhood.',
            'Occupied overlay codes. Inventory and letter gates read this mass.',
            'Isolated compose slot states after graph encode.',
            'Compose states after filing identity is added.',
        ),
    },
    schema_overrides={
        'scope': pl.Enum((
            'claim_and_composed', 'composed', 'fusion_pair_token_mean',
            'norms', 'token_mean',
        )),
        'bytes_per_row': pl.UInt64,
    },
)
# fmt: on


def retention_budget_bytes() -> int:
    """Per-condition per-document ceiling implied by ``SEAM_RETENTION``."""
    return int(SEAM_RETENTION['bytes_per_row'].sum())


def _unnamed_cpu(tensor: Tensor) -> Tensor:
    names = getattr(tensor, 'names', None)
    dropped = tensor.rename(None) if names and any(names) else tensor
    return cast(Tensor, dropped.detach().cpu())


def compact_seam_tensor(name: str, value: Tensor, scope: str) -> Tensor:
    """Reduce one seam to the declared retention scope."""
    tensor = _unnamed_cpu(value)
    if scope in {'token_mean', 'fusion_pair_token_mean'} and tensor.ndim >= 4:
        return cast(Tensor, tensor.mean(dim=-2))
    if (
        scope in {'token_mean', 'fusion_pair_token_mean'}
        and name != 'adapter_delta'
        and tensor.ndim >= 3
    ):
        return cast(Tensor, tensor.mean(dim=-2))
    if scope == 'norms':
        reduced = tensor.flatten(start_dim=2).norm(dim=-1) if tensor.ndim >= 3 else tensor.norm()
        return cast(Tensor, reduced)
    return tensor


def compact_encode_seams(
    tensors: Mapping[str, Tensor],
    keep: frozenset[str] | None = None,
) -> dict[str, Tensor]:
    """Keep actor-returned seams at the first declared scope for each name.

    Names on an explicit keep list that are missing from the retention
    table still return on CPU so occupy and compose gates are not empty.
    """
    scopes = dict(
        reversed(
            tuple(
                zip(
                    SEAM_RETENTION['name'].to_list(),
                    SEAM_RETENTION['scope'].to_list(),
                    strict=True,
                )
            )
        )
    )
    return {
        name: (
            compact_seam_tensor(name, value, scopes[name])
            if name in scopes
            else _unnamed_cpu(value)
        )
        for name, value in tensors.items()
        if name in scopes or (keep is not None and name in keep)
    }


def retain_seams(
    trace: CoveringTrace,
    condition: str,
    *,
    labeled_composed: Tensor | None = None,
    collapsed_composed: Tensor | None = None,
) -> dict[str, Tensor]:
    """Keep the declared seams for ``condition`` at the declared scope."""
    replacements = {
        'labeled_endpoint': labeled_composed,
        'collapsed_endpoint': collapsed_composed,
    }
    rows = tuple(
        zip(
            SEAM_RETENTION['name'].to_list(),
            SEAM_RETENTION['conditions'].to_list(),
            SEAM_RETENTION['scope'].to_list(),
            strict=True,
        )
    )

    def compact(name: str, scope: str) -> Tensor | None:
        raw = replacements[name] if replacements.get(name) is not None else trace.tensors.get(name)
        if raw is None:
            return None
        return compact_seam_tensor(name, raw, scope)

    return {
        name: kept
        for name, conditions, scope in rows
        if condition in conditions and (kept := compact(name, scope)) is not None
    }


def write_frozen_campaign(
    output_dir: Path,
    *,
    conditions: Mapping[str, tuple[Mapping[float, RelationalArms], Mapping[str, Tensor]]],
    sweep: tuple[float, ...] = DECLARED_EDGE_SWEEP,
) -> Path:
    """Write one complete artifact directory per declared condition."""
    root = output_dir / 'frozen_campaign'
    root.mkdir(parents=True, exist_ok=True)

    def write_one(
        name: str,
        payload: tuple[Mapping[float, RelationalArms], Mapping[str, Tensor]],
    ) -> str:
        arms, seams = payload
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(dict(arms), dest / 'arms.pt')
        torch.save(dict(seams), dest / 'seams.pt')
        write_json(
            dest / 'complete.json',
            {
                'condition': name,
                'complete': True,
                'sweep': list(sweep),
                'arm_scales': list(arms),
                'seams': list(seams),
            },
        )
        return name

    written = tuple(
        write_one(name, conditions[name]) for name in RELATIONAL_CONDITIONS if name in conditions
    )
    write_json(
        root / 'campaign.json',
        {
            'conditions': list(RELATIONAL_CONDITIONS),
            'declared': len(RELATIONAL_CONDITIONS),
            'written': list(written),
            'complete': written == RELATIONAL_CONDITIONS,
            'sweep': list(sweep),
            'retention_bytes': retention_budget_bytes(),
        },
    )
    return root


def endpoint_shuffle(labeled: Tensor, *, shift: int = 1) -> Tensor:
    """Cycle destination slots. Source mass, relation totals, and support stay put."""
    return labeled.roll(int(shift), dims=-2)


def relation_label_shuffle(labeled: Tensor, *, shift: int = 1) -> Tensor:
    """Cycle relation codes. Source and destination pairs stay put."""
    return labeled.roll(int(shift), dims=-1)


def endpoint_marginals(labeled: Tensor) -> EndpointMarginals:
    """Source, destination, relation, support, and mass totals of ``W``."""
    pair = labeled.sum(dim=-1)
    return EndpointMarginals(
        source_mass=pair.sum(dim=-1),
        dest_mass=pair.sum(dim=-2),
        relation_totals=labeled.sum(dim=(-3, -2)),
        edge_count=(pair > 0).to(dtype=labeled.dtype).sum(dim=(-2, -1)),
        total_mass=pair.sum(dim=(-2, -1)),
        collapsed=pair,
    )


def endpoint_tables(
    assignment: Tensor,
    token_mask: Tensor,
    vocab: SoftVocabModule,
    inventory: Inventory,
) -> tuple[Tensor, Tensor]:
    """Labeled and collapsed endpoint tables from one late-assignment mask."""
    bundle = inventory.relation_bundle(assignment, token_mask, vocab)
    labeled = _EXPLAIN.graphs_from_bundle(
        assignment,
        token_mask,
        bundle,
        keep_relation=True,
        relation_bank=int(vocab.relation_bank_size),
    )
    collapsed = _EXPLAIN.graphs_from_bundle(assignment, token_mask, bundle)
    return labeled, collapsed


def owner_mass_rows(values: Tensor, owners: Tensor, n_docs: int) -> Tensor:
    """One row per owner: the window with the largest total mass, earliest on ties."""
    mass = values.reshape(values.size(0), -1).sum(dim=-1)
    owner = owners.to(dtype=torch.long)
    best = mass.new_full((n_docs,), float('-inf'))
    best = best.scatter_reduce(0, owner, mass, reduce='amax')
    idx = torch.arange(values.size(0), device=values.device, dtype=torch.long)
    picked = torch.where(mass == best[owner], idx, idx.new_full(idx.shape, values.size(0)))
    row = idx.new_full((n_docs,), values.size(0))
    row = row.scatter_reduce(0, owner, picked, reduce='amin')
    valid = row < values.size(0)
    safe = row.clamp(max=max(values.size(0) - 1, 0))
    gathered = values[safe] if values.size(0) else values
    mask = valid.reshape((n_docs, *([1] * max(values.ndim - 1, 0))))
    return torch.where(mask, gathered, torch.zeros_like(gathered))


def strongest_payment_window(
    demand: Tensor,
    window_supply: Tensor,
    covering: Covering,
    *,
    sigma: Tensor | None = None,
    owners: Tensor | None = None,
    n_docs: int | None = None,
) -> Tensor:
    """Window index of the largest per-cell payment.

    Without owners this is the argmax over the supplied window stack. With
    owners it is one index per document, still pointing into that stack.
    Tied payments keep the earliest window. A document with no windows
    receives the stack length.
    """
    scale = covering.sigma if sigma is None else sigma
    paid = covering.presence(window_supply, scale) * (demand if owners is None else demand[owners])
    if owners is None:
        return cast(Tensor, paid.movedim(0, -1).argmax(dim=-1))
    docs = int(n_docs if n_docs is not None else int(owners.max().item()) + 1)
    ids = owners.to(dtype=torch.long)
    best = scatter(paid, ids, dim=0, dim_size=docs, reduce='max')
    sentinel = float(paid.size(0))
    grouped = (-1,) + (1,) * (paid.ndim - 1)
    index = ids.reshape(grouped).expand_as(paid)
    win = torch.arange(paid.size(0), device=paid.device, dtype=paid.dtype).reshape(grouped)
    chosen = torch.where(paid == best[ids], win, paid.new_full(paid.shape, sentinel))
    first = chosen.new_full((docs, *paid.shape[1:]), sentinel)
    return first.scatter_reduce(0, index, chosen, reduce='amin').to(dtype=torch.long)


def relational_arms(
    *,
    n_query: Tensor,
    n_document: Tensor,
    n_rel_query: Tensor,
    n_rel_document: Tensor,
    labeled_query: Tensor,
    labeled_document: Tensor,
    collapsed_query: Tensor,
    collapsed_document: Tensor,
    covering: Covering,
    window_supply: Tensor | None = None,
    window_edges: Tensor | None = None,
    window_owners: Tensor | None = None,
    passage_ids: tuple[str, ...] = (),
    sigma_edge: Tensor | None = None,
) -> RelationalArms:
    """Score the five arms and, when windows are given, name strongest payment."""
    prior = covering.sigma_edge.detach().clone()
    if sigma_edge is not None:
        covering.sigma_edge.copy_(sigma_edge.to(dtype=prior.dtype, device=prior.device))
    edge_scale = covering.sigma_edge

    def edge_fields(query: Tensor, document: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        unpaid = covering.edge_unpaid_fraction(
            query.reshape(query.size(0), -1, query.size(-1)),
            document.reshape(document.size(0), -1, document.size(-1)),
        )
        return (
            unpaid,
            covering.paid_edges(query, document),
            query * (1.0 - covering.presence(document, covering.sigma_edge)),
        )

    def strongest(
        supply: Tensor | None,
        demand: Tensor,
        sigma: Tensor | None = None,
    ) -> Tensor | None:
        return (
            None
            if supply is None
            else strongest_payment_window(
                demand,
                supply,
                covering,
                sigma=sigma,
                owners=window_owners,
                n_docs=int(demand.size(0)),
            )
        )

    try:
        vertex = covering(n_query, n_document)
        relation = covering.relation_covering(n_rel_query, n_rel_document)
        collapsed_unpaid, collapsed_paid, collapsed_residual = edge_fields(
            collapsed_query, collapsed_document
        )
        labeled_unpaid, labeled_paid, labeled_residual = edge_fields(
            labeled_query, labeled_document
        )
        scored = RelationalArms(
            vertex=vertex,
            relation_type=relation,
            collapsed_unpaid=collapsed_unpaid,
            labeled_unpaid=labeled_unpaid,
            joint_covering=vertex.covering * (1.0 - labeled_unpaid),
            collapsed_paid=collapsed_paid,
            collapsed_residual=collapsed_residual,
            labeled_paid=labeled_paid,
            labeled_residual=labeled_residual,
            strongest_vertex_window=strongest(window_supply, n_query),
            strongest_edge_window=strongest(window_edges, collapsed_query, edge_scale),
            passage_ids=passage_ids,
            window_owners=tuple(int(owner) for owner in window_owners.tolist())
            if window_owners is not None
            else (),
        )
    finally:
        covering.sigma_edge.copy_(prior)
    return scored


OCCUPANCY_TOP_N = 32
DIFFUSE_PAIR_FRACTION = 0.25
DIFFUSE_TOP_N_FRACTION = 0.10
EDGE_QUANTILES = (0.25, 0.50, 0.75)
SCALE_DERIVATION = (
    "Median of occupied labeled cells W[k,k',r] on held-out non-letter "
    'full-mask encodes. Sweep is the 25th to 75th percentile of the same '
    'cells. Document L1 is recorded and unused because presence is per-cell.'
)


class ClaimOccupancyReport(BaseModel):
    """Claim labeled-endpoint occupancy. No covering score."""

    model_config = ConfigDict(frozen=True)

    n_claims: int = Field(ge=0)
    n_nonzero: int = Field(ge=0)
    n_distinct_pairs_union: int = Field(ge=0)
    mean_occupied_pairs: float = Field(ge=0.0)
    top_n: int = Field(ge=1)
    mean_top_n_fraction: float = Field(ge=0.0, le=1.0)
    mean_top_1_fraction: float = Field(ge=0.0, le=1.0)
    mean_top_8_fraction: float = Field(ge=0.0, le=1.0)
    total_mass: float = Field(ge=0.0)
    bank: int = Field(ge=1)


class EdgeScaleReport(BaseModel):
    """Declared edge scale from held-out occupied cell masses."""

    model_config = ConfigDict(frozen=True)

    sigma_edge: float | None = Field(default=None, gt=0.0)
    sweep_low: float | None = Field(default=None, ge=0.0)
    sweep_high: float | None = Field(default=None, ge=0.0)
    n_documents: int = Field(ge=0)
    n_documents_nonzero: int = Field(ge=0)
    n_occupied_cells: int = Field(ge=0)
    quantile_p25: float | None = None
    quantile_p50: float | None = None
    quantile_p75: float | None = None
    document_l1_median: float | None = None
    derivation: str = SCALE_DERIVATION


class GateAReport(BaseModel):
    """Gate A: nonempty non-degenerate claim demand and a fixed edge scale."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    locus: str | None = None
    occupancy: ClaimOccupancyReport
    scale: EdgeScaleReport


def occupied_labeled_cells(labeled: Tensor) -> Tensor:
    """Positive finite cells of a labeled endpoint tensor."""
    flat = labeled.reshape(-1)
    return flat[torch.isfinite(flat) & (flat > 0)]


def claim_endpoint_occupancy(
    labeled: Tensor,
    *,
    top_n: int = OCCUPANCY_TOP_N,
) -> ClaimOccupancyReport:
    """Count nonzero claims, occupied pairs, and top-N mass concentration."""
    pair = labeled.sum(dim=-1)
    bank = int(pair.size(-1)) if pair.ndim >= 2 else 0
    mass = pair.reshape(pair.size(0), -1) if pair.ndim >= 2 else pair.reshape(pair.size(0), 0)
    row_mass = mass.sum(dim=-1)
    nonzero = (row_mass > 0) & torch.isfinite(row_mass)
    occupied = mass > 0
    n_pairs = occupied.to(dtype=mass.dtype).sum(dim=-1)
    n_live = int(nonzero.sum().item())
    width = int(mass.size(-1))
    span = min(max(int(top_n), 8), width)
    ranked = (
        mass.topk(span, dim=-1, sorted=True).values.cumsum(dim=-1)
        if span >= 1 and n_live
        else mass.new_zeros((mass.size(0), max(span, 0)))
    )
    idx = torch.tensor((0, min(7, span - 1), min(int(top_n) - 1, span - 1)), device=mass.device)
    pad = ranked.new_zeros((mass.size(0), 3))
    gathered = ranked.index_select(-1, idx) if n_live and span >= 1 else pad
    frac = torch.where(row_mass.unsqueeze(-1) > 0, gathered / row_mass.unsqueeze(-1), pad)
    means = frac[nonzero].mean(dim=0) if n_live else pad.new_zeros(3)
    union = int(occupied.any(dim=0).sum().item()) if mass.size(0) and width else 0
    total = float(row_mass[torch.isfinite(row_mass)].sum().item()) if mass.size(0) else 0.0
    return ClaimOccupancyReport(
        n_claims=int(labeled.size(0)),
        n_nonzero=n_live,
        n_distinct_pairs_union=union,
        mean_occupied_pairs=float(n_pairs[nonzero].mean().item()) if n_live else 0.0,
        top_n=int(top_n),
        mean_top_n_fraction=float(means[2].item()),
        mean_top_1_fraction=float(means[0].item()),
        mean_top_8_fraction=float(means[1].item()),
        total_mass=max(total, 0.0),
        bank=max(bank, 1),
    )


def edge_scale_from_labeled(labeled: Tensor) -> EdgeScaleReport:
    """Fix sigma_edge from occupied labeled cells. Document L1 is not the scale."""
    n_docs = int(labeled.size(0))
    pair = labeled.sum(dim=-1).reshape(n_docs, -1) if n_docs else labeled.new_zeros((0, 0))
    live_docs = occupied_labeled_cells(pair.sum(dim=-1))
    cells = occupied_labeled_cells(labeled)
    points = torch.tensor(EDGE_QUANTILES, device=cells.device, dtype=torch.float32)
    quantiles = torch.quantile(cells.to(dtype=torch.float32), points) if cells.numel() else None
    p25, median, p75 = (
        (float(quantiles[0].item()), float(quantiles[1].item()), float(quantiles[2].item()))
        if quantiles is not None
        else (None, None, None)
    )
    sigma = median if median is not None and median > 0.0 else None
    return EdgeScaleReport(
        sigma_edge=sigma,
        sweep_low=p25 if sigma is not None else None,
        sweep_high=p75 if sigma is not None else None,
        n_documents=n_docs,
        n_documents_nonzero=int(live_docs.numel()),
        n_occupied_cells=int(cells.numel()),
        quantile_p25=p25,
        quantile_p50=median,
        quantile_p75=p75,
        document_l1_median=float(live_docs.median().item()) if live_docs.numel() else None,
    )


def gate_a(occupancy: ClaimOccupancyReport, scale: EdgeScaleReport) -> GateAReport:
    """Fail closed on empty demand or a missing inventory scale."""
    empty = (
        occupancy.n_nonzero < 1
        or occupancy.total_mass <= 0.0
        or scale.sigma_edge is None
        or scale.n_occupied_cells < 1
    )
    area = max(occupancy.bank * occupancy.bank, 1)
    dense = occupancy.mean_occupied_pairs / area > DIFFUSE_PAIR_FRACTION
    flat = occupancy.mean_top_n_fraction < DIFFUSE_TOP_N_FRACTION
    locus = (
        'RELATION_INVENTORY_EMPTY' if empty else 'RELATION_CODE_DIFFUSE' if dense and flat else None
    )
    return GateAReport(passed=locus is None, locus=locus, occupancy=occupancy, scale=scale)


__all__ = [
    'CLAIM_MASK',
    'DECLARED_EDGE_SWEEP',
    'DIFFUSE_PAIR_FRACTION',
    'DIFFUSE_TOP_N_FRACTION',
    'EDGE_QUANTILES',
    'ENTITY_BANK',
    'FULL_MASK',
    'FUSION_PAIR',
    'OCCUPANCY_TOP_N',
    'RELATIONAL_CONDITIONS',
    'RELATION_BANK',
    'SCALE_DERIVATION',
    'SEAM_RETENTION',
    'ClaimOccupancyReport',
    'EdgeScaleReport',
    'EndpointMarginals',
    'GateAReport',
    'RelationalArms',
    'claim_endpoint_occupancy',
    'compact_encode_seams',
    'compact_seam_tensor',
    'edge_scale_from_labeled',
    'endpoint_marginals',
    'endpoint_shuffle',
    'endpoint_tables',
    'gate_a',
    'occupied_labeled_cells',
    'owner_mass_rows',
    'relation_label_shuffle',
    'relational_arms',
    'retain_seams',
    'retention_budget_bytes',
    'strongest_payment_window',
    'write_frozen_campaign',
]
