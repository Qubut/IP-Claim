"""Offline mechanism, fusion, and letter judgment on a frozen campaign.

Reads persisted arms and seams. Does not encode, train, or retune scales.
Query-level unpaid contrasts resample queries, not cited pairs.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import bootstrap
from torch import Tensor

from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.relational_arms import (
    CLAIM_MASK,
    DECLARED_EDGE_SWEEP,
    FULL_MASK,
    RELATIONAL_CONDITIONS,
    RelationalArms,
    claim_endpoint_occupancy,
    endpoint_marginals,
    owner_mass_rows,
)
from ip_claim.collision.artefacts import write_json
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.cover import Covering, CoveringKnobs, CoveringScore
from ip_claim.collision.data.citation_pairs import (
    CitationPair,
    EpoProcessorCitationPairSource,
    split_pairs_by_query,
)
from ip_claim.collision.diagnose import (
    QueryPartnerScore,
    citation_letter,
    query_macro_unpaid_fraction_xa,
)

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42
BOOTSTRAP_ALPHA = 0.05
MECHANISM_REL_L2 = 1e-5
MARGINAL_ATOL = 1e-5
SATURATION_UNPAID = 1e-3
TRANSFORM_COSINE = 0.99
SEMANTIC_PARITY_COSINE = 0.99
ADAPTER_REL_L2 = 1e-3
PROBE_RIDGE = 1.0
PROBE_TRAIN_FRACTION = 0.8
THIRD_OUTCOME = 'ARRANGEMENT_PRESENT_AT_THIS_GRANULARITY'
_SEAM_DROP = frozenset({
    'labeled_endpoint',
    'collapsed_endpoint',
    'late_assignment',
    'host_text_state',
    'adapter_delta',
    'projected_prefix',
    'token_residual',
    'disclosure_intensity',
    'claim_intensity',
})
_SEAM_KEEP = (
    'late_assignment',
    'host_text_state',
    'adapter_delta',
    'projected_prefix',
    'token_residual',
)
_GATE_C_SEAMS = (
    'compose_state',
    'mixed_compose',
    'projected_prefix',
    'token_residual',
    'host_text_state',
    'adapter_delta',
    'late_assignment',
    'document_labeled',
)
_PREFIX_SEAM = 'projected_prefix'
_UPSTREAM_OF_PREFIX = frozenset({'compose_state', 'mixed_compose'})
MechanismLocus = Literal[
    'RELATION_INVENTORY_EMPTY',
    'RELATION_CODE_DIFFUSE',
    'ENDPOINT_BINDING_INVARIANT',
    'SHUFFLE_MARGINAL_UNPRESERVED',
    'EDGE_SCORE_SATURATED',
    'WINDOW_MASS_CONFOUND',
    'NO_NONREDUNDANT_EDGE_SIGNAL',
]
FusionLocus = Literal[
    'GRAPH_PROJECTION_LOSS',
    'TRAIN_EXPORT_CONDITION_MISMATCH',
    'ALIGNMENT_GRADIENT_REDUNDANT',
    'STRUCTURAL_KE_GRADIENT_REDUNDANT',
    'SOFT_RELATION_KE_GRADIENT_REDUNDANT',
    'FROZEN_HOST_ONLY_RESPONSE',
    'LORA_DELTA_NEGLIGIBLE',
    'LORA_GRADIENT_STARVED',
    'ENDPOINT_NOT_DECODABLE',
    'RELATION_LABEL_NOT_DECODABLE',
    'HOST_LAYER_ERASURE',
    'LATE_ASSIGNMENT_ERASURE',
    'READOUT_ONLY_ERASURE',
    'OBJECTIVE_MISMATCH',
]
_GATE_C_LOCUS: dict[str, FusionLocus] = {
    'compose_state': 'GRAPH_PROJECTION_LOSS',
    'mixed_compose': 'GRAPH_PROJECTION_LOSS',
    'projected_prefix': 'READOUT_ONLY_ERASURE',
    'token_residual': 'HOST_LAYER_ERASURE',
    'host_text_state': 'HOST_LAYER_ERASURE',
    'adapter_delta': 'FROZEN_HOST_ONLY_RESPONSE',
    'late_assignment': 'LATE_ASSIGNMENT_ERASURE',
    'document_labeled': 'READOUT_ONLY_ERASURE',
}


class FrozenReport(BaseModel):
    """Immutable JSON envelope for one frozen-campaign judgment."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        ser_json_inf_nan='null',
    )


class Interval(FrozenReport):
    """Percentile bootstrap interval of a query-level mean."""

    low: float | None = None
    high: float | None = None
    mean: float | None = None
    n_queries: int = Field(ge=0)
    excludes_zero: bool = False


class ConditionView(FrozenReport):
    """Compact tensors kept after discarding window-major seams."""

    name: str
    stems: tuple[str, ...]
    window_count: Tensor
    claim_labeled: Tensor
    document_labeled: Tensor
    claim_intensity: Tensor
    document_intensity: Tensor
    arms: dict[float, RelationalArms]
    seams: dict[str, Tensor]


class ScaleMechanism(FrozenReport):
    """One edge-scale slice of the mechanism gate."""

    sigma_edge: float
    demand_finite: bool
    supply_finite: bool
    rolled_rel_l2: float
    shuffle_cw_rel: float
    shuffle_cv_rel: float
    shuffle_cr_rel: float
    marginals_preserved: bool
    window_count_matched: bool
    edge_mass_matched: bool
    matching_unpaid_mean: float | None
    shuffled_unpaid_mean: float | None
    delta_interval: Interval
    cw_vs_cv_cosine: float | None
    cw_vs_cr_cosine: float | None
    locus: MechanismLocus | None = None


class GateBReport(FrozenReport):
    """Mechanism gate across the declared edge-scale sweep."""

    passed: bool
    locus: MechanismLocus | None = None
    occupancy_nonzero: int
    occupancy_mass: float
    scales: tuple[ScaleMechanism, ...]


class ProbeReport(FrozenReport):
    """Held-out linear decode of endpoint identity versus a label permutation."""

    accuracy: float | None = None
    permutation_accuracy: float | None = None
    n_train: int = 0
    n_test: int = 0
    above_permutation: bool = False


class SeamDelta(FrozenReport):
    """Matching versus counterfactual geometry at one retained seam."""

    name: str
    relative_l2: float | None = None
    cosine: float | None = None
    cka: float | None = None
    separates: bool = False


class GateCReport(FrozenReport):
    """Fusion gate on persisted seams. Missing train or gradient rows are loci."""

    passed: bool
    locus: FusionLocus | None = None
    probe: ProbeReport
    earliest_losing_seam: str | None = None
    seam_deltas: tuple[SeamDelta, ...]
    adapter_rel_l2: float | None = None
    adapter_off_reduces: bool = False
    train_export_present: bool = False
    gradient_cosines_present: bool = False


class LetterArm(FrozenReport):
    """Query-macro unpaid fraction by ST.14 letter, plus unused documents."""

    vertex: dict[str, float | None]
    relation_type: dict[str, float | None]
    endpoint: dict[str, float | None]
    joint: dict[str, float | None]


class LetterReport(FrozenReport):
    """X, Y, A, and random unpaid plus query-resampled X versus A intervals."""

    n_pairs: int
    n_queries_xa: int
    arms: dict[str, LetterArm]
    xa_endpoint_matching: Interval
    xa_vertex_matching: Interval
    matching_vs_rolled_endpoint: Interval
    matching_vs_shuffle_endpoint: Interval
    xa_improved: bool
    heatmap_slots: tuple[dict[str, object], ...]
    heatmap_edges: tuple[dict[str, object], ...]


class FrozenJudgment(FrozenReport):
    """Gate B, Gate C, letters, and the named third outcome when it applies."""

    campaign: str
    sweep: tuple[float, ...]
    gate_b: GateBReport
    gate_c: GateCReport
    letters: LetterReport
    third_outcome: str | None = None


def unnamed(tensor: Tensor) -> Tensor:
    """Drop named dimensions so later reshape and compare stay unnamed."""
    names = getattr(tensor, 'names', None)
    dropped = tensor.rename(None) if names and any(names) else tensor  # type: ignore[no-untyped-call]
    return cast(Tensor, dropped)


def relative_l2(left: Tensor, right: Tensor) -> float:
    """Relative Euclidean gap after flattening both tensors to the same length."""
    base = unnamed(left).reshape(-1).to(dtype=torch.float32)
    other = unnamed(right).reshape(-1).to(dtype=torch.float32)
    if base.numel() != other.numel() or base.numel() == 0:
        return float('nan')
    return float(((base - other).norm() / base.norm().clamp_min(1e-12)).item())


def cosine(left: Tensor, right: Tensor) -> float | None:
    """Cosine of two flattened tensors, or None when shapes or counts disagree."""
    a = unnamed(left).reshape(1, -1).to(dtype=torch.float32)
    b = unnamed(right).reshape(1, -1).to(dtype=torch.float32)
    if a.numel() != b.numel() or a.numel() == 0:
        return None
    return float(F.cosine_similarity(a, b, dim=-1).item())


def query_mean_interval(
    values: Tensor,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> Interval:
    """Percentile interval of the mean after resampling query rows."""
    finite = values.to(dtype=torch.float32)
    finite = finite[torch.isfinite(finite)]
    n = int(finite.numel())
    if n < 1:
        return Interval(n_queries=0)
    sample = np.asarray(finite.detach().cpu().numpy(), dtype=np.float64)
    mean = float(sample.mean())
    if n < 2:
        return Interval(
            low=mean,
            high=mean,
            mean=mean,
            n_queries=n,
            excludes_zero=(mean > 0.0) or (mean < 0.0),
        )
    result = bootstrap(
        (sample,),
        np.mean,
        n_resamples=int(draws),
        confidence_level=1.0 - float(alpha),
        method='percentile',
        rng=int(seed),
    )
    lo = float(result.confidence_interval.low)
    hi = float(result.confidence_interval.high)
    return Interval(
        low=lo,
        high=hi,
        mean=mean,
        n_queries=n,
        excludes_zero=(lo > 0.0) or (hi < 0.0),
    )


def unpaid_fraction(score: CoveringScore) -> Tensor:
    """Per-row unpaid mass over demand, NaN where demand is empty."""
    unpaid = unnamed(score.unpaid_mass).reshape(-1)
    demand = unnamed(score.demand_l1).reshape(-1)
    return torch.where(demand > 0, unpaid / demand, torch.full_like(unpaid, float('nan')))


def edge_unpaid(covering: Covering, query: Tensor, document: Tensor, sigma: float) -> Tensor:
    """Labeled-endpoint unpaid fraction at a temporary edge scale."""
    prior = covering.sigma_edge.detach().clone()
    covering.sigma_edge.copy_(query.new_tensor(sigma).to(dtype=prior.dtype, device=prior.device))
    try:
        return cast(
            Tensor,
            covering.edge_unpaid_fraction(
                query.reshape(query.size(0), -1, query.size(-1)),
                document.reshape(document.size(0), -1, document.size(-1)),
            ),
        )
    finally:
        covering.sigma_edge.copy_(prior)


def vertex_unpaid(covering: Covering, query: Tensor, document: Tensor) -> Tensor:
    """Vertex unpaid fraction from product covering on intensity rows."""
    return unpaid_fraction(covering(query, document))


def relation_unpaid(covering: Covering, query: Tensor, document: Tensor, sigma: float) -> Tensor:
    """Relation-type unpaid fraction at a temporary edge scale."""
    prior = covering.sigma_edge.detach().clone()
    covering.sigma_edge.copy_(query.new_tensor(sigma).to(dtype=prior.dtype, device=prior.device))
    try:
        return unpaid_fraction(covering.relation_covering(query, document))
    finally:
        covering.sigma_edge.copy_(prior)


def pair_indices(
    pairs: Sequence[CitationPair],
    stems: Sequence[str],
) -> tuple[tuple[CitationPair, ...], Tensor, Tensor, tuple[str, ...]]:
    """Join cited pairs onto composed document stems that carry an ST.14 letter."""
    index = {stem: offset for offset, stem in enumerate(stems) if stem}
    live = tuple(
        pair
        for pair in pairs
        if citation_letter(pair) is not None
        and pair.query_application_number in index
        and pair.partner_application_number in index
    )
    if not live:
        empty = torch.zeros(0, dtype=torch.long)
        return (), empty, empty, ()
    query = torch.tensor(tuple(index[pair.query_application_number] for pair in live))
    partner = torch.tensor(tuple(index[pair.partner_application_number] for pair in live))
    marks = tuple(str(citation_letter(pair)) for pair in live)
    return live, query, partner, marks


def query_grouped_mean(queries: Sequence[str], values: Tensor) -> Tensor:
    """Mean unpaid delta per query, sorted by query id."""
    if not queries or values.numel() == 0:
        return values.new_zeros(0)
    frame = pl.DataFrame({
        'query': tuple(queries),
        'value': values.reshape(-1).to(dtype=torch.float64).tolist(),
    })
    grouped = frame.group_by('query').agg(pl.col('value').mean()).sort('query')
    return values.new_tensor(grouped['value'].to_list(), dtype=torch.float32)


def finite_mean(values: Tensor) -> float | None:
    """Mean of finite entries, or None when the slice is empty."""
    finite = values[torch.isfinite(values)]
    return None if finite.numel() == 0 else float(finite.mean().item())


def document_stems(
    passage_ids: tuple[str, ...],
    owners: tuple[int, ...],
    n_docs: int,
) -> tuple[str, ...]:
    """Application numbers, one per composed document, from window passage ids."""
    if not passage_ids or not owners or n_docs < 1:
        return ()
    apps = tuple(item.split(':', 1)[0] for item in passage_ids)
    owner = torch.tensor(owners, dtype=torch.long)
    idx = torch.arange(len(owners), dtype=torch.long)
    first = torch.full((n_docs,), len(owners), dtype=torch.long)
    first = first.scatter_reduce(0, owner.clamp(min=0, max=n_docs - 1), idx, reduce='amin')
    return tuple(apps[int(row)] if 0 <= int(row) < len(apps) else '' for row in first.tolist())


def load_condition(root: Path, name: str) -> ConditionView:  # noqa: C901
    """Load one condition and compact window seams onto composed document rows."""

    def mask_axis(labeled: Tensor) -> Tensor:
        if labeled.ndim >= 5:
            return labeled[:, FULL_MASK] if labeled.size(1) > FULL_MASK else labeled[:, 0]
        return labeled

    def document_labeled(seams: Mapping[str, Tensor], owners: Tensor, n_docs: int) -> Tensor:
        raw = seams.get('labeled_endpoint')
        if raw is None:
            return torch.zeros(n_docs, 1, 1, 1)
        table = mask_axis(unnamed(raw))
        already = table.size(0) == n_docs and (
            owners.numel() == 0 or int(owners.max().item()) + 1 == n_docs
        )
        if already and (owners.numel() == 0 or owners.numel() == n_docs):
            return table
        if owners.numel() == 0:
            return table[:n_docs] if table.size(0) >= n_docs else table
        return owner_mass_rows(table, owners, n_docs)

    def claim_labeled(seams: Mapping[str, Tensor], n_docs: int, fallback: Tensor) -> Tensor:
        raw = seams.get('labeled_endpoint')
        if raw is None:
            return fallback
        table = unnamed(raw)
        if table.ndim >= 5 and table.size(1) > CLAIM_MASK:
            claim = table[:, CLAIM_MASK]
            if claim.size(0) == n_docs:
                return claim
        if fallback.numel():
            return fallback
        return mask_axis(table)[:n_docs]

    def composed_intensity(
        seams: Mapping[str, Tensor],
        key: str,
        n_docs: int,
        owners: Tensor,
    ) -> Tensor:
        raw = seams.get(key)
        if raw is None:
            return torch.zeros(n_docs, 1)
        value = unnamed(raw)
        if value.ndim > 2:
            value = value.reshape(value.size(0), -1)
        if value.size(0) == n_docs:
            return value
        if owners.numel() == 0:
            return value[:n_docs]
        return owner_mass_rows(value, owners, n_docs)

    def window_count(owners: tuple[int, ...], n_docs: int) -> Tensor:
        counts = torch.zeros(n_docs, dtype=torch.float32)
        if not owners or n_docs < 1:
            return counts
        ids = torch.tensor(owners, dtype=torch.long)
        ones = torch.ones(ids.size(0), dtype=torch.float32)
        return counts.scatter_add(0, ids.clamp(min=0, max=n_docs - 1), ones)

    def compact_seam(value: Tensor, n_docs: int, owners: Tensor) -> Tensor:
        if owners.numel() == 0 or value.size(0) != owners.numel() or value.size(0) == n_docs:
            return value
        return owner_mass_rows(value, owners, n_docs)

    dest = root / name
    raw_arms = torch.load(dest / 'arms.pt', map_location='cpu', weights_only=False)
    arms = {float(key): cast(RelationalArms, value) for key, value in dict(raw_arms).items()}
    seams = {
        key: unnamed(value)
        for key, value in dict(
            torch.load(dest / 'seams.pt', map_location='cpu', weights_only=False)
        ).items()
    }
    first = next(iter(arms.values()))
    n_docs = int(unnamed(first.vertex.covering).reshape(-1).size(0))
    owners = torch.tensor(first.window_owners, dtype=torch.long)
    paid = unnamed(first.labeled_paid)
    residual = unnamed(first.labeled_residual)
    claim = paid + residual
    kept = {
        **{
            key: compact_seam(value, n_docs, owners)
            for key, value in seams.items()
            if key not in _SEAM_DROP
        },
        **{key: compact_seam(seams[key], n_docs, owners) for key in _SEAM_KEEP if key in seams},
    }
    return ConditionView(
        name=name,
        stems=document_stems(first.passage_ids, first.window_owners, n_docs),
        window_count=window_count(first.window_owners, n_docs),
        claim_labeled=claim_labeled(seams, n_docs, claim),
        document_labeled=document_labeled(seams, owners, n_docs),
        claim_intensity=composed_intensity(seams, 'claim_intensity', n_docs, owners),
        document_intensity=composed_intensity(seams, 'disclosure_intensity', n_docs, owners),
        arms=arms,
        seams=kept,
    )


def load_campaign(root: Path) -> dict[str, ConditionView]:
    """Load every declared condition that has a complete marker."""
    return {
        name: load_condition(root, name)
        for name in RELATIONAL_CONDITIONS
        if (root / name / 'complete.json').is_file()
    }


def gate_b(
    campaign: Mapping[str, ConditionView],
    covering: Covering,
    pairs: Sequence[CitationPair],
    sweep: tuple[float, ...] = DECLARED_EDGE_SWEEP,
) -> GateBReport:
    """Judge occupancy, shuffle sensitivity, and query-resampled unpaid deltas."""

    def finite_positive(tensor: Tensor) -> bool:
        flat = unnamed(tensor).reshape(-1)
        live = flat[torch.isfinite(flat)]
        return bool(live.numel() and float(live.clamp_min(0).sum().item()) > 0.0)

    def marginals_hold(matching: Tensor, shuffled: Tensor) -> bool:
        before = endpoint_marginals(matching)
        after = endpoint_marginals(shuffled)
        return all((
            torch.allclose(after.source_mass, before.source_mass, atol=MARGINAL_ATOL, rtol=0.0),
            torch.allclose(
                after.relation_totals, before.relation_totals, atol=MARGINAL_ATOL, rtol=0.0
            ),
            torch.allclose(after.edge_count, before.edge_count, atol=MARGINAL_ATOL, rtol=0.0),
            torch.allclose(after.total_mass, before.total_mass, atol=MARGINAL_ATOL, rtol=0.0),
        ))

    def scale_slice(
        matching: ConditionView,
        shuffled: ConditionView,
        rolled: ConditionView | None,
        sigma: float,
    ) -> ScaleMechanism:
        match_arms = matching.arms[min(matching.arms, key=lambda key: abs(key - sigma))]
        shuffle_arms = shuffled.arms[min(shuffled.arms, key=lambda key: abs(key - sigma))]
        demand_ok = finite_positive(matching.claim_labeled)
        supply_ok = finite_positive(matching.document_labeled)
        rolled_l2 = (
            relative_l2(matching.document_labeled, rolled.document_labeled)
            if rolled is not None
            else float('nan')
        )
        live, query_idx, partner_idx, _marks = pair_indices(pairs, matching.stems)
        if live:
            q = matching.claim_labeled[query_idx]
            match_u = edge_unpaid(covering, q, matching.document_labeled[partner_idx], sigma)
            shuffle_u = edge_unpaid(covering, q, shuffled.document_labeled[partner_idx], sigma)
            cw = (shuffle_u - match_u).to(dtype=torch.float32)
            cv = (
                vertex_unpaid(
                    covering,
                    shuffled.claim_intensity[query_idx],
                    shuffled.document_intensity[partner_idx],
                )
                - vertex_unpaid(
                    covering,
                    matching.claim_intensity[query_idx],
                    matching.document_intensity[partner_idx],
                )
            ).to(dtype=torch.float32)
            cr = (
                relation_unpaid(
                    covering,
                    shuffled.claim_labeled[query_idx].sum(dim=(-3, -2)),
                    shuffled.document_labeled[partner_idx].sum(dim=(-3, -2)),
                    sigma,
                )
                - relation_unpaid(
                    covering,
                    matching.claim_labeled[query_idx].sum(dim=(-3, -2)),
                    matching.document_labeled[partner_idx].sum(dim=(-3, -2)),
                    sigma,
                )
            ).to(dtype=torch.float32)
            queries = tuple(pair.query_application_number for pair in live)
            interval = query_mean_interval(query_grouped_mean(queries, cw))
            finite = torch.isfinite(cw) & torch.isfinite(cv) & torch.isfinite(cr)
            cw_cv = cosine(cw[finite], cv[finite]) if int(finite.sum()) else None
            cw_cr = cosine(cw[finite], cr[finite]) if int(finite.sum()) else None
        else:
            match_u = unnamed(match_arms.labeled_unpaid).reshape(-1)
            shuffle_u = unnamed(shuffle_arms.labeled_unpaid).reshape(-1)
            interval = query_mean_interval(shuffle_u - match_u)
            cw_cv = cosine(
                shuffle_u - match_u,
                unpaid_fraction(shuffle_arms.vertex) - unpaid_fraction(match_arms.vertex),
            )
            cw_cr = cosine(
                shuffle_u - match_u,
                unpaid_fraction(shuffle_arms.relation_type)
                - unpaid_fraction(match_arms.relation_type),
            )
        occupancy = claim_endpoint_occupancy(matching.claim_labeled)
        empty = not demand_ok or not supply_ok or occupancy.n_nonzero < 1
        match_intensity = (
            matching.claim_intensity if matching.claim_intensity.numel() else match_arms.vertex.paid
        )
        shuffle_intensity = (
            shuffled.claim_intensity
            if shuffled.claim_intensity.numel()
            else shuffle_arms.vertex.paid
        )
        shuffle_cv = relative_l2(match_intensity, shuffle_intensity)
        shuffle_cr = relative_l2(
            matching.claim_labeled.sum(dim=(-3, -2)),
            shuffled.claim_labeled.sum(dim=(-3, -2)),
        )
        match_mean = finite_mean(match_u)
        shuffle_mean = finite_mean(shuffle_u)
        saturated = (match_mean is not None and match_mean <= SATURATION_UNPAID) and (
            shuffle_mean is not None and shuffle_mean <= SATURATION_UNPAID
        )
        windows_match = bool(
            matching.window_count.shape == shuffled.window_count.shape
            and torch.equal(matching.window_count, shuffled.window_count)
        )
        mass_match = bool(
            torch.allclose(
                endpoint_marginals(matching.document_labeled).total_mass,
                endpoint_marginals(shuffled.document_labeled).total_mass,
                atol=MARGINAL_ATOL,
                rtol=0.0,
            )
        )
        transform = (cw_cv is not None and abs(cw_cv) >= TRANSFORM_COSINE) or (
            cw_cr is not None and abs(cw_cr) >= TRANSFORM_COSINE
        )
        preserved = marginals_hold(matching.document_labeled, shuffled.document_labeled)
        rolled_silent = rolled is not None and not (rolled_l2 > MECHANISM_REL_L2)
        flags: tuple[tuple[bool, MechanismLocus], ...] = (
            (empty, 'RELATION_INVENTORY_EMPTY'),
            (saturated, 'EDGE_SCORE_SATURATED'),
            (not interval.excludes_zero, 'ENDPOINT_BINDING_INVARIANT'),
            (not preserved, 'SHUFFLE_MARGINAL_UNPRESERVED'),
            (not windows_match or not mass_match, 'WINDOW_MASS_CONFOUND'),
            (transform or rolled_silent, 'NO_NONREDUNDANT_EDGE_SIGNAL'),
        )
        return ScaleMechanism(
            sigma_edge=sigma,
            demand_finite=demand_ok,
            supply_finite=supply_ok,
            rolled_rel_l2=rolled_l2,
            shuffle_cw_rel=abs(interval.mean or 0.0),
            shuffle_cv_rel=(
                shuffle_cv if torch.isfinite(torch.tensor(shuffle_cv)) else float('nan')
            ),
            shuffle_cr_rel=(
                shuffle_cr if torch.isfinite(torch.tensor(shuffle_cr)) else float('nan')
            ),
            marginals_preserved=preserved,
            window_count_matched=windows_match,
            edge_mass_matched=mass_match,
            matching_unpaid_mean=match_mean,
            shuffled_unpaid_mean=shuffle_mean,
            delta_interval=interval,
            cw_vs_cv_cosine=cw_cv,
            cw_vs_cr_cosine=cw_cr,
            locus=next((name for hit, name in flags if hit), None),
        )

    matching = campaign['matching']
    shuffled = campaign['endpoint_shuffle']
    rolled = campaign.get('rolled_filing')
    occupancy = claim_endpoint_occupancy(matching.claim_labeled)
    scales = tuple(scale_slice(matching, shuffled, rolled, sigma) for sigma in sweep)
    named = next((item for item in scales if item.locus is not None), None)
    return GateBReport(
        passed=named is None,
        locus=None if named is None else cast(MechanismLocus, named.locus),
        occupancy_nonzero=occupancy.n_nonzero,
        occupancy_mass=occupancy.total_mass,
        scales=scales,
    )


def endpoint_probe(matching: ConditionView) -> ProbeReport:  # noqa: C901
    """Ridge decode of claim dest-slot identity on a held-out filing split."""
    assignment = matching.seams.get('late_assignment')
    n_docs = matching.claim_labeled.size(0)
    if assignment is None or n_docs < 4:
        return ProbeReport()

    def features_of(value: Tensor, owners: Tensor) -> Tensor:
        table = unnamed(value)
        if table.ndim >= 4:
            table = table[:, CLAIM_MASK] if table.size(1) > CLAIM_MASK else table[:, 0]
        if table.ndim >= 3:
            table = table.mean(dim=-2)
        if table.size(0) != n_docs and owners.numel():
            table = owner_mass_rows(table, owners, n_docs)
        return table.reshape(table.size(0), -1).to(dtype=torch.float32)

    owners = torch.tensor(next(iter(matching.arms.values())).window_owners, dtype=torch.long)
    features = features_of(assignment, owners)
    labels = matching.claim_labeled.sum(dim=(-3, -1)).reshape(n_docs, -1).argmax(dim=-1)
    if features.size(0) != n_docs or len(set(labels.tolist())) < 2:
        return ProbeReport()
    n = features.size(0)
    cut = int(n * PROBE_TRAIN_FRACTION)
    x_train, x_test = features[:cut], features[cut:]
    y_train, y_test = labels[:cut], labels[cut:]
    if len(set(y_train.tolist())) < 2 or x_test.size(0) < 1:
        return ProbeReport(n_train=cut, n_test=n - cut)
    classes = int(labels.max().item()) + 1
    eye = torch.eye(features.size(1), dtype=features.dtype, device=features.device)

    def fit_accuracy(train_y: Tensor) -> float:
        targets = F.one_hot(train_y, num_classes=classes).to(dtype=torch.float32)
        weights = torch.linalg.solve(x_train.T @ x_train + PROBE_RIDGE * eye, x_train.T @ targets)
        return float((x_test @ weights).argmax(dim=-1).eq(y_test).float().mean().item())

    test_acc = fit_accuracy(y_train)
    perm = torch.randperm(y_train.size(0), generator=torch.Generator().manual_seed(BOOTSTRAP_SEED))
    perm_acc = fit_accuracy(y_train[perm])
    return ProbeReport(
        accuracy=test_acc,
        permutation_accuracy=perm_acc,
        n_train=cut,
        n_test=n - cut,
        above_permutation=test_acc > perm_acc,
    )


def gate_c(campaign: Mapping[str, ConditionView]) -> GateCReport:
    """Judge dest-slot decodability and whether dest arrangement reaches the prefix."""

    def linear_cka(left: Tensor, right: Tensor) -> float | None:
        # Linear CKA is HSIC of centered features with a linear kernel.
        # We use the n-by-n Gram when n <= d; the feature-side form OOMs on wide seams.
        x = unnamed(left).reshape(left.size(0), -1).to(dtype=torch.float32)
        y = unnamed(right).reshape(right.size(0), -1).to(dtype=torch.float32)
        if x.size(0) != y.size(0) or x.size(0) < 2:
            return None
        if x.numel() > 2_000_000 or y.numel() > 2_000_000:
            return None
        xc = x - x.mean(dim=0)
        yc = y - y.mean(dim=0)
        if xc.size(0) <= xc.size(1) and yc.size(0) <= yc.size(1):
            xx = xc @ xc.transpose(0, 1)
            yy = yc @ yc.transpose(0, 1)
            hsic = (xx * yy).sum()
            denom = ((xx * xx).sum() * (yy * yy).sum()).clamp_min(1e-12).sqrt()
            return float((hsic / denom).item())
        xy = xc.transpose(0, 1) @ yc
        xx = xc.transpose(0, 1) @ xc
        yy = yc.transpose(0, 1) @ yc
        denom = ((xx * xx).sum() * (yy * yy).sum()).clamp_min(1e-12).sqrt()
        return float(((xy * xy).sum() / denom).item())

    def seam_delta(name: str, left: Tensor | None, right: Tensor | None) -> SeamDelta:
        if left is None or right is None:
            return SeamDelta(name=name)
        rel = relative_l2(left, right)
        cos = cosine(left, right)
        cka = linear_cka(left, right)
        flat_left = unnamed(left).reshape(-1)
        flat_right = unnamed(right).reshape(-1)
        comparable = flat_left.numel() == flat_right.numel() and flat_left.numel() > 0
        identical = bool(comparable and torch.equal(flat_left, flat_right))
        both_zero = bool(
            comparable and not torch.any(flat_left.bool()) and not torch.any(flat_right.bool())
        )
        moved = bool(math.isfinite(rel) and rel > MECHANISM_REL_L2)
        angled = bool(cos is not None and math.isfinite(cos) and cos < SEMANTIC_PARITY_COSINE)
        separates = bool(not identical and not both_zero and (moved or angled))
        return SeamDelta(name=name, relative_l2=rel, cosine=cos, cka=cka, separates=separates)

    matching = campaign['matching']
    shuffled = campaign['endpoint_shuffle']
    adapter = campaign.get('adapter_off')
    probe = endpoint_probe(matching)
    deltas = tuple(
        seam_delta(
            name,
            matching.document_labeled if name == 'document_labeled' else matching.seams.get(name),
            shuffled.document_labeled if name == 'document_labeled' else shuffled.seams.get(name),
        )
        for name in _GATE_C_SEAMS
    )
    comparable = tuple(
        item
        for item in deltas
        if item.name == 'document_labeled'
        or (item.name in matching.seams and item.name in shuffled.seams)
    )
    losing = next((item.name for item in comparable if not item.separates), None)
    prefix_at = next(
        (index for index, item in enumerate(comparable) if item.name == _PREFIX_SEAM),
        None,
    )
    losing_at = next(
        (index for index, item in enumerate(comparable) if item.name == losing),
        None,
    )
    reached_prefix = (
        losing_at is None
        or (prefix_at is not None and losing_at > prefix_at)
        or (prefix_at is None and losing not in _UPSTREAM_OF_PREFIX)
    )
    flags: tuple[tuple[bool, FusionLocus | None], ...] = (
        (not probe.above_permutation, 'ENDPOINT_NOT_DECODABLE'),
        (
            not reached_prefix,
            _GATE_C_LOCUS.get(losing or '', 'GRAPH_PROJECTION_LOSS'),
        ),
    )
    locus = next((name for hit, name in flags if hit), None)
    adapter_rel = (
        None
        if adapter is None
        else relative_l2(
            matching.seams.get('adapter_delta', matching.document_labeled),
            adapter.seams.get('adapter_delta', adapter.document_labeled),
        )
    )
    adapter_prefix = (
        None
        if adapter is None or 'projected_prefix' not in matching.seams
        else relative_l2(
            matching.seams['projected_prefix'],
            adapter.seams.get('projected_prefix', matching.seams['projected_prefix']),
        )
    )
    reduces = bool(
        adapter is not None
        and (
            (adapter_rel is not None and adapter_rel > ADAPTER_REL_L2)
            or (adapter_prefix is not None and adapter_prefix > ADAPTER_REL_L2)
        )
    )
    return GateCReport(
        passed=bool(probe.above_permutation and reached_prefix),
        locus=locus,
        probe=probe,
        earliest_losing_seam=losing,
        seam_deltas=deltas,
        adapter_rel_l2=adapter_rel,
        adapter_off_reduces=reduces,
        train_export_present=False,
        gradient_cosines_present=False,
    )


def letter_report(  # noqa: C901
    campaign: Mapping[str, ConditionView],
    covering: Covering,
    pairs: Sequence[CitationPair],
    *,
    heatmap: Mapping[str, object] | None = None,
    sweep: tuple[float, ...] = DECLARED_EDGE_SWEEP,
) -> LetterReport:
    """Vertex, relation, endpoint, and joint unpaid for X, Y, A, and random."""

    def letter_frame(unpaid: Tensor, live: Sequence[CitationPair]) -> pl.DataFrame:
        return pl.DataFrame({
            'query': tuple(pair.query_application_number for pair in live),
            'mark': tuple(str(citation_letter(pair)) for pair in live),
            'unpaid': unpaid.reshape(-1).to(dtype=torch.float64).tolist(),
        })

    def letter_map(unpaid: Tensor, live: Sequence[CitationPair]) -> dict[str, float | None]:
        frame = letter_frame(unpaid, live)

        def mean_of(letter: str) -> float | None:
            chosen = (
                frame
                .filter(pl.col('mark') == letter)
                .group_by('query')
                .agg(pl.col('unpaid').mean())
            )
            if chosen.is_empty():
                return None
            return finite_mean(unpaid.new_tensor(chosen['unpaid'].to_list(), dtype=torch.float32))

        return {letter: mean_of(letter) for letter in ('X', 'Y', 'A')}

    def random_unpaid(
        claim: Tensor,
        document: Tensor,
        live: Sequence[CitationPair],
        stems: Sequence[str],
        *,
        kind: Literal['vertex', 'edge'],
        sigma: float,
    ) -> float | None:
        cited = {(pair.query_application_number, pair.partner_application_number) for pair in live}
        index = {stem: offset for offset, stem in enumerate(stems) if stem}
        queries = tuple(dict.fromkeys(pair.query_application_number for pair in live))
        unused = tuple(
            stem
            for stem in stems
            if stem and queries and (queries[0], stem) not in cited and stem != queries[0]
        )
        if not queries or not unused:
            return None
        q_row = claim[index[queries[0]]].unsqueeze(0)
        d_row = document[index[unused[0]]].unsqueeze(0)
        unpaid = (
            vertex_unpaid(covering, q_row, d_row)
            if kind == 'vertex'
            else edge_unpaid(covering, q_row, d_row, sigma)
        )
        return finite_mean(unpaid)

    def xa_interval(unpaid: Tensor, live: Sequence[CitationPair]) -> Interval:
        scores = tuple(
            QueryPartnerScore(
                query=pair.query_application_number,
                partner=pair.partner_application_number,
                mark=cast(Literal['X', 'Y', 'A'], str(citation_letter(pair))),
                unpaid=(
                    float(unpaid[index].item()) if torch.isfinite(unpaid[index]) else float('nan')
                ),
                covering=0.0,
                demand_l1=1.0,
            )
            for index, pair in enumerate(live)
        )
        xa = query_macro_unpaid_fraction_xa(scores)
        x_frame = (
            letter_frame(unpaid, live)
            .filter(pl.col('mark').is_in(('X', 'A')))
            .group_by(['query', 'mark'])
            .agg(pl.col('unpaid').mean())
            .pivot(on='mark', index='query', values='unpaid', aggregate_function='first')
        )
        if x_frame.is_empty() or 'X' not in x_frame.columns or 'A' not in x_frame.columns:
            return Interval(n_queries=0, mean=xa)
        delta = torch.tensor(
            x_frame
            .filter(pl.col('X').is_not_null() & pl.col('A').is_not_null())
            .select((pl.col('A') - pl.col('X')).alias('delta'))
            .to_series()
            .to_list(),
            dtype=torch.float32,
        )
        interval = query_mean_interval(delta)
        return interval.model_copy(update={'mean': xa if xa is not None else interval.mean})

    def heatmap_rows(raw: object) -> tuple[dict[str, object], ...]:
        if not isinstance(raw, list):
            return ()
        return tuple(cast(list[dict[str, object]], raw)[:8])

    matching = campaign['matching']
    shuffled = campaign['endpoint_shuffle']
    rolled = campaign.get('rolled_filing')
    live, query_idx, partner_idx, _marks = pair_indices(pairs, matching.stems)
    sigma = next(iter(sweep[1:] or sweep), float('nan'))
    slots = heatmap_rows(heatmap.get('slots') if heatmap else None)
    edges = heatmap_rows(heatmap.get('edges') if heatmap else None)
    if not live:
        empty = Interval(n_queries=0)
        return LetterReport(
            n_pairs=0,
            n_queries_xa=0,
            arms={},
            xa_endpoint_matching=empty,
            xa_vertex_matching=empty,
            matching_vs_rolled_endpoint=empty,
            matching_vs_shuffle_endpoint=empty,
            xa_improved=False,
            heatmap_slots=slots,
            heatmap_edges=edges,
        )

    def arm_of(view: ConditionView) -> LetterArm:
        vertex = vertex_unpaid(
            covering, view.claim_intensity[query_idx], view.document_intensity[partner_idx]
        )
        relation = relation_unpaid(
            covering,
            view.claim_labeled[query_idx].sum(dim=(-3, -2)),
            view.document_labeled[partner_idx].sum(dim=(-3, -2)),
            sigma,
        )
        edge = edge_unpaid(
            covering, view.claim_labeled[query_idx], view.document_labeled[partner_idx], sigma
        )
        vertex_map = letter_map(vertex, live)
        relation_map = letter_map(relation, live)
        edge_map = letter_map(edge, live)
        joint_map = letter_map(1.0 - (1.0 - vertex) * (1.0 - edge), live)
        vertex_map['random'] = random_unpaid(
            view.claim_intensity,
            view.document_intensity,
            live,
            view.stems,
            kind='vertex',
            sigma=sigma,
        )
        edge_map['random'] = random_unpaid(
            view.claim_labeled,
            view.document_labeled,
            live,
            view.stems,
            kind='edge',
            sigma=sigma,
        )
        relation_map['random'] = None
        joint_map['random'] = None
        return LetterArm(
            vertex=vertex_map,
            relation_type=relation_map,
            endpoint=edge_map,
            joint=joint_map,
        )

    views = (
        ('matching', matching),
        ('endpoint_shuffle', shuffled),
        *((('rolled_filing', rolled),) if rolled is not None else ()),
    )
    q = matching.claim_labeled[query_idx]
    match_edge = edge_unpaid(covering, q, matching.document_labeled[partner_idx], sigma)
    shuffle_edge = edge_unpaid(covering, q, shuffled.document_labeled[partner_idx], sigma)
    rolled_edge = (
        edge_unpaid(covering, q, rolled.document_labeled[partner_idx], sigma)
        if rolled is not None
        else match_edge
    )
    match_vertex = vertex_unpaid(
        covering, matching.claim_intensity[query_idx], matching.document_intensity[partner_idx]
    )
    queries = tuple(pair.query_application_number for pair in live)
    xa_edge = xa_interval(match_edge, live)
    return LetterReport(
        n_pairs=len(live),
        n_queries_xa=xa_edge.n_queries,
        arms={name: arm_of(view) for name, view in views},
        xa_endpoint_matching=xa_edge,
        xa_vertex_matching=xa_interval(match_vertex, live),
        matching_vs_rolled_endpoint=query_mean_interval(
            query_grouped_mean(queries, rolled_edge - match_edge)
        ),
        matching_vs_shuffle_endpoint=query_mean_interval(
            query_grouped_mean(queries, shuffle_edge - match_edge)
        ),
        xa_improved=bool(xa_edge.mean is not None and xa_edge.mean > 0.0 and xa_edge.excludes_zero),
        heatmap_slots=slots,
        heatmap_edges=edges,
    )


def judge_frozen_campaign(
    root: Path,
    pairs: Sequence[CitationPair],
    covering: Covering,
    *,
    heatmap: Mapping[str, object] | None = None,
    sweep: tuple[float, ...] = DECLARED_EDGE_SWEEP,
) -> FrozenJudgment:
    """Run Gate B, then Gate C, then the letter report on ``root``."""
    campaign = load_campaign(root)
    mechanism = gate_b(campaign, covering, pairs, sweep)
    fusion = gate_c(campaign)
    letters = letter_report(campaign, covering, pairs, heatmap=heatmap, sweep=sweep)
    third = THIRD_OUTCOME if mechanism.passed and not letters.xa_improved else None
    return FrozenJudgment(
        campaign=str(root),
        sweep=sweep,
        gate_b=mechanism,
        gate_c=fusion,
        letters=letters,
        third_outcome=third,
    )


def judge_from_env(output: Path | None = None) -> FrozenJudgment:
    """Load campaign, citations, and covering knobs from the process environment."""

    def campaign_root() -> Path | None:
        listed = os.environ.get('COVERING_FROZEN_CAMPAIGN', '').strip()
        ledger = os.environ.get('COVERING_LEDGER_ROOT', '').strip()
        candidates = tuple(
            path
            for path in (
                Path(listed) if listed else None,
                Path(ledger) / 'letter_graph_contribution' / 'frozen_campaign' if ledger else None,
            )
            if path is not None
        )
        return next((path for path in candidates if (path / 'campaign.json').is_file()), None)

    def citation_pairs() -> tuple[CitationPair, ...]:
        env = os.environ.get('COVERING_CITATIONS', '').strip()
        configured = CollisionEvalConfig.from_yaml().dataset
        path = Path(env) if env else Path(configured or '')
        return cast(tuple[CitationPair, ...], EpoProcessorCitationPairSource().load_pairs(path))

    def development_pairs(pairs: Sequence[CitationPair]) -> tuple[CitationPair, ...]:
        spec = CoveringPilotSpec.from_yaml()
        eval_config = CollisionEvalConfig.from_yaml()
        _train, development, _held = split_pairs_by_query(
            pairs,
            train=eval_config.split.train,
            eval_fraction=eval_config.split.eval,
            test=eval_config.split.test,
            seed=eval_config.split.seed,
            query_limit=eval_config.query_limit,
        )
        return spec.letter_scored_pairs(development)

    root = campaign_root()
    if root is None:
        msg = 'frozen campaign path is missing'
        raise FileNotFoundError(msg)
    covering = Covering(CoveringKnobs())
    heatmap_path = root.parent / 'matching_heatmap.json'
    heatmap = None
    if heatmap_path.is_file():
        raw = json.loads(heatmap_path.read_text(encoding='utf-8'))
        heatmap = raw if isinstance(raw, dict) else None
    report = judge_frozen_campaign(
        root,
        development_pairs(citation_pairs()),
        covering,
        heatmap=heatmap,
    )
    dest = output or root
    write_json(dest / 'judgment.json', report.model_dump(mode='json'))
    return report


__all__ = [
    'BOOTSTRAP_ALPHA',
    'BOOTSTRAP_DRAWS',
    'BOOTSTRAP_SEED',
    'THIRD_OUTCOME',
    'ConditionView',
    'FrozenJudgment',
    'GateBReport',
    'GateCReport',
    'Interval',
    'LetterReport',
    'document_stems',
    'endpoint_probe',
    'gate_b',
    'gate_c',
    'judge_from_env',
    'judge_frozen_campaign',
    'letter_report',
    'load_campaign',
    'load_condition',
    'query_mean_interval',
]
