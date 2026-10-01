"""On-disk covering tables and explain envelopes.

The detector artefact is covering.json. Cosine and InfoNCE scores are not written
as the claim. Token contour and slot-graph community are written only when the
caller already recomputed assignment and pair-graph W for that pair. Covering n
is occupy mass plus kept-edge addends.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch_geometric.data import Data

from ip_claim.collision.collide import (
    RANK_QUERY_TILE,
    CollisionEvalRequest,
    CollisionEvalResult,
    PatentEmbeddingRecord,
    prepare_ranking_batch,
)
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.cover import Covering
from ip_claim.collision.data.citation_pairs import CitationPair
from ip_claim.collision.explain import CollisionCommunity, CoveringContour, Explain

DETECTOR_NAME = 'saturation_covering'
COVERING_JSON = 'covering.json'
RUN_JSON = 'run.json'
EXPLAIN_DIR = 'explain'
SPLIT_NAMES = ('train', 'eval', 'test')


class ExplainState(BaseModel):
    """Late assignment and slot-graph W for one filing, held only at explain time."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    application_number: str
    assignment: Tensor
    attention_mask: Tensor
    claim_mask: Tensor
    claim_edges: Tensor
    full_edges: Tensor


class ExplainArtefact(BaseModel):
    """One explained query-document pair: covering scores plus optional fields."""

    model_config = ConfigDict(frozen=True)

    query_application: str
    document_application: str
    covering: float
    unpaid: float
    unpaid_edge: float
    paid_demand: float
    paid: tuple[float, ...]
    residual: tuple[float, ...]
    paid_mask: tuple[bool, ...]
    gap_mask: tuple[bool, ...]
    conductance: float = 0.0
    spans: tuple[tuple[int, int], ...] = ()
    query_field: tuple[float, ...] = ()
    document_field: tuple[float, ...] = ()


class CoveringReport(BaseModel):
    """Train, eval, and test covering tables with locked knobs."""

    model_config = ConfigDict(frozen=True)

    detector: str = DETECTOR_NAME
    checkpoint: str | None = None
    encoded_apps: int = 0
    knobs: Mapping[str, float | int | None]
    train: Mapping[str, object]
    eval: Mapping[str, object]
    test: Mapping[str, object]


def tensor_floats(values: Tensor) -> tuple[float, ...]:
    """Detach a 1-d tensor to a JSON-safe float tuple."""
    return tuple(float(item) for item in values.detach().cpu().reshape(-1).tolist())


def artefact_stem(query_application: str, document_application: str) -> str:
    """Stable filename stem for one explained pair."""

    def part(application: str) -> str:
        return ''.join(ch if ch.isalnum() or ch in '-._' else '_' for ch in application)

    return f'{part(query_application)}__{part(document_application)}'


def report_from_splits(
    *,
    train: CollisionEvalResult,
    eval_result: CollisionEvalResult,
    test: CollisionEvalResult,
    eval_config: CollisionEvalConfig,
    checkpoint: str | None = None,
    encoded_apps: int = 0,
) -> CoveringReport:
    """Build the covering report from the three split tables."""
    knobs = eval_config.covering.model_dump()
    knobs['contour_tau'] = eval_config.contour_tau
    knobs['explain_top_n'] = eval_config.explain_top_n
    return CoveringReport(
        checkpoint=checkpoint,
        encoded_apps=encoded_apps,
        knobs=knobs,
        train=train.model_dump(mode='json'),
        eval=eval_result.model_dump(mode='json'),
        test=test.model_dump(mode='json'),
    )


def artefact_from_states(
    query: ExplainState,
    document: ExplainState,
    explain: Explain,
    *,
    tau: float,
) -> ExplainArtefact:
    """Contour and community from occupy plus kept-edge intensity."""
    covering = explain.covering
    occupy_query = covering.masked_intensity(query.assignment, query.claim_mask)
    occupy_document = covering.masked_intensity(document.assignment, document.attention_mask)
    n_query = covering.keep_slots(explain.overlay_n(occupy_query, query.claim_edges))
    n_document = explain.overlay_n(occupy_document, document.full_edges)
    contour, community = explain(
        query.assignment,
        document.assignment,
        n_query,
        n_document,
        query.claim_edges,
        document.full_edges,
        query.claim_mask,
        tau=tau,
    )
    return artefact_from_explain(
        query.application_number,
        document.application_number,
        covering,
        n_query,
        n_document,
        contour,
        community,
    )


def artefact_from_explain(
    query_application: str,
    document_application: str,
    covering: Covering,
    n_query: Tensor,
    n_document: Tensor,
    contour: CoveringContour,
    community: CollisionCommunity,
) -> ExplainArtefact:
    """Serialize a recomputed contour and community with the covering scores."""
    scored = covering(n_query, n_document)
    return ExplainArtefact(
        query_application=query_application,
        document_application=document_application,
        covering=float(scored.covering.item()),
        unpaid=float(scored.unpaid_mass.item()),
        unpaid_edge=float(community.unpaid_edge.item()),
        paid_demand=float(community.paid_demand.item()),
        paid=tensor_floats(scored.paid),
        residual=tensor_floats(scored.residual),
        paid_mask=tuple(bool(flag) for flag in community.paid_mask.tolist()),
        gap_mask=tuple(bool(flag) for flag in community.gap_mask.tolist()),
        conductance=float(community.conductance.item()),
        spans=contour.spans,
        query_field=tensor_floats(contour.query_field),
        document_field=tensor_floats(contour.document_field),
    )


def top_hit_keys(
    split_pairs: Sequence[CitationPair],
    records: Sequence[PatentEmbeddingRecord],
    covering: Covering,
    *,
    top_n: int,
    banks: Data | None = None,
) -> tuple[tuple[str, str], ...]:
    """Top covering document ids per query on one split."""
    by_app = {row.application_number: row for row in records}
    eval_pairs = tuple(
        pair
        for pair in split_pairs
        if pair.query_application_number in by_app and pair.partner_application_number in by_app
    )
    if not eval_pairs or top_n < 1:
        return ()
    query_apps = tuple(dict.fromkeys(pair.query_application_number for pair in eval_pairs))
    query_records = tuple(by_app[app] for app in query_apps)
    first_partner = {
        pair.query_application_number: pair.partner_application_number for pair in eval_pairs
    }
    tile = max(int(RANK_QUERY_TILE), 1)
    starts = tuple(range(0, len(query_records), tile))

    def keys_for_start(start: int) -> tuple[tuple[str, str], ...] | None:
        end = min(start + tile, len(query_records))
        batch = prepare_ranking_batch(
            CollisionEvalRequest(
                query_records=query_records[start:end],
                corpus_records=tuple(records),
                partner_apps=tuple(first_partner[app] for app in query_apps[start:end]),
            ),
            covering,
            banks,
        )
        if batch is None:
            return None
        values, indices = batch.scores.topk(min(int(top_n), batch.scores.size(1)), dim=1)
        return tuple(
            (query.application_number, records[int(column)].application_number)
            for query_index, query in enumerate(query_records[start:end])
            for rank, column in enumerate(indices[query_index])
            if bool(torch.isfinite(values[query_index, rank]))
            and records[int(column)].application_number != query.application_number
        )

    pieces = tuple(keys_for_start(start) for start in starts)
    if any(piece is None for piece in pieces):
        return ()
    return tuple(key for piece in pieces if piece is not None for key in piece)


def artefacts_for_keys(
    keys: Sequence[tuple[str, str]],
    states: Mapping[str, ExplainState],
    explain: Explain,
    *,
    tau: float,
) -> tuple[ExplainArtefact, ...]:
    """Build explain envelopes only for pairs whose late states were recomputed."""
    return tuple(
        artefact_from_states(states[query], states[document], explain, tau=tau)
        for query, document in keys
        if query in states and document in states
    )


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a sorted JSON object with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_covering_report(out: Path, report: CoveringReport) -> Path:
    """Write covering.json. This file is the detector claim."""
    path = out / COVERING_JSON
    write_json(path, report.model_dump(mode='json'))
    return path


def write_explain_split(out: Path, split: str, hits: Sequence[ExplainArtefact]) -> Path:
    """Write one JSON file per explained pair under explain/<split>/."""
    split_dir = out / EXPLAIN_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for hit in hits:
        write_json(
            split_dir / f'{artefact_stem(hit.query_application, hit.document_application)}.json',
            hit.model_dump(mode='json'),
        )
    return split_dir


def write_collision_artefacts(
    out: Path,
    report: CoveringReport,
    eval_config: CollisionEvalConfig,
    *,
    explain: Mapping[str, Sequence[ExplainArtefact]] | None = None,
) -> CoveringReport:
    """Write covering.json, run.json, and any explain split directories."""
    write_covering_report(out, report)
    run_payload = eval_config.model_dump(mode='json')
    run_payload.pop('hf_token', None)
    write_json(out / RUN_JSON, run_payload)
    hits = explain if explain is not None else {}
    for split in SPLIT_NAMES:
        write_explain_split(out, split, tuple(hits.get(split, ())))
    return report


__all__ = [
    'COVERING_JSON',
    'DETECTOR_NAME',
    'EXPLAIN_DIR',
    'RUN_JSON',
    'SPLIT_NAMES',
    'CoveringReport',
    'ExplainArtefact',
    'ExplainState',
    'artefact_from_explain',
    'artefact_from_states',
    'artefact_stem',
    'artefacts_for_keys',
    'report_from_splits',
    'tensor_floats',
    'top_hit_keys',
    'write_collision_artefacts',
    'write_covering_report',
    'write_explain_split',
    'write_json',
]
