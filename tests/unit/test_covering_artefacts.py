"""covering.json is the detector write; explain/ uses recomputed A and W."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ip_claim.collision.artefacts import (
    COVERING_JSON,
    DETECTOR_NAME,
    ExplainState,
    artefact_from_states,
    artefact_stem,
    artefacts_for_keys,
    report_from_splits,
    top_hit_keys,
    write_collision_artefacts,
)
from ip_claim.collision.collide import CollisionEvalResult, PatentEmbeddingRecord
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.data.citation_pairs import CitationPair
from ip_claim.collision.explain import Explain


def _record(app: str, n_claim: list[float], n_full: list[float]) -> PatentEmbeddingRecord:
    return PatentEmbeddingRecord(
        application_number=app,
        z_d=torch.zeros(2),
        n_entity_claim=torch.tensor(n_claim),
        n_entity_full=torch.tensor(n_full),
        n_relation_claim=torch.tensor([1.0, 0.0]),
        n_relation_full=torch.tensor([2.0, 0.0]),
    )


def _empty_split() -> CollisionEvalResult:
    zeros = {5: 0.0}
    return CollisionEvalResult(
        recall_at_k=zeros,
        particular_recall_at_k=zeros,
        cpc_hard_recall_at_k=zeros,
        particular_cpc_hard_recall_at_k=zeros,
        mrr=0.0,
        particular_mrr=0.0,
        ndcg_at_k=zeros,
        queries=0,
        unpaid_x=0.1,
        unpaid_y=0.2,
        unpaid_a=0.4,
        unpaid_random=0.8,
        covering_x=0.9,
        covering_y=0.7,
        covering_a=0.5,
        covering_random=0.2,
    )


def _state(
    app: str,
    assignment: torch.Tensor,
    *,
    claim_edges: torch.Tensor,
    full_edges: torch.Tensor,
    claim_mask: torch.Tensor | None = None,
) -> ExplainState:
    tokens = assignment.size(0)
    return ExplainState(
        application_number=app,
        assignment=assignment,
        attention_mask=torch.ones(tokens),
        claim_mask=claim_mask if claim_mask is not None else torch.ones(tokens),
        claim_edges=claim_edges,
        full_edges=full_edges,
    )


def test_covering_json_names_saturation_covering_not_cosine(tmp_path: Path) -> None:
    config = CollisionEvalConfig(output_dir=str(tmp_path), explain_top_n=1)
    empty = _empty_split()
    report = write_collision_artefacts(
        tmp_path,
        report_from_splits(
            train=empty,
            eval_result=empty,
            test=empty,
            eval_config=config,
            encoded_apps=3,
        ),
        config,
    )
    payload = json.loads((tmp_path / COVERING_JSON).read_text(encoding='utf-8'))
    text = json.dumps(payload)
    assert report.detector == DETECTOR_NAME
    assert payload['detector'] == DETECTOR_NAME
    assert payload['encoded_apps'] == 3
    assert 'unpaid_x' in payload['train']
    assert 'unpaid_y' in payload['train']
    assert 'covering_x' in payload['train']
    assert 'queries_x' in payload['train']
    assert 'cosine' not in text.lower()
    assert 'infonce' not in text.lower()
    assert 'z_d' not in payload
    assert (tmp_path / 'run.json').is_file()
    assert (tmp_path / 'explain' / 'train').is_dir()


def test_top_hit_keys_rank_by_covering() -> None:
    covering = Covering(CoveringKnobs())
    query = _record('100', [4.0, 0.0], [4.0, 0.0])
    paid = _record('200', [0.0, 1.0], [8.0, 0.0])
    miss = _record('300', [0.0, 1.0], [0.0, 4.0])
    pairs = (CitationPair(query_application_number='100', partner_application_number='200'),)
    keys = top_hit_keys(pairs, (query, paid, miss), covering, top_n=1)
    assert keys == (('100', '200'),)


def test_artefact_from_states_writes_contour_and_community(tmp_path: Path) -> None:
    covering = Covering(CoveringKnobs())
    explain = Explain(covering)
    assignment_q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assignment_d = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query_edges = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
    document_edges = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    query = _state(
        '100',
        assignment_q,
        claim_edges=query_edges,
        full_edges=query_edges,
        claim_mask=torch.tensor([1.0, 1.0, 0.0]),
    )
    document = _state(
        '200',
        assignment_d,
        claim_edges=document_edges,
        full_edges=document_edges,
    )
    artefact = artefact_from_states(query, document, explain, tau=0.1)
    assert artefact.query_field
    assert artefact.spans
    assert artefact.unpaid_edge > 0.0
    missing = artefacts_for_keys((('100', '999'),), {'100': query}, explain, tau=0.1)
    assert missing == ()
    config = CollisionEvalConfig(output_dir=str(tmp_path))
    write_collision_artefacts(
        tmp_path,
        report_from_splits(
            train=_empty_split(),
            eval_result=_empty_split(),
            test=_empty_split(),
            eval_config=config,
        ),
        config,
        explain={'eval': (artefact,)},
    )
    written = tmp_path / 'explain' / 'eval' / f'{artefact_stem("100", "200")}.json'
    body = json.loads(written.read_text(encoding='utf-8'))
    assert body['query_field']
    assert body['spans']
    assert 'cosine' not in body
    assert artefact.conductance == pytest.approx(float(body['conductance']))


def test_artefact_from_states_slot_keep_unpaid_matches_contour() -> None:
    explain = Explain(Covering(CoveringKnobs(slot_mass_keep=0.35)))
    assignment_q = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.2, 0.4, 0.4],
    ])
    assignment_d = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.2, 0.8],
    ])
    claim_mask = torch.tensor([1.0, 1.0, 0.0])
    before = assignment_q.clone()
    query = _state(
        '100',
        assignment_q,
        claim_edges=torch.zeros(3, 3),
        full_edges=torch.zeros(3, 3),
        claim_mask=claim_mask,
    )
    document = _state(
        '200',
        assignment_d,
        claim_edges=torch.zeros(3, 3),
        full_edges=torch.zeros(3, 3),
    )
    artefact = artefact_from_states(query, document, explain, tau=0.3)
    covering = explain.covering
    occupy_q = covering.masked_intensity(assignment_q, claim_mask)
    occupy_d = covering.masked_intensity(assignment_d, torch.ones(3))
    n_query = covering.keep_slots(explain.overlay_n(occupy_q, torch.zeros(3, 3)))
    n_document = explain.overlay_n(occupy_d, torch.zeros(3, 3))
    scored = covering(n_query, n_document)
    query_field = torch.tensor(artefact.query_field)
    document_field = torch.tensor(artefact.document_field)
    assert torch.equal(assignment_q, before)
    assert artefact.unpaid == pytest.approx(float(scored.unpaid_mass.item()))
    assert artefact.covering == pytest.approx(float(scored.covering.item()))
    assert torch.allclose((claim_mask * query_field).sum(), scored.unpaid_mass)
    assert torch.allclose((torch.ones(3) * document_field).sum(), scored.paid.sum())
    assert int((covering.keep_slots(n_query) > 0).sum().item()) == 1
