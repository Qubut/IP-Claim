"""Unit tests for SSV graph_batch (HUPD CPC prefixes + claim deps)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ip_claim.collision.data import (
    CitationPair,
    CitationPairSource,
    StubCitationPairSource,
)
from ip_claim.ingestion.adapters.hupd_json.loader import patent_from_hupd_dict
from ip_claim.ssv.graph_batch import (
    CPC_IDENTITY_UNKNOWN,
    PatentGraphBatch,
    build_hetero_from_patent,
    cpc_identity_depth,
    cpc_identity_index,
    cpc_prefix_labels,
    graph_batch_from_hupd_dict,
    patent_claim_blob,
    patent_training_text,
    strip_soh_markup,
)

_FIXTURES = ('13817165.json', '14111139.json', '14112715.json')


@pytest.fixture(params=_FIXTURES)
def hupd_raw(request: pytest.FixtureRequest, fixtures_dir: Path) -> dict[str, Any]:
    path = fixtures_dir / 'hupd' / request.param
    return json.loads(path.read_text(encoding='utf-8'))


def test_cpc_prefix_labels_coarse_only() -> None:
    assert cpc_prefix_labels('A61M51723') == ('A', 'A61', 'A61M')
    assert cpc_prefix_labels('H01T2300') == ('H', 'H01', 'H01T')
    assert cpc_prefix_labels('') == ()
    assert cpc_prefix_labels('not-a-cpc') == ()


def test_cpc_identity_index_is_keyed_by_prefix_with_unknown_bucket() -> None:
    assert cpc_identity_index('_UNC') == CPC_IDENTITY_UNKNOWN
    assert cpc_identity_index('') == CPC_IDENTITY_UNKNOWN
    assert cpc_identity_index('not-a-cpc') == CPC_IDENTITY_UNKNOWN
    assert cpc_identity_index('A') != cpc_identity_index('H')
    assert cpc_identity_index('A61') != cpc_identity_index('A')
    assert cpc_identity_index('A61M') != cpc_identity_index('A61')
    assert cpc_identity_depth(cpc_identity_index('A')) == pytest.approx(0.0)
    assert cpc_identity_depth(cpc_identity_index('A61')) == pytest.approx(1.0)
    assert cpc_identity_depth(cpc_identity_index('A61M')) == pytest.approx(2.0)


def test_strip_soh_markup_plain_replace() -> None:
    wrapped = '<SOH>background text<EOH>'
    assert strip_soh_markup(wrapped) == 'background text'
    assert '<SOH>' not in strip_soh_markup(wrapped)
    assert '<EOH>' not in strip_soh_markup(wrapped)


def test_graph_batch_from_real_fixtures(hupd_raw: dict[str, Any]) -> None:
    batch = graph_batch_from_hupd_dict(hupd_raw)
    assert isinstance(batch, PatentGraphBatch)
    assert batch.application_number == str(hupd_raw['application_number']).strip()
    assert batch.application_number.isdigit()
    assert '<SOH>' not in batch.text
    assert '<EOH>' not in batch.text
    if batch.claim_blob:
        assert batch.text.startswith(batch.claim_blob)

    data = batch.data
    assert 'cpc' in data.node_types
    assert 'claim' in data.node_types
    assert 'soft' not in data.node_types
    assert ('cpc', 'parent_of', 'cpc') in data.edge_types
    assert ('claim', 'depends_on', 'claim') in data.edge_types
    assert not any('mention' in str(t).lower() for t in data.edge_types)

    labels: list[str] = list(data['cpc'].label)
    assert all(len(lab) <= 4 or lab == '_UNC' for lab in labels)
    assert not any(lab.endswith(('51723', '2300')) for lab in labels)
    assert data['cpc'].cpc_id.shape == (data['cpc'].num_nodes,)
    assert data['cpc'].cpc_id.dtype == torch.long
    assert int(data['cpc'].cpc_id.unique().numel()) >= 2

    parent_ei = data['cpc', 'parent_of', 'cpc'].edge_index
    assert parent_ei.ndim == 2
    assert parent_ei.shape[0] == 2
    assert parent_ei.shape[1] >= 1

    claim_ei = data['claim', 'depends_on', 'claim'].edge_index
    assert claim_ei.ndim == 2
    assert claim_ei.shape[0] == 2
    assert data['claim'].num_nodes == len(patent_from_hupd_dict(hupd_raw).claims)


def test_empty_background_and_summary_tolerated(fixtures_dir: Path) -> None:
    raw = json.loads((fixtures_dir / 'hupd' / '14112715.json').read_text(encoding='utf-8'))
    assert not (raw.get('background') or '').strip()
    assert not (raw.get('summary') or '').strip()
    batch = graph_batch_from_hupd_dict(raw)
    assert batch.application_number == '14112715'
    assert batch.text
    assert '<SOH>' not in batch.text


def test_soh_stripped_on_sections_with_markup(fixtures_dir: Path) -> None:
    raw = json.loads((fixtures_dir / 'hupd' / '13817165.json').read_text(encoding='utf-8'))
    assert '<SOH>' in (raw.get('background') or '')
    assert '<SOH>' in (raw.get('summary') or '')
    patent = patent_from_hupd_dict(raw)
    text = patent_training_text(patent)
    assert '<SOH>' not in text
    assert '<EOH>' not in text
    assert patent.summary.strip()
    stripped_summary = strip_soh_markup(patent.summary)
    assert stripped_summary
    assert stripped_summary in text
    blob = patent_claim_blob(patent)
    assert blob
    assert text.startswith(blob)


def test_claim_depends_on_uses_parser_parents(fixtures_dir: Path) -> None:
    raw = json.loads((fixtures_dir / 'hupd' / '13817165.json').read_text(encoding='utf-8'))
    patent = patent_from_hupd_dict(raw)
    data = build_hetero_from_patent(patent)
    number_to_idx = {c.number: i for i, c in enumerate(patent.claims)}
    ei = data['claim', 'depends_on', 'claim'].edge_index
    edges = {(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])}
    expected = {
        (number_to_idx[c.number], number_to_idx[c.parent_number])
        for c in patent.claims
        if c.parent_number is not None and c.parent_number in number_to_idx
    }
    assert edges == expected
    assert expected, 'fixture should yield at least one claim dependency edge'


def test_multi_claim_range_ref_falls_back_without_mentions(
    fixtures_dir: Path,
) -> None:
    """``any of claims 1 to 3`` is not a single parent; stay independent."""
    raw = json.loads((fixtures_dir / 'hupd' / '14111139.json').read_text(encoding='utf-8'))
    patent = patent_from_hupd_dict(raw)
    multi = next(c for c in patent.claims if 'any of claims 1 to 3' in c.text.lower())
    assert multi.parent_number is None
    assert multi.is_independent is True

    data = build_hetero_from_patent(patent)
    number_to_idx = {c.number: i for i, c in enumerate(patent.claims)}
    multi_idx = number_to_idx[multi.number]
    ei = data['claim', 'depends_on', 'claim'].edge_index
    srcs = {int(ei[0, i]) for i in range(ei.shape[1])}
    assert multi_idx not in srcs
    assert 'soft' not in data.node_types


def test_main_cpc_section_from_compact_code(fixtures_dir: Path) -> None:
    raw = json.loads((fixtures_dir / 'hupd' / '13817165.json').read_text(encoding='utf-8'))
    batch = graph_batch_from_hupd_dict(raw)
    assert batch.main_cpc_section == 'A'
    labels = list(batch.data['cpc'].label)
    assert 'A' in labels
    assert 'A61' in labels
    assert 'A61M' in labels


def test_citation_pairs_protocol_stub(tmp_path: Path) -> None:
    stub: CitationPairSource = StubCitationPairSource()
    assert stub.load_pairs(tmp_path / 'missing.parquet') == ()
    present = tmp_path / 'pairs.parquet'
    present.write_bytes(b'')
    assert stub.load_pairs(present) == ()
    pair = CitationPair(
        query_application_number='13817165',
        partner_application_number='14111139',
    )
    assert pair.grade == 0
    assert pair.marks == ()
