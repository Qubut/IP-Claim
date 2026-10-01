"""cited_hupd locators fail closed; absence is unmapped, not a zero collision."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from ip_claim.collision.data import (
    AnalyzeCited,
    CitationPair,
    EpoProcessorCitationPairSource,
    StubCitationPairSource,
)

_SHIPPED_CITED_FIELDS = frozenset({'cited_id', 'categories', 'paths'})
_SHIPPED_STRUCT = [
    ('cited_id', pa.string()),
    ('categories', pa.list_(pa.string())),
    ('paths', pa.list_(pa.string())),
]
_SHIPPED_CITED = {
    'cited_id': '14111139',
    'categories': ['X'],
    'paths': ['14111139.json'],
}


def _write_cited_parquet(
    path: Path,
    cited: dict[str, object],
    struct_fields: list[tuple[str, pa.DataType]],
) -> None:
    table = pa.table({
        'epo_hupd_paths': pa.array([['13817165.json']], type=pa.list_(pa.string())),
        'cited_hupd': pa.array(
            [[cited]],
            type=pa.list_(pa.struct(struct_fields)),
        ),
    })
    pq.write_table(table, path)


def test_shipped_cited_hupd_fields_have_no_locators() -> None:
    assert set(_SHIPPED_CITED) == _SHIPPED_CITED_FIELDS
    assert 'claims' not in _SHIPPED_CITED_FIELDS
    assert 'passages' not in _SHIPPED_CITED_FIELDS
    assert {'claims', 'passages'} <= set(AnalyzeCited.model_fields)


def test_missing_locators_are_unmapped_not_grade_zero() -> None:
    cited = AnalyzeCited.model_validate(_SHIPPED_CITED)
    assert cited.grade == 2
    assert cited.marks == ('X',)
    assert cited.claims == ()
    assert cited.passages == ()
    assert cited.locator_status == 'unmapped'
    with pytest.raises(ValueError, match='claim locators are unmapped'):
        cited.mapped_claims()
    with pytest.raises(ValueError, match='passage locators are unmapped'):
        cited.mapped_passages()


def test_unmapped_pair_load_is_not_an_empty_collision(tmp_path: Path) -> None:
    dataset = tmp_path / 'shipped.parquet'
    _write_cited_parquet(dataset, _SHIPPED_CITED, _SHIPPED_STRUCT)
    pairs = EpoProcessorCitationPairSource().load_pairs(dataset)
    assert pairs == (
        CitationPair(
            query_application_number='13817165',
            partner_application_number='14111139',
            marks=('X',),
            grade=2,
        ),
    )
    assert pairs[0].locator_status == 'unmapped'
    with pytest.raises(ValueError, match='claim locators are unmapped'):
        pairs[0].mapped_claims()
    assert StubCitationPairSource().load_pairs(dataset) == ()
    assert EpoProcessorCitationPairSource().load_pairs(tmp_path / 'missing.parquet') == ()


def test_empty_letter_row_is_unmapped_and_not_a_locator() -> None:
    cited = AnalyzeCited(categories=('L',))
    assert cited.grade == 0
    assert cited.locator_status == 'unmapped'
    assert cited.claims == ()
    assert cited.passages == ()
    with pytest.raises(ValueError, match='claim locators are unmapped'):
        cited.mapped_claims()


@pytest.mark.parametrize(
    'payload',
    [
        {'cited_id': '14111139', 'claim_numbers': [1]},
        {'cited_id': '14111139', 'relevant_claims': ['1']},
        {'cited_id': '14111139', 'cited_passages': ['col. 3']},
    ],
)
def test_undeclared_locator_column_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AnalyzeCited.model_validate(payload)


@pytest.mark.parametrize(
    'claims',
    [
        'X',
        '1-3',
        ('not-a-number',),
        ('',),
        (0,),
        (-1,),
        (None,),
        (True,),
    ],
)
def test_unparseable_claim_locators_fail_closed(claims: object) -> None:
    with pytest.raises(ValidationError):
        AnalyzeCited.model_validate({'cited_id': '14111139', 'claims': claims})


@pytest.mark.parametrize(
    'passages',
    [
        'X',
        ('X',),
        ('Y',),
        ('A',),
        ('',),
        (1,),
        (None,),
    ],
)
def test_unparseable_passage_locators_fail_closed(passages: object) -> None:
    with pytest.raises((TypeError, ValidationError)):
        AnalyzeCited.model_validate({'cited_id': '14111139', 'passages': passages})


def test_category_letter_is_not_a_claim_or_passage_locator() -> None:
    cited = AnalyzeCited(categories=('XY',))
    assert cited.marks == ('X', 'Y')
    assert cited.locator_status == 'unmapped'
    assert cited.claims == ()
    assert cited.passages == ()
    with pytest.raises(ValidationError):
        AnalyzeCited(categories=('X',), passages=('X',))
    with pytest.raises(ValidationError):
        AnalyzeCited(categories=('X',), claims=('X',))


def test_parseable_locators_are_mapped() -> None:
    cited = AnalyzeCited.model_validate({
        'cited_id': '14111139',
        'categories': ['X'],
        'paths': ['14111139.json'],
        'claims': ['1', 3],
        'passages': ['col. 3, lines 5-10'],
    })
    assert cited.locator_status == 'mapped'
    assert cited.mapped_claims() == (1, 3)
    assert cited.mapped_passages() == ('col. 3, lines 5-10',)


def test_passage_only_row_does_not_map_claims() -> None:
    cited = AnalyzeCited(passages=('Fig. 2',))
    assert cited.locator_status == 'mapped'
    assert cited.mapped_passages() == ('Fig. 2',)
    with pytest.raises(ValueError, match='claim locators are unmapped'):
        cited.mapped_claims()


def test_load_pairs_keeps_mapped_locators(tmp_path: Path) -> None:
    dataset = tmp_path / 'mapped.parquet'
    cited = {
        **_SHIPPED_CITED,
        'claims': [1, 2],
        'passages': ['Fig. 2'],
    }
    fields = [
        *_SHIPPED_STRUCT,
        ('claims', pa.list_(pa.int64())),
        ('passages', pa.list_(pa.string())),
    ]
    _write_cited_parquet(dataset, cited, fields)
    pairs = EpoProcessorCitationPairSource().load_pairs(dataset)
    assert len(pairs) == 1
    assert pairs[0].locator_status == 'mapped'
    assert pairs[0].mapped_claims() == (1, 2)
    assert pairs[0].mapped_passages() == ('Fig. 2',)


def test_load_pairs_does_not_skip_unparseable_locators(tmp_path: Path) -> None:
    dataset = tmp_path / 'bad.parquet'
    cited = {**_SHIPPED_CITED, 'claims': ['not-a-number']}
    fields = [*_SHIPPED_STRUCT, ('claims', pa.list_(pa.string()))]
    _write_cited_parquet(dataset, cited, fields)
    with pytest.raises(ValidationError):
        EpoProcessorCitationPairSource().load_pairs(dataset)
