"""Paired X/A deltas and union-Y on planted encode-shard intensities."""

from __future__ import annotations

import pytest
import torch

from ip_claim.collision.collide import PatentEmbeddingRecord
from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.data.citation_pairs import CitationPair, St14Mark
from ip_claim.collision.diagnose import (
    QueryPartnerScore,
    intensity_index,
    judge_shards,
    n_entropy,
    paired_delta_split,
    query_macro_unpaid_fraction_xa,
    query_macro_xa_filter,
    score_cited_pairs,
    top_slot_overlap,
    union_y_split,
)


def _record(app: str, n_claim: list[float], n_full: list[float]) -> PatentEmbeddingRecord:
    return PatentEmbeddingRecord(
        application_number=app,
        z_d=torch.zeros(2),
        n_entity_claim=torch.tensor(n_claim),
        n_entity_full=torch.tensor(n_full),
        n_relation_claim=torch.zeros(2),
        n_relation_full=torch.zeros(2),
    )


def _pair(query: str, partner: str, mark: St14Mark) -> CitationPair:
    return CitationPair(
        query_application_number=query,
        partner_application_number=partner,
        marks=(mark,),
    )


def test_intensity_index_keeps_only_claim_and_full_product() -> None:
    kept = _record('1', [1.0, 0.0], [0.0, 1.0])
    claim_only = PatentEmbeddingRecord(
        application_number='2',
        z_d=torch.zeros(2),
        n_entity_claim=torch.tensor([1.0, 0.0]),
    )
    full_only = PatentEmbeddingRecord(
        application_number='3',
        z_d=torch.zeros(2),
        n_entity_full=torch.tensor([0.0, 1.0]),
    )
    index = intensity_index((kept, claim_only, full_only))
    assert set(index.claim) == {'1'}
    assert set(index.full) == {'1'}


def test_score_cited_pairs_drops_missing_letter_or_intensity() -> None:
    covering = Covering(CoveringKnobs())
    records = (
        _record('1', [8.0, 0.0], [8.0, 0.0]),
        _record('11', [8.0, 0.0], [8.0, 0.0]),
    )
    pairs = (
        _pair('1', '11', 'X'),
        CitationPair(query_application_number='1', partner_application_number='11'),
        _pair('9', '11', 'A'),
    )
    scores = score_cited_pairs(pairs, intensity_index(records), covering)
    assert len(scores) == 1
    assert scores[0].query == '1'
    assert scores[0].partner == '11'
    assert scores[0].mark == 'X'
    leftover = covering.reciprocal_score(
        torch.tensor([[8.0, 0.0]]),
        torch.tensor([[8.0, 0.0]]),
    ).residual.reshape(-1)
    assert scores[0].residual == pytest.approx(tuple(float(slot) for slot in leftover.tolist()))


def test_query_macro_xa_names_empty_join_versus_null_frac() -> None:
    disjoint = (
        QueryPartnerScore(
            query='1',
            partner='11',
            mark='X',
            unpaid=1.59,
            covering=0.2,
            demand_l1=4.0,
        ),
        QueryPartnerScore(
            query='2',
            partner='21',
            mark='A',
            unpaid=0.0,
            covering=1.0,
            demand_l1=4.0,
        ),
    )
    empty_demand = (
        QueryPartnerScore(
            query='1',
            partner='11',
            mark='X',
            unpaid=0.0,
            covering=0.0,
            demand_l1=0.0,
        ),
        QueryPartnerScore(
            query='1',
            partner='13',
            mark='A',
            unpaid=0.0,
            covering=0.0,
            demand_l1=0.0,
        ),
    )
    missing = query_macro_xa_filter(disjoint)
    null_frac = query_macro_xa_filter(empty_demand)
    assert query_macro_unpaid_fraction_xa(disjoint) is None
    assert query_macro_unpaid_fraction_xa(empty_demand) is None
    assert missing.reason == 'empty_xa'
    assert missing.n_queries_xa == 0
    assert null_frac.reason == 'null_frac'
    assert null_frac.n_queries_xa == 1


def test_paired_delta_fraction_is_not_the_mean_tie() -> None:
    covering = Covering(CoveringKnobs())
    records = (
        _record('1', [8.0, 0.0], [8.0, 0.0]),
        _record('2', [8.0, 0.0], [8.0, 0.0]),
        _record('11', [8.0, 0.0], [8.0, 0.0]),
        _record('12', [8.0, 0.0], [0.0, 8.0]),
        _record('13', [8.0, 0.0], [2.0, 0.0]),
    )
    pairs = (
        _pair('1', '11', 'X'),
        _pair('1', '13', 'A'),
        _pair('2', '12', 'X'),
        _pair('2', '13', 'A'),
    )
    scores = score_cited_pairs(pairs, intensity_index(records), covering)
    report = paired_delta_split(scores)
    assert report.queries_with_x_and_a == 2
    assert report.fraction_x_below_a == pytest.approx(0.5)
    assert report.median_u_x_minus_a is not None


def test_union_y_drops_unpaid_below_a() -> None:
    covering = Covering(CoveringKnobs())
    records = (
        _record('1', [4.0, 4.0], [0.0, 0.0]),
        _record('21', [0.0, 0.0], [8.0, 0.0]),
        _record('22', [0.0, 0.0], [0.0, 8.0]),
        _record('31', [0.0, 0.0], [1.0, 1.0]),
        _record('41', [0.0, 0.0], [8.0, 8.0]),
    )
    pairs = (
        _pair('1', '21', 'Y'),
        _pair('1', '22', 'Y'),
        _pair('1', '31', 'A'),
        _pair('1', '41', 'X'),
    )
    index = intensity_index(records)
    scores = score_cited_pairs(pairs, index, covering)
    report = union_y_split(scores, index, covering)
    assert report.queries_with_two_y == 1
    assert report.median_union_u is not None
    assert report.median_a_u is not None
    assert report.median_union_u < report.median_a_u
    assert report.fraction_union_below_a == pytest.approx(1.0)
    single = covering(index.claim['1'], index.full['21'])
    assert report.median_union_u < float(single.unpaid_mass.item())


def test_inventory_entropy_and_overlap_distinguish_smear() -> None:
    peaked = torch.tensor([8.0, 0.0, 0.0, 0.0])
    smear = torch.ones(4)
    assert n_entropy(smear) > n_entropy(peaked)
    assert top_slot_overlap(peaked, peaked, top_k=1) == pytest.approx(1.0)
    assert top_slot_overlap(peaked, smear, top_k=1) < 1.0


def test_judge_shards_fills_eval_and_test() -> None:
    covering = Covering(CoveringKnobs())
    records = (
        _record('1', [4.0, 4.0], [0.0, 0.0]),
        _record('21', [0.0, 0.0], [8.0, 0.0]),
        _record('22', [0.0, 0.0], [0.0, 8.0]),
        _record('31', [0.0, 0.0], [1.0, 1.0]),
        _record('41', [0.0, 0.0], [8.0, 8.0]),
    )
    pairs = (
        _pair('1', '21', 'Y'),
        _pair('1', '22', 'Y'),
        _pair('1', '31', 'A'),
        _pair('1', '41', 'X'),
    )
    judgment = judge_shards(
        records,
        eval_pairs=pairs,
        test_pairs=pairs,
        covering=covering,
        overlap_per_letter=2,
        seed=0,
    )
    assert judgment.encoded_apps == 5
    assert judgment.scored_pairs == 8
    assert judgment.eval_union_y.fraction_union_below_a == pytest.approx(1.0)
    assert judgment.eval_paired.queries_with_x_and_a == 1
    assert 'X' in judgment.inventory.letters


def test_stored_covering_n_leftover_moves_when_kept_addend_is_present() -> None:
    """Occupy-only stored n is not leftover of occupy plus a kept addend."""
    covering = Covering(CoveringKnobs(sigma=1.0))
    occupy = [1.0, 1.0]
    phi = [1.5, 1.5]
    occupy_only = score_cited_pairs(
        (_pair('1', '11', 'X'),),
        intensity_index((_record('1', occupy, occupy), _record('11', occupy, occupy))),
        covering,
    )
    with_kept = score_cited_pairs(
        (_pair('1', '11', 'X'),),
        intensity_index((_record('1', occupy, occupy), _record('11', occupy, phi))),
        covering,
    )
    assert occupy_only[0].unpaid != with_kept[0].unpaid
    assert occupy_only[0].residual != with_kept[0].residual
