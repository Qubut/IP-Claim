"""Unit tests for covering rank metrics and epo-processor pair loading."""

from __future__ import annotations

import hashlib
import json
import math
import pickle  # noqa: S403
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning.pytorch as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch_geometric.data import Data

from ip_claim.collision.collide import (
    CollisionEvalRequest,
    CoveringRankPool,
    PatentEmbeddingRecord,
    corpus_rank_banks,
    cpc_hard_rank_positions,
    empty_collision_eval_result,
    evaluate_collision_ranking,
    prepare_ranking_batch,
    retrieval_labels,
    unique_query_records,
)
from ip_claim.collision.config import (
    SHIPPED_COLLISION_EVAL_PATHS,
    CollisionEvalConfig,
    CollisionPrefixMode,
    KeepGridCell,
)
from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.data import (
    HUPD_APPLICATION,
    AnalyzeCited,
    CitationPair,
    EpoProcessorCitationPairSource,
    StubCitationPairSource,
    split_pairs_by_query,
)
from ip_claim.collision.eval import (
    CollisionEncodeCollate,
    CollisionEncodeProbe,
    CollisionEncodeShardStore,
    CollisionEncodeStep,
    CollisionEncodeWriter,
    CollisionEvalQueue,
    encode_collision_records,
    encode_shard_root,
    explain_predict_strategy,
    hupd_stem_index,
    rank_covering_from_records,
    rank_split_pairs,
    resolve_encode_devices,
    resolve_encoded_corpus,
    resolve_explain_device,
    resolve_rank_devices,
)
from ip_claim.collision.rank_grid import KEEP_GRID_JSON, rank_keep_grid
from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmExample
from ip_claim.ssv.graph_batch import PatentGraphBatch, graph_batch_from_hupd_dict
from ip_claim.ssv.model import SoftTrunkModel, TrunkExport
from tests._ssv_fixtures import lightning_trainer_stub


def _stub_mlm_collator(examples: Sequence[SoftMlmExample]) -> SoftMlmBatch:
    n = len(examples)
    ones = torch.ones(n, 4, dtype=torch.long)
    return SoftMlmBatch(
        input_ids=ones,
        unmasked_input_ids=ones,
        attention_mask=ones,
        labels=ones,
        graphs=tuple(example.graph for example in examples),
    )


def _n_record(
    app: str,
    n_claim: list[float],
    n_full: list[float],
    *,
    cpc: str | None = None,
    n_rel_claim: list[float] | None = None,
    n_rel_full: list[float] | None = None,
) -> PatentEmbeddingRecord:
    claim = torch.tensor(n_claim)
    full = torch.tensor(n_full)
    rel_q = torch.tensor(n_rel_claim if n_rel_claim is not None else [0.0, 0.0])
    rel_d = torch.tensor(n_rel_full if n_rel_full is not None else [0.0, 0.0])
    return PatentEmbeddingRecord(
        application_number=app,
        z_d=torch.zeros(2),
        n_entity_claim=claim,
        n_entity_full=full,
        n_relation_claim=rel_q,
        n_relation_full=rel_d,
        cpc_section=cpc,
    )


def test_application_number_from_fixture_path(fixtures_dir: Path) -> None:
    path = fixtures_dir / 'hupd' / '13817165.json'
    assert HUPD_APPLICATION.validate_python(str(path)) == '13817165'
    assert HUPD_APPLICATION.validate_python('/data/hupd/all-years/2013/2013/14033096.json') == (
        '14033096'
    )
    assert HUPD_APPLICATION.validate_python('not-an-app.json') is None


def test_analyze_cited_clef_ip_grade_from_categories() -> None:
    cited_x = AnalyzeCited(categories=('X',))
    assert cited_x.marks == ('X',)
    assert cited_x.grade == 2
    cited_y = AnalyzeCited(categories=('Y',))
    assert cited_y.marks == ('Y',)
    assert cited_y.grade == 2
    cited_xy = AnalyzeCited(categories=('XY',))
    assert cited_xy.marks == ('X', 'Y')
    assert cited_xy.grade == 2
    cited_a = AnalyzeCited(categories=('A',))
    assert cited_a.marks == ('A',)
    assert cited_a.grade == 1
    cited_empty = AnalyzeCited()
    assert cited_empty.marks == ()
    assert cited_empty.grade == 0
    cited_letter = AnalyzeCited(categories=('L',))
    assert cited_letter.marks == ('L',)
    assert cited_letter.grade == 0


def test_split_pairs_by_query_is_disjoint_and_seeded() -> None:
    pairs = tuple(
        CitationPair(
            query_application_number=f'{10_000_000 + index}',
            partner_application_number=f'{20_000_000 + index}',
            marks=('X',),
            grade=2,
        )
        for index in range(10)
    )
    train, eval_pairs, test = split_pairs_by_query(
        pairs,
        train=0.6,
        eval_fraction=0.2,
        test=0.2,
        seed=42,
    )
    train_q = {pair.query_application_number for pair in train}
    eval_q = {pair.query_application_number for pair in eval_pairs}
    test_q = {pair.query_application_number for pair in test}
    assert len(train_q) == 6
    assert len(eval_q) == 2
    assert len(test_q) == 2
    assert train_q.isdisjoint(eval_q)
    assert train_q.isdisjoint(test_q)
    assert eval_q.isdisjoint(test_q)
    again = split_pairs_by_query(pairs, train=0.6, eval_fraction=0.2, test=0.2, seed=42)
    assert tuple(pair.query_application_number for pair in again[0]) == tuple(
        pair.query_application_number for pair in train
    )
    capped = split_pairs_by_query(
        pairs,
        train=0.6,
        eval_fraction=0.2,
        test=0.2,
        seed=42,
        query_limit=1,
    )
    assert len({pair.query_application_number for pair in capped[0]}) == 1
    assert len({pair.query_application_number for pair in capped[1]}) == 1
    assert len({pair.query_application_number for pair in capped[2]}) == 1


def test_query_limit_prefers_x_bearing_queries_inside_each_split() -> None:
    apps = tuple(f'{10_000_000 + index}' for index in range(10))
    ordered = sorted(
        apps,
        key=lambda app: hashlib.sha256(f'42:{app}'.encode()).hexdigest(),
    )
    eval_first, eval_second = ordered[6], ordered[7]
    pairs = tuple(
        CitationPair(
            query_application_number=app,
            partner_application_number=f'{20_000_000 + index}',
            marks=('Y',) if app == eval_first else ('X',),
            grade=2,
        )
        for index, app in enumerate(apps)
    )
    unlimited = split_pairs_by_query(
        pairs,
        train=0.6,
        eval_fraction=0.2,
        test=0.2,
        seed=42,
    )
    assert {pair.query_application_number for pair in unlimited[1]} == {
        eval_first,
        eval_second,
    }
    limited = split_pairs_by_query(
        pairs,
        train=0.6,
        eval_fraction=0.2,
        test=0.2,
        seed=42,
        query_limit=1,
    )
    assert {pair.query_application_number for pair in limited[1]} == {eval_second}
    assert limited[1][0].marks == ('X',)


def test_rank_split_pairs_three_way_covering_order() -> None:
    slots = 10
    decoy = _n_record('decoy', [0.0] * slots, [0.0] * slots, cpc='H')
    records = [decoy]
    pairs: list[CitationPair] = []
    for index in range(slots):
        demand = [0.0] * slots
        demand[index] = 2.0
        paid = [0.0] * slots
        paid[index] = 8.0
        partial = [0.0] * slots
        partial[index] = 1.0
        query_app = f'{10_000_000 + index}'
        x_app = f'{20_000_000 + index}'
        a_app = f'{30_000_000 + index}'
        records.extend((
            _n_record(query_app, demand, [0.0] * slots, cpc='A'),
            _n_record(x_app, [0.0] * slots, paid, cpc='A'),
            _n_record(a_app, [0.0] * slots, partial, cpc='A'),
        ))
        pairs.extend((
            CitationPair(
                query_application_number=query_app,
                partner_application_number=x_app,
                marks=('X',),
                grade=2,
            ),
            CitationPair(
                query_application_number=query_app,
                partner_application_number=a_app,
                marks=('A',),
                grade=1,
            ),
        ))
    corpus = tuple(records)
    by_app = {row.application_number: row for row in corpus}
    covering = Covering(CoveringKnobs())
    sparse = Covering(CoveringKnobs(slot_top_k=1, row_top_k=1))
    empty = empty_collision_eval_result((1, 5))
    train, eval_pairs, test = split_pairs_by_query(
        tuple(pairs),
        train=0.6,
        eval_fraction=0.2,
        test=0.2,
        seed=42,
    )
    assert {pair.query_application_number for pair in train}
    assert {pair.query_application_number for pair in eval_pairs}
    assert {pair.query_application_number for pair in test}
    for split_pairs in (train, eval_pairs, test):
        table = rank_split_pairs(
            split_pairs,
            by_app,
            corpus,
            empty=empty,
            recall_k=(1, 5),
            covering=covering,
        )
        kept = rank_split_pairs(
            split_pairs,
            by_app,
            corpus,
            empty=empty,
            recall_k=(1, 5),
            covering=sparse,
        )
        assert table.queries > 0
        assert table.unpaid_x < table.unpaid_a < table.unpaid_random
        assert kept.unpaid_x < kept.unpaid_a < kept.unpaid_random


def test_dense_smear_inverts_x_a_sparse_demand_restores_order() -> None:
    tail = [2.0] * 15
    query = _n_record('q', [20.0, *tail], [0.0] * 16, cpc='A')
    paid = _n_record('x', [0.0] * 16, [20.0, *([0.3] * 15)], cpc='A')
    background = _n_record('a', [0.0] * 16, [1.0, *([4.0] * 15)], cpc='A')
    decoy = _n_record('r', [0.0] * 16, [0.0, *([4.0] * 15)], cpc='A')
    records = (query, paid, background, decoy)
    queries = (query, query)
    dense = evaluate_collision_ranking(
        queries,
        records,
        ('x', 'a'),
        relevances=(2.0, 1.0),
        marks=(('X',), ('A',)),
        k_values=(1,),
        covering=Covering(CoveringKnobs()),
    )
    assert dense.unpaid_x > dense.unpaid_a
    sparse = evaluate_collision_ranking(
        queries,
        records,
        ('x', 'a'),
        relevances=(2.0, 1.0),
        marks=(('X',), ('A',)),
        k_values=(1,),
        covering=Covering(CoveringKnobs(slot_mass_keep=0.35)),
    )
    assert sparse.unpaid_x < sparse.unpaid_a < sparse.unpaid_random
    planted = evaluate_collision_ranking(
        queries,
        records,
        ('x', 'a'),
        relevances=(2.0, 1.0),
        marks=(('X',), ('A',)),
        k_values=(1,),
        covering=Covering(CoveringKnobs(slot_top_k=1)),
    )
    assert planted.unpaid_x < planted.unpaid_a < planted.unpaid_random


def test_hupd_stem_index_finds_nested_json(tmp_path: Path) -> None:
    nested = tmp_path / 'all-years' / '2013' / '2013'
    nested.mkdir(parents=True)
    path = nested / '14033096.json'
    path.write_text('{}', encoding='utf-8')
    assert hupd_stem_index(tmp_path)['14033096'] == path


def test_evaluate_collision_ranking_perfect_match() -> None:
    records = (
        _n_record('13817165', [2.0, 1.0, 0.0], [0.0, 0.0, 0.0], cpc='A'),
        _n_record('14111139', [0.0, 0.0, 0.0], [8.0, 8.0, 0.0], cpc='A'),
        _n_record('14112715', [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], cpc='H'),
    )
    result = evaluate_collision_ranking(
        (records[0],),
        records,
        ('14111139',),
        relevances=(2.0,),
        k_values=(1, 2),
    )
    assert result.queries == 1
    assert result.recall_at_k[1] == pytest.approx(1.0)
    assert result.particular_recall_at_k[1] == pytest.approx(1.0)
    assert result.cpc_hard_recall_at_k[1] == pytest.approx(1.0)
    assert result.particular_cpc_hard_recall_at_k[1] == pytest.approx(1.0)
    assert result.mrr == pytest.approx(1.0)
    assert result.particular_mrr == pytest.approx(1.0)
    assert result.ndcg_at_k[2] == pytest.approx(1.0)
    assert result.unpaid_x < result.unpaid_random


def test_evaluate_query_tile_matches_full_table() -> None:
    records = (
        _n_record('q1', [2.0, 1.0, 0.0], [0.0, 0.0, 0.0], cpc='A'),
        _n_record('q2', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], cpc='A'),
        _n_record('p1', [0.0, 0.0, 0.0], [8.0, 8.0, 0.0], cpc='A'),
        _n_record('p2', [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], cpc='H'),
    )
    queries = (records[0], records[0], records[1])
    partners = ('p1', 'p2', 'p1')
    shared = {
        'relevances': (2.0, 1.0, 2.0),
        'marks': (('X',), ('A',), ('X',)),
        'k_values': (1, 2),
    }
    full = evaluate_collision_ranking(queries, records, partners, query_tile=8, **shared)
    tiled = evaluate_collision_ranking(queries, records, partners, query_tile=1, **shared)
    assert tiled.queries_x == full.queries_x
    assert tiled.unpaid_x == pytest.approx(full.unpaid_x)
    assert tiled.unpaid_a == pytest.approx(full.unpaid_a)
    assert tiled.unpaid_random == pytest.approx(full.unpaid_random)
    assert tiled.covering_x == pytest.approx(full.covering_x)
    assert tiled.recall_at_k[1] == pytest.approx(full.recall_at_k[1])
    assert tiled.mrr == pytest.approx(full.mrr)


def test_unique_query_records_keeps_first_seen() -> None:
    first = _n_record('q1', [1.0], [1.0])
    second = _n_record('q2', [2.0], [2.0])
    again = _n_record('q1', [9.0], [9.0])
    unique, pair_rows = unique_query_records((first, second, again))
    assert tuple(row.application_number for row in unique) == ('q1', 'q2')
    assert pair_rows == (0, 1, 0)
    assert unique[0].n_entity_claim is not None
    assert unique[0].n_entity_claim.tolist() == [1.0]


def test_retrieval_labels_stay_on_score_device() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.2]])
    labeled = retrieval_labels(
        Data(
            scores=scores,
            pair_rows=torch.tensor([0]),
            partner_indices=torch.tensor([1]),
            relevances=torch.tensor([2.0]),
        ),
        min_grade=1.0,
    )
    assert labeled.preds.device == scores.device
    assert labeled.target.device == scores.device
    assert labeled.query_index.device == scores.device
    assert labeled.target[0, 1].item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_retrieval_labels_follow_cuda_scores() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.2]], device='cuda')
    labeled = retrieval_labels(
        Data(
            scores=scores,
            pair_rows=torch.tensor([0], device='cuda'),
            partner_indices=torch.tensor([1], device='cuda'),
            relevances=torch.tensor([2.0], device='cuda'),
        ),
        min_grade=1.0,
    )
    assert labeled.preds.device.type == 'cuda'
    assert labeled.target.device == scores.device
    assert labeled.query_index.device == scores.device


def test_prepare_ranking_batch_unique_query_rows() -> None:
    records = (
        _n_record('q1', [2.0, 1.0, 0.0], [0.0, 0.0, 0.0], cpc='A'),
        _n_record('p1', [0.0, 0.0, 0.0], [8.0, 8.0, 0.0], cpc='A'),
        _n_record('p2', [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], cpc='H'),
    )
    batch = prepare_ranking_batch(
        CollisionEvalRequest(
            query_records=(records[0], records[0]),
            corpus_records=records,
            partner_apps=('p1', 'p2'),
            relevances=(2.0, 1.0),
            marks=(('X',), ('A',)),
        ),
        Covering(CoveringKnobs()),
    )
    assert batch is not None
    assert batch.scores.size(0) == 1
    assert batch.pair_rows.tolist() == [0, 0]
    assert batch.partner_indices.numel() == 2


def test_cpc_hard_rank_positions_restricted_pool() -> None:
    records = (
        _n_record('13817165', [2.0, 0.0], [0.0, 0.0], cpc='A'),
        _n_record('14111139', [0.0, 0.0], [1.0, 0.0], cpc='A'),
        _n_record('14111140', [0.0, 0.0], [8.0, 0.0], cpc='A'),
        _n_record('14112715', [0.0, 0.0], [8.0, 0.0], cpc='H'),
    )
    positions = cpc_hard_rank_positions(
        (records[0],),
        records,
        ('14111139',),
    )
    assert positions.tolist() == [2]

    result = evaluate_collision_ranking(
        (records[0],),
        records,
        ('14111139',),
        k_values=(1, 2),
    )
    assert result.cpc_hard_recall_at_k[1] == pytest.approx(0.0)
    assert result.cpc_hard_recall_at_k[2] == pytest.approx(1.0)


def test_evaluate_collision_ranking_mixed_clef_ip_grades() -> None:
    records = (
        _n_record('qA', [2.0, 0.0], [0.0, 0.0], cpc='A'),
        _n_record('pA', [0.0, 0.0], [8.0, 0.0], cpc='A'),
        _n_record('qX', [0.0, 2.0], [0.0, 0.0], cpc='H'),
        _n_record('pX', [0.0, 0.0], [0.0, 1.0], cpc='H'),
        _n_record('hard', [0.0, 0.0], [0.0, 8.0], cpc='H'),
    )
    result = evaluate_collision_ranking(
        (records[0], records[2]),
        records,
        ('pA', 'pX'),
        relevances=(1.0, 2.0),
        k_values=(1, 2),
    )
    assert result.recall_at_k[1] == pytest.approx(0.5)
    assert result.recall_at_k[2] == pytest.approx(1.0)
    assert result.particular_recall_at_k[1] == pytest.approx(0.0)
    assert result.particular_recall_at_k[2] == pytest.approx(1.0)
    assert result.mrr == pytest.approx(0.75)
    assert result.particular_mrr == pytest.approx(0.5)
    assert result.unpaid_x > result.unpaid_a
    grade_zero = evaluate_collision_ranking(
        (records[0],),
        records,
        ('pA',),
        relevances=(0.0,),
        k_values=(1,),
    )
    assert grade_zero.recall_at_k[1] == pytest.approx(0.0)
    assert grade_zero.particular_recall_at_k[1] == pytest.approx(0.0)
    assert grade_zero.mrr == pytest.approx(0.0)
    assert grade_zero.ndcg_at_k[1] == pytest.approx(0.0)


def test_evaluate_collision_ranking_y_is_not_novelty_x() -> None:
    records = (
        _n_record('q', [4.0, 0.0], [0.0, 0.0], cpc='A'),
        _n_record('pX', [0.0, 0.0], [8.0, 0.0], cpc='A'),
        _n_record('pY', [0.0, 0.0], [0.0, 0.0], cpc='A'),
        _n_record('pA', [0.0, 0.0], [1.0, 0.0], cpc='A'),
        _n_record('decoy', [0.0, 0.0], [0.0, 0.0], cpc='A'),
    )
    queries = (records[0], records[0], records[0])
    result = evaluate_collision_ranking(
        queries,
        records,
        ('pX', 'pY', 'pA'),
        relevances=(2.0, 2.0, 1.0),
        marks=(('X',), ('Y',), ('A',)),
        k_values=(1, 2),
    )
    assert result.queries == 1
    assert result.unpaid_y > result.unpaid_x
    assert result.unpaid_x < result.unpaid_a < result.unpaid_random
    assert result.covering_x > result.covering_a > result.covering_y
    xy = evaluate_collision_ranking(
        (records[0],),
        records,
        ('pY',),
        relevances=(2.0,),
        marks=(('X', 'Y'),),
        k_values=(1,),
    )
    assert xy.unpaid_x == pytest.approx(result.unpaid_y)
    assert math.isnan(xy.unpaid_y)
    assert xy.queries_x == 1
    assert xy.queries_y == 0


def test_evaluate_collision_ranking_empty_x_is_not_paid() -> None:
    records = (
        _n_record('q', [4.0, 0.0], [0.0, 0.0], cpc='A'),
        _n_record('pY', [0.0, 0.0], [0.0, 0.0], cpc='A'),
        _n_record('pA', [0.0, 0.0], [1.0, 0.0], cpc='A'),
        _n_record('decoy', [0.0, 0.0], [0.0, 0.0], cpc='A'),
    )
    result = evaluate_collision_ranking(
        (records[0], records[0]),
        records,
        ('pY', 'pA'),
        relevances=(2.0, 1.0),
        marks=(('Y',), ('A',)),
        k_values=(1,),
    )
    assert result.queries_x == 0
    assert result.queries_y == 1
    assert result.queries_a == 1
    assert math.isnan(result.unpaid_x)
    assert math.isnan(result.covering_x)
    assert not (result.unpaid_x < result.unpaid_a < result.unpaid_random)


def test_evaluate_collision_ranking_query_macro_not_pair_weighted() -> None:
    slots = 2
    heavy = _n_record('qHeavy', [2.0, 0.0], [0.0, 0.0], cpc='A')
    light = _n_record('qLight', [10.0, 0.0], [0.0, 0.0], cpc='A')
    decoy = _n_record('decoy', [0.0, 0.0], [0.0, 0.0], cpc='A')
    partners = [_n_record(f'a{index}', [0.0] * slots, [0.0] * slots, cpc='A') for index in range(5)]
    light_partner = _n_record('aLight', [0.0, 0.0], [0.0, 0.0], cpc='A')
    x_one = _n_record('x1', [0.0, 0.0], [0.0, 0.0], cpc='A')
    x_two = _n_record('x2', [0.0, 0.0], [0.0, 0.0], cpc='A')
    x_other = _n_record('xOther', [0.0, 0.0], [0.0, 0.0], cpc='A')
    records = (heavy, light, decoy, light_partner, x_one, x_two, x_other, *partners)
    queries = (heavy,) * 5 + (light,) + (heavy, heavy, light)
    result = evaluate_collision_ranking(
        queries,
        records,
        (
            *(row.application_number for row in partners),
            'aLight',
            'x1',
            'x2',
            'xOther',
        ),
        relevances=(1.0,) * 6 + (2.0, 2.0, 2.0),
        marks=(('A',),) * 6 + (('X',), ('X',), ('X',)),
        k_values=(1,),
    )
    assert result.queries == 2
    assert result.unpaid_a == pytest.approx(6.0)
    assert result.unpaid_x == pytest.approx(6.0)
    assert result.covering_x == pytest.approx(0.0)
    assert result.covering_a == pytest.approx(0.0)


def test_relation_mix_can_reorder_covering_ranks() -> None:
    records = (
        _n_record(
            'q',
            [2.0, 0.0],
            [0.0, 0.0],
            n_rel_claim=[2.0, 0.0],
            n_rel_full=[0.0, 0.0],
        ),
        _n_record(
            'entity_hit',
            [0.0, 0.0],
            [1.0, 0.0],
            n_rel_claim=[0.0, 0.0],
            n_rel_full=[0.0, 0.0],
        ),
        _n_record(
            'relation_hit',
            [0.0, 0.0],
            [0.0, 0.0],
            n_rel_claim=[0.0, 0.0],
            n_rel_full=[8.0, 0.0],
        ),
    )
    entity_only = evaluate_collision_ranking(
        (records[0],),
        records,
        ('relation_hit',),
        k_values=(1,),
        covering=Covering(CoveringKnobs(lambda_relation=0.0)),
    )
    mixed = evaluate_collision_ranking(
        (records[0],),
        records,
        ('relation_hit',),
        k_values=(1,),
        covering=Covering(CoveringKnobs(lambda_relation=1.0)),
    )
    assert entity_only.recall_at_k[1] == pytest.approx(0.0)
    assert mixed.recall_at_k[1] == pytest.approx(1.0)


def test_evaluate_collision_ranking_skips_cosine_when_intensities_missing() -> None:
    records = (
        PatentEmbeddingRecord(
            application_number='q', z_d=torch.tensor([1.0, 0.0]), cpc_section='A'
        ),
        PatentEmbeddingRecord(
            application_number='p', z_d=torch.tensor([0.99, 0.01]), cpc_section='A'
        ),
    )
    result = evaluate_collision_ranking((records[0],), records, ('p',), k_values=(1,))
    assert result.queries == 0
    assert result.recall_at_k[1] == pytest.approx(0.0)


def test_evaluate_collision_ranking_empty_corpus() -> None:
    result = evaluate_collision_ranking((), (), ())
    assert result.queries == 0
    assert result.recall_at_k[5] == pytest.approx(0.0)
    assert result.mrr == pytest.approx(0.0)
    assert result.ndcg_at_k[5] == pytest.approx(0.0)


def _write_analyze_fixture(path: Path) -> None:
    table = pa.table({
        'epo_patent_id': ['US13817165A1'],
        'epo_hupd_paths': pa.array([['13817165.json']], type=pa.list_(pa.string())),
        'cited_hupd': pa.array(
            [[{'cited_id': '14111139', 'categories': ['X'], 'paths': ['14111139.json']}]],
            type=pa.list_(
                pa.struct([
                    ('cited_id', pa.string()),
                    ('categories', pa.list_(pa.string())),
                    ('paths', pa.list_(pa.string())),
                ])
            ),
        ),
        'family_hupd': pa.array(
            [[]], type=pa.list_(pa.struct([('id', pa.string()), ('paths', pa.list_(pa.string()))]))
        ),
    })
    pq.write_table(table, path)


def test_epo_processor_pair_loader(tmp_path: Path) -> None:
    dataset = tmp_path / 'dataset.parquet'
    _write_analyze_fixture(dataset)
    pairs = EpoProcessorCitationPairSource().load_pairs(dataset)
    assert pairs == (
        CitationPair(
            query_application_number='13817165',
            partner_application_number='14111139',
            marks=('X',),
            grade=2,
        ),
    )


def test_epo_processor_pair_loader_nested_container_paths(tmp_path: Path) -> None:
    dataset = tmp_path / 'dataset.parquet'
    table = pa.table({
        'epo_patent_id': ['US2014075987A1'],
        'epo_hupd_paths': pa.array(
            [['/data/hupd/all-years/2013/2013/14033096.json']],
            type=pa.list_(pa.string()),
        ),
        'cited_hupd': pa.array(
            [
                [
                    {
                        'cited_id': '2012000245',
                        'categories': ['Y'],
                        'paths': ['/data/hupd/all-years/2011/2011/13174002.json'],
                    }
                ]
            ],
            type=pa.list_(
                pa.struct([
                    ('cited_id', pa.string()),
                    ('categories', pa.list_(pa.string())),
                    ('paths', pa.list_(pa.string())),
                ])
            ),
        ),
        'family_hupd': pa.array(
            [[]], type=pa.list_(pa.struct([('id', pa.string()), ('paths', pa.list_(pa.string()))]))
        ),
    })
    pq.write_table(table, dataset)
    pairs = EpoProcessorCitationPairSource().load_pairs(dataset)
    assert pairs == (
        CitationPair(
            query_application_number='14033096',
            partner_application_number='13174002',
            marks=('Y',),
            grade=2,
        ),
    )


def test_resolve_encode_devices_cpu_when_cuda_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_encode_devices(4, use_gpu=True) == ('cpu', 1, 'auto')


def test_resolve_encode_devices_auto_strategy_on_multi_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    assert resolve_encode_devices(4, use_gpu=True) == ('gpu', 4, 'auto')


def test_resolve_rank_devices_cpu_when_cuda_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_rank_devices(4, use_gpu=True) == (torch.device('cpu'),)


def test_resolve_rank_devices_all_visible_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    assert resolve_rank_devices(4, use_gpu=True) == tuple(
        torch.device(f'cuda:{index}') for index in range(4)
    )


def test_evaluate_two_cpu_shards_match_one_device() -> None:
    records = (
        _n_record('q1', [2.0, 1.0, 0.0], [0.0, 0.0, 0.0], cpc='A'),
        _n_record('q2', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], cpc='A'),
        _n_record('p1', [0.0, 0.0, 0.0], [8.0, 8.0, 0.0], cpc='A'),
        _n_record('p2', [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], cpc='H'),
    )
    queries = (records[0], records[0], records[1])
    partners = ('p1', 'p2', 'p1')
    shared = {
        'relevances': (2.0, 1.0, 2.0),
        'marks': (('X',), ('A',), ('X',)),
        'k_values': (1, 2),
    }
    one = evaluate_collision_ranking(
        queries, records, partners, devices=(torch.device('cpu'),), **shared
    )
    two = evaluate_collision_ranking(
        queries,
        records,
        partners,
        devices=(torch.device('cpu'), torch.device('cpu')),
        **shared,
    )
    assert two.unpaid_x == pytest.approx(one.unpaid_x)
    assert two.recall_at_k[1] == pytest.approx(one.recall_at_k[1])
    assert two.mrr == pytest.approx(one.mrr)


def test_corpus_rank_banks_keeps_full_replica_on_cpu_for_many_cuda() -> None:
    records = (
        _n_record('q1', [2.0, 1.0, 0.0], [1.0, 0.0, 0.0], cpc='A'),
        _n_record('p1', [0.0, 0.0, 0.0], [8.0, 8.0, 0.0], cpc='A'),
    )
    banks = corpus_rank_banks(
        records,
        tuple(torch.device(f'cuda:{index}') for index in range(4)),
    )
    assert banks is not None
    assert len(banks.entity_shards) == 1
    assert banks.entity_shards[0].device.type == 'cpu'
    assert banks.entity_shards[0].size(0) == len(records)
    assert banks.relation_shards[0].size(0) == len(records)


def test_evaluate_ray_cpu_workers_match_sequential() -> None:
    records = (
        _n_record('q1', [2.0, 1.0, 0.0], [0.0, 0.0, 0.0], cpc='A'),
        _n_record('q2', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], cpc='A'),
        _n_record('p1', [0.0, 0.0, 0.0], [8.0, 8.0, 0.0], cpc='A'),
        _n_record('p2', [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], cpc='H'),
    )
    queries = (records[0], records[0], records[1])
    partners = ('p1', 'p2', 'p1')
    shared = {
        'relevances': (2.0, 1.0, 2.0),
        'marks': (('X',), ('A',), ('X',)),
        'k_values': (1, 2),
        'query_tile': 1,
    }
    one = evaluate_collision_ranking(
        queries, records, partners, devices=(torch.device('cpu'),), **shared
    )
    banks = corpus_rank_banks(records, (torch.device('cpu'),))
    assert banks is not None
    pool = CoveringRankPool.ray_workers(
        Covering(CoveringKnobs()),
        banks,
        count=2,
        use_gpu=False,
    )
    try:
        two = evaluate_collision_ranking(
            queries,
            records,
            partners,
            devices=(torch.device('cpu'),),
            banks=banks,
            pool=pool,
            **shared,
        )
    finally:
        pool.close()
    assert pool.backend == 'ray'
    assert two.unpaid_x == pytest.approx(one.unpaid_x)
    assert two.recall_at_k[1] == pytest.approx(one.recall_at_k[1])
    assert two.mrr == pytest.approx(one.mrr)


def test_resolve_explain_device_cpu_when_gpu_not_requested() -> None:
    assert resolve_explain_device(use_gpu=False) == torch.device('cpu')


def test_resolve_explain_device_fails_loud_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        resolve_explain_device(use_gpu=True)


def test_resolve_explain_device_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert resolve_explain_device(use_gpu=True) == torch.device('cuda')


def test_explain_predict_strategy_spawns_on_many_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    spec = explain_predict_strategy(4, use_gpu=True)
    assert spec['accelerator'] == 'gpu'
    assert spec['devices'] == 4
    assert spec['strategy'] == 'ddp_spawn'


def test_explain_predict_strategy_stays_auto_on_one_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    spec = explain_predict_strategy(1, use_gpu=True)
    assert spec['devices'] == 1
    assert spec['strategy'] == 'auto'


def test_explain_collate_pickles_for_ddp_spawn() -> None:
    payload = CollisionEncodeCollate(collator=_stub_mlm_collator, inventory=None)
    clone = pickle.loads(pickle.dumps(payload))  # noqa: S301
    assert clone.inventory is None
    assert clone.collator is _stub_mlm_collator


def _probe(*, row_entropy: float = 0.5, apps_step: int = 1) -> CollisionEncodeProbe:
    return CollisionEncodeProbe.from_bank_stats(
        row_entropy=row_entropy,
        batch_usage=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        utilization_eps=1e-3,
        relation_row_entropy=0.4,
        relation_batch_usage=torch.tensor([1.0, 0.0]),
        prefix_l2=1.2,
        prefix_pairwise_cosine=0.0,
        zd_zg_cosine=0.35,
        apps_step=apps_step,
    )


def test_collision_encode_probe_peaked_vs_uniform() -> None:
    peaked = CollisionEncodeProbe.from_bank_stats(
        row_entropy=0.0,
        batch_usage=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        utilization_eps=1e-3,
        relation_row_entropy=0.0,
        relation_batch_usage=torch.tensor([1.0, 0.0]),
        prefix_l2=1.2,
        prefix_pairwise_cosine=0.0,
        zd_zg_cosine=0.4,
        apps_step=2,
    )
    assert peaked.assignment_perplexity == pytest.approx(1.0)
    assert peaked.entity_batch_utilization == pytest.approx(0.25)
    assert peaked.relation_batch_utilization == pytest.approx(0.5)
    assert peaked.usage_entropy == pytest.approx(0.0)

    uniform = CollisionEncodeProbe.from_bank_stats(
        row_entropy=math.log(4),
        batch_usage=torch.full((4,), 0.25),
        utilization_eps=1e-3,
        relation_row_entropy=math.log(2),
        relation_batch_usage=torch.full((2,), 0.5),
        prefix_l2=0.0,
        prefix_pairwise_cosine=0.0,
        zd_zg_cosine=0.0,
        apps_step=2,
    )
    assert uniform.assignment_perplexity == pytest.approx(4.0)
    assert uniform.entity_batch_utilization == pytest.approx(1.0)
    assert uniform.relation_batch_utilization == pytest.approx(1.0)
    assert uniform.usage_entropy == pytest.approx(math.log(4))


def test_prefix_l2_does_not_imply_constant_prefix() -> None:
    def probe(
        soft_tokens: torch.Tensor,
        z_d: torch.Tensor,
        z_g: torch.Tensor,
    ) -> CollisionEncodeProbe:
        return CollisionEncodeProbe.from_inventory(
            SimpleNamespace(soft_vocab=None),
            TrunkExport(soft_tokens=soft_tokens, z_d=z_d, z_g=z_g),
            None,
            torch.zeros(soft_tokens.size(0), 1, dtype=torch.long),
        )

    shared = torch.tensor([[22.8, 0.0, 0.0], [0.0, 22.8, 0.0]])
    same_energy = probe(shared.unsqueeze(1), shared, shared)
    assert same_energy.prefix_l2 == pytest.approx(22.8)
    assert same_energy.prefix_pairwise_cosine == pytest.approx(0.0)
    cloned = torch.tensor([[22.8, 0.0, 0.0], [22.8, 0.0, 0.0]])
    constant = probe(cloned.unsqueeze(1), cloned, cloned)
    assert constant.prefix_l2 == pytest.approx(22.8)
    assert constant.prefix_pairwise_cosine == pytest.approx(1.0)
    host_only = probe(torch.zeros(2, 4, 3), torch.ones(2, 3), torch.zeros(2, 3))
    assert host_only.prefix_l2 == pytest.approx(0.0)
    assert host_only.prefix_pairwise_cosine == pytest.approx(0.0)


def test_trunk_export_field_set_unchanged() -> None:
    assert set(TrunkExport.model_fields) == {'z_d', 'z_g', 'soft_tokens'}


def test_collision_eval_config_owns_encode_knobs(tmp_path: Path) -> None:
    path = tmp_path / 'collision.yaml'
    _ = path.write_text(
        """encode_batch_size: 64
dataloader_num_workers: 50
shard_flush_every: 50
prefix_mode: host_only
""",
        encoding='utf-8',
    )
    loaded = CollisionEvalConfig.from_yaml(path)
    assert loaded.encode_batch_size == 64
    assert loaded.dataloader_num_workers == 50
    assert loaded.shard_flush_every == 50
    assert loaded.prefix_mode == CollisionPrefixMode.host_only
    assert loaded.covering.sigma == pytest.approx(1.0)
    assert loaded.covering.sigma_edge == pytest.approx(1.0)
    assert loaded.covering.lambda_relation == pytest.approx(0.0)
    assert loaded.covering.row_top_k is None
    assert loaded.covering.slot_mass_keep is None
    assert loaded.contour_tau == pytest.approx(0.3)
    assert loaded.explain_top_n == 5
    assert loaded.reuse_encode is True
    assert loaded.rank_query_tile == 256


def test_collision_eval_config_loads_covering_knobs(tmp_path: Path) -> None:
    path = tmp_path / 'covering.yaml'
    _ = path.write_text(
        """covering:
  sigma: 0.5
  sigma_edge: 3.0
  lambda_relation: 0.25
  slot_mass_keep: 0.7
  row_top_k: 1
contour_tau: 0.4
explain_top_n: 3
""",
        encoding='utf-8',
    )
    loaded = CollisionEvalConfig.from_yaml(path)
    assert loaded.covering.sigma == pytest.approx(0.5)
    assert loaded.covering.sigma_edge == pytest.approx(3.0)
    assert loaded.covering.lambda_relation == pytest.approx(0.25)
    assert loaded.covering.slot_mass_keep == pytest.approx(0.7)
    assert loaded.covering.row_top_k == 1
    assert loaded.contour_tau == pytest.approx(0.4)
    assert loaded.explain_top_n == 3


def test_shipped_collision_yaml_keeps_demand_dense() -> None:
    root = Path(__file__).resolve().parents[2] / 'configs'
    assert (
        root / 'collision_eval.yaml',
        root / 'collision_eval.smoke.yaml',
        root / 'collision_eval.smoke.host_only.yaml',
    ) == SHIPPED_COLLISION_EVAL_PATHS
    loaded = tuple(CollisionEvalConfig.from_yaml(path) for path in SHIPPED_COLLISION_EVAL_PATHS)
    assert all(
        cfg.covering.row_top_k is None
        and cfg.covering.row_mass_keep is None
        and cfg.covering.slot_top_k is None
        and cfg.covering.slot_mass_keep is None
        and cfg.disclosure_max_chunks is None
        and cfg.encode_shards is None
        and cfg.keep_grid == ()
        for cfg in loaded
    )


def test_disclosure_probe_yaml_declares_median_chunk_cap() -> None:
    probe = Path(__file__).resolve().parents[2] / 'configs' / 'collision_eval.disclosure.yaml'
    loaded = CollisionEvalConfig.from_yaml(probe)
    assert loaded.disclosure_max_chunks == 8
    assert loaded.reuse_encode is False
    assert loaded.covering.row_top_k is None
    assert loaded.covering.slot_mass_keep is None
    assert loaded.keep_grid == ()


def test_keep_probe_yaml_declares_slot_grid() -> None:
    probe = Path(__file__).resolve().parents[2] / 'configs' / 'collision_eval.disclosure.keep.yaml'
    loaded = CollisionEvalConfig.from_yaml(probe)
    assert loaded.reuse_encode is True
    assert loaded.encode_shards == '/outputs/collision-disclosure/encode-shards'
    assert loaded.covering.slot_mass_keep is None
    assert loaded.covering.slot_top_k is None
    assert tuple((cell.slot_mass_keep, cell.slot_top_k) for cell in loaded.keep_grid) == (
        (0.35, None),
        (0.50, None),
        (0.70, None),
        (None, 8),
        (None, 16),
    )


def test_encode_writer_rewrites_shard_and_progress_each_batch(tmp_path: Path) -> None:
    store = CollisionEncodeShardStore(root=tmp_path)
    writer = CollisionEncodeWriter(store, shard_flush_every=1)
    trainer = lightning_trainer_stub(global_rank=1, logger=None, is_global_zero=False)
    pl_module = pl.LightningModule()
    first = PatentEmbeddingRecord(
        application_number='13817165',
        z_d=torch.ones(3),
        cpc_section='H',
    )
    second = PatentEmbeddingRecord(
        application_number='14111139',
        z_d=torch.zeros(3),
        cpc_section='G',
    )
    writer.write_on_batch_end(
        trainer,
        pl_module,
        CollisionEncodeStep(records=(first,), probe=_probe(apps_step=1)),
        (),
        None,
        0,
        0,
    )
    writer.write_on_batch_end(
        trainer,
        pl_module,
        CollisionEncodeStep(records=(second,), probe=_probe(row_entropy=0.58, apps_step=1)),
        (),
        None,
        1,
        0,
    )
    loaded = torch.load(store.shard_path(1), weights_only=False)
    assert tuple(record.application_number for record in loaded) == ('13817165', '14111139')
    progress = json.loads((tmp_path / 'progress-rank-01.json').read_text(encoding='utf-8'))
    assert progress['apps'] == 2
    assert progress['batch'] == 1
    assert progress['rank'] == 1
    assert progress['row_entropy'] == pytest.approx(0.58)


def test_encode_writer_flushes_shard_only_on_cadence(tmp_path: Path) -> None:
    store = CollisionEncodeShardStore(root=tmp_path)
    writer = CollisionEncodeWriter(store, shard_flush_every=2)
    trainer = lightning_trainer_stub(global_rank=0, logger=None, is_global_zero=True)
    pl_module = pl.LightningModule()
    first = PatentEmbeddingRecord(
        application_number='13817165',
        z_d=torch.ones(3),
        cpc_section='H',
    )
    second = PatentEmbeddingRecord(
        application_number='14111139',
        z_d=torch.zeros(3),
        cpc_section='G',
    )
    writer.write_on_batch_end(
        trainer,
        pl_module,
        CollisionEncodeStep(records=(first,), probe=_probe()),
        (),
        None,
        0,
        0,
    )
    assert not store.shard_path(0).is_file()
    progress = json.loads((tmp_path / 'progress-rank-00.json').read_text(encoding='utf-8'))
    assert progress['apps'] == 1
    assert 'row_entropy' in progress
    writer.write_on_batch_end(
        trainer,
        pl_module,
        CollisionEncodeStep(records=(second,), probe=_probe()),
        (),
        None,
        1,
        0,
    )
    loaded = torch.load(store.shard_path(0), weights_only=False)
    assert tuple(record.application_number for record in loaded) == ('13817165', '14111139')


def test_stub_pair_source_returns_empty(tmp_path: Path) -> None:
    stub = StubCitationPairSource()
    assert stub.load_pairs(tmp_path / 'missing.parquet') == ()


def test_encode_collision_records_one_graph_parse_per_row(
    fixtures_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = json.loads((fixtures_dir / 'hupd' / '13817165.json').read_text(encoding='utf-8'))
    expected_cpc = graph_batch_from_hupd_dict(row).main_cpc_section
    parses = 0
    real_parse = graph_batch_from_hupd_dict

    def counted_parse(raw: dict[str, object]) -> PatentGraphBatch:
        nonlocal parses
        parses += 1
        return real_parse(raw)

    monkeypatch.setattr(
        'ip_claim.collision.encode_job.graph_batch_from_hupd_dict',
        counted_parse,
    )

    class _Trunk(torch.nn.Module):
        def export_trunk(
            self,
            graphs: object,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            texts: Sequence[str] = (),
            claim_texts: Sequence[str] = (),
            living: torch.Tensor | None = None,
            claim_mask: torch.Tensor | None = None,
        ) -> TrunkExport:
            del graphs, attention_mask, texts, claim_texts, living, claim_mask
            rows = int(input_ids.shape[0])
            assert rows > 0
            return TrunkExport(
                z_d=torch.arange(rows * 3, dtype=torch.float).reshape(rows, 3),
                z_g=torch.zeros(rows, 3),
                soft_tokens=torch.zeros(rows, 1, 3),
            )

    records = encode_collision_records(
        (('13817165', row), ('13817166', row)),
        collator=_stub_mlm_collator,
        model=cast(SoftTrunkModel, cast(object, _Trunk())),
        batch_size=2,
        num_devices=1,
        shard_dir=tmp_path / 'encode-shards',
        log_dir=tmp_path,
    )
    assert records is not None
    assert parses == 2
    apps = tuple(record.application_number for record in records)
    assert frozenset(apps) == frozenset(('13817165', '13817166'))
    assert apps.count('13817165') >= 1
    assert apps.count('13817166') >= 1
    assert {record.cpc_section for record in records} == {expected_cpc}
    assert records[0].z_d.tolist() == [0.0, 1.0, 2.0]
    assert records[0].n_entity_claim is None
    assert records[0].n_entity_full is None
    jsonl = (tmp_path / 'encode-metrics.jsonl').read_text(encoding='utf-8')
    assert 'row_entropy' in jsonl
    progress = json.loads((tmp_path / 'encode-shards' / 'progress-rank-00.json').read_text())
    assert progress['apps'] == len(records)
    assert 'row_entropy' in progress


def test_try_read_complete_loads_contiguous_shards(tmp_path: Path) -> None:
    store = CollisionEncodeShardStore(root=tmp_path)
    first = _n_record('13817165', [1.0, 0.0], [2.0, 0.0], cpc='A')
    second = _n_record('14111139', [0.0, 1.0], [0.0, 3.0], cpc='H')
    store.write(0, (first,))
    store.write(1, (second,))
    loaded = store.try_read_complete()
    assert loaded is not None
    assert tuple(row.application_number for row in loaded) == ('13817165', '14111139')


def test_try_read_complete_rejects_rank_gap(tmp_path: Path) -> None:
    store = CollisionEncodeShardStore(root=tmp_path)
    store.write(0, (_n_record('13817165', [1.0], [1.0]),))
    store.write(2, (_n_record('14111139', [1.0], [1.0]),))
    assert store.try_read_complete() is None


def test_resolve_encoded_corpus_reuses_shards_without_encode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CollisionEncodeShardStore(root=tmp_path / 'encode-shards')
    row = _n_record('13817165', [1.0, 0.0], [2.0, 0.0], cpc='A')
    store.write(0, (row,))

    def _encode_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError('encode_collision_records must not run when shards are complete')

    monkeypatch.setattr('ip_claim.collision.eval.encode_collision_records', _encode_must_not_run)
    loaded = resolve_encoded_corpus(
        CollisionEvalConfig(output_dir=str(tmp_path), reuse_encode=True),
        CollisionEvalQueue(train_pairs=(), eval_pairs=(), test_pairs=(), items=()),
        collator=_stub_mlm_collator,
        model=cast(SoftTrunkModel, cast(object, SimpleNamespace())),
    )
    assert loaded is not None
    assert tuple(record.application_number for record in loaded) == ('13817165',)


def test_encode_shard_root_prefers_explicit_path(tmp_path: Path) -> None:
    other = tmp_path / 'disclosure-shards'
    loaded = CollisionEvalConfig(
        output_dir=str(tmp_path / 'keep-out'),
        encode_shards=str(other),
    )
    assert encode_shard_root(loaded) == other
    assert encode_shard_root(CollisionEvalConfig(output_dir=str(tmp_path))) == (
        tmp_path / 'encode-shards'
    )


def _smear_corpus() -> tuple[
    tuple[PatentEmbeddingRecord, ...],
    tuple[CitationPair, ...],
]:
    tail = [2.0] * 15
    query = _n_record('10000001', [20.0, *tail], [0.0] * 16, cpc='A')
    paid = _n_record('10000002', [0.0] * 16, [20.0, *([0.3] * 15)], cpc='A')
    background = _n_record('10000003', [0.0] * 16, [1.0, *([4.0] * 15)], cpc='A')
    decoy = _n_record('10000004', [0.0] * 16, [0.0, *([4.0] * 15)], cpc='A')
    pairs = (
        CitationPair(
            query_application_number='10000001',
            partner_application_number='10000002',
            marks=('X',),
        ),
        CitationPair(
            query_application_number='10000001',
            partner_application_number='10000003',
            marks=('A',),
        ),
    )
    return (query, paid, background, decoy), pairs


def test_rank_covering_from_records_applies_slot_keep(tmp_path: Path) -> None:
    records, pairs = _smear_corpus()
    dense = rank_covering_from_records(
        CollisionEvalConfig(output_dir=str(tmp_path / 'dense'), use_gpu=False),
        records,
        train_pairs=(),
        eval_pairs=pairs,
        test_pairs=(),
    )['eval']
    assert dense.unpaid_x > dense.unpaid_a
    sparse = rank_covering_from_records(
        CollisionEvalConfig(
            output_dir=str(tmp_path / 'sparse'),
            use_gpu=False,
            covering=CoveringKnobs(slot_mass_keep=0.35),
        ),
        records,
        train_pairs=(),
        eval_pairs=pairs,
        test_pairs=(),
    )['eval']
    assert sparse.unpaid_x < sparse.unpaid_a < sparse.unpaid_random
    assert (tmp_path / 'sparse' / 'covering.json').is_file()


def test_rank_keep_grid_writes_cell_tables(tmp_path: Path) -> None:
    records, pairs = _smear_corpus()
    report = rank_keep_grid(
        CollisionEvalConfig(
            output_dir=str(tmp_path),
            use_gpu=False,
            encode_shards=str(tmp_path / 'shards'),
            keep_grid=(
                KeepGridCell(slot_mass_keep=0.35),
                KeepGridCell(slot_top_k=1),
            ),
        ),
        records,
        train_pairs=(),
        eval_pairs=pairs,
        test_pairs=(),
    )
    assert tuple(cell.stem for cell in report.cells) == ('mass035', 'topk1')
    assert (tmp_path / KEEP_GRID_JSON).is_file()
    for cell in report.cells:
        assert cell.queries_x > 0
        assert cell.unpaid_x < cell.unpaid_a < cell.unpaid_random
        assert Path(cell.covering).is_file()
