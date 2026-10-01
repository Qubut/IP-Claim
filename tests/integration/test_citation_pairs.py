"""Integration smoke for epo-processor-shaped analyze dataset Parquet."""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ip_claim.app.container.collision import CollisionContainer
from ip_claim.collision.config import CollisionEvalConfig, CollisionSplitSpec
from ip_claim.collision.data import EpoProcessorCitationPairSource
from ip_claim.collision.eval import run_collision_eval
from ip_claim.ssv.config import ArchSpec, HostSpec, SsvTrainConfig


def _write_analyze_fixture(path: Path) -> None:
    table = pa.table({
        'epo_patent_id': ['US13817165A1'],
        'epo_hupd_paths': pa.array([['13817165.json']], type=pa.list_(pa.string())),
        'cited_hupd': pa.array(
            [[{'cited_id': '14111139', 'categories': ['Y'], 'paths': ['14111139.json']}]],
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


def _ssv_eval_config() -> SsvTrainConfig:
    return SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        arch=ArchSpec(
            gnn_hidden=32,
            gnn_heads=4,
            gnn_layers=1,
            n_soft_tokens=4,
            entity_bank_size=16,
            soft_dim=32,
            max_length=48,
        ),
    )


def test_collision_eval_on_fixture_parquet(tmp_path: Path, fixtures_dir: Path) -> None:
    dataset = tmp_path / 'analyze_dataset.parquet'
    _write_analyze_fixture(dataset)
    ssv_config = _ssv_eval_config()
    eval_config = CollisionEvalConfig(
        recall_k=(1, 5),
        split=CollisionSplitSpec(train=0.0, eval=1.0, test=0.0),
        num_devices=1,
        output_dir=str(tmp_path / 'collision-out'),
    )
    wiring = CollisionContainer(ssv_config=ssv_config)
    result = run_collision_eval(
        eval_config,
        dataset_path=dataset,
        pair_source=EpoProcessorCitationPairSource(),
        hupd_dir=fixtures_dir / 'hupd',
        collator=wiring.ssv.eval_collator(),
        model=wiring.ssv.soft_trunk(),
    )
    assert result is not None
    assert result.eval.queries == 1
    assert 0.0 <= result.eval.mrr <= 1.0
    assert result.eval.recall_at_k[1] >= 0.0
    assert result.eval.cpc_hard_recall_at_k[1] >= 0.0


def test_collision_eval_walks_nested_all_years(tmp_path: Path, fixtures_dir: Path) -> None:
    dataset = tmp_path / 'analyze_dataset.parquet'
    _write_analyze_fixture(dataset)
    hupd = tmp_path / 'hupd'
    for app, year in (('13817165', '2014'), ('14111139', '2015')):
        dest = hupd / 'all-years' / year / year
        dest.mkdir(parents=True)
        shutil.copy2(fixtures_dir / 'hupd' / f'{app}.json', dest / f'{app}.json')
    ssv_config = _ssv_eval_config()
    eval_config = CollisionEvalConfig(
        recall_k=(1, 5),
        split=CollisionSplitSpec(train=0.0, eval=1.0, test=0.0),
        num_devices=1,
        output_dir=str(tmp_path / 'collision-out'),
    )
    wiring = CollisionContainer(ssv_config=ssv_config)
    result = run_collision_eval(
        eval_config,
        dataset_path=dataset,
        pair_source=EpoProcessorCitationPairSource(),
        hupd_dir=hupd,
        collator=wiring.ssv.eval_collator(),
        model=wiring.ssv.soft_trunk(),
    )
    assert result is not None
    assert result.eval.queries == 1
