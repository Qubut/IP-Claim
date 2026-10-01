"""HUPD directory load in ingestion; SSV maps patents to SoftMlmExample."""

from __future__ import annotations

from pathlib import Path

import pytest

from ip_claim.ingestion.adapters.hupd_json.paths import load_hupd_patents, sample_hupd_paths
from ip_claim.ingestion.models import Patent
from ip_claim.ssv.dataset import examples_from_patents

_FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'hupd'


def test_load_hupd_patents_reads_package_fixtures() -> None:
    patents = load_hupd_patents(_FIXTURES)
    assert all(isinstance(patent, Patent) for patent in patents)
    assert {patent.application_number for patent in patents} == {
        '13817165',
        '14111139',
        '14112715',
    }
    examples = examples_from_patents(patents)
    assert len(examples) == len(patents)
    assert all(example.text for example in examples)


def test_load_hupd_patents_respects_limit(tmp_path: Path) -> None:
    (tmp_path / 'a.json').write_text('{"application_number": "1"}\n', encoding='utf-8')
    (tmp_path / 'b.json').write_text('{"application_number": "2"}\n', encoding='utf-8')
    (tmp_path / 'c.json').write_text('{"application_number": "3"}\n', encoding='utf-8')
    patents = load_hupd_patents(tmp_path, limit=2)
    assert len(patents) == 2
    assert all(isinstance(patent, Patent) for patent in patents)


def test_load_hupd_patents_rejects_non_object_json(tmp_path: Path) -> None:
    (tmp_path / 'row.json').write_text('[1, 2]\n', encoding='utf-8')
    with pytest.raises(TypeError, match='HUPD JSON must be an object'):
        load_hupd_patents(tmp_path)


def test_hupd_json_files_walks_nested_year_tree(tmp_path: Path) -> None:
    nested = tmp_path / 'all-years' / '2013'
    nested.mkdir(parents=True)
    path = nested / '14033096.json'
    path.write_text('{"application_number": "14033096"}\n', encoding='utf-8')
    patents = load_hupd_patents(tmp_path)
    assert [patent.application_number for patent in patents] == ['14033096']


def test_sample_hupd_paths_builds_and_reuses_index_cache(tmp_path: Path) -> None:
    (tmp_path / 'a.json').write_text('{"application_number": "1"}\n', encoding='utf-8')
    (tmp_path / 'b.json').write_text('{"application_number": "2"}\n', encoding='utf-8')
    cache = tmp_path / 'paths.idx'
    first, n_pool = sample_hupd_paths(tmp_path, limit=2, seed=3, index_cache=cache)
    assert cache.is_file()
    assert n_pool == 2
    again, again_pool = sample_hupd_paths(tmp_path, limit=2, seed=3, index_cache=cache)
    assert first == again
    assert again_pool == 2


def test_sample_hupd_paths_full_limit_returns_the_pool(tmp_path: Path) -> None:
    (tmp_path / 'a.json').write_text('{"application_number": "1"}\n', encoding='utf-8')
    (tmp_path / 'b.json').write_text('{"application_number": "2"}\n', encoding='utf-8')
    chosen, n_pool = sample_hupd_paths(tmp_path, limit=10, seed=1)
    assert n_pool == 2
    assert len(chosen) == 2
    assert set(chosen) == {tmp_path / 'a.json', tmp_path / 'b.json'}


def test_sample_hupd_paths_is_deterministic() -> None:
    first, n_pool = sample_hupd_paths(_FIXTURES, limit=2, seed=7)
    again, again_pool = sample_hupd_paths(_FIXTURES, limit=2, seed=7)
    other, _ = sample_hupd_paths(_FIXTURES, limit=2, seed=8)
    assert n_pool == again_pool
    assert first == again
    assert len(first) == 2
    assert first != other or n_pool < 2
