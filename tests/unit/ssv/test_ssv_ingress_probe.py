"""Extract-termhood occupancy probe: join, report, CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
from patent_ate import write_termhood
from patent_ate.nlp import TermSpan
from patent_ate.termhood import TERMHOOD_META_NAME, TermhoodStore, TermhoodTable
from structlog.testing import capture_logs

from ip_claim.ssv import __main__ as ssv_main
from ip_claim.ssv.graph_ingress import (
    GraphIngress,
    spans_from_extract,
)
from ip_claim.ssv.ingress_probe import (
    IngressProbeRequest,
    OccupiedFiling,
    occupy_extract_parts,
    run_ingress_probe,
    summarize_filings,
    write_probe,
)


def _extract_part(path: Path, rows: tuple[tuple[str, str], ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        'patent_id': [patent_id for patent_id, _term in rows],
        'n_docs': [1] * len(rows),
        'terms': [
            [{'term': term, 'frequency': 1, 'surfaces': [term]}] for _patent_id, term in rows
        ],
    }).write_parquet(path)
    return path


def test_summarize_filings_counts_stop_leaks_and_empty() -> None:
    report = summarize_filings(
        (
            OccupiedFiling(
                application_number='1',
                n_live=10,
                n_occupied=2,
                labels=('vehicle interior', 'comprising'),
            ),
            OccupiedFiling(
                application_number='2',
                n_live=8,
                n_occupied=0,
                labels=(),
            ),
        ),
        {'vehicle interior': 1.5, 'comprising': 0.2},
        n_pool=40,
        seed=1,
    )
    assert report.n_patents == 2
    assert report.n_occupy == 2
    assert report.n_empty == 1
    assert report.occupy_rate == pytest.approx(2 / 18)
    assert report.n_stop_leaks == 1
    assert report.stop_leaks == ('comprising',)
    assert report.top_by_score[0].label == 'vehicle interior'


def test_summarize_filings_ranks_corpus_termhood() -> None:
    report = summarize_filings(
        (
            OccupiedFiling(
                application_number='1',
                n_live=4,
                n_occupied=1,
                labels=('coil spring',),
            ),
        ),
        {'coil spring': 3.0, 'analyte sensor': 9.0, 'first end': 0.0},
        n_pool=100_000,
        seed=0,
        n_patents=100_000,
        n_occupy=1,
        document_frequency={'coil spring': 12, 'analyte sensor': 2, 'first end': 400},
    )
    assert report.n_patents == 100_000
    assert report.n_occupy == 1
    assert report.n_unique_labels == 2
    assert report.top_by_score[0].label == 'analyte sensor'
    assert report.top_by_docs[0].label == 'coil spring'


def test_write_termhood_emits_parquet_and_meta(tmp_path: Path) -> None:
    path = write_termhood(
        TermhoodTable(
            c_values={'coil spring': 3.0},
            document_frequency={'coil spring': 2},
            total_docs=10,
        ),
        tmp_path,
    )
    assert path == tmp_path
    assert (tmp_path / TERMHOOD_META_NAME).is_file()
    assert not (tmp_path / 'termhood.json').exists()
    store = TermhoodStore.open(path)
    assert store.meta.total_docs == 10
    assert pl.scan_parquet(store.parquet).select('key').collect().to_series().to_list() == [
        'coil spring'
    ]


def test_write_probe_emits_json_and_html(tmp_path: Path) -> None:
    report = summarize_filings(
        (
            OccupiedFiling(
                application_number='1',
                n_live=4,
                n_occupied=1,
                labels=('coil spring',),
            ),
        ),
        {'coil spring': 3.0},
        n_pool=1,
        seed=0,
    )
    html = write_probe(report, tmp_path)
    assert html.is_file()
    assert (tmp_path / 'probe.json').is_file()
    assert 'coil spring' in html.read_text(encoding='utf-8')


def test_probe_ingress_cli_forwards_args(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    report = summarize_filings((), {}, n_pool=0, seed=3)

    def fake_run(request: IngressProbeRequest) -> object:
        captured['request'] = request
        return report

    with (
        patch.object(ssv_main, 'run_ingress_probe', fake_run),
        patch.object(ssv_main, 'write_probe', lambda _report, output: output / 'probe.html'),
    ):
        code = ssv_main.main([
            'probe-ingress',
            '--output',
            str(tmp_path / 'out'),
            '--occupy-limit',
            '4',
            '--seed',
            '9',
            '--termhood',
            str(tmp_path / 'termhood.json'),
            '--termhood-docs',
            '100000',
            '--extract',
            str(tmp_path / 'extract'),
        ])

    assert code == 0
    request = captured['request']
    assert request.occupy_limit == 4
    assert request.seed == 9
    assert request.output_dir == tmp_path / 'out'
    assert request.termhood_path == tmp_path / 'termhood.json'
    assert request.termhood_docs == 100_000
    assert request.extract_dir == tmp_path / 'extract'


def _boom_spacy(*_args: object, **_kwargs: object) -> object:
    raise AssertionError('occupy must not load spaCy')


def _boom_from_texts(*_args: object, **_kwargs: object) -> None:
    raise AssertionError('occupy must not call JateDraw.from_texts')


def _boom_json_or_cuda(*_args: object, **_kwargs: object) -> object:
    raise AssertionError('occupy must not read host JSON or CUDA occupy_batch')


def _install_occupy_booms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('ip_claim.ssv.graph_ingress.host_language', _boom_spacy)
    monkeypatch.setattr('patent_ate.nlp.host_language', _boom_spacy)
    monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', _boom_from_texts)
    monkeypatch.setattr('ip_claim.ssv.graph_ingress.spacy.load', _boom_spacy)
    monkeypatch.setattr(GraphIngress, 'nlp', lambda self, **_: _boom_spacy())
    monkeypatch.setattr('patent_ate.extract.patent_rows', _boom_json_or_cuda)
    monkeypatch.setattr('patent_ate.corpus.patent_text_from_path', _boom_json_or_cuda)
    monkeypatch.setattr('patent_ate.corpus.sample_json_paths', _boom_json_or_cuda)
    monkeypatch.setattr(
        'ip_claim.ingestion.adapters.hupd_json.paths.sample_hupd_paths',
        _boom_json_or_cuda,
    )


def test_spans_from_extract_locates_surfaces_without_spacy() -> None:
    haystack = '\u00fchelix extra'
    docs = spans_from_extract(
        ('coil spring assembly', haystack),
        (
            ({'term': 'coil spring', 'frequency': 1, 'surfaces': ['coil spring']},),
            ({'term': 'helix extra', 'frequency': 1, 'surfaces': ['helix extra']},),
        ),
    )
    assert docs[0] == (
        TermSpan(lemma_key='coil spring', start_char=0, end_char=11, token_length=2),
    )
    assert docs[1][0].lemma_key == 'helix extra'
    assert docs[1][0].start_char == haystack.find('helix extra')
    assert docs[1][0].end_char == docs[1][0].start_char + len('helix extra')
    assert docs[1][0].start_char != haystack.encode('utf-8').find(b'helix extra')


def test_occupy_extract_skips_host_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_occupy_booms(monkeypatch)
    missing = tmp_path / 'does-not-exist.json'
    extract_dir = tmp_path / 'extract'
    part = _extract_part(
        extract_dir / 'part.parquet',
        ((str(missing.resolve()), 'coil spring'),),
    )
    assert not missing.exists()
    tally = occupy_extract_parts(
        (part,),
        TermhoodTable(
            c_values={'coil spring': 4.0},
            document_frequency={'coil spring': 1},
            total_docs=10,
        ),
        occupy_limit=1,
    )
    assert tally.live > 0
    assert tally.occupied > 0
    assert tally.examples[0].labels == ('coil spring',)
    assert tally.examples[0].application_number == str(missing.resolve())


def test_occupy_extract_parts_logs_patent_progress(tmp_path: Path) -> None:
    extract_dir = tmp_path / 'extract'
    first = _extract_part(extract_dir / 'a.parquet', (('p1', 'coil spring'),))
    second = _extract_part(extract_dir / 'b.parquet', (('p2', 'coil spring'),))
    with capture_logs() as logs:
        occupy_extract_parts(
            (first, second),
            TermhoodTable(
                c_values={'coil spring': 4.0},
                document_frequency={'coil spring': 1},
                total_docs=10,
            ),
            occupy_limit=4518254,
            part_window=1,
        )
    windows = tuple(row for row in logs if row['event'] == 'ssv.probe.occupy.window')
    assert len(windows) == 2
    assert windows[0]['patents_done'] == 1
    assert windows[1]['patents_done'] == 2
    assert windows[0]['occupy_limit'] == 4518254
    assert windows[0]['n_workers'] == 1
    assert windows[0]['device'] == 'cpu'
    assert 'wall_s' in windows[0]
    assert windows[0]['n_occupied'] >= 1


def test_occupy_extract_empty_when_termhood_has_no_score(tmp_path: Path) -> None:
    extract_dir = tmp_path / 'extract'
    part = _extract_part(
        extract_dir / 'part.parquet',
        (('p1', 'coil spring'),),
    )
    table = TermhoodTable(total_docs=10)
    tally = occupy_extract_parts(
        (part,),
        table,
        occupy_limit=1,
    )
    assert tally.occupied == 0
    assert tally.empty == 1
    assert tally.examples[0].labels == ()
    termhood = write_termhood(table, tmp_path / 'termhood')
    with pytest.raises(ValueError, match='extract keys and termhood do not match'):
        run_ingress_probe(
            IngressProbeRequest(
                output_dir=tmp_path / 'out',
                termhood_path=termhood,
                extract_dir=extract_dir,
                occupy_limit=1,
            )
        )
    code = ssv_main.main([
        'probe-ingress',
        '--output',
        str(tmp_path / 'cli-out'),
        '--termhood',
        str(termhood),
        '--extract',
        str(extract_dir),
        '--occupy-limit',
        '1',
    ])
    assert code != 0


def test_run_ingress_probe_requires_termhood(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom_termhood(*_args: object, **_kwargs: object) -> None:
        raise AssertionError('occupy must not call corpus_termhood')

    monkeypatch.setattr('patent_ate.extract.corpus_termhood', boom_termhood)
    extract_dir = tmp_path / 'extract'
    _extract_part(
        extract_dir / 'part.parquet',
        ((str((tmp_path / 'a.json').resolve()), 'coil spring'),),
    )
    with pytest.raises(ValueError, match='occupy requires an existing termhood'):
        run_ingress_probe(
            IngressProbeRequest(
                output_dir=tmp_path / 'out',
                extract_dir=extract_dir,
            )
        )


def test_run_ingress_probe_occupies_without_json_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_occupy_booms(monkeypatch)
    extract_dir = tmp_path / 'extract'
    _extract_part(
        extract_dir / 'a.parquet',
        (('p1', 'coil spring'),),
    )
    _extract_part(
        extract_dir / 'b.parquet',
        (('p2', 'coil spring'),),
    )
    termhood = write_termhood(
        TermhoodTable(
            c_values={'coil spring': 4.0},
            document_frequency={'coil spring': 1},
            total_docs=10,
        ),
        tmp_path / 'termhood',
    )
    report = run_ingress_probe(
        IngressProbeRequest(
            output_dir=tmp_path / 'out',
            termhood_path=termhood,
            extract_dir=extract_dir,
            occupy_limit=4518254,
        )
    )
    assert report.n_pool == 2
    assert report.n_occupy == 2
    assert report.n_patents == 2
    assert report.n_empty == 0
    assert {row.application_number for row in report.filings} == {'p1', 'p2'}
    assert all(row.labels == ('coil spring',) for row in report.filings)
    code = ssv_main.main([
        'probe-ingress',
        '--output',
        str(tmp_path / 'cli-out'),
        '--termhood',
        str(termhood),
        '--extract',
        str(extract_dir),
        '--occupy-limit',
        '4518254',
    ])
    assert code == 0
    payload = (tmp_path / 'cli-out' / 'probe.json').read_text(encoding='utf-8')
    assert '"n_occupy": 2' in payload
    assert '"n_pool": 2' in payload


def test_probe_ingress_cli_rejects_missing_termhood(tmp_path: Path) -> None:
    code = ssv_main.main([
        'probe-ingress',
        '--output',
        str(tmp_path / 'out'),
        '--extract',
        str(tmp_path / 'extract'),
    ])
    assert code != 0
