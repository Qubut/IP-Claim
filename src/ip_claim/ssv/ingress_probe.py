"""Corpus occupancy probe: scored extract terms joined to committed termhood.

This module owns occupy and the occupancy report. Termhood comes from the
ATE package. Occupy reads compact extract parquet, not host JSON.
It does not load a trunk checkpoint.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import batched
from pathlib import Path

import polars as pl
import structlog
from patent_ate import TermhoodIndex, TermhoodStore, extract_parquet_parts, write_termhood
from patent_ate.extract import EXTRACT_DIRNAME
from patent_ate.plan import PATENT_ID_COLUMN
from patent_ate.termhood import (
    TermhoodTable,
    is_stop_surface,
    product_score_expr,
)
from pydantic import BaseModel, ConfigDict, Field

from ip_claim.ssv.inspect import ReportTable

_EXAMPLE_ROWS = 12
_TOP_K = 40
_PART_WINDOW = 256
_EMPTY_LABELS = pl.lit([], dtype=pl.List(pl.String))
_log = structlog.get_logger(__name__)


class OccupiedFiling(BaseModel):
    """Occupy outcome for one extract patent."""

    model_config = ConfigDict(frozen=True)

    application_number: str
    n_live: int = Field(ge=0)
    n_occupied: int = Field(ge=0)
    labels: tuple[str, ...]


class TermCount(BaseModel):
    """One term with termhood and document frequency in the occupied set."""

    model_config = ConfigDict(frozen=True)

    label: str
    score: float
    n_docs: int = Field(ge=0)


class IngressProbeReport(BaseModel):
    """Occupancy summary plus per-filing labels."""

    model_config = ConfigDict(frozen=True)

    n_patents: int = Field(ge=0)
    n_occupy: int = Field(ge=0)
    n_pool: int = Field(ge=0)
    seed: int
    n_empty: int = Field(ge=0)
    occupy_rate: float
    n_unique_labels: int = Field(ge=0)
    n_stop_leaks: int = Field(ge=0)
    stop_leaks: tuple[str, ...]
    top_by_score: tuple[TermCount, ...]
    top_by_docs: tuple[TermCount, ...]
    filings: tuple[OccupiedFiling, ...]

    def table_specs(self) -> tuple[ReportTable, ...]:
        """Declared report sections for Great Tables."""
        return (
            ReportTable(
                title='Occupancy summary',
                rows=(
                    {
                        'n_patents': self.n_patents,
                        'n_occupy': self.n_occupy,
                        'n_pool': self.n_pool,
                        'seed': self.seed,
                        'n_empty': self.n_empty,
                        'occupy_rate': round(self.occupy_rate, 4),
                        'n_unique_labels': self.n_unique_labels,
                        'n_stop_leaks': self.n_stop_leaks,
                    },
                ),
            ),
            ReportTable(
                title=f'Top {_TOP_K} by termhood',
                rows=tuple(row.model_dump() for row in self.top_by_score)
                or ({'note': 'no scored terms'},),
            ),
            ReportTable(
                title=f'Top {_TOP_K} by document count',
                rows=tuple(row.model_dump() for row in self.top_by_docs)
                or ({'note': 'no occupied labels'},),
            ),
            ReportTable(
                title='Stop-surface leaks',
                rows=tuple({'label': label} for label in self.stop_leaks) or ({'note': 'none'},),
            ),
            ReportTable(
                title=f'Example filings ({_EXAMPLE_ROWS})',
                rows=tuple(
                    {
                        'application': row.application_number,
                        'n_live': row.n_live,
                        'n_occupied': row.n_occupied,
                        'labels': ', '.join(row.labels),
                    }
                    for row in self.filings[:_EXAMPLE_ROWS]
                )
                or ({'note': 'empty occupy'},),
            ),
        )


class IngressProbeRequest(BaseModel):
    """Validated envelope for extract-termhood occupy."""

    model_config = ConfigDict(frozen=True)

    output_dir: Path
    occupy_limit: int = Field(default=1000, ge=1)
    seed: int = 20260830
    termhood_path: Path | None = None
    termhood_docs: int = Field(default=100_000, ge=1)
    extract_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class OccupyTally:
    """Running occupy totals plus a short example prefix."""

    examples: tuple[OccupiedFiling, ...] = ()
    live: int = 0
    occupied: int = 0
    empty: int = 0

    def merged(self, other: OccupyTally) -> OccupyTally:
        """Add ``other`` into this tally."""
        return OccupyTally(
            examples=(*self.examples, *other.examples)[:_EXAMPLE_ROWS],
            live=self.live + other.live,
            occupied=self.occupied + other.occupied,
            empty=self.empty + other.empty,
        )


def run_ingress_probe(request: IngressProbeRequest) -> IngressProbeReport:
    """Occupy extract parquet rows joined to committed termhood. No host JSON."""
    if request.termhood_path is None:
        raise ValueError(
            'occupy requires an existing termhood artifact; extract parquet is not scored here'
        )
    if request.termhood_path.suffix == '.json':
        payload = json.loads(request.termhood_path.read_text(encoding='utf-8'))
        if not isinstance(payload, Mapping):
            raise ValueError('termhood json is not an object')
        write_termhood(
            TermhoodTable.load({
                **payload,
                'total_docs': request.termhood_docs,
            }),
            request.output_dir,
        )
        store = TermhoodStore.open(request.output_dir)
    else:
        store = TermhoodStore.open(request.termhood_path)
    extract_dir = request.extract_dir or (store.root / EXTRACT_DIRNAME)
    parts = extract_parquet_parts(extract_dir)
    if not parts:
        raise ValueError('extract parquet is required to occupy')
    n_extract = int(
        pl.scan_parquet([str(path) for path in parts]).select(pl.len()).collect().item()
    )
    occupy_n = min(request.occupy_limit, n_extract)
    _log.info(
        'ssv.probe.occupy.start',
        n_occupy=occupy_n,
        occupy_limit=request.occupy_limit,
        n_extract=n_extract,
        backend='extract_join',
        worker_count=1,
        extract_dir=str(extract_dir),
        n_extract_parts=len(parts),
    )
    tally = occupy_extract_parts(parts, store, occupy_limit=occupy_n)
    if tally.occupied == 0:
        raise ValueError('occupy produced no scored terms; extract keys and termhood do not match')
    return summarize_filings(
        tally.examples,
        {},
        n_pool=n_extract,
        seed=request.seed,
        n_patents=occupy_n,
        n_occupy=occupy_n,
        n_empty=tally.empty,
        live_total=tally.live,
        occ_total=tally.occupied,
        store=store,
    )


def occupy_score_frame(termhood: TermhoodTable | TermhoodIndex | TermhoodStore) -> pl.DataFrame:
    """Return positive occupy scores keyed like extract lemmas. No host JSON."""

    def frame_from_scores(ranked: Mapping[str, float]) -> pl.DataFrame:
        keys = tuple(ranked)
        if not keys:
            return pl.DataFrame({
                'key': pl.Series(dtype=pl.String),
                'score': pl.Series(dtype=pl.Float64),
            })
        return pl.DataFrame({
            'key': pl.Series(keys, dtype=pl.String),
            'score': pl.Series(
                tuple(float(ranked[key]) for key in keys),
                dtype=pl.Float64,
            ),
        }).filter(pl.col('score') > 0)

    match termhood:
        case TermhoodStore():
            frame = (
                pl
                .scan_parquet(termhood.parquet)
                .with_columns(product_score_expr(termhood.meta.total_docs))
                .select('key', 'score')
                .filter(pl.col('score') > 0)
                .collect()
            )
        case TermhoodIndex() | TermhoodTable():
            frame = frame_from_scores(dict(termhood.model_dump()['scores']))
        case _:
            raise TypeError('occupy scores need a termhood table, index, or store')
    if frame.is_empty():
        return frame
    keep = tuple(not is_stop_surface(str(key)) for key in frame.get_column('key').to_list())
    return frame.filter(pl.Series('keep', keep))


def occupy_extract_frame(frame: pl.DataFrame, scores: pl.DataFrame) -> OccupyTally:
    """Join one extract table to termhood scores. Does not read host JSON."""
    patents = frame.select(pl.col(PATENT_ID_COLUMN).cast(pl.String).alias('patent_id'))
    if frame.is_empty():
        return OccupyTally()
    exploded = (
        frame
        .select(pl.col(PATENT_ID_COLUMN).cast(pl.String).alias('patent_id'), 'terms')
        .explode('terms')
        .filter(pl.col('terms').is_not_null())
        .unnest('terms')
        .filter(pl.col('term').is_not_null())
        .with_columns(
            pl
            .col('term')
            .cast(pl.String)
            .str.replace_all('Ġ', '')
            .str.replace_all('▁', '')
            .str.replace_all('##', '')
            .str.strip_chars()
            .str.to_lowercase()
            .str.strip_chars('.,;:!?()[]{}"\'`-_/\\')
            .alias('key')
        )
        .filter(pl.col('key').str.len_chars() > 0)
    )
    joined = exploded.join(scores, on='key', how='left') if not exploded.is_empty() else exploded
    occupied_terms = (
        joined.group_by('patent_id', maintain_order=True).agg(
            n_live=pl.col('key').n_unique(),
            labels=pl
            .col('key')
            .filter(pl.col('score').fill_null(0.0) > 0)
            .unique(maintain_order=True),
        )
        if not joined.is_empty() and 'score' in joined.columns
        else pl.DataFrame({
            'patent_id': pl.Series(dtype=pl.String),
            'n_live': pl.Series(dtype=pl.UInt32),
            'labels': pl.Series(dtype=pl.List(pl.String)),
        })
    )
    occupied = (
        patents
        .join(occupied_terms, on='patent_id', how='left')
        .with_columns(
            pl.col('n_live').fill_null(0),
            labels=pl.col('labels').fill_null(_EMPTY_LABELS),
        )
        .with_columns(n_occupied=pl.col('labels').list.len())
    )
    filings = tuple(
        OccupiedFiling(
            application_number=str(row['patent_id']),
            n_live=int(row['n_live']),
            n_occupied=int(row['n_occupied']),
            labels=tuple(row['labels'] or ()),
        )
        for row in occupied.to_dicts()
    )
    return OccupyTally(
        examples=filings[:_EXAMPLE_ROWS],
        live=sum(row.n_live for row in filings),
        occupied=sum(row.n_occupied for row in filings),
        empty=sum(1 for row in filings if row.n_occupied == 0),
    )


def occupy_extract_parts(
    parts: Sequence[Path],
    termhood: TermhoodTable | TermhoodIndex | TermhoodStore,
    *,
    occupy_limit: int = 0,
    part_window: int = _PART_WINDOW,
) -> OccupyTally:
    """Occupy extract Parquet parts by joining termhood. No host JSON, no CUDA."""
    if not parts:
        return OccupyTally()
    scores = occupy_score_frame(termhood)
    width = max(1, part_window)
    windows = tuple(tuple(block) for block in batched(tuple(parts), width))

    def occupy_window(
        state: tuple[OccupyTally, int, int],
        window: tuple[Path, ...],
    ) -> tuple[OccupyTally, int, int]:
        tally, done, window_i = state
        if occupy_limit > 0 and done >= occupy_limit:
            return state
        remaining = occupy_limit - done if occupy_limit > 0 else 0
        started = time.perf_counter()
        lazy = pl.scan_parquet([str(path) for path in window]).select(PATENT_ID_COLUMN, 'terms')
        frame = lazy.head(remaining).collect() if remaining > 0 else lazy.collect()
        folded = occupy_extract_frame(frame, scores)
        patents_done = done + frame.height
        _log.info(
            'ssv.probe.occupy.window',
            stride_id=0,
            window_i=window_i,
            n_docs=frame.height,
            patents_done=patents_done,
            occupy_limit=occupy_limit,
            n_workers=1,
            n_occupied=folded.occupied,
            device='cpu',
            wall_s=round(time.perf_counter() - started, 6),
        )
        return (tally.merged(folded), patents_done, window_i + 1)

    tally, _, _ = reduce(occupy_window, windows, (OccupyTally(), 0, 0))
    return tally


def summarize_filings(
    filings: Sequence[OccupiedFiling],
    scores: Mapping[str, float],
    *,
    n_pool: int,
    seed: int,
    n_patents: int | None = None,
    n_occupy: int | None = None,
    document_frequency: Mapping[str, int] | None = None,
    n_empty: int | None = None,
    live_total: int | None = None,
    occ_total: int | None = None,
    store: TermhoodStore | None = None,
) -> IngressProbeReport:
    """Aggregate occupy rows and optional corpus termhood into a report."""
    occupy_labels = Counter(label for filing in filings for label in filing.labels)
    leaks = tuple(sorted({label for label in occupy_labels if is_stop_surface(label)}))
    live_total = sum(filing.n_live for filing in filings) if live_total is None else live_total
    occ_total = sum(filing.n_occupied for filing in filings) if occ_total is None else occ_total
    if store is not None:
        by_score_rows, by_docs_rows, n_unique = store.report_ranks(_TOP_K)
        by_score = tuple(
            TermCount(label=label, score=score, n_docs=n_docs)
            for label, score, n_docs in by_score_rows
        )
        by_docs = tuple(
            TermCount(label=label, score=score, n_docs=n_docs)
            for label, score, n_docs in by_docs_rows
        )
    elif document_frequency is None:
        ranked = tuple(
            TermCount(label=label, score=float(scores.get(label, 0.0)), n_docs=count)
            for label, count in occupy_labels.items()
        )
        n_unique = len(occupy_labels)
        by_score = tuple(sorted(ranked, key=lambda row: (-row.score, -row.n_docs, row.label)))[
            :_TOP_K
        ]
        by_docs = tuple(sorted(ranked, key=lambda row: (-row.n_docs, -row.score, row.label)))[
            :_TOP_K
        ]
    else:
        ranked = tuple(
            TermCount(
                label=label,
                score=float(score),
                n_docs=int(document_frequency.get(label, 0)),
            )
            for label, score in scores.items()
            if score > 0
        )
        n_unique = len(ranked)
        by_score = tuple(sorted(ranked, key=lambda row: (-row.score, -row.n_docs, row.label)))[
            :_TOP_K
        ]
        by_docs = tuple(sorted(ranked, key=lambda row: (-row.n_docs, -row.score, row.label)))[
            :_TOP_K
        ]
    return IngressProbeReport(
        n_patents=len(filings) if n_patents is None else n_patents,
        n_occupy=len(filings) if n_occupy is None else n_occupy,
        n_pool=n_pool,
        seed=seed,
        n_empty=sum(1 for filing in filings if filing.n_occupied == 0)
        if n_empty is None
        else n_empty,
        occupy_rate=(occ_total / live_total) if live_total else 0.0,
        n_unique_labels=n_unique,
        n_stop_leaks=len(leaks),
        stop_leaks=leaks,
        top_by_score=by_score,
        top_by_docs=by_docs,
        filings=tuple(filings),
    )


def write_probe(report: IngressProbeReport, output_dir: Path) -> Path:
    """Write ``probe.json`` and ``probe.html`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'probe.json'
    _ = json_path.write_text(
        json.dumps(report.model_dump(mode='json'), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    html_path = output_dir / 'probe.html'
    body = ''.join(spec.as_html() for spec in report.table_specs())
    _ = html_path.write_text(
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        f'<title>SSV occupancy probe</title></head><body>{body}</body></html>',
        encoding='utf-8',
    )
    return html_path
