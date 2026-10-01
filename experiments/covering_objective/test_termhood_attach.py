"""Host-free termhood attach: requested keys only, no full-table collect."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from patent_ate.extract import write_termhood
from patent_ate.termhood import TermhoodIndex, TermhoodStore, TermhoodTable

from experiments.covering_objective.corpus import (
    bind_termhood_attach,
    termhood_row_group_index,
)
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = pytest.mark.experiment


def _ingress_model() -> SoftTrunkModel:
    return cast(
        SoftTrunkModel,
        cast(object, SimpleNamespace(graph_ingress=SimpleNamespace(termhood=None))),
    )


class TestTermhoodAttachHost:
    """Bounded attach against a committed store. No encode."""

    def test_missing_store_is_zero(self) -> None:
        """No store publishes nothing."""
        model = _ingress_model()
        attach = bind_termhood_attach(model, None)
        assert attach(('coil spring',)) == 0
        assert model.graph_ingress.termhood is None

    def test_empty_keys_publish_empty_index(self, tmp_path: Path) -> None:
        """Empty request clears the published scores."""
        table = TermhoodTable(
            c_values={'coil spring': 3.5},
            document_frequency={'coil spring': 2},
            total_docs=12,
        )
        store = TermhoodStore.open(write_termhood(table, tmp_path))
        model = _ingress_model()
        attach = bind_termhood_attach(model, store)
        assert attach(()) == 0
        published = model.graph_ingress.termhood
        assert isinstance(published, TermhoodIndex)
        assert published.scores == {}
        assert published.source == store.root

    def test_attach_returns_only_requested_keys(self, tmp_path: Path) -> None:
        """Scores come from the store product, keyed by the request only."""
        table = TermhoodTable(
            c_values={
                'coil spring': 3.5,
                'vehicle interior': 1.25,
                'other phrase': 8.0,
                'comprising': 9.0,
            },
            document_frequency={
                'coil spring': 2,
                'vehicle interior': 4,
                'other phrase': 1,
                'comprising': 2,
            },
            total_docs=12,
        )
        store = TermhoodStore.open(write_termhood(table, tmp_path))
        model = _ingress_model()
        attach = bind_termhood_attach(model, store)
        wanted = ('coil spring', 'missing', 'coil spring', 'comprising')
        assert attach(wanted) == 1
        published = model.graph_ingress.termhood
        assert isinstance(published, TermhoodIndex)
        assert tuple(published.scores) == ('coil spring',)
        assert published.scores['coil spring'] == pytest.approx(table.score('coil spring'))
        assert 'comprising' not in published.scores

    def test_attach_does_not_scan_collect(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The lazy full-table collect path stays unused."""
        table = TermhoodTable(
            c_values={'coil spring': 3.5},
            document_frequency={'coil spring': 2},
            total_docs=12,
        )
        store = TermhoodStore.open(write_termhood(table, tmp_path))

        def boom(*_args: object, **_kwargs: object) -> pl.LazyFrame:
            raise AssertionError('scan_parquet must not run')

        monkeypatch.setattr(pl, 'scan_parquet', boom)
        model = _ingress_model()
        attach = bind_termhood_attach(model, store)
        assert attach(('coil spring',)) == 1

    def test_attach_reads_only_matching_row_group(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After the key index exists, attach reads one matching row group."""
        keys = tuple(f'k{index:02d}' for index in range(8))
        table = pa.table({
            'key': pa.array(keys, type=pa.string()),
            'c_value': pa.array(tuple(float(index) for index in range(8)), type=pa.float64()),
            'df': pa.array((1,) * 8, type=pa.int64()),
        })
        partial = tmp_path / 'termhood.parquet.partial'
        pq.write_table(table, partial, row_group_size=2)
        store = TermhoodStore.commit(partial, tmp_path, total_docs=100)
        assert termhood_row_group_index(store).height == 8
        model = _ingress_model()
        attach = bind_termhood_attach(model, store)
        seen: list[int] = []
        original_one = pq.ParquetFile.read_row_group
        original_many = pq.ParquetFile.read_row_groups

        def spy_one(
            self: pq.ParquetFile,
            i: int,
            columns: list[str] | None = None,
            *,
            use_threads: bool = True,
            use_pandas_metadata: bool = False,
        ) -> pa.Table:
            seen.append(int(i))
            return original_one(
                self,
                i,
                columns=columns,
                use_threads=use_threads,
                use_pandas_metadata=use_pandas_metadata,
            )

        def spy_many(
            self: pq.ParquetFile,
            indices: list[int],
            columns: list[str] | None = None,
            *,
            use_threads: bool = True,
            use_pandas_metadata: bool = False,
        ) -> pa.Table:
            seen.extend(int(index) for index in indices)
            return original_many(
                self,
                indices,
                columns=columns,
                use_threads=use_threads,
                use_pandas_metadata=use_pandas_metadata,
            )

        monkeypatch.setattr(pq.ParquetFile, 'read_row_group', spy_one)
        monkeypatch.setattr(pq.ParquetFile, 'read_row_groups', spy_many)
        assert attach(('k05',)) == 1
        assert seen == [2]
