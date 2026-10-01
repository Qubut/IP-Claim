"""Seeded HUPD draws and termhood publish for covering encodes.

``HupdDraw`` is one ``CollisionEncodeDataset`` sample with claim and
disclosure views. The loader never binds the full termhood table. Attach
publishes requested scores from the row-group sidecar.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from patent_ate.termhood import (
    TermhoodIndex,
    TermhoodStore,
    is_stop_surface,
    product_score_expr,
)
from torch_geometric.data import HeteroData

from ip_claim.collision.disclosure import disclosure_windows
from ip_claim.collision.encode_job import CollisionEncodeDataset, CollisionEncodeRow
from ip_claim.collision.eval import hupd_stem_index
from ip_claim.ingestion.adapters.hupd_json.paths import sample_hupd_paths
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.graph_ingress import termhood_row_group_index
from ip_claim.ssv.model import SoftTrunkModel

HupdLoader = Callable[..., 'HupdDraw']


@dataclass(frozen=True)
class HupdDraw:
    """One seeded ``CollisionEncodeDataset`` draw with claim and window views."""

    rows: tuple[CollisionEncodeRow, ...]
    n_pool: int
    claim_rows: tuple[CollisionEncodeRow, ...]
    disc_rows: tuple[CollisionEncodeRow, ...]

    def graphs(self) -> tuple[HeteroData, ...]:
        """Filing graphs in draw order."""
        return tuple(row.example.graph for row in self.rows)

    def disc_texts(self) -> tuple[str, ...]:
        """Full disclosure strings on the disclosure rows."""
        return tuple(row.disclosure for row in self.disc_rows)

    def window_rows(
        self,
        tokenizer: object,
        *,
        max_length: int,
        max_chunks: int,
        graphs: Sequence[HeteroData] | None = None,
    ) -> tuple[tuple[CollisionEncodeRow, ...], tuple[int, ...]]:
        """One encode row per disclosure window, tagged with the filing owner."""
        filing = tuple(self.graphs() if graphs is None else graphs)
        packed = tuple(
            (owner, window)
            for owner, row in enumerate(self.rows)
            for window in disclosure_windows(
                tokenizer,
                row.disclosure,
                max_length=max_length,
                max_chunks=max_chunks,
            )
        )
        rows = tuple(
            self.rows[owner].model_copy(
                update={
                    'example': self.rows[owner].example.model_copy(
                        update={
                            'text': window,
                            'graph': filing[owner],
                            'disclosure_text': window,
                        }
                    ),
                    'disclosure': window,
                }
            )
            for owner, window in packed
        )
        return rows, tuple(owner for owner, _window in packed)

    def lemma_keys(
        self,
        ssv_model: SoftTrunkModel,
        rows: Sequence[CollisionEncodeRow] | None = None,
    ) -> tuple[str, ...]:
        """Candidate lemma keys from the texts on ``rows``.

        Defaults to claim-row encode text. Pass disclosure rows when the
        encode is full-text. Never returns the published store key set.
        Remembered spans stay on the ingress for the later encode.
        """
        listed = self.claim_rows if rows is None else tuple(rows)
        texts = tuple(row.example.text for row in listed if row.example.text.strip())
        return tuple(
            span.lemma_key for doc in ssv_model.graph_ingress.candidates(texts) for span in doc
        )

    def rolled(self, offset: int = 1) -> tuple[int, ...]:
        """Cyclic foreign index used by crossed and patched cells."""
        count = len(self.rows)
        return tuple((index + offset) % count for index in range(count))

    def take(self, start: int, stop: int) -> HupdDraw:
        """Keep a contiguous slice and the original pool size."""
        return HupdDraw(
            rows=self.rows[start:stop],
            n_pool=self.n_pool,
            claim_rows=self.claim_rows[start:stop],
            disc_rows=self.disc_rows[start:stop],
        )

    def views(
        self,
        texts: Sequence[str],
        graphs: Sequence[HeteroData],
    ) -> tuple[CollisionEncodeRow, ...]:
        """Reuse the product row, swapping only the encoded text and graph."""
        return tuple(
            row.model_copy(
                update={
                    'example': row.example.model_copy(
                        update={'text': text, 'graph': graph, 'disclosure_text': text}
                    ),
                    'disclosure': text,
                }
            )
            for row, text, graph in zip(self.rows, texts, graphs, strict=True)
        )

    @classmethod
    def from_dataset(cls, dataset: CollisionEncodeDataset, n_pool: int) -> HupdDraw:
        """Claim and disclosure views from one encode dataset draw."""
        rows = tuple(dataset[index] for index in range(len(dataset)))
        claims = tuple(
            row.model_copy(
                update={'example': row.example.model_copy(update={'text': row.claim_blob})}
            )
            for row in rows
        )
        discs = tuple(
            row.model_copy(
                update={
                    'example': row.example.model_copy(
                        update={
                            'text': row.disclosure,
                            'disclosure_text': row.disclosure,
                        }
                    ),
                    'disclosure': row.disclosure,
                }
            )
            for row in rows
        )
        return cls(rows=rows, n_pool=n_pool, claim_rows=claims, disc_rows=discs)


def published_termhood_n(ssv_model: SoftTrunkModel) -> int:
    """Count keys already published on the session ingress. Does not re-extract."""
    scores = getattr(ssv_model.graph_ingress.termhood, 'scores', None)
    return 0 if not isinstance(scores, dict) else len(scores)


def hupd_loader(
    ssv_job: SsvTrainConfig,
    *,
    hygiene_n: int,
    seed: int,
) -> HupdLoader:
    """Seeded HUPD draw. Occupancy scores are attached later, not loaded here."""
    draw_seed = seed

    def cache_path() -> Path | None:
        listed = os.environ.get('COVERING_HUPD_INDEX', '').strip()
        return next(
            (
                path
                for path in (
                    Path(listed) if listed else None,
                    Path('/outputs/ssv/hupd_path_index.txt'),
                )
                if path is not None and path.is_file()
            ),
            None,
        )

    def load(
        limit: int | None = None,
        *,
        seed: int | None = None,
        stems: Sequence[str] | None = None,
        occupy: bool = True,
    ) -> HupdDraw:
        _ = occupy
        cache = cache_path()
        if stems is not None:
            listed = hupd_stem_index(ssv_job.hupd_root(), cache)
            items = tuple((app, listed[app]) for app in dict.fromkeys(stems) if app in listed)
            return HupdDraw.from_dataset(CollisionEncodeDataset(items), len(items))
        paths, n_pool = sample_hupd_paths(
            ssv_job.hupd_root(),
            limit=hygiene_n if limit is None else limit,
            seed=draw_seed if seed is None else seed,
            index_cache=cache,
        )
        return HupdDraw.from_dataset(
            CollisionEncodeDataset(tuple((path.stem, path) for path in paths)),
            n_pool,
        )

    return load


def bind_termhood_attach(
    ssv_model: SoftTrunkModel,
    termhood_store: TermhoodStore | None,
) -> Callable[[Sequence[str]], int]:
    """Publish product scores for requested keys only. Zero when no store."""
    pa.set_cpu_count(os.cpu_count() or 1)
    index = None if termhood_store is None else termhood_row_group_index(termhood_store)

    def attach(keys: Sequence[str]) -> int:
        if termhood_store is None or index is None:
            return 0
        wanted = tuple(dict.fromkeys(key for key in keys if key))
        if not wanted:
            ssv_model.graph_ingress.termhood = TermhoodIndex(
                scores={},
                source=termhood_store.root,
            )
            return 0
        key_col = index.get_column('key')
        probe = pl.DataFrame({
            'key': list(wanted),
            'pos': key_col.search_sorted(pl.Series(wanted, dtype=pl.String), side='left'),
        }).filter(pl.col('pos') < key_col.len())
        found = key_col.gather(probe.get_column('pos'))
        groups = tuple(
            sorted({
                int(group)
                for group in index
                .get_column('row_group')
                .gather(probe.filter(probe.get_column('key') == found).get_column('pos'))
                .to_list()
            })
        )
        if not groups:
            ssv_model.graph_ingress.termhood = TermhoodIndex(
                scores={},
                source=termhood_store.root,
            )
            return 0
        facts = cast(
            pl.DataFrame,
            pl.from_arrow(
                pq.ParquetFile(termhood_store.parquet).read_row_groups(
                    list(groups),
                    columns=['key', 'c_value', 'df'],
                    use_threads=True,
                )
            ),
        )
        scored = (
            facts
            .filter(pl.col('key').is_in(list(wanted)))
            .with_columns(product_score_expr(termhood_store.meta.total_docs))
            .select('key', 'score')
        )
        published = {
            str(key): float(score)
            for key, score in scored.iter_rows()
            if not is_stop_surface(str(key))
        }
        ssv_model.graph_ingress.termhood = TermhoodIndex(
            scores=published,
            source=termhood_store.root,
        )
        return len(published)

    return attach
