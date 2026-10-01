"""Score-time keep grid on existing encode shards.

Each cell sparsifies query demand only. Document supply stays dense.
Does not encode, explain, or load the trunk.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

from ip_claim.collision.collide import CollisionEvalResult, PatentEmbeddingRecord
from ip_claim.collision.config import CollisionEvalConfig, CoveringKnobs, KeepGridCell
from ip_claim.collision.data.citation_pairs import CitationPair, CitationPairSource
from ip_claim.collision.encode_job import CollisionEncodeShardStore
from ip_claim.collision.eval import (
    encode_shard_root,
    rank_covering_from_records,
    split_collision_pairs,
)

_log = structlog.get_logger(__name__)

KEEP_GRID_JSON = 'keep-grid.json'


class KeepGridCellResult(BaseModel):
    """Eval unpaid and covering for one keep-grid cell."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan='null')

    stem: str
    covering: str
    slot_mass_keep: float | None
    slot_top_k: int | None
    queries_x: int
    unpaid_x: float
    unpaid_y: float
    unpaid_a: float
    unpaid_random: float
    covering_x: float
    covering_y: float
    covering_a: float
    covering_random: float


class KeepGridReport(BaseModel):
    """All keep-grid cells scored on one encoded corpus."""

    model_config = ConfigDict(frozen=True)

    encoded_apps: int
    shard_dir: str
    cells: tuple[KeepGridCellResult, ...]


def rank_keep_grid(
    eval_config: CollisionEvalConfig,
    records: tuple[PatentEmbeddingRecord, ...],
    *,
    train_pairs: tuple[CitationPair, ...],
    eval_pairs: tuple[CitationPair, ...],
    test_pairs: tuple[CitationPair, ...],
) -> KeepGridReport:
    """Rank every declared keep cell and write one covering.json per cell."""
    if not eval_config.keep_grid:
        msg = 'keep_grid is empty'
        raise ValueError(msg)

    def cell_stem(knobs: CoveringKnobs) -> str:
        parts: list[str] = []
        if knobs.slot_mass_keep is not None:
            parts.append(f'mass{round(knobs.slot_mass_keep * 100):03d}')
        if knobs.slot_top_k is not None:
            parts.append(f'topk{knobs.slot_top_k}')
        return '-'.join(parts) or 'dense'

    def cell_result(
        stem: str,
        path: Path,
        knobs: CoveringKnobs,
        ranked: CollisionEvalResult,
    ) -> KeepGridCellResult:
        return KeepGridCellResult(
            stem=stem,
            covering=str(path / 'covering.json'),
            slot_mass_keep=knobs.slot_mass_keep,
            slot_top_k=knobs.slot_top_k,
            queries_x=ranked.queries_x,
            unpaid_x=ranked.unpaid_x,
            unpaid_y=ranked.unpaid_y,
            unpaid_a=ranked.unpaid_a,
            unpaid_random=ranked.unpaid_random,
            covering_x=ranked.covering_x,
            covering_y=ranked.covering_y,
            covering_a=ranked.covering_a,
            covering_random=ranked.covering_random,
        )

    root = Path(eval_config.output_dir)

    def rank_cell(cell: KeepGridCell) -> KeepGridCellResult:
        knobs = cell.as_covering(eval_config.covering)
        stem = cell_stem(knobs)
        ranked = rank_covering_from_records(
            eval_config.model_copy(
                update={
                    'covering': knobs,
                    'output_dir': str(root / stem),
                }
            ),
            records,
            train_pairs=train_pairs,
            eval_pairs=eval_pairs,
            test_pairs=test_pairs,
        )
        return cell_result(stem, root / stem, knobs, ranked['eval'])

    cells = tuple(rank_cell(cell) for cell in eval_config.keep_grid)
    report = KeepGridReport(
        encoded_apps=len(records),
        shard_dir=str(encode_shard_root(eval_config)),
        cells=cells,
    )
    root.mkdir(parents=True, exist_ok=True)
    written = root / KEEP_GRID_JSON
    _ = written.write_text(report.model_dump_json(indent=2) + '\n', encoding='utf-8')
    _log.info(
        'collision.rank.keep_grid_done',
        path=str(written),
        cells=len(cells),
        encoded_apps=len(records),
    )
    return report


def rank_keep_grid_from_paths(
    eval_config: CollisionEvalConfig,
    *,
    dataset_path: Path,
    pair_source: CitationPairSource,
    shard_dir: Path | None = None,
) -> KeepGridReport:
    """Load complete shards and pair splits, then sweep the keep grid."""
    root = shard_dir if shard_dir is not None else encode_shard_root(eval_config)
    records = CollisionEncodeShardStore(root=root).try_read_complete()
    if records is None:
        msg = f'complete encode shards are required under {root}'
        raise FileNotFoundError(msg)
    split = split_collision_pairs(
        eval_config,
        dataset_path=dataset_path,
        pair_source=pair_source,
    )
    if split is None:
        msg = f'no citation pairs in {dataset_path}'
        raise ValueError(msg)
    train_pairs, eval_pairs, test_pairs = split
    return rank_keep_grid(
        eval_config.model_copy(update={'encode_shards': str(root)}),
        records,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        test_pairs=test_pairs,
    )


__all__ = [
    'KEEP_GRID_JSON',
    'KeepGridCellResult',
    'KeepGridReport',
    'rank_keep_grid',
    'rank_keep_grid_from_paths',
]
