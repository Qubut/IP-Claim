"""Lazy HUPD dataset: domain patents become SoftMlmExample graph rows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import Dataset

from ip_claim.ingestion.adapters.hupd_json.paths import (
    HupdPathIndex,
    iter_hupd_json_paths,
    patent_from_hupd_path,
)
from ip_claim.ingestion.models import Patent
from ip_claim.ssv.collate import SoftMlmExample
from ip_claim.ssv.graph_batch import (
    PatentGraphBatch,
    dest_claim_texts,
    graph_batch_from_patent,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FIXTURES = _PACKAGE_ROOT / 'tests' / 'fixtures' / 'hupd'


def example_from_graph_batch(
    batch: PatentGraphBatch,
    *,
    claim_text: str | None = None,
) -> SoftMlmExample:
    """Keep MLM text separate from numbered-claim and disclosure views."""
    independents = dest_claim_texts(batch)
    demand = independents[0] if independents else ''
    return SoftMlmExample(
        text=batch.text,
        graph=batch.data,
        claim_text=demand if claim_text is None else claim_text,
        disclosure_text=batch.disclosure,
    )


def dest_examples_from_graph_batch(batch: PatentGraphBatch) -> tuple[SoftMlmExample, ...]:
    """One dest example per independent numbered claim; empty demand when none."""
    independents = dest_claim_texts(batch)
    if not independents:
        return (example_from_graph_batch(batch, claim_text=''),)
    return tuple(example_from_graph_batch(batch, claim_text=text) for text in independents)


def soft_mlm_example_from_path(path: Path) -> SoftMlmExample:
    """Parse one HUPD JSON file into a collator-ready example."""
    return example_from_graph_batch(graph_batch_from_patent(patent_from_hupd_path(path)))


def examples_from_patents(patents: Sequence[Patent]) -> tuple[SoftMlmExample, ...]:
    """Map domain patents into SoftMlmExample rows."""
    return tuple(example_from_graph_batch(graph_batch_from_patent(patent)) for patent in patents)


class LazyHupdMlmDataset(Dataset[SoftMlmExample]):
    """Patent MLM dataset that parses JSON and builds graphs on each access."""

    def __init__(
        self,
        hupd_dir: Path | None = None,
        *,
        limit: int | None = None,
        index_cache: Path | None = None,
    ) -> None:
        root = hupd_dir if hupd_dir is not None else _DEFAULT_FIXTURES
        if limit is None and index_cache is not None:
            self._paths = HupdPathIndex(root=root, cache_path=index_cache).load()
        else:
            self._paths = iter_hupd_json_paths(root, limit=limit)

    def __len__(self) -> int:
        """Number of indexed HUPD JSON files."""
        return len(self._paths)

    def __getitem__(self, index: int) -> SoftMlmExample:
        """Load and graph-build one patent by file index."""
        return soft_mlm_example_from_path(self._paths[index])


__all__ = [
    'LazyHupdMlmDataset',
    'dest_examples_from_graph_batch',
    'example_from_graph_batch',
    'examples_from_patents',
    'soft_mlm_example_from_path',
]
