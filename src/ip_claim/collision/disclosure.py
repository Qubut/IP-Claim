"""Collision-only disclosure windows and length judgment.

Training text stays claims, abstract, and summary. Description mass is split
into non-overlapping host windows of the collator length, then summed into
``n_full``. Overlap would double-count the intensity sum.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import polars as pl
import structlog
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from ip_claim.ingestion.adapters.hupd_json.loader import patent_from_hupd_dict
from ip_claim.ingestion.adapters.hupd_json.paths import HupdPathIndex, hupd_json_files
from ip_claim.ssv.graph_batch import patent_disclosure_text

_log = structlog.get_logger(__name__)

DEFAULT_DISCLOSURE_JUDGMENT = 'disclosure-length.json'
DEFAULT_MEASURE_SAMPLE = 2048


class DisclosureLengthReport(BaseModel):
    """Token and window histogram for a sampled HUPD disclosure set."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan='null')

    sampled: int
    nonempty: int
    max_length: int
    window_width: int
    empty_fraction: float
    token_p50: float | None = None
    token_p90: float | None = None
    token_p99: float | None = None
    token_max: int | None = None
    chunks_p50: float | None = None
    chunks_p90: float | None = None
    chunks_p99: float | None = None
    fraction_tokens_beyond_8: float | None = None
    fraction_tokens_beyond_16: float | None = None


def content_window_width(tokenizer: Any, max_length: int) -> int:
    """Content tokens per window after the host's special-token budget."""
    special = int(tokenizer.num_special_tokens_to_add(pair=False))
    return max(int(max_length) - special, 1)


def disclosure_token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    """Host content ids for one disclosure string. Empty text is an empty tuple.

    The host tokenizer warns when a single encode exceeds ``model_max_length``
    (ModernBERT: 8192). These ids are sliced to the collator window before any
    model forward, so the call is length-unbounded and silent.
    """
    if not text.strip():
        return ()
    return tuple(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
    )


def disclosure_windows(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    max_chunks: int,
) -> tuple[str, ...]:
    """Non-overlapping decoded windows, capped at ``max_chunks``."""
    if max_chunks < 1:
        return ()
    ids = disclosure_token_ids(tokenizer, text)
    if not ids:
        return ()
    width = content_window_width(tokenizer, max_length)
    parts = tuple(ids[offset : offset + width] for offset in range(0, len(ids), width))
    kept = parts[: int(max_chunks)]
    return tuple(tokenizer.decode(part, skip_special_tokens=True) for part in kept if part)


def add_chunk_intensities(base: Tensor, chunks: Tensor, owners: Tensor) -> Tensor:
    """Add each chunk row onto the owner row of the first-window intensity."""
    added = base.clone()
    return added.index_add(0, owners, chunks)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    frame = pl.DataFrame({'v': list(values)})
    return float(frame.select(pl.col('v').quantile(q)).item())


def _beyond_cap(tokens: Sequence[int], width: int, cap: int) -> float | None:
    if not tokens:
        return None
    kept = width * cap
    dropped = tuple(max(count - kept, 0) / count if count else 0.0 for count in tokens)
    return float(sum(dropped) / len(dropped))


def sample_hupd_paths(
    hupd_dir: Path,
    *,
    sample: int,
    seed: int,
    index_cache: Path | None = None,
) -> tuple[Path, ...]:
    """Seeded sample of HUPD JSON paths. Prefer an on-disk path index when present."""
    if index_cache is not None and index_cache.is_file():
        pool = HupdPathIndex(root=hupd_dir, cache_path=index_cache).load()
    else:
        pool = tuple(islice(hupd_json_files(hupd_dir), max(int(sample) * 20, int(sample))))
    if not pool:
        msg = f'No HUPD JSON files under {hupd_dir}'
        raise FileNotFoundError(msg)
    take = min(int(sample), len(pool))
    if take == len(pool):
        return pool
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(int(seed)))
    return tuple(pool[int(index)] for index in order[:take].tolist())


def measure_disclosure_lengths(
    paths: Sequence[Path],
    tokenizer: Any,
    *,
    max_length: int,
) -> DisclosureLengthReport:
    """Token counts and implied window counts for one disclosure sample."""
    width = content_window_width(tokenizer, max_length)
    tokens = tuple(
        len(disclosure_token_ids(tokenizer, patent_disclosure_text(patent_from_hupd_dict(row))))
        for path in paths
        if (row := json.loads(path.read_text(encoding='utf-8')))
    )
    live = tuple(count for count in tokens if count > 0)
    chunks = tuple((count + width - 1) // width for count in live)
    return DisclosureLengthReport(
        sampled=len(tokens),
        nonempty=len(live),
        max_length=int(max_length),
        window_width=width,
        empty_fraction=(1.0 - (len(live) / len(tokens))) if tokens else 1.0,
        token_p50=_quantile(live, 0.5),
        token_p90=_quantile(live, 0.9),
        token_p99=_quantile(live, 0.99),
        token_max=max(live) if live else None,
        chunks_p50=_quantile(chunks, 0.5),
        chunks_p90=_quantile(chunks, 0.9),
        chunks_p99=_quantile(chunks, 0.99),
        fraction_tokens_beyond_8=_beyond_cap(live, width, 8),
        fraction_tokens_beyond_16=_beyond_cap(live, width, 16),
    )


def write_disclosure_length_report(path: Path, report: DisclosureLengthReport) -> Path:
    """Write the disclosure-length JSON next to covering artefacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(report.model_dump_json(indent=2) + '\n', encoding='utf-8')
    return path


def measure_disclosure_from_paths(
    *,
    hupd_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    max_length: int,
    sample: int = DEFAULT_MEASURE_SAMPLE,
    seed: int = 42,
    index_cache: Path | None = None,
) -> Path:
    """Sample HUPD JSON, histogram disclosure tokens, write disclosure-length.json."""
    paths = sample_hupd_paths(
        hupd_dir,
        sample=sample,
        seed=seed,
        index_cache=index_cache,
    )
    report = measure_disclosure_lengths(paths, tokenizer, max_length=max_length)
    written = write_disclosure_length_report(output_dir / DEFAULT_DISCLOSURE_JUDGMENT, report)
    _log.info(
        'collision.disclosure.measured',
        path=str(written),
        sampled=report.sampled,
        nonempty=report.nonempty,
        token_p90=report.token_p90,
        chunks_p90=report.chunks_p90,
    )
    return written
