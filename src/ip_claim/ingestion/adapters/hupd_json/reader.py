"""HUPD JSON streaming patent source.

Walks ``data_dir`` for ``*.json`` files, lazily decodes each one, and
emits a typed :class:`Patent` value object via async iteration. Optional
``subset``/``subset_seed`` enable deterministic random sampling without
loading the whole directory into memory.

``HupdJsonPatentSource.stream`` exposes the decode pipeline as an
``AsyncIterator[Patent]``. Streaming keeps an HUPD slice that can exceed
RAM off the heap. Per-file open/close is bounded by ``aiofiles`` buffered
reads; HUPD records are typically 50-300 KB.
"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from aiostream import pipe, stream
from returns.result import Failure, Result, Success, safe

from ip_claim.ingestion.models import Patent

from .loader import patent_from_hupd_dict
from .paths import hupd_json_files

_log = structlog.get_logger(__name__)

HupdDecode = Result[tuple[Path, dict[str, Any]], Exception]


@dataclass(frozen=True, slots=True)
class HupdJsonSourceConfig:
    """Wiring for :class:`HupdJsonPatentSource`."""

    data_dir: Path
    subset: int | None = None
    """``None`` or 0 → stream every file. Positive ``N`` → at most ``N`` files."""
    subset_seed: int = 42


@safe
def _decode_sync(path: Path) -> dict[str, Any]:
    """Synchronous fallback decode (used only by tests / direct callers)."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding='utf-8'))
    return payload


async def _decode_one(path: Path, *_extra: Path) -> HupdDecode:
    """Asynchronously read+decode one HUPD JSON file as a :class:`Result`."""
    try:
        async with aiofiles.open(path, encoding='utf-8') as fh:
            payload = json.loads(await fh.read())
    except (OSError, json.JSONDecodeError) as exc:
        return Failure(exc)
    return Success((path, payload))


def _log_or_pass(result: HupdDecode) -> bool:
    """Side-effect filter: log decode failures, keep successes."""
    match result:
        case Success(_):
            return True
        case Failure(exc):
            _log.warning('hupd_stream.skip', error=repr(exc))
            return False
        case _:  # pragma: no cover - exhaustive on Result variants
            return False


def _to_patent(result: HupdDecode, *_extra: HupdDecode) -> Patent:
    """Project a successful decode :class:`Result` into a :class:`Patent`."""
    _path, payload = result.unwrap()
    return patent_from_hupd_dict(payload)


class HupdJsonPatentSource:
    """Async iterator over HUPD patent JSON files on disk."""

    def __init__(self, config: HupdJsonSourceConfig) -> None:
        self._config = config

    def _select_files(self) -> list[Path]:
        """Resolve the file list, honoring ``subset`` + ``subset_seed``.

        Discover JSON paths, optionally sample, then sort. Deterministic given seed.
        """
        all_files = sorted(hupd_json_files(self._config.data_dir))
        n = self._config.subset or 0
        if n <= 0 or n >= len(all_files):
            return all_files
        # Comment placement matters for ruff: noqa goes on the call site.
        rng = random.Random(self._config.subset_seed)  # noqa: S311 - deterministic sampling, not cryptographic
        return sorted(rng.sample(all_files, n))

    async def stream(self) -> AsyncIterator[Patent]:
        """Yield patents from the configured ``data_dir``.

        Pipeline: ``iterate(files) | amap(_decode_one) | filter(_log_or_pass)
        | map(_to_patent)``. The trailing ``async for`` simply re-exposes
        the pipeline as an ``AsyncIterator[Patent]`` so callers do not have
        to know about ``aiostream``.
        """
        if not self._config.data_dir.is_dir():
            msg = f'HUPD data dir does not exist: {self._config.data_dir}'
            raise FileNotFoundError(msg)

        files = self._select_files()
        _log.info(
            'hupd_stream.start',
            count=len(files),
            data_dir=str(self._config.data_dir),
        )

        pipeline = (
            stream.iterate(files)
            | pipe.amap(_decode_one, task_limit=4)
            | pipe.filter(_log_or_pass)
            | pipe.map(_to_patent)
        )
        async with pipeline.stream() as streamer:
            async for patent in streamer:
                yield patent

        _log.info('hupd_stream.done', count=len(files))
