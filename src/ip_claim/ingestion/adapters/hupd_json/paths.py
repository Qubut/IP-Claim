"""Recursive ``*.json`` walk and FileLock path-index cache for nested HUPD dumps."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path

import structlog
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from ip_claim.ingestion.adapters.hupd_json.loader import patent_from_hupd_dict
from ip_claim.ingestion.models import Patent

_log = structlog.get_logger(__name__)

_JSON_SUFFIX = '.json'


def hupd_json_files(root: Path) -> Iterator[Path]:
    """Stream regular-file HUPD JSON paths under root via ``Path.walk``."""
    return (
        dirpath / name
        for dirpath, _dirnames, filenames in root.walk()
        for name in filenames
        if name.endswith(_JSON_SUFFIX)
    )


class HupdPathIndex(BaseModel):
    """On-disk path list for a large HUPD JSON tree (shared across DDP ranks)."""

    model_config = ConfigDict(frozen=True)

    root: Path
    cache_path: Path
    progress_log_interval: int = Field(default=100_000, ge=1)

    def load(self) -> tuple[Path, ...]:
        """Return cached paths or build the index once under an exclusive file lock."""
        if cached := self.read_cache():
            _log.info('hupd.index.cache_hit', count=len(cached), path=str(self.cache_path))
            return cached

        lock_path = self.cache_path.with_suffix('.lock')
        with FileLock(lock_path):
            if cached := self.read_cache():
                _log.info('hupd.index.cache_hit', count=len(cached), path=str(self.cache_path))
                return cached
            _log.info('hupd.index.build', root=str(self.root), cache=str(self.cache_path))
            paths = self.discover_paths()
            self.write_cache(paths)
            return paths

    def read_cache(self) -> tuple[Path, ...] | None:
        """Load a previously written path index, or None when missing or empty."""
        if not self.cache_path.is_file():
            return None
        lines = self.cache_path.read_text(encoding='utf-8').splitlines()
        paths = tuple(Path(line) for line in lines if line.strip())
        return paths or None

    def discover_paths(self) -> tuple[Path, ...]:
        """Collect JSON paths under the corpus root, with periodic progress on long walks."""
        paths = tuple(
            tqdm(
                hupd_json_files(self.root),
                desc='hupd.index',
                unit='file',
                miniters=self.progress_log_interval,
                mininterval=10.0,
            )
        )
        if not paths:
            msg = f'No HUPD JSON files under {self.root}'
            raise FileNotFoundError(msg)
        _log.info('hupd.index.complete', count=len(paths), root=str(self.root))
        return paths

    def write_cache(self, paths: Sequence[Path]) -> None:
        """Atomically persist discovered paths for later rank-local reads."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix('.tmp')
        _ = tmp.write_text('\n'.join(str(path) for path in paths) + '\n', encoding='utf-8')
        _ = tmp.replace(self.cache_path)


def sample_hupd_paths(
    root: Path,
    *,
    limit: int,
    seed: int,
    index_cache: Path | None = None,
) -> tuple[tuple[Path, ...], int]:
    """Return ``limit`` paths sampled without replacement, plus the pool size."""
    pool = (
        HupdPathIndex(root=root, cache_path=index_cache).load()
        if index_cache is not None
        else tuple(hupd_json_files(root))
    )
    if not pool:
        msg = f'No HUPD JSON files under {root}'
        raise FileNotFoundError(msg)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    chosen = tuple(rng.sample(list(pool), k=min(limit, len(pool))))
    return chosen, len(pool)


def iter_hupd_json_paths(hupd_dir: Path, *, limit: int | None = None) -> tuple[Path, ...]:
    """Collect HUPD JSON file paths without reading file contents."""
    if not hupd_dir.is_dir():
        msg = f'No HUPD JSON directory at {hupd_dir}'
        raise FileNotFoundError(msg)

    paths = (
        tuple(islice(hupd_json_files(hupd_dir), limit))
        if limit is not None
        else tuple(hupd_json_files(hupd_dir))
    )
    if not paths:
        msg = f'No HUPD JSON files under {hupd_dir}'
        raise FileNotFoundError(msg)
    return paths


def patent_from_hupd_path(path: Path) -> Patent:
    """Load one HUPD JSON file through the existing dict-to-Patent mapper."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        msg = 'HUPD JSON must be an object'
        raise TypeError(msg)
    return patent_from_hupd_dict(payload)


def load_hupd_patents(hupd_dir: Path, *, limit: int | None = None) -> tuple[Patent, ...]:
    """Load domain patents from a HUPD JSON directory."""
    paths = iter_hupd_json_paths(hupd_dir, limit=limit)
    return tuple(patent_from_hupd_path(path) for path in paths)


__all__ = [
    'HupdPathIndex',
    'hupd_json_files',
    'iter_hupd_json_paths',
    'load_hupd_patents',
    'patent_from_hupd_path',
    'sample_hupd_paths',
]
