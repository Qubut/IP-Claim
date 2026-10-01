"""Shared pytest fixtures.

Convention:
    * ``unit`` tests use in-memory fakes implementing the port Protocols.
    * ``integration`` tests are gated by the ``integration`` marker and use
      ``testcontainers`` (Neo4j).
    * ``property`` tests use ``hypothesis``.
    * ``contract`` tests verify that real-world fixture HUPD JSON files still
      conform to the parser's expected schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV'] = '0'

import pytest

from ip_claim.ingestion.adapters.hupd_json.loader import patent_from_hupd_dict
from ip_claim.ingestion.models import Patent

# The three real HUPD patent fixtures committed under ``tests/fixtures/hupd``.
# Unit and integration tests MUST drive patent input from these instead of
# hand-rolled ``Patent(...)`` literals — synthetic patents are not accepted.
_HUPD_FIXTURE_NAMES = ('13817165.json', '14111139.json', '14112715.json')


@pytest.fixture(scope='session')
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return Path(__file__).resolve().parent / 'fixtures'


@pytest.fixture(scope='session')
def real_patents(fixtures_dir: Path) -> tuple[Patent, ...]:
    """All three real HUPD fixtures, loaded as domain :class:`Patent` objects."""
    return tuple(
        patent_from_hupd_dict(
            json.loads((fixtures_dir / 'hupd' / name).read_text(encoding='utf-8'))
        )
        for name in _HUPD_FIXTURE_NAMES
    )


@pytest.fixture(scope='session')
def real_patent(real_patents: tuple[Patent, ...]) -> Patent:
    """The first real HUPD fixture as a domain :class:`Patent`."""
    return real_patents[0]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear inherited ``IP_CLAIM_*`` env vars so tests stay deterministic."""
    for key in list(os.environ):
        if key.startswith('IP_CLAIM_'):
            monkeypatch.delenv(key, raising=False)
