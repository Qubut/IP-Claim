# ruff: noqa: N818
"""Cross-cutting error types for ip_claim.

Two complementary shapes live here, both framework-free and dependency-free
(so ``shared`` never depends on any capability):

* **Exception hierarchy** (``IpClaimError`` and friends) — raised at adapter
  boundaries and inside the domain/application layers.
* **Result payloads** (``AppError`` and friends) — frozen-dataclass error
  *values* carried inside ``returns.result.Result[T, AppError]`` at the
  adapter edge, so failures become ordinary values.

Callers get a stable taxonomy for ``except`` clauses across adapter swaps,
while the boundary stays raise-free. A new exception subclass needs a
corresponding Result mapping so the two shapes stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---- Exception hierarchy -------------------------------------------------
class IpClaimError(Exception):
    """Root of the ip_claim exception hierarchy."""


class ConfigurationError(IpClaimError):
    """Raised when configuration is missing, malformed, or fails validation."""


class SourceError(IpClaimError):
    """Raised by source adapters (HUPD JSON reader, etc.)."""


class SourceParseError(SourceError):
    """Raised when a source record cannot be parsed into a domain Patent."""


class ExtractionError(IpClaimError):
    """Raised by NER / relation-extraction adapters."""


class BudgetExceededError(ExtractionError):
    """Raised when LLM budget guard trips (cost / runaway protection)."""


class KGRepositoryError(IpClaimError):
    """Raised by Neo4j repository on persistence failures."""


class SchemaMigrationError(KGRepositoryError):
    """Raised when Neo4j schema migrations fail to apply."""


class MemoryAdapterError(IpClaimError):
    """Raised by the mem0 memory adapter."""


# Historic alias kept for callers that imported ``domain.MemoryError``.
MemoryError = MemoryAdapterError  # noqa: A001 — domain-scoped, intentional


# ---- Result payloads (adapter boundary) ----------------------------------
@dataclass(frozen=True, slots=True)
class AppError:
    """Base error payload for adapter-boundary failures.

    Attributes:
        message: Human-readable failure summary.
        cause: Optional original exception raised by an underlying library.
        context: Optional structured metadata for debugging/reporting.
    """

    message: str
    cause: Exception | None = None
    context: dict[str, Any] | None = None

    def __str__(self) -> str:
        """Return a formatted error string with cause and context inline."""
        msg = self.message
        if self.cause is not None:
            msg += f' (caused by: {type(self.cause).__name__}: {self.cause})'
        if self.context:
            msg += f' {self.context}'
        return msg


@dataclass(frozen=True, slots=True)
class ConfigPayload(AppError):
    """Configuration loading / parsing / validation failure payload."""


@dataclass(frozen=True, slots=True)
class SourceReadPayload(AppError):
    """Patent-source read / parse failure payload (HUPD JSON, etc.).

    Attributes:
        file_path: Optional path to the file that failed.
    """

    file_path: str | None = None

    def __str__(self) -> str:
        """Format including ``file_path`` when present."""
        msg = super().__str__()
        if self.file_path:
            msg += f' (file: {self.file_path})'
        return msg


@dataclass(frozen=True, slots=True)
class KGRepositoryPayload(AppError):
    """Knowledge-graph repository (Neo4j) failure payload.

    Attributes:
        statement: Optional Cypher statement that failed.
    """

    statement: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryAdapterPayload(AppError):
    """Mem0 adapter failure payload."""


@dataclass(frozen=True, slots=True)
class LLMClientPayload(AppError):
    """LLM client failure payload (timeouts, rate limits, schema errors).

    Attributes:
        endpoint: Optional endpoint URL or model identifier.
    """

    endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetryPayload(AppError):
    """Telemetry sink failure payload (OTel exporter)."""


DomainErrorPayload = (
    ConfigPayload
    | SourceReadPayload
    | KGRepositoryPayload
    | MemoryAdapterPayload
    | LLMClientPayload
    | TelemetryPayload
)

__all__ = [
    'AppError',
    'BudgetExceededError',
    'ConfigPayload',
    'ConfigurationError',
    'DomainErrorPayload',
    'ExtractionError',
    'IpClaimError',
    'KGRepositoryError',
    'KGRepositoryPayload',
    'LLMClientPayload',
    'MemoryAdapterError',
    'MemoryAdapterPayload',
    'MemoryError',
    'SchemaMigrationError',
    'SourceError',
    'SourceParseError',
    'SourceReadPayload',
    'TelemetryPayload',
]
