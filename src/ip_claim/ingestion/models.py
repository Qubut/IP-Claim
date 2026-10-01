"""Patent, claim, inventor, and classification value objects for ingest."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


def _blank_as_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _none_as_blank(value: object) -> str:
    if value is None:
        return ''
    return str(value).strip()


def _ymd_or_none(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[0:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def _label_codes(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, (list, tuple)):
        return tuple(text for item in value if (text := str(item).strip()))
    return ()


ApplicationNumber = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
OptionalText = Annotated[str | None, BeforeValidator(_blank_as_none)]
PlainText = Annotated[str, BeforeValidator(_none_as_blank)]
OptionalYmd = Annotated[date | None, BeforeValidator(_ymd_or_none)]
LabelCodes = Annotated[tuple[str, ...], BeforeValidator(_label_codes)]


class _Frozen(BaseModel):
    """Base for immutable, hashable, validated value objects."""

    model_config = ConfigDict(
        frozen=True,
        extra='ignore',  # ignore unknown HUPD keys (forward compat)
        str_strip_whitespace=True,
        validate_assignment=False,
    )


class Inventor(_Frozen):
    """Single inventor on a patent."""

    last_name: PlainText = ''
    first_name: PlainText = ''
    city: OptionalText = None
    state: OptionalText = None
    country: OptionalText = None

    @property
    def full_name(self) -> str:
        """Canonical full name (used as Neo4j ``:Inventor`` MERGE key)."""
        return f'{self.first_name.strip()} {self.last_name.strip()}'.strip()


class Examiner(_Frozen):
    """USPTO examiner."""

    examiner_id: PlainText = ''
    last_name: PlainText = ''
    first_name: PlainText = ''
    middle_name: OptionalText = None

    @property
    def full_name(self) -> str:
        """Concatenated first / middle / last names with single spaces."""
        parts = [self.first_name, self.middle_name, self.last_name]
        return ' '.join(p for p in parts if p and p.strip())


class Classification(_Frozen):
    """CPC / IPCR / USPC classification of a patent."""

    main_cpc: OptionalText = None
    cpc_codes: LabelCodes = Field(default_factory=tuple)
    main_ipcr: OptionalText = None
    ipcr_codes: LabelCodes = Field(default_factory=tuple)
    uspc_class: OptionalText = None
    uspc_subclass: OptionalText = None


class Claim(_Frozen):
    """A single claim within a patent."""

    number: int = Field(ge=1)
    text: str
    is_independent: bool = True
    parent_number: int | None = None
    """For dependent claims: the claim number this one depends on."""


class SectionKind(StrEnum):
    """Closed set of patent text-section kinds."""

    ABSTRACT = 'ABSTRACT'
    BACKGROUND = 'BACKGROUND'
    SUMMARY = 'SUMMARY'
    DESCRIPTION = 'DESCRIPTION'


class TextSection(_Frozen):
    """One free-text section of a patent (abstract, background, summary, ...).

    Distinct from :class:`Claim` because claims have legal status; sections
    are descriptive prose.
    """

    kind: SectionKind
    text: str

    @property
    def is_empty(self) -> bool:
        """True iff the section text is whitespace-only."""
        return not self.text.strip()


class Patent(_Frozen):
    """Top-level patent value object — the domain root for KG ingestion.

    Built from one HUPD JSON file. Carries everything the KG-build pipeline
    needs; nothing more. Persistence-shape (Neo4j Cypher params) is the
    repository's concern, not this object's.
    """

    application_number: ApplicationNumber
    publication_number: OptionalText = None
    patent_number: OptionalText = None
    title: PlainText = ''
    decision: str

    filing_date: OptionalYmd = None
    publication_date: OptionalYmd = None
    issue_date: OptionalYmd = None
    abandon_date: OptionalYmd = None

    classification: Classification = Field(default_factory=Classification)
    inventors: tuple[Inventor, ...] = Field(default_factory=tuple)
    examiner: Examiner | None = None

    abstract: PlainText = ''
    background: PlainText = ''
    summary: PlainText = ''
    full_description: PlainText = ''
    claims: tuple[Claim, ...] = Field(default_factory=tuple)

    @field_validator('decision', mode='before')
    @classmethod
    def _normalize_decision(cls, v: object) -> str:
        return str(v).strip().upper() if v is not None else 'UNKNOWN'

    @property
    def filing_year(self) -> int | None:
        """Year of the filing date, or None if not filed yet."""
        return self.filing_date.year if self.filing_date else None

    @property
    def inventor_countries(self) -> tuple[str, ...]:
        """Unique inventor country codes (None values dropped)."""
        return tuple({inv.country for inv in self.inventors if inv.country})

    @property
    def has_text_content(self) -> bool:
        """True iff at least one text section is non-empty (skip-empty guard)."""
        return any((self.abstract, self.background, self.summary, self.full_description)) or bool(
            self.claims
        )
