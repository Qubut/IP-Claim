"""Citation-pair loading seam for epo-processor ``analyze --dataset`` Parquet."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol, get_args, runtime_checkable

import pyarrow.parquet as pq
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    StringConstraints,
    TypeAdapter,
    computed_field,
    model_validator,
)

_DIGIT_STEM = StringConstraints(strip_whitespace=True, pattern=r'^\d+$')
_CATEGORY_TOKEN = StringConstraints(strip_whitespace=True, to_upper=True)

ClefIpGrade = Literal[0, 1, 2]
LocatorStatus = Literal['mapped', 'unmapped']
St14Mark = Literal['A', 'D', 'E', 'L', 'O', 'P', 'T', 'X', 'Y']
_ST14_MARKS = frozenset(get_args(St14Mark))
_UNMAPPED_CLAIMS = 'citation claim locators are unmapped'
_UNMAPPED_PASSAGES = 'citation passage locators are unmapped'


def _hupd_json_stem(value: object) -> str | None:
    if value is None:
        return None
    stem = Path(str(value)).stem
    return stem if stem.isdigit() else None


def _clef_ip_grade(marks: Sequence[str]) -> ClefIpGrade:
    letters = set(marks)
    if letters & {'X', 'Y'}:
        return 2
    if 'A' in letters:
        return 1
    return 0


def _locator_items(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return () if not value else (value,)
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        return (value,)
    return tuple(value)


def _parse_claim_locators(value: object) -> tuple[int, ...]:
    def parse_one(item: object) -> int:
        if isinstance(item, bool) or item is None:
            raise ValueError('unparseable claim locator')
        if isinstance(item, int) and item >= 1:
            return item
        if isinstance(item, str) and item.strip().isdigit():
            number = int(item.strip())
            if number >= 1:
                return number
        raise ValueError('unparseable claim locator')

    return tuple(parse_one(item) for item in _locator_items(value))


def _parse_passage_locators(value: object) -> tuple[str, ...]:
    def parse_one(item: object) -> str:
        if not isinstance(item, str):
            raise TypeError('unparseable passage locator')
        passage = item.strip()
        if not passage or passage.upper() in _ST14_MARKS:
            raise ValueError('unparseable passage locator')
        return passage

    return tuple(parse_one(item) for item in _locator_items(value))


HupdApplication = Annotated[str | None, BeforeValidator(_hupd_json_stem)]
HUPD_APPLICATION: TypeAdapter[str | None] = TypeAdapter(HupdApplication)
ApplicationNumber = Annotated[str, _DIGIT_STEM]
CitationCategory = Annotated[str, _CATEGORY_TOKEN]
ClaimLocators = Annotated[tuple[int, ...], BeforeValidator(_parse_claim_locators)]
PassageLocators = Annotated[tuple[str, ...], BeforeValidator(_parse_passage_locators)]


class CitedLocators(BaseModel):
    """Claim and passage locators on a cited partner or loaded pair."""

    model_config = ConfigDict(frozen=True)

    claims: tuple[int, ...] = ()
    passages: tuple[str, ...] = ()

    @computed_field
    @property
    def locator_status(self) -> LocatorStatus:
        """Mapped only when at least one claim or passage locator parsed."""
        return 'mapped' if self.claims or self.passages else 'unmapped'

    def mapped_claims(self) -> tuple[int, ...]:
        """Claim numbers the citation stamps, or closed failure if none."""
        if not self.claims:
            raise ValueError(_UNMAPPED_CLAIMS)
        return self.claims

    def mapped_passages(self) -> tuple[str, ...]:
        """Passage strings the citation stamps, or closed failure if none."""
        if not self.passages:
            raise ValueError(_UNMAPPED_PASSAGES)
        return self.passages


class CitationPair(CitedLocators):
    """One weak collision pair (query application and cited partner)."""

    model_config = ConfigDict(frozen=True)

    query_application_number: ApplicationNumber
    partner_application_number: ApplicationNumber
    marks: tuple[St14Mark, ...] = ()
    grade: ClefIpGrade = 0

    @model_validator(mode='before')
    @classmethod
    def _grade_from_marks(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        marks = tuple(payload.get('marks') or ())
        if marks:
            payload['grade'] = _clef_ip_grade(marks)
        return payload


class AnalyzeCited(CitedLocators):
    """One cited partner row from an analyze dataset record.

    Claim and passage locators fail closed: a missing stamp is unmapped,
    and a present but unparseable stamp is a validation error. Extra
    columns are rejected so a locator field cannot be dropped.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    cited_id: str = ''
    categories: tuple[CitationCategory, ...] = ()
    paths: tuple[str, ...] = ()
    claims: ClaimLocators = ()
    passages: PassageLocators = ()
    marks: tuple[St14Mark, ...] = ()
    grade: ClefIpGrade = 0
    application_number: ApplicationNumber | None = None

    @model_validator(mode='before')
    @classmethod
    def _derive_partner(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        tokens = tuple(
            str(item).strip().upper()
            for item in (payload.get('categories') or ())
            if str(item).strip()
        )
        marks = tuple(sorted({mark for token in tokens for mark in token if mark in _ST14_MARKS}))
        payload['marks'] = marks
        payload['grade'] = _clef_ip_grade(marks)
        paths = payload.get('paths') or ()
        cited_id = payload.get('cited_id') or ''
        raw = paths[0] if paths else cited_id
        payload['application_number'] = _hupd_json_stem(raw) if raw else None
        return payload


class AnalyzeRecord(BaseModel):
    """One analyze dataset row: query HUPD paths and cited partners."""

    model_config = ConfigDict(frozen=True, extra='ignore')

    epo_hupd_paths: tuple[str, ...] = ()
    cited_hupd: tuple[AnalyzeCited, ...] = ()
    query_application_number: ApplicationNumber | None = None

    @model_validator(mode='before')
    @classmethod
    def _derive_query(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        paths = payload.get('epo_hupd_paths') or ()
        payload['query_application_number'] = _hupd_json_stem(paths[0]) if paths else None
        return payload


@runtime_checkable
class CitationPairSource(Protocol):
    """Port for epo-processor linked citation-overlap datasets."""

    def load_pairs(self, path: Path) -> tuple[CitationPair, ...]:
        """Load pairs from an ``analyze --dataset`` Parquet path."""
        ...


class StubCitationPairSource:
    """Returns no citation pairs when no dataset path is configured or available."""

    def load_pairs(self, path: Path) -> tuple[CitationPair, ...]:
        """Returns an empty tuple for any path, including missing files."""
        _ = path
        return ()


def split_pairs_by_query(
    pairs: Sequence[CitationPair],
    *,
    train: float,
    eval_fraction: float,
    test: float,
    seed: int,
    query_limit: int | None = None,
) -> tuple[tuple[CitationPair, ...], tuple[CitationPair, ...], tuple[CitationPair, ...]]:
    """Partition pairs by query application using a seeded shuffle.

    A finite ``query_limit`` keeps X-bearing queries first inside each split
    (same seeded order), then fills remaining slots. The 60/20/20 cut does
    not change. Unlimited splits stay a prefix of the shuffle.
    """
    if abs(train + eval_fraction + test - 1.0) > 1e-6:
        msg = f'split fractions must sum to 1.0, got {train + eval_fraction + test}'
        raise ValueError(msg)

    def take_limited(ordered_ids: Sequence[str]) -> list[str]:
        if query_limit is None:
            return list(ordered_ids)
        x_apps = {pair.query_application_number for pair in pairs if 'X' in pair.marks}
        preferred = [app for app in ordered_ids if app in x_apps]
        others = [app for app in ordered_ids if app not in x_apps]
        return [*preferred, *others][:query_limit]

    queries = sorted({pair.query_application_number for pair in pairs})
    queries = sorted(
        queries,
        key=lambda app: hashlib.sha256(f'{seed}:{app}'.encode()).hexdigest(),
    )
    n_train = int(len(queries) * train)
    n_eval = int(len(queries) * eval_fraction)
    train_ids = take_limited(queries[:n_train])
    eval_ids = take_limited(queries[n_train : n_train + n_eval])
    test_ids = take_limited(queries[n_train + n_eval :])
    train_set, eval_set, test_set = set(train_ids), set(eval_ids), set(test_ids)
    return (
        tuple(pair for pair in pairs if pair.query_application_number in train_set),
        tuple(pair for pair in pairs if pair.query_application_number in eval_set),
        tuple(pair for pair in pairs if pair.query_application_number in test_set),
    )


class EpoProcessorCitationPairSource:
    """Load weak collision pairs from epo-processor ``analyze --dataset`` Parquet."""

    def load_pairs(self, path: Path) -> tuple[CitationPair, ...]:
        """Validate each row as ``AnalyzeRecord`` and emit query/partner pairs."""
        if not path.is_file():
            return ()
        return tuple(
            CitationPair(
                query_application_number=query,
                partner_application_number=partner,
                marks=cited.marks,
                grade=cited.grade,
                claims=cited.claims,
                passages=cited.passages,
            )
            for record in (
                AnalyzeRecord.model_validate(row) for row in pq.read_table(path).to_pylist()
            )
            if (query := record.query_application_number) is not None
            for cited in record.cited_hupd
            if (partner := cited.application_number) is not None and partner != query
        )


__all__ = [
    'HUPD_APPLICATION',
    'AnalyzeCited',
    'AnalyzeRecord',
    'ApplicationNumber',
    'CitationPair',
    'CitationPairSource',
    'CitedLocators',
    'ClefIpGrade',
    'EpoProcessorCitationPairSource',
    'HupdApplication',
    'LocatorStatus',
    'St14Mark',
    'StubCitationPairSource',
    'split_pairs_by_query',
]
