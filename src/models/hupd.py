from datetime import datetime
from typing import Annotated, Any, ClassVar

from beanie import Document, Indexed
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ApplicationMetadata(BaseModel):
    """Core application identification information."""

    application_number: Annotated[str, Indexed(unique=True)]
    publication_number: str | None = None
    patent_number: str
    title: str
    decision: str


class ApplicationDates(BaseModel):
    """All date-related fields."""

    date_produced: datetime | None = None
    date_published: datetime | None = None
    filing_date: datetime
    patent_issue_date: datetime | None = None
    abandon_date: datetime | None = None

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode='before')
    @classmethod
    def convert_date_fields(cls, data: Any) -> Any:  # noqa: C901
        """Convert string dates to datetime objects, handling empty strings."""
        date_fields = [
            'date_produced',
            'date_published',
            'filing_date',
            'patent_issue_date',
            'abandon_date',
        ]

        for field in date_fields:
            if field not in data:
                continue

            value = data[field]
            if isinstance(value, str):
                # Handle empty strings
                if not value.strip():
                    data[field] = None
                else:
                    try:
                        # Try parsing YYYYMMDD format
                        data[field] = datetime.strptime(value, '%Y%m%d')  # noqa: DTZ007
                    except ValueError:
                        try:
                            # Try parsing YYYY-MM-DD format
                            data[field] = datetime.strptime(value, '%Y-%m-%d')  # noqa: DTZ007
                        except ValueError:
                            # Fallback to None if parsing fails
                            data[field] = None
        return data


class ClassificationInfo(BaseModel):
    """Patent classification systems."""

    main_cpc_label: str | None = None
    cpc_labels: list[str] = Field(default_factory=list)
    main_ipcr_label: str | None = None
    ipcr_labels: list[str] = Field(default_factory=list)
    uspc_class: str | None = None
    uspc_subclass: str | None = None


class ExaminerInfo(BaseModel):
    """Examiner details."""

    examiner_id: str
    examiner_name_last: str
    examiner_name_first: str
    examiner_name_middle: str | None = None


class Inventor(BaseModel):
    """Individual inventor information."""

    inventor_name_last: str
    inventor_name_first: str
    inventor_city: str | None = None
    inventor_state: str | None = None
    inventor_country: str | None = None


class PatentContent(BaseModel):
    """Text content sections of the patent."""

    abstract: str
    claims: str
    background: str
    summary: str
    full_description: str


class PatentApplication(Document):
    """Full patent application document."""

    metadata: ApplicationMetadata
    dates: ApplicationDates
    classification: ClassificationInfo
    examiner: ExaminerInfo
    inventors: list[Inventor] = Field(default_factory=list, alias='inventor_list')
    content: PatentContent

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator('inventors', mode='before')
    @classmethod
    def ensure_inventor_list(cls, value: Any) -> list[Inventor]:
        """Ensure inventor_list is always a list."""
        return value or []

    @model_validator(mode='after')
    def validate_abandonment_date(self) -> 'PatentApplication':
        """Validate abandonment logic."""
        if self.dates.abandon_date and self.metadata.decision.lower() == 'accepted':
            raise ValueError('Accepted patents cannot have an abandon date')
        return self

    class Settings:
        name = 'applications'
        indexes: ClassVar = [
            'dates.filing_date',
            'classification.main_cpc_label',
            'metadata.decision',
            'examiner.examiner_id',
            'inventors.inventor_country',
            [('metadata.title', 'text')],
        ]

    @property
    def application_number(self) -> str:
        """Return the unique application number."""
        return self.metadata.application_number

    @property
    def filing_year(self) -> int:
        """Extract the year from the filing date."""
        return self.dates.filing_date.year

    @property
    def inventor_countries(self) -> list[str]:
        """List of unique countries from all inventors (excluding nulls)."""
        return list({inv.inventor_country for inv in self.inventors if inv.inventor_country})
