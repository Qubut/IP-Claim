from typing import Any

from pydantic import BaseModel, Field, model_validator


class LangChainPatentMetadata(BaseModel):
    """Pydantic model for LangChain document metadata."""

    application_number: str
    publication_number: str | None = None
    patent_number: str
    title: str
    decision: str
    filing_date: str | None = None
    patent_issue_date: str | None = None
    abandon_date: str | None = None
    main_cpc_label: str | None = None
    cpc_labels: list[str] = Field(default_factory=list)
    main_ipcr_label: str | None = None
    ipcr_labels: list[str] = Field(default_factory=list)
    uspc_class: str | None = None
    uspc_subclass: str | None = None
    examiner_id: str
    examiner_name: str
    inventor_countries: list[str] = Field(default_factory=list)
    inventor_count: int
    filing_year: int

    @model_validator(mode='before')
    @classmethod
    def handle_null_values(cls, data: Any) -> Any:
        """Convert None values to appropriate defaults."""
        if isinstance(data, dict):
            for field in cls.model_fields:
                if field in data and data[field] is None:
                    # Handle different field types
                    if field in {'cpc_labels', 'ipcr_labels', 'inventor_countries'}:
                        data[field] = []
                    elif field == 'inventor_count':
                        data[field] = 0
        return data


class LangChainPatentContent(BaseModel):
    """Pydantic model for LangChain patent document content."""

    Title: str
    Abstract: str
    Claims: str
    Background: str
    Summary: str
    Description: str
