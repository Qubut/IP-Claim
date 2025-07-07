from collections.abc import AsyncGenerator
from typing import Any

from beanie import Document as BeanieDocument
from langchain_core.documents import Document as LangChainDocument

from models.hupd import PatentApplication
from models.langchain_patent_doc import LangChainPatentContent, LangChainPatentMetadata


class PatentConverter:
    """Converts PatentApplication documents to LangChain documents with batching."""

    @staticmethod
    def _build_page_content(patent: BeanieDocument) -> str:
        """Construct page content from specified sections."""
        return LangChainPatentContent(
            Title=patent.metadata.title,
            Abstract=patent.content.abstract,
            Claims=patent.content.claims,
            Background=patent.content.background,
            Summary=patent.content.summary,
            Description=patent.content.full_description,
        ).model_dump_json()

    @staticmethod
    def _build_metadata(patent: BeanieDocument) -> dict[str, Any]:
        """Extract and format metadata."""
        return LangChainPatentMetadata(
            application_number=patent.metadata.application_number,
            publication_number=patent.metadata.publication_number,
            patent_number=patent.metadata.patent_number,
            title=patent.metadata.title,
            decision=patent.metadata.decision,
            filing_date=patent.dates.filing_date.isoformat() if patent.dates.filing_date else None,
            patent_issue_date=patent.dates.patent_issue_date.isoformat()
            if patent.dates.patent_issue_date
            else None,
            abandon_date=patent.dates.abandon_date.isoformat()
            if patent.dates.abandon_date
            else None,
            main_cpc_label=patent.classification.main_cpc_label,
            cpc_labels=patent.classification.cpc_labels,
            main_ipcr_label=patent.classification.main_ipcr_label,
            ipcr_labels=patent.classification.ipcr_labels,
            uspc_class=patent.classification.uspc_class,
            uspc_subclass=patent.classification.uspc_subclass,
            examiner_id=patent.examiner.examiner_id,
            examiner_name=f'{patent.examiner.examiner_name_first} {patent.examiner.examiner_name_last}',
            inventor_countries=patent.inventor_countries,
            inventor_count=len(patent.inventors),
            filing_year=patent.filing_year,
        ).model_dump()

    @classmethod
    def convert_document(cls, patent: BeanieDocument) -> LangChainDocument:
        """Convert a single PatentApplication to LangChain Document."""
        return LangChainDocument(
            page_content=cls._build_page_content(patent), metadata=cls._build_metadata(patent)
        )

    @classmethod
    async def batch_convert(
        cls, query: Any = None, batch_size: int = 1000, projection: dict[str, int] | None = None
    ) -> AsyncGenerator[list[LangChainDocument], None]:
        """
        Convert documents in batches for memory efficiency.

        Args:
            query: Beanie query object for filtering
            batch_size: Number of documents per batch
            projection: MongoDB projection to limit fields retrieved
        """
        query = query or PatentApplication.find_all()
        current_batch = []

        async for patent in query.project(projection_model=projection):
            current_batch.append(cls.convert_document(patent))
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if current_batch:
            yield current_batch
