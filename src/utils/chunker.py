import ast
import re

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import (
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
)
from pydantic import BaseModel, Field


class PatentChunkerConfig(BaseModel):
    """Configuration for patent document chunking."""

    chunk_size: int = Field(
        default=1000, description='Target size of each chunk (in characters or tokens)'
    )
    chunk_overlap: int = Field(default=200, description='Overlap between chunks')
    splitter_type: str = Field(
        default='spacy',
        description='Type of splitter to use: spacy, sentence_transformers, nltk, or recursive',
    )
    model_name: str = Field(
        default='en_core_web_sm', description='Model name for spaCy or sentence transformers'
    )
    split_claims: bool = Field(default=True, description='Whether to split claims individually')
    preserve_claim_numbers: bool = Field(
        default=True, description='Keep claim numbers when splitting claims'
    )


class PatentChunker:
    """Chunker specialized for patent documents with JSON-like content structure."""

    def __init__(self, config: PatentChunkerConfig):
        self.config = config
        self.claim_delimiter = re.compile(r'(\d+\.\s)')  # Regex to capture claim numbers

        if config.splitter_type == 'spacy':
            self.splitter = SpacyTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                pipeline=config.model_name,
            )
        elif config.splitter_type == 'sentence_transformers':
            self.splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=config.chunk_overlap,
                model_name=config.model_name,
                tokens_per_chunk=config.chunk_size,
            )
        elif config.splitter_type == 'nltk':
            self.splitter = NLTKTextSplitter(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )
        else:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )

    def _chunk_claims(self, claims_text: str) -> list[str]:
        """Split claims section while preserving claim numbers."""
        if not self.config.split_claims:
            return [claims_text]

        parts = self.claim_delimiter.split(claims_text)
        if len(parts) < 2:
            return [claims_text]

        # Skip initial empty part if exists
        start_index = 1 if not parts[0] else 0
        chunks = []

        # Recombine claim numbers with their content
        for i in range(start_index, len(parts), 2):
            if i + 1 < len(parts) and (chunk := (parts[i] + parts[i + 1]).strip()):
                chunks.append(chunk)
            elif parts[i].strip():
                chunks.append(parts[i].strip())

        return chunks

    def _chunk_section(self, section_name: str, section_text: str) -> list[str]:
        """Applies appropriate chunking strategy based on section type."""
        if section_name == 'claims':
            return self._chunk_claims(section_text)
        return self.splitter.split_text(section_text)

    def chunk_document(self, document: LangChainDocument) -> list[LangChainDocument]:
        """Splits a patent document into chunks preserving structure."""
        try:
            # Convert document content back to structured format
            content = (
                ast.literal_eval(document.page_content)
                if isinstance(document.page_content, str)
                else document.page_content
            )
        except:
            # Fallback to text splitting if parsing fails
            return self.splitter.split_documents([document])

        chunks = []
        metadata = document.metadata.copy()

        # Process each section independently
        for section_name, section_text in content.items():
            if not section_text:
                continue

            section_chunks = self._chunk_section(section_name, section_text)

            for i, chunk in enumerate(section_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'section': section_name,
                    'chunk_index': i,
                    'total_chunks': len(section_chunks),
                })

                if (
                    section_name == 'claims'
                    and self.config.preserve_claim_numbers
                    and (match := re.match(r'^(\d+)\.', chunk))
                ):
                    chunk_metadata['claim_number'] = int(match.group(1))

                chunks.append(LangChainDocument(page_content=chunk, metadata=chunk_metadata))

        return chunks

    def chunk_documents(self, documents: list[LangChainDocument]) -> list[LangChainDocument]:
        """Batch process multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
