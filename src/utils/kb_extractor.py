"""
Knowledge Graph Extraction Pipeline with spaCy and LLM.

This pipeline:
1. Extracts entities from text using spaCy with chunking for large documents
2. Uses an LLM to identify relationships between extracted entities
3. Stores entities and relationships in pandas DataFrames
4. Can be extended to generate Cypher or other outputs
"""

from functools import reduce
from typing import Literal

import dspy
import spacy


def extract_entities(
    text: str, model: str = 'en_core_web_lg', chunk_size: int = 100000, window: int = 200
) -> list[tuple]:  # TODO: Add Coreference
    """
    Extract entities from large text by splitting into chunks and avoiding boundary errors.

    Args:
        text: Input text to process
        model: spaCy model name (default: "en_core_web_lg")
        chunk_size: Max characters per chunk (default: 100,000)
        window: Context window to find sentence boundaries (default: 200)

    Returns:
        List of entity tuples: (text, label, start_char, end_char)
    """
    nlp = spacy.load(model)
    nlp.add_pipe('entity_ruler', after='ner', config={'overwrite_ents': True})
    # Short-circuit for small texts
    if len(text) <= chunk_size:
        doc = nlp(text)
        return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

    text_length = len(text)

    def process_chunk(acc, start_idx):
        entities, _ = acc
        end_idx = start_idx + chunk_size
        is_end_of_text = end_idx >= text_length

        if is_end_of_text:
            chunk = text[start_idx:]
        else:
            # Find the nearest sentence boundary to avoid splitting entities
            boundary = max(
                text.rfind('.', end_idx - window, end_idx),
                text.rfind('?', end_idx - window, end_idx),
                text.rfind('!', end_idx - window, end_idx),
                text.rfind('\n', end_idx - window, end_idx),
            )
            end_idx = boundary + 1 if boundary > 0 else end_idx
            chunk = text[start_idx:end_idx]

        doc = nlp(chunk)
        new_entities = [
            (ent.text, ent.label_, start_idx + ent.start_char, start_idx + ent.end_char)
            for ent in doc.ents
        ]
        return (entities + new_entities, end_idx)

    result, _ = reduce(
        process_chunk,
        range(0, text_length, chunk_size),
        ([], 0),  # Initial accumulator: (empty entities list, last end_idx)
    )
    return result


class EntityExtraction(dspy.Signature):
    """Extract all possible entities from a given text."""
    text: str = dspy.InputField(desc='Text to analyze for entities')
    entities: list[str] = dspy.OutputField(desc='List of string entities')


class RelationExtraction(dspy.Signature):
    """
    Identify relationships between entities in a given text.

    Instructions:
    - Only identify relationships between the provided entities
    - Use common relationship types like:  material_of, method_step, component_of, cites, improves, alternative_to, comprises, depends_on, etc.
    - For ambiguous relationships, choose the most specific type
    - Output JSON format: {"relations": [{"e_1": str, "rel": str, "e_2": str}]}
    """

    text: str = dspy.InputField(desc='Text to analyze for relationships')
    entities: list[str] = dspy.InputField(desc='List of string entities')
    relations: dict[
        Literal['relations'], list[dict[Literal['e_1'] | Literal['rel'] | Literal['e_2'], str]]
    ] = dspy.OutputField(desc='Relationships in JSON format')


class KGBuilder(dspy.Module):
    """Builds knowledge graphs from text using entity and relation extraction."""

    def __init__(self):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(EntityExtraction)
        self.extract_relations = dspy.ChainOfThought(RelationExtraction)

    def __call__(self, text: str) ->  dict[
        Literal['relations'], list[dict[Literal['e_1'] | Literal['rel'] | Literal['e_2'], str]]
    ]:
        """
        Extract relationships between entities using LLM.

        Args:
            text: Original text containing entities
            entities: List of entity tuples from spaCy

        Returns:
            Dictionary of parsed relationships
        """
        entities = self.extract_entities(text=text)
        return self.extract_relations(text=text, entities=entities)
