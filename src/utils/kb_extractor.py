"""
Knowledge Graph Extraction Pipeline with spaCy and LLM.

This pipeline:
1. Extracts entities from text using spaCy with chunking for large documents
2. Uses an LLM to identify relationships between extracted entities
3. Stores entities and relationships in pandas DataFrames
4. Can be extended to generate Cypher or other outputs
"""

import json
from functools import reduce

import dspy
import pandas as pd
import spacy
from loguru import logger


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
        ([], 0)  # Initial accumulator: (empty entities list, last end_idx)
    )
    return result


class RelationExtraction(dspy.Signature):
    """
    Identify relationships between entities in a given text.

    Instructions:
    - Only identify relationships between the provided entities
    - Use common relationship types like:  material_of, method_step, component_of, cites, improves, alternative_to, comprises, depends_on, etc.
    - For ambiguous relationships, choose the most specific type
    - Output JSON format: {"relations": [{"entity1": str, "relation": str, "entity2": str}]}
    """

    text = dspy.InputField(desc='Text to analyze for relationships')
    entities = dspy.InputField(desc='List of entities in JSON format')
    relations = dspy.OutputField(desc='Relationships in JSON format', prefix='```json\n')


class RelationExtractor(dspy.Module):
    """LLM-based relation extractor between entities."""

    def __init__(self):
        super().__init__()
        self.extract_relations = dspy.ChainOfThought(RelationExtraction)

    def forward(self, text: str, entities: list[tuple]) -> dict:
        """
        Extract relationships between entities using LLM.

        Args:
            text: Original text containing entities
            entities: List of entity tuples from spaCy

        Returns:
            Dictionary of parsed relationships
        """
        entity_list = [{'text': e[0], 'label': e[1]} for e in entities]
        entity_json = json.dumps(entity_list)

        result = self.extract_relations(text=text, entities=entity_json)

        try:
            relations_json = result.relations.strip().replace('```json', '').strip('`', '')
            return json.loads(relations_json)
        except json.JSONDecodeError:
            return {'relations': []}


class KnowledgeGraphBuilder:
    """Builds knowledge graphs from text using entity and relation extraction."""

    def __init__(self, spacy_model: str = 'en_core_web_lg'):
        """
        Initialize the knowledge graph builder.

        Args:
            spacy_model: spaCy model to use for entity extraction
        """
        self.spacy_model = spacy_model
        self.relation_extractor = RelationExtractor()

    def build_knowledge_graph(self, text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process text to extract entities and relations.

        Args:
            text: Input text to process

        Returns:
            Tuple of DataFrames: (entities_df, relations_df)
        """
        entities = extract_entities(text, model=self.spacy_model)
        entities_df = pd.DataFrame(entities, columns=['text', 'label', 'start_char', 'end_char'])

        if not entities:
            # Return empty relations if no entities are found
            relations_df = pd.DataFrame(columns=['entity1', 'relation', 'entity2'])
            return entities_df, relations_df

        if (relations := self.relation_extractor(text, entities).get('relations', [])):
            relations_df = pd.DataFrame(relations)
            required_cols = ['entity1', 'relation', 'entity2']
            for col in required_cols:
                if col not in relations_df.columns:
                    relations_df[col] = None
            relations_df = relations_df[required_cols]
            return entities_df, relations_df
        # Return empty relations if no relations are found
        relations_df = pd.DataFrame(columns=required_cols)
        return entities_df, relations_df


if __name__ == '__main__':
    kg_builder = KnowledgeGraphBuilder()

    medical_text = (
        'Patient John Smith, age 45, presented with hypertension and diabetes. '
        'Dr. Emily Johnson prescribed Lisinopril 10mg daily and Metformin 500mg twice daily. '
        'John works at Microsoft in Redmond, Washington.'
    )

    entities_df, relations_df = kg_builder.build_knowledge_graph(medical_text)

    logger.info(f'\nEntities DataFrame:\n{entities_df}')
    logger.info(f'\nRelations DataFrame:\n{relations_df}')

    tech_text = (
        'Patent US1234567 describes a quantum computing method using superconducting qubits. '
        'The invention was created by Dr. Alan Turing at MIT in 2023. '
        'The system operates at temperatures below 50mK using liquid helium cooling.'
    )

    entities_df, relations_df = kg_builder.build_knowledge_graph(tech_text)

    logger.info(f'\nTechnology Entities:\n{entities_df}')
    logger.info(f'\nTechnology Relations:\n{relations_df}')
