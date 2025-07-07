from functools import reduce

import spacy
from loguru import logger


def extract_entities(
    text: str,
    model: str = 'en_core_web_trf',  # 'en_core_sci_lg'
    chunk_size: int = 100000,
    window: int = 200,
) -> list[tuple[str, str, int, int]]:
    """
    Extract entities from large text with cross-chunk coreference resolution.

    Args:
        text: Input text to process
        model: spaCy model name (default: "en_core_web_trf")
        chunk_size: Max characters per chunk (default: 100,000)
        window: Context window for sentence boundaries (default: 200)

    Returns:
        List of entity tuples: (text, label, start_char, end_char)
    """
    nlp = spacy.load(model)

    # Add coreference resolution components
    if 'experimental_coref' not in nlp.pipe_names:
        nlp.add_pipe('experimental_coref')
    if 'experimental_span_resolver' not in nlp.pipe_names:
        nlp.add_pipe('experimental_span_resolver')

    # Process full document for coreference resolution
    logger.info('Performing full-document coreference resolution...')
    full_doc = nlp(text)

    # Extract coreference clusters from span groups
    coref_chains = {}
    prefix = "coref_clusters"
    for key, span_group in full_doc.spans.items():
        if key.startswith(f"{prefix}_"):
            try:
                chain_id = int(key.split('_')[-1])
            except ValueError:
                continue

            mentions = [(
                    span.start_char,
                    span.end_char,
                    span.text
                ) for span in span_group]

            if mentions:
                coref_chains[chain_id] = {
                    'main_mention': 0,  # Use first mention as main
                    'mentions': mentions
                }

    logger.info(f'Found {len(coref_chains)} coreference clusters')

    # Short-circuit for small texts
    if len(text) <= chunk_size:
        return _process_chunk_entities(nlp, text, 0, coref_chains, set(), {})

    # Prepare for chunk processing
    text_length = len(text)
    processed_mentions = set()
    chain_entities = {}  # Track entity labels for coreference chains

    def process_chunk(
        acc: tuple[list[tuple[str, str, int, int]], int], start_idx: int
    ) -> tuple[list[tuple[str, str, int, int]], int]:
        entities, last_end = acc
        end_idx = start_idx + chunk_size
        is_end_of_text = end_idx >= text_length

        if is_end_of_text:
            chunk = text[start_idx:]
        else:
            # Find nearest sentence boundary
            boundary = max(
                text.rfind('.', end_idx - window, end_idx),
                text.rfind('?', end_idx - window, end_idx),
                text.rfind('!', end_idx - window, end_idx),
                text.rfind('\n\n', end_idx - window, end_idx),  # Section breaks
            )
            end_idx = boundary + 1 if boundary > 0 else end_idx
            chunk = text[start_idx:end_idx]

        # Process chunk
        chunk_entities = _process_chunk_entities(
            nlp, chunk, start_idx, coref_chains, processed_mentions, chain_entities
        )
        return (entities + chunk_entities, end_idx)

    # Process all chunks
    result, _ = reduce(
        process_chunk,
        range(0, text_length, chunk_size),
        ([], 0),  # (entities, last_end_idx)
    )
    return result


def _process_chunk_entities(
    nlp: spacy.Language,
    chunk: str,
    offset: int,
    coref_chains: dict,
    processed_mentions: set,
    chain_entities: dict,
) -> list[tuple[str, str, int, int]]:
    """
    Process entities in a chunk with coreference awareness.

    Args:
        nlp: spaCy language model
        chunk: Text chunk to process
        offset: Character offset in original text
        coref_chains: Pre-resolved coreference chains
        processed_mentions: Set of processed mentions
        chain_entities: Entity labels for chains

    Returns:
        List of entity tuples
    """
    doc = nlp(chunk)
    entities = []
    local_mentions = set()

    # Process coreference chains
    if coref_chains:
        for chain_id, chain_info in coref_chains.items():
            for _mention_idx, mention in enumerate(chain_info['mentions']):
                start, end, mention_text = mention
                # Adjust indices for chunk boundaries
                if not (offset <= start < offset + len(chunk)):
                    continue

                # Calculate relative position in chunk
                rel_start = start - offset
                rel_end = end - offset
                span = doc.char_span(rel_start, rel_end, alignment_mode='expand')

                if not span:
                    continue

                # Create mention identifier
                mention_id = (start, end)

                # Skip already processed mentions
                if mention_id in processed_mentions:
                    continue

                # Get or determine entity label
                if chain_id in chain_entities:
                    label = chain_entities[chain_id]
                elif span.ents:
                    label = span.ents[0].label_
                    chain_entities[chain_id] = label
                else:
                    # Try to get label from main mention
                    main_mention = chain_info['mentions'][chain_info['main_mention']]
                    main_start, main_end, _ = main_mention
                    if offset <= main_start < offset + len(chunk):
                        main_span = doc.char_span(
                            main_start - offset, main_end - offset, alignment_mode='expand'
                        )
                        if main_span and main_span.ents:
                            label = main_span.ents[0].label_
                            chain_entities[chain_id] = label
                        else:
                            label = 'CORE'
                    else:
                        label = 'CORE'

                # Add entity
                entities.append((mention_text, label, start, end))
                processed_mentions.add(mention_id)
                local_mentions.add(mention_id)

    # Process regular entities not in coref chains
    for ent in doc.ents:
        start = offset + ent.start_char
        end = offset + ent.end_char
        ent_id = (start, end)

        if ent_id not in processed_mentions:
            entities.append((ent.text, ent.label_, start, end))
            processed_mentions.add(ent_id)
            local_mentions.add(ent_id)

    return entities
