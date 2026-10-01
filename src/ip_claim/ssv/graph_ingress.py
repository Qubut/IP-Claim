"""Occupancy overlay from scored terms onto the host token grid.

Termhood artefacts and JATE draws live in the ATE package. This module
maps those spans onto assignment tensors. Callers keep the raw host
string for MLM.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import cache
from itertools import chain, groupby, starmap
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple, Never, Protocol, cast

import polars as pl
import pyarrow.parquet as pq
import spacy
import torch
import torch.nn.functional as F
from patent_ate.nlp import JateDraw, TermSpan
from patent_ate.spec import AteSpec
from patent_ate.termhood import (
    TermhoodIndex,
    TermhoodStore,
    TermhoodTable,
    is_stop_surface,
    normalize_surface,
)
from pydantic import BaseModel, ConfigDict
from returns.converters import result_to_maybe
from returns.maybe import Maybe, Nothing
from returns.pipeline import flow
from returns.pointfree import lash
from returns.result import safe
from spacy.language import Language
from spacy.tokens import Doc, Span
from thinc.api import set_gpu_allocator
from thinc.util import require_gpu
from torch import Tensor, nn
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_PAD = TermSpan(lemma_key='', start_char=0, end_char=0, token_length=1)
_NOUN_CHUNK_ATTR = 'cached_noun_chunks'
_HIDDEN_CUDA = frozenset({'void', '-1'})


def occupy_eligible_spans(spans: Sequence[TermSpan]) -> tuple[TermSpan, ...]:
    """Keep noun-group spans that are not published stop surfaces."""
    return tuple(
        span
        for span in spans
        if span.lemma_key and not is_stop_surface(span.lemma_key)
    )


def gpu_job_refuses_spacy_fork() -> bool:
    """True when spaCy pipe must stay in-process.

    A listed ``CUDA_VISIBLE_DEVICES`` value other than the hide sentinels
    ``void`` and ``-1`` marks a GPU job. Fixtures and ``from_pretrained``
    enter the CUDA driver through ``is_available`` and ``device_count``
    without leaving ``is_initialized`` true. This predicate does not call
    either. A live CUDA context is the late path after a tensor exists.
    """
    raw = os.environ.get('CUDA_VISIBLE_DEVICES')
    tokens = () if raw is None else tuple(part.strip() for part in raw.split(',') if part.strip())
    advertised = bool(tokens) and not _HIDDEN_CUDA.issuperset(tokens)
    return advertised or torch.cuda.is_initialized()


if not Doc.has_extension(_NOUN_CHUNK_ATTR):
    Doc.set_extension(_NOUN_CHUNK_ATTR, default=None)


@Language.component('cache_noun_chunks')
def cache_noun_chunks(doc: Doc) -> Doc:
    """Store noun-chunk token and character spans on the doc in the spaCy worker."""
    doc._.cached_noun_chunks = tuple(
        (
            int(chunk.start),
            int(chunk.end),
            int(chunk.start_char),
            int(chunk.end_char),
            str(chunk.text),
        )
        for chunk in doc.noun_chunks
    )
    return doc


class ChunkView(NamedTuple):
    """Noun-chunk token and character spans materialized in the spaCy worker."""

    start: int
    end: int
    start_char: int
    end_char: int
    text: str


class CachedNounChunks:
    """Doc proxy that reuses worker-cached noun chunks; never walks the tree."""

    __slots__ = ('_chunks', '_doc')
    _chunks: tuple[ChunkView, ...]
    _doc: Doc

    def __init__(self, doc: Doc) -> None:
        packed = getattr(getattr(doc, '_', None), _NOUN_CHUNK_ATTR, None)
        if packed is None:
            raise RuntimeError('noun chunks were not materialized in the spaCy worker pipe')
        self._doc = doc
        rows = cast(Iterable[tuple[int, int, int, int, str]], packed)
        self._chunks = tuple(starmap(ChunkView, rows))

    def __getattr__(self, name: str) -> object:
        return getattr(self._doc, name)

    @property
    def noun_chunks(self) -> tuple[ChunkView, ...]:
        return self._chunks

    @property
    def sents(self) -> Iterator[Span]:
        return self._doc.sents


@cache
def host_language(model: str, *, gpu: bool = False) -> Language:
    """Return the spaCy Language for ``model``, with NER disabled.

    ``gpu=True`` sets the PyTorch GPU allocator and calls ``require_gpu``
    before load so transformer and Thinc ops share the visible card. On CPU,
    host threads are pinned to one before load so forked spaCy workers do not
    oversubscribe the box. Noun-chunk offsets are materialized in-pipe so
    worker processes send packed spans, not a tree for the driver to walk.
    """
    if gpu:
        set_gpu_allocator('pytorch')
        require_gpu()
    else:
        torch.set_num_threads(1)
    loaded = spacy.load(model, disable=['ner'])
    if 'cache_noun_chunks' not in loaded.pipe_names:
        loaded.add_pipe('cache_noun_chunks', last=True)
    return loaded


class OccupyScores(Protocol):
    """Lookup occupy scores for requested lemma keys without a corpus scan."""

    def scores_for(self, keys: Sequence[str]) -> dict[str, float]:
        """Return occupy scores for ``keys`` only."""
        ...

    def score(self, key: str) -> float:
        """Return the occupy score for a single ``key``."""
        ...


def termhood_row_group_index(store: TermhoodStore) -> pl.DataFrame:
    """Sorted key and row-group map for one committed fact table."""
    sidecar = Path(f'{store.parquet}.rowgroup')
    if sidecar.is_file():
        existing = pl.read_parquet(sidecar)
        if existing.columns == ['key', 'row_group'] and existing.height == store.meta.n_keys:
            return existing
    meta = pq.ParquetFile(store.parquet).metadata
    n_groups = meta.num_row_groups
    sizes = tuple(meta.row_group(index).num_rows for index in range(n_groups))
    groups = tuple(chain.from_iterable((index,) * size for index, size in enumerate(sizes)))
    frame = (
        pl
        .read_parquet(store.parquet, columns=['key'])
        .with_columns(pl.Series('row_group', groups, dtype=pl.Int32))
        .lazy()
        .sort('key')
        .collect(engine='streaming')
    )
    partial = Path(f'{sidecar}.partial')
    frame.write_parquet(partial)
    partial.replace(sidecar)
    return frame


class OccupancyMap(BaseModel):
    """Assignment and termhood weights aligned to the host sequence."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    assignment: Tensor
    weights: Tensor
    labels: tuple[tuple[str, ...], ...]


class SpanTable(BaseModel):
    """Padded term rows for one batch, aligned to host tokens."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    starts: Tensor
    ends: Tensor
    lengths: Tensor
    scores: Tensor
    keys: tuple[tuple[str, ...], ...]

    @classmethod
    def from_spans(
        cls,
        docs: Sequence[Sequence[TermSpan]],
        termhood: OccupyScores,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpanTable:
        """Build a padded span table scored by ``termhood``."""
        width = max((len(doc) for doc in docs), default=0)
        padded = tuple(tuple(doc) + (_PAD,) * (width - len(doc)) for doc in docs)
        by_key = termhood.scores_for(
            tuple(dict.fromkeys(span.lemma_key for doc in padded for span in doc))
        )
        return cls(
            starts=torch.tensor(
                [[span.start_char for span in doc] for doc in padded],
                device=device,
            ),
            ends=torch.tensor(
                [[span.end_char for span in doc] for doc in padded],
                device=device,
            ),
            lengths=torch.tensor(
                [[span.token_length for span in doc] for doc in padded],
                device=device,
                dtype=dtype,
            ),
            scores=torch.tensor(
                [
                    [
                        0.0
                        if is_stop_surface(span.lemma_key)
                        else float(by_key.get(span.lemma_key, 0.0))
                        for span in doc
                    ]
                    for doc in padded
                ],
                device=device,
                dtype=dtype,
            ),
            keys=tuple(tuple(span.lemma_key for span in doc) for doc in padded),
        )

    def overlap(self, offsets: Tensor, live_mask: Tensor) -> Tensor:
        """Return whether each span interval overlaps a live host token."""
        tok_start = offsets[..., 0]
        tok_end = offsets[..., 1]
        live = live_mask.bool() & tok_end.gt(0)
        return (
            tok_start.unsqueeze(1).lt(self.ends.unsqueeze(2))
            & tok_end.unsqueeze(1).gt(self.starts.unsqueeze(2))
            & live.unsqueeze(1)
        )

    def occupy(self, assignment: Tensor, live_mask: Tensor, offsets: Tensor) -> OccupancyMap:
        """Place each live token on its longest scored span and pool assignment."""
        hit = self.overlap(offsets, live_mask)
        token_width = int(assignment.size(1))
        rank = torch.arange(1, token_width + 1, device=assignment.device)
        heads = (hit.to(dtype=torch.long) * rank).amax(dim=-1) - 1
        valid = self.scores.gt(0) & hit.any(dim=-1) & heads.ge(0)
        safe = heads.clamp(min=0)
        best = self.lengths.new_zeros(int(heads.size(0)), token_width).scatter_reduce(
            1, safe, self.lengths.where(valid, 0), reduce='amax'
        )
        keep = valid & self.lengths.eq(best.gather(1, safe))
        span_rank = torch.arange(heads.size(1), device=heads.device).expand_as(heads)
        vacant = span_rank.new_full(span_rank.shape, int(heads.size(1)))
        first = vacant.new_full((int(heads.size(0)), token_width), int(heads.size(1)))
        first = first.scatter_reduce(1, safe, torch.where(keep, span_rank, vacant), reduce='amin')
        keep &= span_rank.eq(first.gather(1, safe))
        place = F.one_hot(safe, token_width).to(dtype=assignment.dtype) * keep.unsqueeze(-1)
        mass = hit.to(dtype=assignment.dtype) * keep.unsqueeze(-1)
        span_mean = torch.matmul(mass, assignment) / mass.sum(dim=-1).clamp_min(1).unsqueeze(-1)
        used = place.gt(0).any(dim=1)
        placed = {
            (row, int(heads[row, span])): self.keys[row][span]
            for row, doc in enumerate(self.keys)
            for span, _key in enumerate(doc)
            if bool(keep[row, span])
        }
        return OccupancyMap(
            assignment=torch.where(
                used.unsqueeze(-1),
                torch.einsum('bst,bsk->btk', place, span_mean),
                assignment.new_zeros(assignment.shape),
            ),
            weights=torch.einsum('bst,bs->bt', place, self.scores)
            * live_mask.to(dtype=assignment.dtype),
            labels=tuple(
                tuple(placed.get((row, token), '') for token in range(token_width))
                for row in range(int(assignment.size(0)))
            ),
        )


class GraphIngress(nn.Module):
    """Builds occupancy maps from host assignments and claim text."""

    def __init__(
        self,
        tokenizer_id: str | None = None,
        *,
        spacy_model: str | None = None,
    ) -> None:
        super().__init__()
        self.termhood: OccupyScores = TermhoodTable()
        self._tokenizer_id = tokenizer_id
        self._spacy_model = spacy_model or AteSpec().spacy_model
        self._offset_tok: PreTrainedTokenizerBase | None = None
        self._inference_spans: dict[str, tuple[TermSpan, ...]] = {}
        self._parallel_language: Language | None = None

    def get_extra_state(self) -> dict[str, object]:
        """Return a small termhood snapshot for the checkpoint."""
        source = getattr(self.termhood, 'source', None)
        if source is not None:
            return {'termhood_root': str(source)}
        if isinstance(self.termhood, TermhoodTable):
            return self.termhood.dump()
        return {}

    def set_extra_state(self, state: object) -> None:
        """Restore termhood from a checkpoint extra state or artifact root."""
        if not isinstance(state, Mapping):
            return
        root = state.get('termhood_root') or state.get('termhood_parquet')
        if isinstance(root, str) and root:
            source = getattr(self.termhood, 'source', None)
            if source is not None and Path(str(source)) == Path(root):
                return
            self.termhood = TermhoodIndex.from_store(TermhoodStore.open(Path(root)))
            return
        self.termhood = TermhoodTable.load(state)

    def nlp(self, *, gpu: bool = False) -> Language:
        """Return the spaCy Language used for term extraction."""
        return host_language(self._spacy_model, gpu=gpu)

    def parallel_language(self) -> Language:
        """Return a Language whose pipe fans spaCy across CPU workers.

        The pipe is built once per ingress and cached, so the spaCy worker pool
        and model load are paid a single time instead of per ``candidates`` call.
        Worker count follows the host core count on a CPU job. A GPU job stays
        at one process so spaCy never forks this process. Noun-chunk offsets
        come from the spaCy workers; a missing cache raises rather than walking
        the tree here.
        """
        if self._parallel_language is not None:
            return self._parallel_language
        host = self.nlp()
        spec = AteSpec()
        cores = os.cpu_count() or 1

        class ParallelPipe:
            def pipe(self, stream: Iterable[str], **kwargs: object) -> Iterable[CachedNounChunks]:
                del kwargs
                listed = tuple(stream)
                count = max(1, len(listed))
                workers = 1 if gpu_job_refuses_spacy_fork() else max(1, min(count, cores))
                width = max(
                    1,
                    min(count, spec.pipe_docs, (count + workers - 1) // workers),
                )
                parsed = tuple(host.pipe(listed, n_process=workers, batch_size=width))
                return tuple(map(CachedNounChunks, parsed))

        self._parallel_language = cast(Language, ParallelPipe())
        return self._parallel_language

    def offset_tokenizer(
        self,
        fallback: PreTrainedTokenizerBase | None = None,
    ) -> PreTrainedTokenizerBase:
        """Return a tokenizer that can emit character offsets for host ids."""

        @safe(exceptions=(OSError, TypeError, ValueError))
        def load_pretrained(tokenizer_id: str) -> PreTrainedTokenizerBase:
            return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)

        def load_fast(tokenizer_id: str | None) -> Maybe[PreTrainedTokenizerBase]:
            def from_id(tid: str) -> Maybe[PreTrainedTokenizerBase]:
                return result_to_maybe(load_pretrained(tid))

            return Maybe.from_optional(tokenizer_id).bind(from_id)

        def missing() -> Never:
            msg = 'offset mapping needs a host tokenizer id or a fallback tokenizer'
            raise RuntimeError(msg)

        path = self._tokenizer_id or getattr(fallback, 'name_or_path', None)
        loaded = flow(
            Maybe.from_optional(self._offset_tok),
            lash(lambda _: load_fast(str(path) if path else None)),
            lash(lambda _: Maybe.from_optional(fallback)),
        ).or_else_call(missing)
        self._offset_tok = loaded
        return loaded

    def token_offsets(
        self,
        texts: Sequence[str],
        *,
        max_length: int,
        device: torch.device,
        input_ids: Tensor,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Tensor:
        """Return character start and end for each host token in ``input_ids``.

        Fast HuggingFace offset mapping must match ``input_ids``. Missing or
        mismatched fast offsets raise.
        """

        @safe(exceptions=(TypeError, ValueError, NotImplementedError))
        def encode_offsets(encoder: PreTrainedTokenizerBase) -> Mapping[str, object]:
            return encoder(
                list(texts),
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt',
            )

        def mapping_if_matched(encoded: Mapping[str, object]) -> Tensor | None:
            ids = encoded['input_ids']
            mapping = encoded['offset_mapping']
            if not isinstance(ids, Tensor) or not isinstance(mapping, Tensor):
                return None
            matched = ids.shape == input_ids.shape and torch.equal(ids, input_ids.detach().cpu())
            return mapping.to(device=device) if matched else None

        def hf_map(encoder: PreTrainedTokenizerBase) -> Maybe[Tensor]:
            if not getattr(encoder, 'is_fast', False):
                return Nothing
            return result_to_maybe(encode_offsets(encoder)).bind_optional(mapping_if_matched)

        def missing_map() -> Never:
            msg = 'host token ids have no matching fast offset map; occupy will not spaCy-align'
            raise RuntimeError(msg)

        return flow(
            hf_map(tokenizer),
            lash(lambda _: hf_map(self.offset_tokenizer(tokenizer))),
        ).or_else_call(missing_map)

    def missing_inference_texts(self, texts: Sequence[str]) -> tuple[str, ...]:
        """Return host strings that still need a JATE draw."""
        return tuple(text for text in dict.fromkeys(texts) if text not in self._inference_spans)

    def candidates(self, texts: Sequence[str]) -> tuple[tuple[TermSpan, ...], ...]:
        """Return term spans for each row of ``texts`` and remember them per string."""
        if not texts:
            return ()
        needed = self.missing_inference_texts(texts)
        if needed:
            drawn = JateDraw.from_texts(needed, self.parallel_language()).docs
            self._inference_spans.update({
                text: tuple(doc) for text, doc in zip(needed, drawn, strict=True)
            })
        return tuple(
            occupy_eligible_spans(self._inference_spans[text]) for text in texts
        )

    def remember_inference_spans(self, spans: Mapping[str, Sequence[TermSpan]]) -> None:
        """Store term spans keyed by host string without a spaCy redraw."""
        self._inference_spans.update({text: tuple(doc) for text, doc in spans.items()})

    def discard_inference_spans(self, texts: Sequence[str] | None = None) -> None:
        """Drop remembered JATE spans so a later worker copy stays small."""
        if texts is None:
            self._inference_spans.clear()
            return
        _ = tuple(self._inference_spans.pop(text, None) for text in dict.fromkeys(texts))

    def occupy(
        self,
        assignment: Tensor,
        live_mask: Tensor,
        texts: Sequence[str],
        offsets: Tensor,
        *,
        update_stats: bool = True,
        spans: Sequence[Sequence[TermSpan]] | None = None,
    ) -> OccupancyMap:
        """Return occupancy pooled onto scored term heads.

        Missing spans come from the remembered candidate table. A cache miss
        draws JATE once through a multi-worker spaCy pipe and, when the
        termhood object is a mutable table, merges that same draw. Remembered
        strings skip the pipe.
        """
        rows = int(assignment.size(0))
        if spans is not None:
            self.remember_inference_spans({
                text: tuple(doc) for text, doc in zip(texts, spans, strict=True)
            })
        needed = self.missing_inference_texts(texts)
        if needed:
            drawn = JateDraw.from_texts(needed, self.parallel_language())
            self._inference_spans.update({
                text: tuple(doc) for text, doc in zip(needed, drawn.docs, strict=True)
            })
            if update_stats and isinstance(self.termhood, TermhoodTable):
                self.termhood = self.termhood.merged(drawn.termhood)
        held = tuple(self._inference_spans.get(text, ()) for text in texts)
        per_doc: tuple[tuple[TermSpan, ...], ...] = tuple(
            occupy_eligible_spans(doc) for doc in held
        )
        docs = tuple(per_doc[row] if row < len(per_doc) else () for row in range(rows))
        return (
            empty_occupancy(assignment)
            if not any(docs)
            else SpanTable.from_spans(
                docs,
                self.termhood,
                device=assignment.device,
                dtype=assignment.dtype,
            ).occupy(assignment, live_mask, offsets)
        )


def spans_from_extract(
    texts: Sequence[str],
    term_rows: Sequence[Sequence[Mapping[str, object]]],
) -> tuple[tuple[TermSpan, ...], ...]:
    """Locate extract surfaces on the host character grid. No spaCy."""
    if len(texts) != len(term_rows):
        raise ValueError('extract term rows must align with host texts')
    if not texts:
        return ()
    exploded = (
        pl
        .DataFrame({
            'row': list(range(len(texts))),
            'text': list(texts),
            'terms': list(term_rows),
        })
        .explode('terms')
        .filter(pl.col('terms').is_not_null())
        .unnest('terms')
        .explode('surfaces')
        .rename({'surfaces': 'surface'})
        .filter(
            pl.col('surface').is_not_null()
            & pl.col('surface').cast(pl.String).str.len_chars().gt(0)
            & pl.col('term').is_not_null()
        )
    )
    if exploded.is_empty():
        return tuple(() for _ in texts)
    located = (
        exploded
        .group_by('row', maintain_order=True)
        .agg(pl.col('text').first(), pl.col('surface').alias('patterns'))
        .with_columns(
            starts=pl.col('text').str.find_many(pl.col('patterns'), overlapping=True),
            hits=pl.col('text').str.extract_many(pl.col('patterns'), overlapping=True),
        )
        .explode('starts', 'hits')
    )
    pairs = exploded.select('row', pl.col('surface').alias('hits'), 'term').unique(
        subset=['row', 'hits'],
        keep='first',
    )
    records = (
        located
        .join(pairs, on=['row', 'hits'])
        .select(
            'row',
            'text',
            'starts',
            'hits',
            'term',
        )
        .to_dicts()
    )

    def as_span(row: Mapping[str, object]) -> TermSpan | None:
        lemma = normalize_surface(str(row['term']))
        if not lemma or is_stop_surface(lemma):
            return None
        text = str(row['text'])
        start_obj = row['starts']
        byte_start = start_obj if isinstance(start_obj, int) else int(str(start_obj))
        start = len(text.encode('utf-8')[:byte_start].decode('utf-8'))
        hit = str(row['hits'])
        return TermSpan(
            lemma_key=lemma,
            start_char=start,
            end_char=start + len(hit),
            token_length=max(1, len(lemma.split())),
        )

    keyed = itemgetter('row')
    grouped = {
        int(row_i): tuple(span for rec in group if (span := as_span(rec)) is not None)
        for row_i, group in groupby(sorted(records, key=keyed), key=keyed)
    }
    return tuple(grouped.get(index, ()) for index in range(len(texts)))


def empty_occupancy(assignment: Tensor) -> OccupancyMap:
    """Return zero assignment and zero occupancy weights on the host grid."""
    width = int(assignment.size(1))
    rows = int(assignment.size(0))
    return OccupancyMap(
        assignment=assignment.new_zeros(assignment.shape),
        weights=assignment.new_zeros(assignment.shape[:-1]),
        labels=tuple(tuple('' for _ in range(width)) for _ in range(rows)),
    )


__all__ = [
    'GraphIngress',
    'OccupancyMap',
    'OccupyScores',
    'SpanTable',
    'empty_occupancy',
    'host_language',
    'occupy_eligible_spans',
    'spans_from_extract',
]
