"""Fixture occupy: content heads on X; stop surfaces and X-only content stay off A.

Occupy heads are termhood-scored noun-group spans after published stop
surfaces. Vacant rows write zeros. Inventory occupies when the trunk owns
occupy. Attach and the extract probe drop published stop keys.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import torch
from experiments.covering_objective.corpus import bind_termhood_attach
from patent_ate.extract import write_termhood
from patent_ate.nlp import TermSpan
from patent_ate.termhood import TermhoodIndex, TermhoodStore, TermhoodTable, is_stop_surface
from tests._ssv_fixtures import SSV_HUPD_FIXTURES, ssv_tiny_vocab_config

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ssv.config import ArchSpec, SsvTrainConfig
from ip_claim.ssv.graph_ingress import (
    GraphIngress,
    OccupancyMap,
    empty_occupancy,
    occupy_eligible_spans,
    spans_from_extract,
)
from ip_claim.ssv.ingress_probe import occupy_extract_parts, occupy_score_frame
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.model import SoftTrunkModel
from ip_claim.ssv.soft_vocab import SoftVocabModule

_X_CONTENT = 'photodiode latchbolt'
_X_TEXT = 'A photodiode latchbolt assembly measures glucose.'
_A_TEXT = 'A method comprising a housing, wherein the apparatus is an apparatus;'
_STOP_KEYS = ('comprising', 'wherein', ';')
_BANK = 8
_TOKENS = 8


def _span(key: str, start: int, end: int, length: int) -> TermSpan:
    return TermSpan(lemma_key=key, start_char=start, end_char=end, token_length=length)


def _content_span() -> TermSpan:
    start = _X_TEXT.find(_X_CONTENT)
    return _span(_X_CONTENT, start, start + len(_X_CONTENT), 2)


def _stop_spans(text: str) -> tuple[TermSpan, ...]:
    return tuple(
        _span(key, start, start + len(key), 1)
        for key in _STOP_KEYS
        if (start := text.find(key)) >= 0
    )


def _live_labels(occupied: OccupancyMap) -> frozenset[str]:
    return frozenset(label for row in occupied.labels for label in row if label)


def _termhood() -> TermhoodTable:
    return TermhoodTable(
        c_values={
            _X_CONTENT: 4.0,
            'comprising': 9.0,
            'wherein': 9.0,
            ';': 9.0,
        },
        document_frequency={
            _X_CONTENT: 1,
            'comprising': 2,
            'wherein': 2,
            ';': 2,
        },
        total_docs=10,
    )


def _offsets(width: int, *, content_end: int) -> torch.Tensor:
    steps = torch.linspace(0, content_end, width, dtype=torch.long)
    ends = torch.cat((steps[1:], steps[-1:] + 1))
    return torch.stack((steps, ends), dim=-1).unsqueeze(0)


def _occupy(
    texts: tuple[str, ...],
    spans: tuple[tuple[TermSpan, ...], ...],
    *,
    assignment: torch.Tensor | None = None,
    live: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> OccupancyMap:
    rows = len(texts)
    width = _TOKENS
    assign = torch.ones(rows, width, _BANK) if assignment is None else assignment
    mask = torch.ones(rows, width) if live is None else live
    mapped = (
        _offsets(width, content_end=max(len(text) for text in texts)).expand(rows, width, 2)
        if offsets is None
        else offsets
    )
    ingress = GraphIngress()
    ingress.termhood = _termhood()
    return ingress.occupy(assign, mask, texts, mapped, spans=spans, update_stats=False)


class TestOccupyPair:
    """X occupies the content span; A does not occupy that span or published stops."""

    def test_x_occupies_content_and_a_drops_stops_and_x_content(self) -> None:
        """Content heads land on X. A keeps neither that span nor published stops."""
        assert all(is_stop_surface(key) for key in _STOP_KEYS)
        assert not is_stop_surface(_X_CONTENT)
        x_spans = (_content_span(), *_stop_spans(_X_TEXT))
        a_spans = _stop_spans(_A_TEXT)
        assert occupy_eligible_spans(x_spans) == (_content_span(),)
        assert occupy_eligible_spans(a_spans) == ()
        occupied = _occupy((_X_TEXT, _A_TEXT), (x_spans, a_spans))
        x_labels = frozenset(label for label in occupied.labels[0] if label)
        a_labels = frozenset(label for label in occupied.labels[1] if label)
        assert _X_CONTENT in x_labels
        assert occupied.weights[0].gt(0).any()
        assert _X_CONTENT not in a_labels
        assert a_labels.isdisjoint(_STOP_KEYS)
        assert not occupied.weights[1].any()
        assert not occupied.assignment[1].any()

    def test_stop_only_a_writes_vacant_not_host_assignment(self) -> None:
        """Stop-only remembered spans on A write vacant occupancy, not host assignment."""
        assignment = torch.arange(1, 1 + _TOKENS * _BANK, dtype=torch.float).reshape(
            1, _TOKENS, _BANK
        )
        occupied = _occupy((_A_TEXT,), (_stop_spans(_A_TEXT),), assignment=assignment)
        vacant = empty_occupancy(assignment)
        assert _live_labels(occupied).isdisjoint({_X_CONTENT, *_STOP_KEYS})
        assert not occupied.weights.any()
        assert not occupied.assignment.any()
        assert torch.equal(occupied.assignment, vacant.assignment)
        assert torch.equal(occupied.weights, vacant.weights)

    def test_candidates_keep_x_content_and_drop_stops_on_a(self) -> None:
        """Noun-group draw after published stops keeps X content and leaves A off that span."""
        ingress = GraphIngress()
        x_docs, a_docs = ingress.candidates((_X_TEXT, _A_TEXT))
        x_keys = frozenset(span.lemma_key for span in x_docs)
        a_keys = frozenset(span.lemma_key for span in a_docs)
        assert any('photodiode' in key and 'latchbolt' in key for key in x_keys)
        assert not any(is_stop_surface(key) for key in x_keys | a_keys)
        assert a_keys.isdisjoint(_STOP_KEYS)
        assert not any(_X_CONTENT in key for key in a_keys)

    def test_extract_spans_drop_stops_and_miss_x_content_on_a(self) -> None:
        """Extract surfaces on A do not become occupy heads for stops or the X-only span."""
        x_term = {'term': _X_CONTENT, 'frequency': 1, 'surfaces': [_X_CONTENT]}
        stop_terms = tuple({'term': key, 'frequency': 1, 'surfaces': [key]} for key in _STOP_KEYS)
        docs = spans_from_extract(
            (_A_TEXT, _X_TEXT),
            ((*stop_terms, x_term), (x_term, *stop_terms)),
        )
        a_keys = frozenset(span.lemma_key for span in docs[0])
        x_keys = frozenset(span.lemma_key for span in docs[1])
        assert a_keys == frozenset()
        assert _X_CONTENT in x_keys
        assert x_keys.isdisjoint(_STOP_KEYS)


class TestVacantOccupy:
    """Empty draw and missing token ids write zeros instead of host assignment."""

    def test_empty_draw_zeros_assignment(self) -> None:
        """An empty JATE draw on A does not pass last-layer assignment through."""
        assignment = torch.ones(1, _TOKENS, _BANK)
        occupied = _occupy((_A_TEXT,), ((),), assignment=assignment)
        assert not occupied.assignment.any()
        assert not occupied.weights.any()
        assert _live_labels(occupied) == frozenset()

    def test_empty_texts_on_trunk_occupy_write_zeros(self) -> None:
        """The trunk occupy path vacates when host strings are missing."""
        assignment = torch.ones(1, _TOKENS, _BANK)
        vacant = SoftTrunkModel.occupy_assignment(
            cast(SoftTrunkModel, SimpleNamespace()),
            assignment,
            torch.ones(1, _TOKENS),
            torch.ones(1, _TOKENS, dtype=torch.long),
            (),
        )
        assert not vacant.assignment.any()
        assert not vacant.weights.any()
        assert torch.equal(vacant.assignment, empty_occupancy(assignment).assignment)

    def test_inventory_missing_ids_do_not_call_occupy(self) -> None:
        """Missing token ids vacate covering intensity without host assignment."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config())
        last_layer = torch.randn(1, _TOKENS, 32)
        attention = torch.ones(1, _TOKENS)
        claim = torch.ones(1, _TOKENS)

        def occupy(
            assignment: torch.Tensor,
            live_mask: torch.Tensor,
            input_ids: torch.Tensor,
            texts: tuple[str, ...],
            **kwargs: object,
        ) -> OccupancyMap:
            del assignment, live_mask, input_ids, texts, kwargs
            raise AssertionError('missing token ids must not call occupy')

        container = SsvContainer(config=SsvTrainConfig())
        payload = container.inventory()(
            last_layer,
            attention,
            claim,
            model=SimpleNamespace(soft_vocab=vocab, occupy_assignment=occupy),
            texts=(_A_TEXT,),
        )
        assert not payload.n_entity_full.any()
        assert not payload.n_entity_claim.any()


class TestCoveringInventoryOccupy:
    """Covering inventory occupies when the trunk owns occupy."""

    def test_stop_only_a_does_not_keep_host_intensity(self) -> None:
        """A stop-only occupy map zeros covering n; a bank-only trunk would keep host n."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config())
        last_layer = torch.randn(1, _TOKENS, 32)
        attention = torch.ones(1, _TOKENS)
        claim = torch.ones(1, _TOKENS)
        host = Inventory(occupied_floor=0.0)(
            last_layer,
            attention,
            claim,
            model=SimpleNamespace(soft_vocab=vocab),
            claim_texts=(_X_TEXT,),
        )
        assert host.n_entity_full.any()

        def occupy(
            assignment: torch.Tensor,
            live_mask: torch.Tensor,
            input_ids: torch.Tensor,
            texts: tuple[str, ...],
            **kwargs: object,
        ) -> OccupancyMap:
            del input_ids, kwargs
            ingress = GraphIngress()
            ingress.termhood = _termhood()
            offsets = _offsets(_TOKENS, content_end=len(_A_TEXT))
            return ingress.occupy(
                assignment,
                live_mask,
                texts,
                offsets,
                spans=(_stop_spans(_A_TEXT),),
                update_stats=False,
            )

        container = SsvContainer(config=SsvTrainConfig())
        payload = container.inventory()(
            last_layer,
            attention,
            claim,
            model=SimpleNamespace(soft_vocab=vocab, occupy_assignment=occupy),
            texts=(_A_TEXT,),
            input_ids=torch.ones(1, _TOKENS, dtype=torch.long),
        )
        assert not payload.n_entity_full.any()
        assert not payload.n_entity_claim.any()

    def test_x_content_occupy_moves_covering_n(self) -> None:
        """When occupy lands the X content span, covering n uses those weights."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config())
        last_layer = torch.randn(1, _TOKENS, 32)
        attention = torch.ones(1, _TOKENS)
        claim = torch.ones(1, _TOKENS)

        def occupy(
            assignment: torch.Tensor,
            live_mask: torch.Tensor,
            input_ids: torch.Tensor,
            texts: tuple[str, ...],
            **kwargs: object,
        ) -> OccupancyMap:
            del input_ids, kwargs
            ingress = GraphIngress()
            ingress.termhood = _termhood()
            offsets = _offsets(_TOKENS, content_end=len(_X_TEXT))
            return ingress.occupy(
                assignment,
                live_mask,
                texts,
                offsets,
                spans=((_content_span(),),),
                update_stats=False,
            )

        container = SsvContainer(config=SsvTrainConfig())
        payload = container.inventory()(
            last_layer,
            attention,
            claim,
            model=SimpleNamespace(soft_vocab=vocab, occupy_assignment=occupy),
            texts=(_X_TEXT,),
            input_ids=torch.ones(1, _TOKENS, dtype=torch.long),
            claim_texts=(_X_TEXT,),
        )
        assert payload.n_entity_full.any()


class TestPublishedStopDrop:
    """Attach and the extract probe omit published stop keys."""

    def test_attach_omits_fixture_stops_and_keeps_x_content(self, tmp_path: Path) -> None:
        """Row-group attach publishes the X content key and drops the fixture stops."""
        store = TermhoodStore.open(write_termhood(_termhood(), tmp_path))
        model = cast(
            SoftTrunkModel,
            cast(object, SimpleNamespace(graph_ingress=SimpleNamespace(termhood=None))),
        )
        attach = bind_termhood_attach(model, store)
        assert attach((_X_CONTENT, *_STOP_KEYS)) == 1
        published = model.graph_ingress.termhood
        assert isinstance(published, TermhoodIndex)
        assert tuple(published.scores) == (_X_CONTENT,)
        assert frozenset(published.scores).isdisjoint(_STOP_KEYS)

    def test_probe_score_frame_and_extract_join_drop_stops(self, tmp_path: Path) -> None:
        """Probe scores omit fixture stops; an extract row of those keys stays vacant."""
        scores = occupy_score_frame(_termhood())
        keys = frozenset(scores.get_column('key').to_list())
        assert _X_CONTENT in keys
        assert keys.isdisjoint(_STOP_KEYS)
        part = tmp_path / 'extract' / 'part.parquet'
        part.parent.mkdir(parents=True)
        pl.DataFrame({
            'patent_id': ['a-stop', 'x-content'],
            'n_docs': [1, 1],
            'terms': [
                [{'term': key, 'frequency': 1, 'surfaces': [key]} for key in _STOP_KEYS],
                [{'term': _X_CONTENT, 'frequency': 1, 'surfaces': [_X_CONTENT]}],
            ],
        }).write_parquet(part)
        tally = occupy_extract_parts((part,), _termhood(), occupy_limit=2)
        by_id = {row.application_number: row for row in tally.examples}
        assert by_id['a-stop'].labels == ()
        assert by_id['a-stop'].n_occupied == 0
        assert by_id['x-content'].labels == (_X_CONTENT,)
        assert 'comprising' not in by_id['x-content'].labels


class TestOccupyCapAndCorpus:
    """Occupy width is leftover-unpaid-sufficient. HUPD claim strings stay."""

    def test_occupy_has_no_integer_cap(self) -> None:
        """Config and inventory do not carry a global occupy integer."""
        container = SsvContainer(config=SsvTrainConfig())
        assert 'soft_occupied_max' not in ArchSpec.model_fields
        assert 'soft_occupied_max' not in type(SsvTrainConfig().arch).model_fields
        assert 'occupied_max' not in dict(container.inventory().named_buffers())

    def test_hupd_fixture_claims_still_contain_comprising(self) -> None:
        """Shipped HUPD claim strings still carry the transition surface."""
        claims = (SSV_HUPD_FIXTURES / '13817165.json').read_text(encoding='utf-8')
        assert 'comprising' in claims.lower()
        assert 'wherein' in claims.lower()
