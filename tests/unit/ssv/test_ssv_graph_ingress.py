"""ATE occupancy: NP heads, JATE C-value times IDF, published stops are weight zero."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import spacy
import torch
from patent_ate.extract import write_termhood
from patent_ate.nlp import JateDraw, TermSpan
from patent_ate.spec import AteSpec
from patent_ate.termhood import (
    AteLexicon,
    TermhoodIndex,
    TermhoodStore,
    TermhoodTable,
    is_stop_surface,
)
from spacy.language import Language
from tests._ssv_fixtures import SSV_HUPD_FIXTURES, ssv_smoke_config, ssv_tiny_vocab_config

from ip_claim.ingestion.adapters.hupd_json.paths import patent_from_hupd_path
from ip_claim.ssv.collate import SoftMlmCollator
from ip_claim.ssv.dataset import examples_from_patents
from ip_claim.ssv.graph_ingress import (
    GraphIngress,
    OccupancyMap,
    empty_occupancy,
    gpu_job_refuses_spacy_fork,
    host_language,
    spans_from_extract,
)
from ip_claim.ssv.host_tokenizer import batch_encoding_tensor, load_host_tokenizer
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.soft_vocab import SoftVocabModule


def _span(key: str, start: int, end: int, length: int) -> TermSpan:
    return TermSpan(lemma_key=key, start_char=start, end_char=end, token_length=length)


def test_published_stops_include_comprising_and_method() -> None:
    catalog = AteLexicon.from_package()
    assert 'comprising' in catalog.surfaces
    assert 'method' in catalog.surfaces
    assert catalog.determiners == frozenset({
        'a',
        'an',
        'the',
        'said',
        'this',
        'that',
        'these',
        'those',
    })
    assert is_stop_surface('comprising')
    assert is_stop_surface(';')
    assert is_stop_surface('method')
    assert not is_stop_surface('vehicle interior')


def test_jate_scores_nested_phrases_and_zeros_stops() -> None:
    table = TermhoodTable.from_texts((
        'The interior panel belongs to the vehicle interior panel assembly.',
        'The interior panel belongs to the vehicle interior panel assembly.',
    ))
    nested = table.score('interior panel')
    outer = table.score('vehicle interior panel assembly')
    assert outer > 0.0
    assert nested < outer
    assert not table.score('comprising')


def test_jate_zeros_unigrams() -> None:
    table = TermhoodTable.from_texts((
        'The portion of the housing covers the portion of the housing.',
        'The portion of the housing covers the portion of the housing.',
    ))
    assert not table.score('portion')
    assert not table.score('housing')


def test_jate_downweights_high_df_boilerplate() -> None:
    boilerplate = 'The present disclosure describes a first end of a shaft.'
    specific = 'An analyte sensor electrode assembly measures glucose.'
    table = TermhoodTable.from_texts((boilerplate,) * 8 + (specific,))
    hapax_keys = tuple(key for key in table.c_values if 'analyte' in key)
    assert hapax_keys
    assert max(table.score(key) for key in hapax_keys) > 0.0
    assert not table.score('present disclosure')
    assert not table.score('first end')


def test_comprising_punct_and_unary_method_do_not_occupy() -> None:
    ingress = GraphIngress()
    ingress.termhood = TermhoodTable(
        c_values={'comprising': 9.0, 'method': 9.0},
        document_frequency={'comprising': 2, 'method': 2},
        total_docs=10,
    )
    assignment = torch.ones(1, 6, 4)
    live = torch.ones(1, 6)
    offsets = torch.tensor([[[0, 0], [0, 1], [2, 12], [13, 14], [15, 21], [22, 28]]])
    occupied = ingress.occupy(
        assignment,
        live,
        ('a comprising ; method housing',),
        offsets,
        spans=(
            (
                _span('comprising', 2, 12, 1),
                _span('method', 15, 21, 1),
            ),
        ),
    )
    assert not occupied.weights.any()
    assert all(not label for label in occupied.labels[0])
    assert not occupied.assignment.any()


def test_vehicle_interior_pools_onto_one_head() -> None:
    ingress = GraphIngress()
    ingress.termhood = TermhoodTable(
        c_values={'vehicle interior': 4.0},
        document_frequency={'vehicle interior': 1},
        total_docs=10,
    )
    assignment = torch.arange(8, dtype=torch.float).reshape(1, 4, 2)
    live = torch.ones(1, 4)
    offsets = torch.tensor([[[0, 7], [8, 16], [17, 20], [0, 0]]])
    span = _span('vehicle interior', 0, 16, 2)
    occupied = ingress.occupy(
        assignment,
        live,
        ('vehicle interior has',),
        offsets,
        spans=((span,),),
    )
    assert occupied.labels[0][1] == 'vehicle interior'
    assert occupied.weights[0, 1].item() > 0.0
    assert not occupied.weights[0, 0]
    assert torch.allclose(occupied.assignment[0, 1], assignment[0, :2].mean(dim=0))
    assert not occupied.assignment[0, 0].any()


def test_empty_texts_leave_assignment_and_zero_weights() -> None:
    assignment = torch.randn(2, 3, 5)
    vacant = empty_occupancy(assignment)
    assert vacant.assignment.shape == assignment.shape
    assert not vacant.assignment.any()
    assert torch.equal(vacant.weights, torch.zeros(2, 3))


def test_jate_chunked_from_texts_matches_one_pass() -> None:
    texts = (
        'The interior panel belongs to the vehicle interior panel assembly.',
        'The interior panel belongs to the vehicle interior panel assembly.',
    )
    one = TermhoodTable.from_texts(texts)
    chunked = JateDraw.from_texts(texts, chunk_size=1).termhood
    keys = tuple(one.c_values)
    assert one.scores_for(keys) == chunked.scores_for(keys)
    assert one.document_frequency == chunked.document_frequency


def test_termhood_table_round_trip() -> None:
    table = TermhoodTable.from_texts(('vehicle interior has',))
    assert table.score('vehicle interior') > 0.0
    assert table.document_frequency.get('vehicle interior', 0) >= 1
    assert table.c_values.get('vehicle interior', 0.0) > 0.0
    assert table.total_docs >= 1
    ingress = GraphIngress()
    dumped = table.dump()
    assert 'scores' not in dumped
    ingress.set_extra_state(dumped)
    published = ingress.termhood
    assert isinstance(published, TermhoodTable)
    assert published.score('vehicle interior') == table.score('vehicle interior')
    assert published.document_frequency.get('vehicle interior') == (
        table.document_frequency.get('vehicle interior')
    )
    assert published.c_values.get('vehicle interior') == table.c_values.get(
        'vehicle interior'
    )
    stale = GraphIngress()
    stale.set_extra_state({
        'freq': {'vehicle interior': 1},
        'df': {'vehicle interior': 1},
        'n_docs': 1,
    })
    assert not stale.termhood.score('vehicle interior')


def test_saturated_ranker_prefers_hapax_over_mid_df_generic() -> None:
    table = TermhoodTable(
        c_values={
            'light emitting device': 6544.0,
            'rare compound phrase': 3101.0,
            'seq id no': 5567.0,
        },
        document_frequency={
            'light emitting device': 315,
            'rare compound phrase': 1,
            'seq id no': 194,
        },
        total_docs=100_000,
    )
    assert table.score('rare compound phrase') > table.score('light emitting device')
    assert not table.score('seq id no')
    assert not table.score('brief description')


def test_sequence_identifier_is_published_form() -> None:
    assert is_stop_surface('SEQ ID NO')
    assert is_stop_surface('seq. id. no.')
    assert is_stop_surface('SEQ NO')
    assert not is_stop_surface('light emitting device')
    assert not is_stop_surface('nh2 seq id')


def test_from_product_dump_recovers_c_value() -> None:
    docs = 100_000
    df = 315
    c_value = 6544.0
    table = TermhoodTable.model_validate({
        'scores': {'light emitting device': c_value * math.log((docs + 1) / df)},
        'document_frequency': {'light emitting device': df},
        'total_docs': docs,
    })
    assert table.c_values['light emitting device'] == pytest.approx(c_value)
    assert table.total_docs == docs
    assert table.score('light emitting device') > 0.0


def test_inventory_covering_n_uses_occupancy_weights() -> None:
    vocab = SoftVocabModule(ssv_tiny_vocab_config())
    last_layer = torch.randn(1, 4, 32)
    attention = torch.ones(1, 4)
    claim = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    weights = torch.tensor([[0.0, 2.5, 0.0, 0.0]])

    def occupy(
        assignment: torch.Tensor,
        live_mask: torch.Tensor,
        input_ids: torch.Tensor,
        texts: tuple[str, ...],
        **kwargs: object,
    ) -> OccupancyMap:
        del live_mask, input_ids, texts, kwargs
        return OccupancyMap(
            assignment=assignment,
            weights=weights,
            labels=(('', 'vehicle interior', '', ''),),
        )

    model = SimpleNamespace(soft_vocab=vocab, occupy_assignment=occupy)
    inventory = Inventory(occupied_floor=0.0)
    payload = inventory(
        last_layer,
        attention,
        claim,
        model=model,
        texts=('vehicle interior',),
        input_ids=torch.ones(1, 4, dtype=torch.long),
        claim_texts=('1. A vehicle interior.',),
    )
    late, _ = vocab.soft_assign(last_layer)
    assert payload.claim_labeled is not None
    assert payload.full_labeled is not None
    assert payload.claim_demand is not None
    occupy_claim = vocab.masked_intensity(late, weights * claim)
    occupy_full = vocab.masked_intensity(late, weights)
    assert torch.allclose(
        payload.n_entity_claim,
        inventory.overlay_intensity(
            occupy_claim,
            payload.claim_labeled,
            demand=payload.claim_demand,
        ),
    )
    assert torch.allclose(
        payload.n_entity_full,
        inventory.overlay_intensity(
            occupy_full,
            payload.full_labeled,
            demand=payload.claim_demand,
        ),
    )


def test_inventory_without_texts_does_not_use_attention_mask() -> None:
    vocab = SoftVocabModule(ssv_tiny_vocab_config())
    last_layer = torch.randn(1, 4, 32)
    attention = torch.ones(1, 4)
    claim = torch.ones(1, 4)

    def occupy(
        assignment: torch.Tensor,
        live_mask: torch.Tensor,
        input_ids: torch.Tensor,
        texts: tuple[str, ...],
        **kwargs: object,
    ) -> OccupancyMap:
        del assignment, live_mask, input_ids, texts, kwargs
        raise AssertionError('missing texts must not call occupy')

    model = SimpleNamespace(soft_vocab=vocab, occupy_assignment=occupy)
    payload = Inventory(occupied_floor=0.0)(
        last_layer,
        attention,
        claim,
        model=model,
    )
    assert not payload.n_entity_full.any()
    assert not payload.n_entity_claim.any()


def test_collate_keeps_raw_comprising_text(tmp_path: Path) -> None:
    config = ssv_smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.0, max_length=48, rho=0.0)
    text = 'A vehicle interior comprising a housing.'
    patents = (patent_from_hupd_path(SSV_HUPD_FIXTURES / '13817165.json'),)
    examples = examples_from_patents(patents)
    patched = examples[0].model_copy(update={'text': text})
    batch = collator((patched,))
    assert batch.texts == (text,)
    decoded = tokenizer.decode(batch.unmasked_input_ids[0].tolist(), skip_special_tokens=True)
    assert 'comprising' in decoded.lower()


def test_candidates_drop_transitions_and_keep_multitoken_np() -> None:
    ingress = GraphIngress()
    keyed = {
        span.lemma_key
        for span in ingress.candidates(('A vehicle interior comprising a housing.',))[0]
    }
    assert not any('comprising' in key.split() for key in keyed)
    assert any('vehicle' in key and 'interior' in key for key in keyed)


def test_host_language_requires_gpu_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    loads: list[str] = []

    class Loaded:
        pipe_names: tuple[str, ...] = ()

        def add_pipe(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        'ip_claim.ssv.graph_ingress.set_gpu_allocator',
        lambda _name: None,
    )
    monkeypatch.setattr(
        'ip_claim.ssv.graph_ingress.require_gpu',
        lambda: called.append(True) or True,
    )
    monkeypatch.setattr(spacy, 'load', lambda model, disable=(): loads.append(model) or Loaded())
    host_language.cache_clear()
    try:
        host_language('en_core_web_sm', gpu=False)
        host_language('en_core_web_sm', gpu=False)
        assert called == []
        assert loads == ['en_core_web_sm']
        host_language('en_core_web_sm', gpu=True)
        assert called == [True]
    finally:
        host_language.cache_clear()


def _boom_spacy(*_args: object, **_kwargs: object) -> object:
    raise AssertionError('occupy must not load spaCy')


def _boom_from_texts(*_args: object, **_kwargs: object) -> None:
    raise AssertionError('occupy must not call JateDraw.from_texts')


def test_occupy_extract_spans_without_spacy_or_jate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('ip_claim.ssv.graph_ingress.host_language', _boom_spacy)
    monkeypatch.setattr('patent_ate.nlp.host_language', _boom_spacy)
    monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', _boom_from_texts)
    monkeypatch.setattr(spacy, 'load', _boom_spacy)
    monkeypatch.setattr(GraphIngress, 'nlp', lambda self, **_: _boom_spacy())
    config = ssv_smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    assert tokenizer.is_fast
    ingress = GraphIngress(config.host.tokenizer_id or config.host.name)
    ingress.termhood = TermhoodTable(
        c_values={'coil spring': 4.0},
        document_frequency={'coil spring': 1},
        total_docs=10,
    )
    texts = ('coil spring assembly',)
    spans = spans_from_extract(
        texts,
        (({'term': 'coil spring', 'frequency': 1, 'surfaces': ['coil spring']},),),
    )
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=16,
        padding='max_length',
        return_tensors='pt',
    )
    live = batch_encoding_tensor(encoded, 'attention_mask')
    occupied = ingress.occupy(
        live.unsqueeze(-1).to(dtype=torch.float),
        live,
        texts,
        ingress.token_offsets(
            texts,
            max_length=16,
            device=torch.device('cpu'),
            input_ids=batch_encoding_tensor(encoded, 'input_ids'),
            tokenizer=tokenizer,
        ),
        update_stats=False,
        spans=spans,
    )
    assert occupied.weights.any()
    assert 'coil spring' in occupied.labels[0]


def test_occupy_reuses_inference_spans_for_same_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {'n': 0}

    def fake_from_texts(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls['n'] += 1
        return SimpleNamespace(docs=((),), termhood=TermhoodTable())

    monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', fake_from_texts)
    ingress = GraphIngress()
    assignment = torch.ones(1, 4, 1)
    live = torch.ones(1, 4)
    offsets = torch.zeros(1, 4, 2)
    texts = ('coil spring assembly',)
    first = ingress.occupy(assignment, live, texts, offsets, update_stats=False)
    second = ingress.occupy(assignment, live, texts, offsets, update_stats=False)
    assert calls['n'] == 1
    assert first.assignment.shape == second.assignment.shape
    ingress.occupy(assignment, live, texts, offsets, update_stats=True)
    assert calls['n'] == 1
    ingress.discard_inference_spans(texts)
    ingress.occupy(assignment, live, texts, offsets, update_stats=False)
    assert calls['n'] == 2


def test_occupy_reuses_candidates_across_batch_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {'n': 0}

    def fake_from_texts(texts: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        calls['n'] += 1
        listed = tuple(texts) if isinstance(texts, (list, tuple)) else (str(texts),)
        return SimpleNamespace(docs=tuple(() for _ in listed), termhood=TermhoodTable())

    monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', fake_from_texts)
    ingress = GraphIngress()
    assignment = torch.ones(2, 4, 1)
    live = torch.ones(2, 4)
    offsets = torch.zeros(2, 4, 2)
    _ = ingress.candidates(('coil spring assembly', 'leaf spring'))
    assert calls['n'] == 1
    occupied = ingress.occupy(
        assignment,
        live,
        ('leaf spring', 'coil spring assembly'),
        offsets,
        update_stats=False,
    )
    assert calls['n'] == 1
    assert occupied.assignment.shape[0] == 2


def test_remember_inference_spans_skips_from_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {'n': 0}

    def fake_from_texts(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls['n'] += 1
        return SimpleNamespace(docs=((),), termhood=TermhoodTable())

    monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', fake_from_texts)
    ingress = GraphIngress()
    span = TermSpan(lemma_key='coil spring', start_char=0, end_char=12, token_length=2)
    ingress.remember_inference_spans({'coil spring assembly': (span,)})
    drawn = ingress.candidates(('coil spring assembly',))
    assert calls['n'] == 0
    assert drawn == ((span,),)


def _pipe_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeHost:
        def pipe(self, texts: object, **kwargs: object) -> tuple[()]:
            del texts
            captured.update(kwargs)
            return ()

    def fake_from_texts(
        texts: Sequence[str],
        nlp: Language | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        listed = tuple(texts)
        if nlp is not None:
            _ = tuple(nlp.pipe(listed))
        return SimpleNamespace(docs=tuple(() for _ in listed), termhood=TermhoodTable())

    def fake_nlp(_self: GraphIngress, *, gpu: bool = False) -> FakeHost:
        del gpu
        return FakeHost()

    monkeypatch.setattr(GraphIngress, 'nlp', fake_nlp)
    monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', fake_from_texts)
    return captured


def _expected_pipe_kwargs(texts: tuple[str, ...]) -> tuple[int, int]:
    spec = AteSpec()
    count = max(1, len(texts))
    cores = os.cpu_count() or 1
    workers = 1 if gpu_job_refuses_spacy_fork() else max(1, min(count, cores))
    width = max(1, min(count, spec.pipe_docs, (count + workers - 1) // workers))
    return workers, width


def test_candidates_pipe_uses_n_process(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _pipe_capture(monkeypatch)
    texts = ('coil spring assembly', 'leaf spring', 'vehicle interior')
    _ = GraphIngress().candidates(texts)
    workers, width = _expected_pipe_kwargs(texts)
    assert captured['n_process'] == workers
    assert captured['batch_size'] == width


def test_occupy_index_skips_from_texts_when_spans_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {'n': 0}

    def fake_from_texts(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls['n'] += 1
        return SimpleNamespace(docs=((),), termhood=TermhoodTable())

    monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', fake_from_texts)
    ingress = GraphIngress()
    ingress.termhood = TermhoodIndex(scores={'coil spring': 1.0})
    span = TermSpan(lemma_key='coil spring', start_char=0, end_char=12, token_length=2)
    ingress.remember_inference_spans({'coil spring assembly': (span,)})
    occupied = ingress.occupy(
        torch.ones(1, 4, 1),
        torch.ones(1, 4),
        ('coil spring assembly',),
        torch.zeros(1, 4, 2),
        update_stats=True,
    )
    assert calls['n'] == 0
    assert occupied.assignment.shape == (1, 4, 1)


def test_occupy_stats_pipe_uses_n_process(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _pipe_capture(monkeypatch)
    ingress = GraphIngress()
    texts = ('coil spring assembly', 'leaf spring', 'vehicle interior')
    _ = ingress.occupy(
        torch.ones(3, 4, 1),
        torch.ones(3, 4),
        texts,
        torch.zeros(3, 4, 2),
        update_stats=True,
    )
    workers, width = _expected_pipe_kwargs(texts)
    assert captured['n_process'] == workers
    assert captured['batch_size'] == width


def test_occupy_table_skips_from_texts_when_spans_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {'n': 0}

    def fake_from_texts(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls['n'] += 1
        return SimpleNamespace(docs=((),), termhood=TermhoodTable())

    monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', fake_from_texts)
    ingress = GraphIngress()
    span = TermSpan(lemma_key='coil spring', start_char=0, end_char=12, token_length=2)
    ingress.remember_inference_spans({'coil spring assembly': (span,)})
    occupied = ingress.occupy(
        torch.ones(1, 4, 1),
        torch.ones(1, 4),
        ('coil spring assembly',),
        torch.zeros(1, 4, 2),
        update_stats=True,
    )
    assert calls['n'] == 0
    assert occupied.assignment.shape == (1, 4, 1)


def test_set_extra_state_store_binds_termhood_index(tmp_path: Path) -> None:
    table = TermhoodTable(
        c_values={'coil spring': 3.5, 'comprising': 9.0},
        document_frequency={'coil spring': 2, 'comprising': 2},
        total_docs=12,
    )
    store = TermhoodStore.open(write_termhood(table, tmp_path))
    ingress = GraphIngress()
    ingress.set_extra_state({'termhood_root': str(store.root)})
    assert isinstance(ingress.termhood, TermhoodIndex)
    got = ingress.termhood.scores_for(('coil spring', 'comprising', 'missing'))
    assert got == table.scores_for(('coil spring', 'comprising', 'missing'))
    assert ingress.get_extra_state() == {'termhood_root': str(store.root)}


def test_set_extra_state_skips_store_already_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TermhoodTable(
        c_values={'coil spring': 3.5},
        document_frequency={'coil spring': 2},
        total_docs=12,
    )
    store = TermhoodStore.open(write_termhood(table, tmp_path))
    ingress = GraphIngress()
    calls = {'n': 0}
    real_from_store = TermhoodIndex.from_store

    def counted_from_store(opened: TermhoodStore) -> TermhoodIndex:
        calls['n'] += 1
        return real_from_store(opened)

    monkeypatch.setattr(TermhoodIndex, 'from_store', counted_from_store)
    ingress.set_extra_state({'termhood_root': str(store.root)})
    bound = ingress.termhood
    ingress.set_extra_state({'termhood_root': str(store.root)})
    assert calls['n'] == 1
    assert ingress.termhood is bound


def test_termhood_index_scores_for_fixture_keys(tmp_path: Path) -> None:
    store = TermhoodStore.write_frame(
        pl.DataFrame({
            'key': ['coil spring', 'comprising', 'vehicle interior'],
            'c_value': [3.5, 9.0, 1.25],
            'df': [2, 2, 1],
        }),
        tmp_path,
        total_docs=12,
    )
    index = TermhoodIndex.from_store(TermhoodStore.open(store.root))
    got = index.scores_for(('coil spring', 'vehicle interior', 'absent key', 'comprising'))
    expected = {
        'coil spring': math.log1p(3.5) * math.log(1.0 + (12 - 2 + 0.5) / (2 + 0.5)),
        'vehicle interior': math.log1p(1.25) * math.log(1.0 + (12 - 1 + 0.5) / (1 + 0.5)),
        'absent key': 0.0,
        'comprising': 0.0,
    }
    assert got == pytest.approx(expected)
