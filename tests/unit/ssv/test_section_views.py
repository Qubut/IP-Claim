"""Independent claim and disclosure views stay off the MLM concatenation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from patent_ate.nlp import TermSpan

from ip_claim.ingestion.adapters.hupd_json.paths import patent_from_hupd_path
from ip_claim.ssv.collate import JATE_DRAW_CHAR_CAP, SoftMlmCollator
from ip_claim.ssv.config import ArchSpec, HostSpec, RuntimeSpec, SsvTrainConfig
from ip_claim.ssv.dataset import examples_from_patents
from ip_claim.ssv.graph_batch import dest_claim_texts, graph_batch_from_patent
from ip_claim.ssv.host_tokenizer import load_host_tokenizer

_FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'hupd'


def test_examples_keep_claim_and_disclosure_off_mlm_text() -> None:
    patents = tuple(
        patent_from_hupd_path(_FIXTURES / name)
        for name in ('13817165.json', '14111139.json', '14112715.json')
    )
    batches = tuple(graph_batch_from_patent(patent) for patent in patents)
    examples = examples_from_patents(patents)
    assert all(
        example.claim_text == (dest_claim_texts(batch)[0] if dest_claim_texts(batch) else '')
        and example.disclosure_text == batch.disclosure
        and example.text == batch.text
        for example, batch in zip(examples, batches, strict=True)
    )
    assert all(
        example.claim_text != batch.claim_blob
        for example, batch in zip(examples, batches, strict=True)
        if len(batch.claim_texts) > 1
    )
    assert all(
        example.claim_text.strip() and example.disclosure_text.strip() for example in examples
    )
    assert all(
        len({example.disclosure_text, example.claim_text, example.text}) == 3
        for example in examples
    )


def test_collate_preserves_section_views_and_tokenizes_mlm_text_only() -> None:
    patent = patent_from_hupd_path(_FIXTURES / '13817165.json')
    example = examples_from_patents((patent,))[0]
    config = SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        arch=ArchSpec(max_length=48),
        runtime=RuntimeSpec(ray=False),
    )
    tokenizer = load_host_tokenizer(config)
    batch = SoftMlmCollator(tokenizer, mlm_probability=0.0, max_length=48)((example,))
    assert batch.texts == (example.text,)
    assert batch.claim_texts == (example.claim_text,)
    assert batch.disclosure_texts == (example.disclosure_text,)
    assert len({example.disclosure_text, example.claim_text, example.text}) == 3
    assert batch.term_spans == ()
    assert batch.termhood_delta is None


def test_collator_primes_jate_spans_without_host_spacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patent = patent_from_hupd_path(_FIXTURES / '13817165.json')
    example = examples_from_patents((patent,))[0]
    config = SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        arch=ArchSpec(max_length=48),
        runtime=RuntimeSpec(ray=False),
    )
    span = TermSpan(lemma_key='coil spring', start_char=0, end_char=12, token_length=2)

    def fake_drawn(self: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        texts = getattr(self, 'texts', ())
        listed = tuple(texts) if isinstance(texts, (list, tuple)) else (str(texts),)
        return SimpleNamespace(docs=tuple((span,) for _ in listed))

    monkeypatch.setattr('patent_ate.nlp.TextWindow.drawn', fake_drawn)
    monkeypatch.setattr('ip_claim.ssv.graph_ingress.host_language', lambda *_a, **_k: object())
    tokenizer = load_host_tokenizer(config)
    batch = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.0,
        max_length=48,
        prime_jate_spans=True,
    )((example,))
    assert batch.term_spans == ((span,),)
    assert batch.termhood_delta is None


def test_collator_caps_jate_draw_text_to_char_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patent = patent_from_hupd_path(_FIXTURES / '13817165.json')
    example = examples_from_patents((patent,))[0]
    overflow = example.model_copy(update={'text': 'a' * (JATE_DRAW_CHAR_CAP + 10_000)})
    seen: list[tuple[str, ...]] = []

    def fake_drawn(self: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        texts = getattr(self, 'texts', ())
        listed = tuple(texts) if isinstance(texts, (list, tuple)) else (str(texts),)
        seen.append(listed)
        return SimpleNamespace(docs=tuple(() for _ in listed))

    monkeypatch.setattr('patent_ate.nlp.TextWindow.drawn', fake_drawn)
    monkeypatch.setattr('ip_claim.ssv.graph_ingress.host_language', lambda *_a, **_k: object())
    config = SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        arch=ArchSpec(max_length=48),
        runtime=RuntimeSpec(ray=False),
    )
    tokenizer = load_host_tokenizer(config)
    batch = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.0,
        max_length=48,
        prime_jate_spans=True,
    )((overflow,))
    assert seen == [(overflow.text[:JATE_DRAW_CHAR_CAP],)]
    assert len(overflow.text) > JATE_DRAW_CHAR_CAP
    assert batch.texts == (overflow.text,)
