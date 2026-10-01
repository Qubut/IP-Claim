"""Host-free JATE candidate cache for covering encodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from patent_ate.nlp import TermSpan
from patent_ate.termhood import TermhoodTable
from torch_geometric.data import HeteroData

from experiments.covering_objective.corpus import HupdDraw
from ip_claim.collision.encode_job import CollisionEncodeRow
from ip_claim.ssv.collate import SoftMlmExample
from ip_claim.ssv.graph_ingress import GraphIngress
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = pytest.mark.experiment

_SPAN = TermSpan(lemma_key='coil spring', start_char=0, end_char=12, token_length=2)


def _capture_pipe_workers(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    seen: dict[str, int] = {}

    class FakeDoc:
        class User:
            cached_noun_chunks: tuple[object, ...] = ()

        _ = User()

    def fake_pipe(
        listed: object,
        n_process: int,
        batch_size: int,
    ) -> tuple[FakeDoc, ...]:
        del batch_size
        docs = tuple(FakeDoc() for _ in cast(tuple[str, ...], listed))
        seen['n_process'] = n_process
        seen['n_docs'] = len(docs)
        return docs

    monkeypatch.setattr(
        GraphIngress,
        'nlp',
        lambda _self, **_kwargs: SimpleNamespace(pipe=fake_pipe),
    )
    return seen


class TestEncodeCandidatesHost:
    """One spaCy draw is shared; actors remember instead of redrawing."""

    def test_lemma_keys_keep_spans_for_later_encode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lemma-key publish leaves the ingress cache for the encode workers."""
        calls = {'n': 0}

        def fake_from_texts(texts: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
            calls['n'] += 1
            listed = tuple(texts) if isinstance(texts, (list, tuple)) else (str(texts),)
            return SimpleNamespace(
                docs=tuple((_SPAN,) for _ in listed),
                termhood=TermhoodTable(),
            )

        monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', fake_from_texts)
        ingress = GraphIngress()
        model = cast(SoftTrunkModel, cast(object, SimpleNamespace(graph_ingress=ingress)))
        text = 'coil spring assembly'
        row = CollisionEncodeRow(
            application_number='host-free',
            example=SoftMlmExample(text=text, graph=HeteroData()),
            claim_blob=text,
        )
        draw = HupdDraw(rows=(), n_pool=0, claim_rows=(row,), disc_rows=())
        keys = draw.lemma_keys(model)
        assert keys == ('coil spring',)
        assert calls['n'] == 1
        drawn = ingress.candidates(('coil spring assembly',))
        assert calls['n'] == 1
        assert drawn == ((_SPAN,),)

    def test_remembered_spans_skip_worker_redraw(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A worker that received driver spans must not call JateDraw.from_texts."""
        calls = {'n': 0}

        def fake_from_texts(*_args: object, **_kwargs: object) -> SimpleNamespace:
            calls['n'] += 1
            return SimpleNamespace(docs=((),), termhood=TermhoodTable())

        monkeypatch.setattr('ip_claim.ssv.graph_ingress.JateDraw.from_texts', fake_from_texts)
        worker = GraphIngress()
        worker.remember_inference_spans({'coil spring assembly': (_SPAN,)})
        drawn = worker.candidates(('coil spring assembly',))
        assert calls['n'] == 0
        assert drawn == ((_SPAN,),)

    def test_parallel_pipe_stays_in_process_after_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A live CUDA context must not fork spaCy workers from this process."""
        seen = _capture_pipe_workers(monkeypatch)
        monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
        parsed = tuple(GraphIngress().parallel_language().pipe(('a', 'b')))
        assert seen['n_process'] == 1
        assert seen['n_docs'] == 2
        assert len(parsed) == 2

    def test_parallel_pipe_stays_in_process_when_cuda_advertised(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An advertised GPU job must not fork even when no tensor context exists."""
        seen = _capture_pipe_workers(monkeypatch)
        monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
        parsed = tuple(GraphIngress().parallel_language().pipe(('a', 'b')))
        assert seen['n_process'] == 1
        assert seen['n_docs'] == 2
        assert len(parsed) == 2
