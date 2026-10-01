"""Disclosure windows stay off the train text and sum into n_full."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch_geometric.data import HeteroData

from ip_claim.collision.config import CollisionPrefixMode
from ip_claim.collision.disclosure import (
    add_chunk_intensities,
    content_window_width,
    disclosure_windows,
    measure_disclosure_lengths,
)
from ip_claim.collision.encode_job import (
    CollisionEncodeBatch,
    fold_disclosure_chunks,
    last_layer_for_inventory,
)
from ip_claim.ingestion.adapters.hupd_json.loader import patent_from_hupd_dict
from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmExample
from ip_claim.ssv.graph_batch import (
    graph_batch_from_hupd_dict,
    patent_disclosure_text,
    patent_training_text,
)
from ip_claim.ssv.inventory import CoveringInventory
from ip_claim.ssv.model import SoftTrunkModel


class _LetterTokenizer:
    """One id per letter; two special tokens, matching a host window budget."""

    def num_special_tokens_to_add(self, *, pair: bool = False) -> int:
        del pair
        return 2

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        verbose: bool = True,
    ) -> list[int]:
        del add_special_tokens, truncation, verbose
        return [ord(char) for char in text if not char.isspace()]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return ''.join(chr(item) for item in token_ids)


def test_disclosure_is_not_in_training_text() -> None:
    raw = {
        'application_number': '1',
        'claims': '1. A widget.',
        'abstract': 'short abstract',
        'background': 'BACKGROUNDMARKER',
        'full_description': 'DESCRIPTIONMARKER',
        'summary': '',
    }
    patent = patent_from_hupd_dict(raw)
    train = patent_training_text(patent)
    disclosure = patent_disclosure_text(patent)
    assert 'BACKGROUNDMARKER' not in train
    assert 'DESCRIPTIONMARKER' not in train
    assert 'BACKGROUNDMARKER' in disclosure
    assert 'DESCRIPTIONMARKER' in disclosure
    batch = graph_batch_from_hupd_dict(raw)
    assert batch.disclosure == disclosure
    assert batch.text == train


class _SilentIfVerboseFalse(_LetterTokenizer):
    """Fails if unbounded encode stays on the host verbose path."""

    model_max_length = 4
    warned = False

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        verbose: bool = True,
    ) -> list[int]:
        ids = super().encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            verbose=verbose,
        )
        if verbose and len(ids) > self.model_max_length:
            self.warned = True
        return ids


def test_unbounded_disclosure_encode_is_silent_and_uncut() -> None:
    tokenizer = _SilentIfVerboseFalse()
    windows = disclosure_windows(tokenizer, 'abcdefghij', max_length=4, max_chunks=8)
    assert tokenizer.warned is False
    assert windows == ('ab', 'cd', 'ef', 'gh', 'ij')


def test_second_window_holds_mass_first_window_drops() -> None:
    tokenizer = _LetterTokenizer()
    assert content_window_width(tokenizer, max_length=4) == 2
    first = disclosure_windows(tokenizer, 'aabbcc', max_length=4, max_chunks=1)
    both = disclosure_windows(tokenizer, 'aabbcc', max_length=4, max_chunks=3)
    assert first == ('aa',)
    assert both == ('aa', 'bb', 'cc')
    oneshot = disclosure_windows(tokenizer, 'aa', max_length=4, max_chunks=3)
    assert oneshot == ('aa',)


def test_chunk_intensities_sum_equals_oneshot_when_one_window() -> None:
    base = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    chunks = torch.tensor([[0.0, 2.0]])
    owners = torch.tensor([0])
    summed = add_chunk_intensities(base, chunks, owners)
    assert summed[0].tolist() == pytest.approx([1.0, 2.0])
    assert summed[1].tolist() == pytest.approx([0.0, 0.0])


def test_measure_fixture_disclosure_is_nonempty(fixtures_dir: Path) -> None:
    paths = tuple((fixtures_dir / 'hupd').glob('*.json'))
    assert paths
    report = measure_disclosure_lengths(paths, _LetterTokenizer(), max_length=4)
    assert report.sampled == len(paths)
    assert report.nonempty >= 1
    assert report.window_width == 2
    assert report.token_max is not None
    assert report.token_max > 0


def test_fold_disclosure_tiles_to_first_window_batch() -> None:
    seen: list[int] = []
    graph = HeteroData()

    class _Collator:
        tokenizer = _LetterTokenizer()
        max_length = 4

        def __call__(self, examples: Sequence[SoftMlmExample]) -> SoftMlmBatch:
            width = len(tuple(examples))
            ones = torch.ones(width, 4)
            return SoftMlmBatch(
                input_ids=ones.long(),
                unmasked_input_ids=ones.long(),
                attention_mask=ones,
                labels=ones.long(),
                graphs=tuple(graph for _ in examples),
            )

    first = torch.ones(2, 4)
    batch = CollisionEncodeBatch(
        application_numbers=('1', '2'),
        cpc_sections=(None, None),
        claim_mask=first,
        disclosures=('aabbccdd', 'eeff'),
        mlm=SoftMlmBatch(
            input_ids=first.long(),
            unmasked_input_ids=first.long(),
            attention_mask=first,
            labels=first.long(),
            graphs=(graph, graph),
        ),
    )

    def fake_last_layer(
        chunk_batch: CollisionEncodeBatch,
        **_kwargs: object,
    ) -> tuple[None, torch.Tensor]:
        width = int(chunk_batch.mlm.attention_mask.size(0))
        seen.append(width)
        return None, torch.zeros(width, 1, 2)

    class _Inventory:
        def __call__(
            self,
            last_layer: torch.Tensor,
            _attention: torch.Tensor,
            _mask: torch.Tensor,
            *,
            model: object,
            texts: object = (),
            input_ids: object = None,
            living: object = None,
            claim_texts: object = (),
            demand: object = None,
        ) -> CoveringInventory:
            del model, texts, input_ids, living, claim_texts, demand
            rows = last_layer.size(0)
            zeros = torch.zeros(rows, 2)
            return CoveringInventory(
                n_entity_claim=zeros,
                n_entity_full=torch.ones(rows, 2),
                n_relation_claim=zeros,
                n_relation_full=torch.ones(rows, 2),
                mean_row_entropy=torch.zeros(()),
                batch_usage=torch.zeros(2),
                relation_row_entropy=0.0,
            )

    payload = CoveringInventory(
        n_entity_claim=torch.zeros(2, 2),
        n_entity_full=torch.zeros(2, 2),
        n_relation_claim=torch.zeros(2, 2),
        n_relation_full=torch.zeros(2, 2),
        mean_row_entropy=torch.zeros(()),
        batch_usage=torch.zeros(2),
        relation_row_entropy=0.0,
    )
    folded = fold_disclosure_chunks(
        payload,
        batch,
        model=cast(SoftTrunkModel, cast(object, SimpleNamespace())),
        device=torch.device('cpu'),
        prefix_mode=CollisionPrefixMode.trunk,
        inventory=cast(Any, _Inventory()),
        collator=_Collator(),
        max_chunks=8,
        last_layer=fake_last_layer,
    )
    assert max(seen) <= 2
    assert sum(seen) == 6
    assert folded.n_entity_full[0].tolist() == pytest.approx([4.0, 4.0])
    assert folded.n_entity_full[1].tolist() == pytest.approx([2.0, 2.0])


def test_last_layer_for_inventory_exports_unmasked_ids() -> None:
    """Inventory last-layer export occupies the ids before MLM corruption."""
    graph = HeteroData()
    masked = torch.tensor([[1, 2, 3, 4]])
    clean = torch.tensor([[5, 6, 7, 8]])
    seen: dict[str, torch.Tensor] = {}

    class _Export:
        soft_tokens = torch.zeros(1, 1, 2)

    class _Model:
        n_soft_tokens = 1
        soft_vocab = object()
        host = object()

        def export_trunk(
            self,
            graphs: object,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            texts: object = (),
            claim_texts: object = (),
            living: object = None,
            claim_mask: object = None,
        ) -> _Export:
            del graphs, attention_mask, texts, claim_texts, living, claim_mask
            seen['ids'] = input_ids.clone()
            return _Export()

    class _Inventory:
        def last_layer_after_prefix(
            self,
            model: object,
            prefix: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            del model, prefix, attention_mask
            seen['layer_ids'] = input_ids.clone()
            return torch.zeros(1, 4, 2)

    batch = CollisionEncodeBatch(
        application_numbers=('1',),
        cpc_sections=(None,),
        claim_mask=torch.ones(1, 4),
        disclosures=('x',),
        mlm=SoftMlmBatch(
            input_ids=masked,
            unmasked_input_ids=clean,
            attention_mask=torch.ones(1, 4),
            labels=torch.ones(1, 4, dtype=torch.long),
            graphs=(graph,),
            texts=('coil',),
        ),
    )
    last_layer_for_inventory(
        batch,
        model=cast(SoftTrunkModel, cast(object, _Model())),
        device=torch.device('cpu'),
        prefix_mode=CollisionPrefixMode.trunk,
        inventory=cast(Any, _Inventory()),
    )
    assert torch.equal(seen['ids'], clean)
    assert torch.equal(seen['layer_ids'], clean)
