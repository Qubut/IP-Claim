"""Post-prefix inventory: claim vs full masks and late pair rebuild."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.soft_vocab import SoftVocabModule
from tests._ssv_fixtures import ssv_tiny_vocab_config


def _tiny_vocab() -> SoftVocabModule:
    return SoftVocabModule(ssv_tiny_vocab_config())


class _VocabTrunk(NamedTuple):
    """Bank-only trunk for late-assignment intensity tests."""

    soft_vocab: SoftVocabModule


class _PrefixHostOut:
    """Fixed last-layer table: five tokens, width 8."""

    hidden_states: tuple[Tensor, ...] = (
        torch.zeros(1, 5, 8),
        torch.arange(40, dtype=torch.float).reshape(1, 5, 8),
    )


class _PrefixHost:
    """Host that embeds ids as zeros and returns `_PrefixHostOut`."""

    def get_input_embeddings(self) -> Callable[[Tensor], Tensor]:
        def embed(ids: Tensor) -> Tensor:
            return torch.zeros(ids.size(0), ids.size(1), 8)

        return embed

    def __call__(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        output_hidden_states: bool,
    ) -> _PrefixHostOut:
        del inputs_embeds, attention_mask, output_hidden_states
        return _PrefixHostOut()


class _PrefixTrunk(NamedTuple):
    """Prefix-only trunk for last-layer slice tests."""

    host: _PrefixHost


def test_inventory_buffers_are_not_parameters() -> None:
    inventory = Inventory()
    assert list(inventory.parameters()) == []
    names = dict(inventory.named_buffers())
    assert set(names) == {'occupied_floor'}
    assert tuple(inventory.children()) == ()


def test_claim_token_mask_marks_offset_prefix() -> None:
    class _FastTok:
        is_fast = True

        def __call__(self, texts: object, **kwargs: object) -> dict[str, torch.Tensor]:
            _ = (texts, kwargs)
            return {
                'offset_mapping': torch.tensor([[[0, 0], [0, 4], [5, 11], [12, 17], [0, 0]]]),
                'attention_mask': torch.ones(1, 5),
            }

    attention = torch.ones(1, 5)
    mask = Inventory().claim_mask(
        _FastTok(),
        ('gear spring motor',),
        ('gear spring',),
        attention,
        max_length=8,
    )
    assert mask.tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0]]


def test_empty_claim_text_is_zero_mask() -> None:
    class _FastTok:
        is_fast = True

        def __call__(self, texts: object, **kwargs: object) -> dict[str, torch.Tensor]:
            _ = (texts, kwargs)
            return {
                'offset_mapping': torch.tensor([[[0, 0], [0, 4], [0, 0]]]),
                'attention_mask': torch.ones(1, 3),
            }

    attention = torch.ones(1, 3)
    mask = Inventory().claim_mask(_FastTok(), ('abstract only',), ('',), attention, max_length=8)
    assert torch.equal(mask, torch.zeros_like(attention))


def test_claim_token_mask_marks_interior_numbered_claim() -> None:
    class _FastTok:
        is_fast = True

        def __call__(self, texts: object, **kwargs: object) -> dict[str, torch.Tensor]:
            _ = (texts, kwargs)
            return {
                'offset_mapping': torch.tensor([[[0, 0], [0, 7], [8, 18], [19, 24], [0, 0]]]),
                'attention_mask': torch.ones(1, 5),
            }

    attention = torch.ones(1, 5)
    mask = Inventory().claim_mask(
        _FastTok(),
        ('claim 1 photodiode extra',),
        ('photodiode',),
        attention,
        max_length=8,
    )
    assert mask.tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0]]


def test_inventory_mask_asymmetry_and_raw_intensities() -> None:
    vocab = _tiny_vocab()
    model = _VocabTrunk(vocab)
    last_layer = torch.randn(1, 6, 32)
    attention = torch.ones(1, 6)
    claim = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    inventory = Inventory(occupied_floor=0.0)
    payload = inventory(
        last_layer,
        attention,
        claim,
        model=model,
        claim_texts=('1. A photodiode.',),
    )
    late_assign, _ = vocab.soft_assign(last_layer)
    assert payload.claim_labeled is not None
    assert payload.full_labeled is not None
    assert payload.claim_demand is not None
    assert torch.allclose(
        payload.n_entity_claim,
        inventory.overlay_intensity(
            vocab.masked_intensity(late_assign, claim),
            payload.claim_labeled,
            demand=payload.claim_demand,
        ),
    )
    assert torch.allclose(
        payload.n_entity_full,
        inventory.overlay_intensity(
            vocab.masked_intensity(late_assign, attention),
            payload.full_labeled,
            demand=payload.claim_demand,
        ),
    )
    assert payload.n_entity_full.sum().item() > payload.n_entity_claim.sum().item()
    assert payload.n_entity_full.sum().item() > 1.0 + 1e-5
    simplex = payload.n_entity_full / payload.n_entity_full.sum()
    assert not torch.allclose(payload.n_entity_full, simplex)
    assert payload.n_relation_full.shape == (1, 4)
    assert payload.n_relation_claim.shape == (1, 4)
    assert payload.n_relation_full.sum() + 1e-4 >= payload.n_relation_claim.sum()
    assert payload.claim_labeled is not None
    assert payload.full_labeled is not None
    assert payload.claim_labeled.shape == (1, 8, 8)
    assert payload.full_labeled.shape == (1, 8, 8)


def test_inventory_export_is_dense_keep_applies_at_rank() -> None:
    vocab = _tiny_vocab()
    model = _VocabTrunk(vocab)
    last_layer = torch.randn(1, 6, 32)
    attention = torch.ones(1, 6)
    claim = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    inventory = Inventory(occupied_floor=0.0)
    payload = inventory(
        last_layer,
        attention,
        claim,
        model=model,
        claim_texts=('1. A photodiode.',),
    )
    late_assign, _ = vocab.soft_assign(last_layer)
    assert payload.claim_labeled is not None
    assert payload.full_labeled is not None
    assert payload.claim_demand is not None
    assert torch.allclose(
        payload.n_entity_claim,
        inventory.overlay_intensity(
            vocab.masked_intensity(late_assign, claim),
            payload.claim_labeled,
            demand=payload.claim_demand,
        ),
    )
    assert torch.allclose(
        payload.n_entity_full,
        inventory.overlay_intensity(
            vocab.masked_intensity(late_assign, attention),
            payload.full_labeled,
            demand=payload.claim_demand,
        ),
    )
    covering = Covering(CoveringKnobs(slot_top_k=1))
    kept_claim = covering.keep_slots(payload.n_entity_claim)
    assert int((kept_claim > 0).sum().item()) <= 1
    scored = covering(kept_claim, payload.n_entity_full)
    assert scored.demand_l1.shape == (1,)
    assert payload.n_entity_full.sum().item() >= payload.n_entity_claim.sum().item()


def test_last_layer_after_prefix_drops_soft_tokens() -> None:
    prefix = torch.zeros(1, 2, 8)
    ids = torch.ones(1, 3, dtype=torch.long)
    attention = torch.ones(1, 3)
    hidden = Inventory().last_layer_after_prefix(
        _PrefixTrunk(_PrefixHost()),
        prefix,
        ids,
        attention,
    )
    assert hidden.shape == (1, 3, 8)
    assert torch.equal(hidden, torch.arange(16, 40, dtype=torch.float).reshape(1, 3, 8))
