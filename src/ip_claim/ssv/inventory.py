"""Covering intensities: occupy mass plus kept typed-pair addends.

The trunk re-assigns last-layer host states onto the entity bank after the
prefix is prepended. When the trunk owns occupy, occupy mass uses those
ingress weights. Covering n adds incidence of kept directed pairs; consumed
refuse does not enter that addend. Missing texts or token ids yield vacant
occupancy instead of the attention mask. The occupancy floor lives as a
buffer so it follows the module device.
The trunk is passed into ``forward`` so Lightning does not register it as
a child of this module.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from itertools import starmap
from typing import Never, Protocol

import torch
from pydantic import BaseModel, ConfigDict
from returns.maybe import Maybe
from returns.result import Result, Success, safe
from torch import Tensor, nn

from ip_claim.ssv.covering_trace import capture_adapter_delta, covering_trace_active, trace_tensor
from ip_claim.ssv.graph_ingress import empty_occupancy
from ip_claim.ssv.host_tokenizer import batch_encoding_tensor
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.soft_graph import (
    SoftRelationBundle,
    build_soft_relation_bundle,
    leftover_unpaid_occupy_mask,
)
from ip_claim.ssv.soft_vocab import SoftVocabModule


class CoveringInventory(BaseModel):
    """Per-row occupy-plus-kept-edge intensities and kept pair-mass tables."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    n_entity_claim: Tensor
    n_entity_full: Tensor
    n_relation_claim: Tensor
    n_relation_full: Tensor
    mean_row_entropy: Tensor
    batch_usage: Tensor
    relation_row_entropy: float
    relation_batch_usage: Tensor | None = None
    claim_labeled: Tensor | None = None
    full_labeled: Tensor | None = None
    claim_demand: Tensor | None = None


SpanTokenizer = Callable[..., Mapping[str, Tensor]]


class InventoryHostOutput(Protocol):
    """Host forward result that exposes last-layer hidden states."""

    @property
    def hidden_states(self) -> Sequence[Tensor] | None:
        """Stacked host layers; the last entry is the inventory slice source."""
        ...


class InventoryPrefixHost(Protocol):
    """Host LM surface used to prepend the soft-token prefix."""

    def get_input_embeddings(self) -> Callable[[Tensor], Tensor]:
        """Token-id embedding table."""
        ...

    def __call__(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        output_hidden_states: bool,
    ) -> InventoryHostOutput:
        """Forward with concatenated prefix and text embeddings."""
        ...


class InventoryPrefixTrunk(Protocol):
    """Trunk fragment that can run a prefix-prepended host forward."""

    @property
    def host(self) -> InventoryPrefixHost:
        """Language-model host used to prepend the prefix."""
        ...


class InventoryVocabTrunk(Protocol):
    """Trunk fragment that exposes the entity and relation banks."""

    soft_vocab: SoftVocabModule


class ClaimSpanEncoder(BaseModel):
    """Token-span claim mask from a numbered claim span inside training text."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tokenizer: SpanTokenizer
    max_length: int

    def token_mask(
        self,
        texts: Sequence[str],
        claim_blobs: Sequence[str],
        attention_mask: Tensor,
    ) -> Tensor:
        """Prefer character offsets; otherwise count claim content tokens."""
        if not getattr(self.tokenizer, 'is_fast', False):
            return self.from_content_span(texts, claim_blobs, attention_mask)

        @safe(exceptions=(KeyError, TypeError, ValueError))
        def offset_mask() -> Tensor:
            return self.from_offsets(texts, claim_blobs, attention_mask)

        def content_span(_: KeyError | TypeError | ValueError) -> Result[Tensor, Exception]:
            return Success(self.from_content_span(texts, claim_blobs, attention_mask))

        return offset_mask().lash(content_span).unwrap()

    def from_offsets(
        self,
        texts: Sequence[str],
        claim_blobs: Sequence[str],
        attention_mask: Tensor,
    ) -> Tensor:
        """Tokens whose character span lies inside the numbered claim."""
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        offsets = batch_encoding_tensor(encoded, 'offset_mapping').to(device=attention_mask.device)
        if offsets.size(1) != attention_mask.size(1):
            return attention_mask

        def char_span(text: str, blob: str) -> tuple[int, int]:
            if not blob:
                return (-1, -1)
            start = text.find(blob)
            if start < 0:
                return (-1, -1)
            return (start, start + len(blob))

        spans = tuple(starmap(char_span, zip(texts, claim_blobs, strict=True)))
        starts = torch.tensor(
            [start for start, _ in spans],
            device=attention_mask.device,
        )
        ends = torch.tensor(
            [end for _, end in spans],
            device=attention_mask.device,
        )
        in_claim = (offsets[..., 0] < ends.unsqueeze(1)) & (offsets[..., 1] > starts.unsqueeze(1))
        return torch.where(
            (starts < 0).unsqueeze(1),
            torch.zeros_like(attention_mask),
            in_claim.to(dtype=attention_mask.dtype) * attention_mask,
        )

    def from_content_span(
        self,
        texts: Sequence[str],
        claim_blobs: Sequence[str],
        attention_mask: Tensor,
    ) -> Tensor:
        """First N non-special tokens when the numbered claim is a prefix."""
        prefix = tuple(
            bool(blob) and text.startswith(blob)
            for text, blob in zip(texts, claim_blobs, strict=True)
        )
        if not any(prefix):
            return torch.zeros_like(attention_mask)
        claim_enc = self.tokenizer(
            list(claim_blobs),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors='pt',
        )
        full_enc = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors='pt',
        )

        special = (
            batch_encoding_tensor(full_enc, 'special_tokens_mask')
            .to(device=attention_mask.device)
            .bool()
        )
        if special.size(1) != attention_mask.size(1):
            return attention_mask
        n_content = (
            (
                (~batch_encoding_tensor(claim_enc, 'special_tokens_mask').bool())
                & batch_encoding_tensor(claim_enc, 'attention_mask').bool()
            )
            .sum(dim=-1)
            .to(device=attention_mask.device)
        )
        content = attention_mask.bool() & ~special
        rank = content.to(dtype=torch.long).cumsum(dim=-1)
        in_claim = content & (rank <= n_content.unsqueeze(1))
        prefix_ok = torch.tensor(prefix, device=attention_mask.device)
        return torch.where(
            (~prefix_ok | (n_content == 0)).unsqueeze(1),
            torch.zeros_like(attention_mask),
            in_claim.to(dtype=attention_mask.dtype),
        )


class Inventory(nn.Module):
    """Claim versus full-text covering n from occupy plus kept-edge addends."""

    occupied_floor: Tensor

    def __init__(
        self,
        *,
        occupied_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.occupied_floor = nn.Buffer(torch.tensor(float(occupied_floor), dtype=torch.float32))

    def last_layer_after_prefix(
        self,
        model: InventoryPrefixTrunk,
        prefix: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Last-layer host states on the text tokens after the prefix is prepended."""
        token_embeds = model.host.get_input_embeddings()(input_ids)
        n_soft = int(prefix.size(1))
        host = model.host
        adapters = capture_adapter_delta(host) if isinstance(host, nn.Module) else nullcontext()
        with adapters:
            host_out = host(
                inputs_embeds=torch.cat([prefix, token_embeds], dim=1),
                attention_mask=torch.cat(
                    [
                        torch.ones(
                            attention_mask.size(0),
                            n_soft,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                        attention_mask,
                    ],
                    dim=1,
                ),
                output_hidden_states=True,
            )

        def missing_hidden() -> Never:
            msg = 'host last-layer states are required for inventory export'
            raise RuntimeError(msg)

        hidden = Maybe.from_optional(host_out.hidden_states).or_else_call(missing_hidden)
        return hidden[-1][:, n_soft:, :]

    def claim_mask(
        self,
        tokenizer: SpanTokenizer | None,
        texts: Sequence[str],
        claim_blobs: Sequence[str],
        attention_mask: Tensor,
        *,
        max_length: int,
    ) -> Tensor:
        """Mark numbered-claim tokens on the collated sequence; else the attention mask."""
        if tokenizer is None or not texts:
            return attention_mask
        return ClaimSpanEncoder(tokenizer=tokenizer, max_length=max_length).token_mask(
            texts, claim_blobs, attention_mask
        )

    def relation_bundle(
        self,
        late_assign: Tensor,
        token_mask: Tensor,
        vocab: SoftVocabModule,
        demand: Tensor | None = None,
        living: Tensor | None = None,
        pair_mass: Tensor | None = None,
    ) -> SoftRelationBundle:
        """Pair assignment rebuilt from late occupancy under one token mask.

        Numbered-claim demand selects leftover-unpaid-sufficient occupy.
        Empty demand is empty occupy.
        """
        return build_soft_relation_bundle(
            vocab,
            late_assign,
            token_mask,
            mass_floor=float(self.occupied_floor.item()),
            demand=demand,
            pair_mass=pair_mass,
            living=living,
        )

    def record_endpoint_seams(
        self,
        late_assign: Tensor,
        claim_mask: Tensor,
        attention_mask: Tensor,
        claim_pairs: SoftRelationBundle,
        full_pairs: SoftRelationBundle,
    ) -> tuple[Tensor, Tensor]:
        """Build kept pair-mass tables; also trace when capture is on."""
        explain = importlib.import_module('ip_claim.collision.explain').Explain()
        claim_labeled = explain.graphs_from_bundle(late_assign, claim_mask, claim_pairs)
        full_labeled = explain.graphs_from_bundle(late_assign, attention_mask, full_pairs)
        if covering_trace_active():
            stacked = torch.stack((claim_labeled, full_labeled), dim=1)
            _ = trace_tensor(
                'labeled_endpoint',
                stacked,
                'batch',
                'mask',
                'src',
                'dst',
            )
            _ = trace_tensor(
                'collapsed_endpoint',
                stacked,
                'batch',
                'mask',
                'src',
                'dst',
            )
        return claim_labeled, full_labeled

    def overlay_intensity(
        self,
        occupy: Tensor,
        labeled: Tensor,
        living: Tensor | None = None,
        demand: Tensor | None = None,
    ) -> Tensor:
        """Occupy mass plus incidence of the dest-labelled kept pair table.

        A living typed pair table is composed into that square addend when
        supplied. Numbered-claim demand restricts occupy to leftover-unpaid-
        sufficient support. Leftover unpaid stays on Covering.
        """
        overlay = PhiIntensity(int(occupy.size(-1))).to(device=occupy.device)
        pair_mass = labeled if labeled.ndim == 3 else labeled.sum(dim=-1)
        floor = float(self.occupied_floor.item())
        live = occupy > floor
        if demand is not None:
            live = leftover_unpaid_occupy_mask(
                occupy,
                demand,
                pair_mass,
                floor,
                living,
            )
        return overlay(occupy, pair_mass, live, living)

    def forward(
        self,
        last_layer: Tensor,
        attention_mask: Tensor,
        claim_mask: Tensor,
        *,
        model: InventoryVocabTrunk,
        texts: Sequence[str] = (),
        input_ids: Tensor | None = None,
        living: Tensor | None = None,
        claim_texts: Sequence[str] = (),
        demand: Tensor | None = None,
    ) -> CoveringInventory:
        """Build claim and full-text covering n from occupy and kept pair tables.

        Numbered-claim texts or an explicit demand tensor select leftover-
        unpaid-sufficient occupy from claim-span top-code mass. Occupancy is
        not that demand. Empty demand is empty occupy. A living typed pair
        table is composed into both covering n maps when supplied. Leftover
        unpaid stays on Covering.
        """
        vocab = model.soft_vocab
        late_assign, projected = vocab.soft_assign(last_layer)
        late_assign = trace_tensor(
            'late_assignment',
            late_assign,
            'batch',
            'token',
            'bank',
        )
        claim_token_mask = claim_mask
        occupy = getattr(model, 'occupy_assignment', None)
        if occupy is not None:
            occupied = (
                empty_occupancy(late_assign)
                if not texts or input_ids is None
                else occupy(
                    late_assign,
                    attention_mask,
                    input_ids,
                    texts,
                    update_stats=False,
                )
            )
            late_assign = trace_tensor(
                'late_assignment',
                occupied.assignment,
                'batch',
                'token',
                'bank',
            )
            occupy_w = trace_tensor(
                'termhood_weights',
                occupied.weights,
                'batch',
                'token',
            )
            claim_mask = occupy_w * claim_mask.to(dtype=occupy_w.dtype)
            attention_mask = occupy_w

        def relation_mass(bundle: SoftRelationBundle) -> Tensor:
            bank = int(vocab.relation_bank_size)
            rows = last_layer.size(0)
            zeros = torch.zeros(rows, bank, device=last_layer.device, dtype=last_layer.dtype)
            assignment = bundle.relation_assignment
            if assignment is None or bundle.pair_count == 0:
                return zeros
            counts = tuple(int(overlay.edge_index.size(1)) for overlay in bundle.overlays)

            def slot_sum(part: Tensor) -> Tensor:
                if part.size(0) == 0:
                    return torch.zeros(bank, device=last_layer.device, dtype=last_layer.dtype)
                return part.sum(dim=0)

            return torch.stack(tuple(slot_sum(part) for part in assignment.split(counts, dim=0)))

        occupy_claim = vocab.masked_intensity(late_assign, claim_mask)
        occupy_full = vocab.masked_intensity(late_assign, attention_mask)
        occupy_demand = (
            demand
            if demand is not None
            else (
                vocab.claim_span_demand(late_assign, claim_token_mask)
                if any(claim_texts)
                else late_assign.new_zeros(late_assign.size(0), late_assign.size(-1))
            )
        )
        claim_pairs = self.relation_bundle(
            late_assign,
            claim_mask,
            vocab,
            demand=occupy_demand,
            living=living,
        )
        full_pairs = self.relation_bundle(
            late_assign,
            attention_mask,
            vocab,
            demand=occupy_demand,
            living=living,
        )
        claim_labeled, full_labeled = self.record_endpoint_seams(
            late_assign,
            claim_mask,
            attention_mask,
            claim_pairs,
            full_pairs,
        )
        _terms, mean_row_entropy, batch_usage = vocab.diversity_loss(
            late_assign,
            projected,
            token_mask=attention_mask,
        )
        relation_entropy = 0.0
        relation_usage = None
        if full_pairs.pair_count > 0 and full_pairs.mean_relation_entropy is not None:
            relation_entropy = float(full_pairs.mean_relation_entropy.detach().item())
            relation_usage = full_pairs.relation_batch_usage
        return CoveringInventory(
            n_entity_claim=trace_tensor(
                'claim_intensity',
                self.overlay_intensity(occupy_claim, claim_labeled, living, occupy_demand),
                'batch',
                'bank',
            ),
            n_entity_full=trace_tensor(
                'disclosure_intensity',
                self.overlay_intensity(occupy_full, full_labeled, living, occupy_demand),
                'batch',
                'bank',
            ),
            n_relation_claim=relation_mass(claim_pairs),
            n_relation_full=relation_mass(full_pairs),
            mean_row_entropy=mean_row_entropy,
            batch_usage=batch_usage,
            relation_row_entropy=relation_entropy,
            relation_batch_usage=relation_usage,
            claim_labeled=claim_labeled,
            full_labeled=full_labeled,
            claim_demand=occupy_demand,
        )


__all__ = [
    'ClaimSpanEncoder',
    'CoveringInventory',
    'Inventory',
    'SpanTokenizer',
]
