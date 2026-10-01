"""Host LM with LoRA and soft-token ``inputs_embeds`` prefixing.

Composes soft-graph overlay, graph encode, host-width projection, entity and
relation soft vocab, and a PEFT host. Soft tokens prepend token embeddings;
MLM labels on those positions are ignored. Forward also exposes trunk exports
``z_d``, ``z_g``, and soft tokens.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Never, cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from pydantic import BaseModel, ConfigDict, Field
from returns.maybe import Maybe
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from transformers import AutoModelForMaskedLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.covering_trace import (
    capture_adapter_delta,
    overlay_edge_shift,
    trace_overlay,
    trace_tensor,
)
from ip_claim.ssv.dea import (
    EntityDenoiseHead,
    entity_denoise_terms,
    pack_token_rows_to_nodes,
    scatter_node_rows_to_tokens,
)
from ip_claim.ssv.encode import ComposeScoreReport, SoftGraphEncoder, row_rms
from ip_claim.ssv.graph_ingress import GraphIngress, OccupancyMap, empty_occupancy
from ip_claim.ssv.host_tokenizer import load_fast_host_tokenizer
from ip_claim.ssv.inventory import ClaimSpanEncoder
from ip_claim.ssv.project import SoftTokenProjector
from ip_claim.ssv.soft_graph import (
    SOFT_RELATED,
    build_soft_relation_bundle,
    merge_soft_overlay,
    pullback_compose_to_tokens,
)
from ip_claim.ssv.soft_vocab import DiversityTerms, SoftVocabModule, SoftVocabOutput

_LIVING_BLEND = 0.1


class TrunkExport(BaseModel):
    """Stable trunk tensors for downstream heads."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    z_d: Tensor
    """Patent text embedding ``(B, d_model)``."""

    z_g: Tensor
    """Graph embedding ``(B, d_model)``."""

    soft_tokens: Tensor
    """Host-width soft tokens ``(B, n_soft, d_model)``."""


class SoftTrunkOutput(BaseModel):
    """MLM, diversity, KE, entity denoise, and trunk exports for one training step."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mlm_loss: Tensor
    diversity_terms: DiversityTerms
    diversity_loss: Tensor
    ke_loss: Tensor
    soft_ke_loss: Tensor
    mean_row_entropy: Tensor
    inventory_entropy: Tensor
    occupancy: Tensor
    n_dead_before_restart: Tensor
    relation_occupancy: Tensor
    soft_vocab: SoftVocabOutput
    z_d: Tensor
    z_g: Tensor
    soft_tokens: Tensor
    mlm_preds: Tensor
    """Argmax token ids over the host sequence including soft prefix ``(B, T)``."""

    mlm_label_ids: Tensor
    """MLM labels with soft-prefix ``-100`` ignore positions ``(B, T)``."""

    mlm_token_nll: Tensor
    """Per-position masked-LM NLL over the soft-prefix-and-text sequence ``(B, T)``.

    Zero at ``-100``-labeled positions.
    """

    n_soft_tokens: int = Field(ge=1)
    compose_scores: ComposeScoreReport
    dea_loss: Tensor
    """Soft cross-entropy of remasked node states against the detached clean assignment."""

    dea_gap: Tensor
    """Hidden-position top-1 agreement minus the batch-modal teacher-code baseline."""

    text_hidden: Tensor
    """Last-layer host states on text tokens after the prefix."""

    def as_trunk_export(self) -> TrunkExport:
        """Project step outputs into the frozen trunk export contract."""
        return TrunkExport(z_d=self.z_d, z_g=self.z_g, soft_tokens=self.soft_tokens)


def masked_mean_pool(values: Tensor, mask: Tensor) -> Tensor:
    """Mean-pool a sequence over positions where ``mask`` is truthy.

    Rows with no active position return a zero vector instead of dividing by zero.
    """
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def mask_local_query(
    clean_embeds: Tensor,
    labels: Tensor,
    attention_mask: Tensor,
    *,
    window: int,
) -> Tensor:
    """Pool unmasked context around this draw's masked positions into one query per row.

    Every source position read here is outside the current MLM draw's masked set,
    so the result never reads content at a position it is trying to describe; it
    varies across draws only because different positions end up in or out of that
    excluded set. Rows with no masked position return a zero vector.
    """
    hidden_mask = labels.ge(0)
    context_mask = attention_mask.bool() & ~hidden_mask
    length = clean_embeds.size(1)
    positions = torch.arange(length, device=clean_embeds.device)
    near = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() <= window
    near = near.to(dtype=clean_embeds.dtype)
    context_values = clean_embeds * context_mask.unsqueeze(-1).to(dtype=clean_embeds.dtype)
    local_sum = torch.einsum('ij,bjd->bid', near, context_values)
    local_count = torch.einsum('ij,bj->bi', near, context_mask.to(dtype=clean_embeds.dtype))
    local_context = local_sum / local_count.clamp_min(1.0).unsqueeze(-1)
    return masked_mean_pool(local_context, hidden_mask)


def apply_overlay_edge_shift(graph: HeteroData, *, bank_size: int) -> HeteroData:
    """Roll overlay destination slots or relation vectors when a covering shift is active."""
    dest_shift, attr_shift = overlay_edge_shift()
    if dest_shift == 0 and attr_shift == 0:
        return graph
    store = graph[SOFT_RELATED]
    if dest_shift != 0 and store.edge_index.numel() > 0 and bank_size > 0:
        shifted = store.edge_index.clone()
        shifted[1] = (shifted[1] + dest_shift).remainder(bank_size)
        store.edge_index = shifted
    if attr_shift != 0 and store.edge_attr.numel() > 0:
        store.edge_attr = store.edge_attr.roll(attr_shift, dims=0)
    return graph


class SoftTrunkModel(nn.Module):
    """Encode overlay graphs, project soft tokens, run LoRA MLM with prefix embeds."""

    encoder: SoftGraphEncoder
    projector: SoftTokenProjector
    slot_projector: SoftTokenProjector
    soft_vocab: SoftVocabModule
    graph_ingress: GraphIngress
    dea_head: EntityDenoiseHead
    host: PreTrainedModel

    def __init__(
        self,
        config: SsvTrainConfig,
        *,
        encoder: SoftGraphEncoder,
        projector: SoftTokenProjector,
        slot_projector: SoftTokenProjector,
        soft_vocab: SoftVocabModule,
        host: PreTrainedModel,
    ) -> None:
        super().__init__()
        self._n_soft = int(config.arch.n_soft_tokens)
        self._soft_occupied_floor = float(config.arch.soft_occupied_floor)
        self._mask_query_window = int(config.arch.mask_query_window)
        self._inject_scale = 0.0
        self._inject_ramp_end = float(config.arch.inject_ramp_end)
        self._tokenizer_id = config.host.tokenizer_id or config.host.name
        self._host_tok: PreTrainedTokenizerBase | None = None
        self.encoder = encoder
        self.projector = projector
        self.slot_projector = slot_projector
        self.soft_vocab = soft_vocab
        self.graph_ingress = GraphIngress(
            self._tokenizer_id,
            spacy_model=config.ate.spacy_model,
        )
        if config.runtime.termhood_store_path is not None:
            self.graph_ingress.set_extra_state({
                'termhood_root': config.runtime.termhood_store_path
            })
        self.dea_head = EntityDenoiseHead(config)
        self.host = host
        slots = int(config.arch.entity_bank_size)
        self.living_pair_mass = nn.Buffer(torch.zeros(slots, slots))

    def living_snapshot(self) -> Tensor:
        """Other-filing leftover typed pair table for this step's overlay read."""
        return self.living_pair_mass.detach().clone()

    def absorb_living(self, this_filing: Tensor) -> None:
        """Blend this-filing kept pair mass into the persistable living table."""
        incoming = this_filing.detach()
        if incoming.ndim == 4:
            incoming = incoming.sum(dim=-1)
        if incoming.ndim == 3:
            incoming = incoming.mean(dim=0)
        if incoming.shape != self.living_pair_mass.shape:
            return
        self.living_pair_mass.mul_(1.0 - _LIVING_BLEND).add_(incoming, alpha=_LIVING_BLEND)

    def overlay_living(self, living: Tensor | None) -> Tensor:
        """Living pair table for this overlay read. Snapshot when none is passed."""
        return self.living_snapshot() if living is None else living

    def numbered_claim_demand(
        self,
        assignment: Tensor,
        attention_mask: Tensor,
        texts: Sequence[str],
        claim_texts: Sequence[str],
        claim_mask: Tensor | None = None,
    ) -> Tensor:
        """Slot demand from numbered-claim token spans. Not occupancy.

        Empty claim texts and no claim mask yield zeros. That is empty
        occupy, not a top-M fallback.
        """
        empty = assignment.new_zeros(assignment.size(0), assignment.size(-1))
        if not any(claim_texts) and claim_mask is None:
            return empty
        span = (
            claim_mask
            if claim_mask is not None
            else ClaimSpanEncoder(
                tokenizer=self.host_tokenizer(),
                max_length=int(assignment.size(1)),
            ).token_mask(tuple(texts), tuple(claim_texts), attention_mask)
        )
        return self.soft_vocab.claim_span_demand(assignment, span)

    @property
    def inject_scale(self) -> float:
        """Fixed residual scale of RMS-matched compose states at hidden slots."""
        return self._inject_scale

    @property
    def covering_inject_scale(self) -> float:
        """Residual scale on covering last-layer text tokens.

        Covering encode has no MLM hidden-slot schedule. The residual uses at
        least the configured ramp ceiling so filing identity reaches inventory
        ``n`` when the train hook still holds the scale at zero.
        """
        return max(self._inject_scale, self._inject_ramp_end)

    def mix_slot_residual(
        self,
        text_embeds: Tensor,
        packed_states: Tensor,
        packed_mask: Tensor,
        attention_mask: Tensor,
        *,
        scale: float,
        slot_mask: Tensor | None = None,
        zero_inject: bool = False,
    ) -> Tensor:
        """Add an RMS-matched compose residual onto selected text embeddings.

        The host embedding stays in the sum. The projected graph state is
        scaled to the host RMS at that position, then multiplied by ``scale``.
        ``slot_mask`` selects positions; omit it to mix every text token.
        """
        projected = self.slot_projector(packed_states).tokens
        slots = scatter_node_rows_to_tokens(
            projected,
            attention_mask,
            packed_mask,
        )
        aligned = slots * (row_rms(text_embeds) / row_rms(slots))
        aligned = torch.zeros_like(aligned) if zero_inject else aligned
        residual = trace_tensor(
            'token_residual',
            text_embeds.new_tensor(scale) * aligned,
            'batch',
            'token',
            'hidden',
        )
        mixed = text_embeds + residual
        if slot_mask is None:
            return mixed
        return cast(Tensor, torch.where(slot_mask.unsqueeze(-1), mixed, text_embeds))

    def set_inject_scale(self, scale: float) -> None:
        """Set the scheduled hidden-slot residual scale.

        The scale is a fixed ramp from the train hook. It is not a parameter.
        """
        scale_value = float(scale)
        if scale_value < 0.0 or scale_value > 1.0:
            msg = f'inject scale must be in [0, 1]; got {scale_value}'
            raise ValueError(msg)
        self._inject_scale = scale_value

    @property
    def n_soft_tokens(self) -> int:
        """Soft-token prefix length prepended to host embeddings."""
        return self._n_soft

    @property
    def soft_occupied_floor(self) -> float:
        """Minimum occupancy for an entity-bank row to enter the overlay."""
        return self._soft_occupied_floor

    def host_tokenizer(self) -> PreTrainedTokenizerBase:
        """Fast tokenizer whose encode matches host ``input_ids`` for offset mapping."""
        if self._host_tok is None:
            self._host_tok = load_fast_host_tokenizer(self._tokenizer_id)
        return self._host_tok

    def occupy_assignment(
        self,
        assignment: Tensor,
        live_mask: Tensor,
        input_ids: Tensor,
        texts: Sequence[str] = (),
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        update_stats: bool | None = None,
    ) -> OccupancyMap:
        """Score noun-phrase heads and pool assignment onto those positions."""
        resolved = tuple(texts)
        if not resolved:
            return empty_occupancy(assignment)
        encoder = tokenizer if tokenizer is not None else self.host_tokenizer()
        offsets = self.graph_ingress.token_offsets(
            resolved,
            max_length=int(assignment.size(1)),
            device=assignment.device,
            input_ids=input_ids,
            tokenizer=encoder,
        )
        return self.graph_ingress.occupy(
            assignment,
            live_mask,
            resolved,
            offsets,
            update_stats=self.training if update_stats is None else update_stats,
        )

    def forward(
        self,
        graphs: HeteroData | Sequence[HeteroData],
        input_ids: Tensor,
        unmasked_input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        texts: Sequence[str] = (),
        claim_texts: Sequence[str] = (),
        living: Tensor | None = None,
        *,
        zero_prefix: bool = False,
        zero_inject: bool = False,
    ) -> SoftTrunkOutput:
        """Run soft-graph conditioned MLM, diversity, KE, and trunk exports.

        Assignment uses unmasked token embeddings. Hidden MLM targets do not
        occupy overlay codes. The host still reads corrupted embeddings.
        """
        graph_list = (graphs,) if isinstance(graphs, HeteroData) else tuple(graphs)
        embed = self.host.get_input_embeddings()
        clean_embeds = embed(unmasked_input_ids)
        masked_embeds = embed(input_ids)
        text_query = masked_mean_pool(clean_embeds, attention_mask) + mask_local_query(
            clean_embeds,
            labels,
            attention_mask,
            window=self._mask_query_window,
        )
        self.soft_vocab.ensure_banks_seeded(clean_embeds)
        early_assign, _ = self.soft_vocab.soft_assign(clean_embeds)
        early_assign = trace_tensor(
            'early_assignment',
            early_assign,
            'batch',
            'token',
            'bank',
        )
        visible = attention_mask * (~labels.ge(0)).to(dtype=attention_mask.dtype)
        occupied = self.occupy_assignment(
            early_assign,
            visible,
            unmasked_input_ids,
            texts,
        )
        termhood = trace_tensor(
            'termhood_weights',
            occupied.weights,
            'batch',
            'token',
        )
        assignment = trace_tensor(
            'termhood_assignment',
            occupied.assignment,
            'batch',
            'token',
            'bank',
        )
        live = self.overlay_living(living)
        demand = self.numbered_claim_demand(
            assignment,
            visible,
            texts,
            claim_texts,
        )
        soft_bundle = build_soft_relation_bundle(
            self.soft_vocab,
            assignment,
            termhood,
            mass_floor=self._soft_occupied_floor,
            demand=demand,
            living=live,
        )
        explain = importlib.import_module('ip_claim.collision.explain').Explain()
        self.absorb_living(explain.graphs_from_bundle(assignment, termhood, soft_bundle))
        trace_overlay('overlay_state', soft_bundle.overlays)
        bank_size = int(self.soft_vocab.entity_bank.size(0))
        merged = tuple(
            apply_overlay_edge_shift(
                merge_soft_overlay(gifted, overlay, bank_size=bank_size),
                bank_size=bank_size,
            )
            for gifted, overlay in zip(graph_list, soft_bundle.overlays, strict=True)
        )
        encoded = self.encoder(merged, text_query=text_query)
        token_states = pullback_compose_to_tokens(
            encoded.node_states,
            encoded.node_mask,
            soft_bundle.overlays,
            occupied.assignment,
        )
        packed_states, packed_mask = pack_token_rows_to_nodes(token_states, attention_mask)
        denoise = entity_denoise_terms(
            self.dea_head,
            packed_states,
            packed_mask,
            self.soft_vocab.entity_bank,
            early_assign.detach(),
            labels,
            attention_mask,
        )
        soft = encoded.soft_tokens
        prefix = trace_tensor(
            'projected_prefix',
            self.projector(soft).tokens,
            'batch',
            'slot',
            'hidden',
        )
        prefix = torch.zeros_like(prefix) if zero_prefix else prefix
        inputs_embeds = torch.cat(
            [
                prefix,
                self.mix_slot_residual(
                    masked_embeds,
                    packed_states,
                    packed_mask,
                    attention_mask,
                    scale=self._inject_scale,
                    slot_mask=labels.ge(0),
                    zero_inject=zero_inject,
                ),
            ],
            dim=1,
        )

        soft_ones = torch.ones(
            attention_mask.size(0),
            self._n_soft,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        soft_ignore = torch.full(
            (labels.size(0), self._n_soft),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        host_mask = torch.cat([soft_ones, attention_mask], dim=1)
        host_labels = torch.cat([soft_ignore, labels], dim=1)

        def missing_mlm() -> Never:
            msg = 'Host MLM forward returned no loss; labels are required'
            raise RuntimeError(msg)

        def host_mlm(embeds: Tensor, mask: Tensor, lab: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            host_out = self.host(
                inputs_embeds=embeds,
                attention_mask=mask,
                labels=lab,
                output_hidden_states=True,
            )

            def missing_hidden() -> Never:
                msg = 'Host MLM forward returned no hidden states'
                raise RuntimeError(msg)

            hidden = Maybe.from_optional(host_out.hidden_states).or_else_call(missing_hidden)
            loss = Maybe.from_optional(host_out.loss).or_else_call(missing_mlm)
            return host_out.logits, loss, hidden[-1]

        with capture_adapter_delta(self.host):
            logits, mlm_loss, last_hidden = host_mlm(inputs_embeds, host_mask, host_labels)
        text_hidden = trace_tensor(
            'host_text_state',
            last_hidden[:, self._n_soft :, :],
            'batch',
            'token',
            'hidden',
        )
        vocab = self.soft_vocab(text_hidden, token_mask=attention_mask)
        _ = trace_tensor(
            'late_assignment',
            vocab.assignment,
            'batch',
            'token',
            'bank',
        )
        rel_vq = Maybe.from_optional(soft_bundle.relation_div_loss).value_or(
            vocab.diversity_terms.rel_vq
        )
        diversity_terms = DiversityTerms(
            vq=vocab.diversity_terms.vq,
            inventory=vocab.diversity_terms.inventory,
            usage_kl=vocab.diversity_terms.usage_kl,
            gap=vocab.diversity_terms.gap,
            rel_vq=rel_vq,
        )
        diversity_loss = self.soft_vocab.combine_diversity(diversity_terms)
        soft_ke_loss = soft_bundle.soft_ke_loss
        ke_loss = encoded.ke_loss + soft_ke_loss

        n_supervised = int(labels.ge(0).sum())
        if n_supervised == 0 or not torch.isfinite(mlm_loss):
            msg = (
                'Host MLM loss is non-finite or has no supervised positions '
                f'(n_supervised={n_supervised})'
            )
            raise RuntimeError(msg)
        mlm_label_ids = torch.cat([soft_ignore, labels], dim=1)
        mlm_preds = logits.argmax(dim=-1)
        mlm_token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            mlm_label_ids,
            ignore_index=-100,
            reduction='none',
        )

        z_d = masked_mean_pool(text_hidden, attention_mask)
        z_g = prefix.mean(dim=1)

        n_relation_restarts = Maybe.from_optional(soft_bundle.n_relation_restarts).value_or(
            vocab.n_entity_restarts.new_zeros(())
        )

        def occupancy_pair() -> tuple[Tensor, Tensor]:
            return self.soft_vocab.occupancy_and_dead(self.soft_vocab.relation_usage_ema)

        relation_occupancy, n_relation_dead = Maybe.do(
            (occ, dead)
            for occ in Maybe.from_optional(soft_bundle.relation_occupancy)
            for dead in Maybe.from_optional(soft_bundle.n_relation_dead_before_restart)
        ).or_else_call(occupancy_pair)
        n_restarts = vocab.n_entity_restarts + n_relation_restarts
        soft_vocab_out = SoftVocabOutput(
            assignment=vocab.assignment,
            projected=vocab.projected,
            soft_entities=vocab.soft_entities,
            diversity_terms=diversity_terms,
            diversity_loss=diversity_loss,
            mean_row_entropy=vocab.mean_row_entropy,
            inventory_entropy=vocab.inventory_entropy,
            batch_usage=vocab.batch_usage,
            occupancy=vocab.occupancy,
            n_dead_before_restart=vocab.n_dead_before_restart,
            usage_entropy=vocab.usage_entropy,
            inverse_simpson=vocab.inverse_simpson,
            relation_assignment=soft_bundle.relation_assignment,
            soft_relations=soft_bundle.soft_relations,
            relation_batch_usage=soft_bundle.relation_batch_usage,
            mean_relation_entropy=soft_bundle.mean_relation_entropy,
            relation_occupancy=relation_occupancy,
            n_relation_dead_before_restart=n_relation_dead,
            n_entity_restarts=vocab.n_entity_restarts,
            n_relation_restarts=n_relation_restarts,
            n_restarts=n_restarts,
        )

        return SoftTrunkOutput(
            mlm_loss=mlm_loss,
            diversity_terms=diversity_terms,
            diversity_loss=diversity_loss,
            ke_loss=ke_loss,
            soft_ke_loss=soft_ke_loss,
            mean_row_entropy=vocab.mean_row_entropy,
            inventory_entropy=vocab.inventory_entropy,
            occupancy=vocab.occupancy,
            n_dead_before_restart=vocab.n_dead_before_restart,
            relation_occupancy=relation_occupancy,
            soft_vocab=soft_vocab_out,
            z_d=z_d,
            z_g=z_g,
            soft_tokens=prefix,
            mlm_preds=mlm_preds,
            mlm_label_ids=mlm_label_ids,
            mlm_token_nll=mlm_token_nll,
            n_soft_tokens=self._n_soft,
            compose_scores=encoded.compose_scores,
            dea_loss=denoise.dea_loss,
            dea_gap=denoise.dea_gap,
            text_hidden=text_hidden,
        )

    def export_trunk(
        self,
        graphs: HeteroData | Sequence[HeteroData],
        input_ids: Tensor,
        attention_mask: Tensor,
        texts: Sequence[str] = (),
        claim_texts: Sequence[str] = (),
        living: Tensor | None = None,
        claim_mask: Tensor | None = None,
    ) -> TrunkExport:
        """Encode without MLM labels and return the trunk export contract."""
        graph_list = (graphs,) if isinstance(graphs, HeteroData) else tuple(graphs)
        token_embeds = self.host.get_input_embeddings()(input_ids)
        self.soft_vocab.ensure_banks_seeded(token_embeds)
        early_assign, _ = self.soft_vocab.soft_assign(token_embeds)
        early_assign = trace_tensor(
            'early_assignment',
            early_assign,
            'batch',
            'token',
            'bank',
        )
        occupied = self.occupy_assignment(
            early_assign,
            attention_mask,
            input_ids,
            texts,
            update_stats=False,
        )
        termhood = trace_tensor(
            'termhood_weights',
            occupied.weights,
            'batch',
            'token',
        )
        assignment = trace_tensor(
            'termhood_assignment',
            occupied.assignment,
            'batch',
            'token',
            'bank',
        )
        live = self.overlay_living(living)
        demand = self.numbered_claim_demand(
            assignment,
            attention_mask,
            texts,
            claim_texts,
            claim_mask,
        )
        soft_bundle = build_soft_relation_bundle(
            self.soft_vocab,
            assignment,
            termhood,
            mass_floor=self._soft_occupied_floor,
            demand=demand,
            living=live,
        )
        trace_overlay('overlay_state', soft_bundle.overlays)
        bank_size = int(self.soft_vocab.entity_bank.size(0))
        merged = tuple(
            apply_overlay_edge_shift(
                merge_soft_overlay(gifted, overlay, bank_size=bank_size),
                bank_size=bank_size,
            )
            for gifted, overlay in zip(graph_list, soft_bundle.overlays, strict=True)
        )
        encoded = self.encoder(merged)
        token_states = pullback_compose_to_tokens(
            encoded.node_states,
            encoded.node_mask,
            soft_bundle.overlays,
            occupied.assignment,
        )
        packed_states, packed_mask = pack_token_rows_to_nodes(token_states, attention_mask)
        prefix = trace_tensor(
            'projected_prefix',
            self.projector(encoded.soft_tokens).tokens,
            'batch',
            'slot',
            'hidden',
        )
        mixed_embeds = self.mix_slot_residual(
            token_embeds,
            packed_states,
            packed_mask,
            attention_mask,
            scale=self.covering_inject_scale,
        )
        inputs_embeds = torch.cat([prefix, mixed_embeds], dim=1)
        soft_ones = torch.ones(
            attention_mask.size(0),
            self._n_soft,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        with capture_adapter_delta(self.host):
            host_out = self.host(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.cat([soft_ones, attention_mask], dim=1),
                output_hidden_states=True,
            )
        text_hidden = trace_tensor(
            'host_text_state',
            host_out.hidden_states[-1][:, self._n_soft :, :],
            'batch',
            'token',
            'hidden',
        )
        z_d = masked_mean_pool(text_hidden, attention_mask)
        return TrunkExport(
            z_d=z_d,
            z_g=prefix.mean(dim=1),
            soft_tokens=prefix,
        )


def build_lora_host(config: SsvTrainConfig) -> PeftModel:
    """Load a masked LM and wrap it with LoRA adapters."""
    token = None
    if config.runtime.hf_token is not None:
        token = config.runtime.hf_token.get_secret_value().strip() or None
    base = AutoModelForMaskedLM.from_pretrained(config.host.name, token=token)
    if int(base.config.hidden_size) != int(config.host.d_model):
        msg = (
            f'host.d_model={config.host.d_model} does not match '
            f'{config.host.name} hidden_size={base.config.hidden_size}'
        )
        raise ValueError(msg)
    lora = LoraConfig(
        r=int(config.arch.lora_r),
        lora_alpha=int(config.arch.lora_alpha),
        lora_dropout=float(config.arch.lora_dropout),
        bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=list(config.host.lora_target_modules),
    )
    checkpoint_kwargs = {'use_reentrant': False}
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
    wrapped = get_peft_model(base, lora)
    host = cast(PreTrainedModel, cast(object, wrapped))
    host.enable_input_require_grads()
    host.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
    return cast(PeftModel, wrapped)


def build_soft_trunk(config: SsvTrainConfig) -> SoftTrunkModel:
    """Construct encoder, projector, soft vocab, and LoRA host for one config."""
    return SoftTrunkModel(
        config,
        encoder=SoftGraphEncoder(config),
        projector=SoftTokenProjector(config),
        slot_projector=SoftTokenProjector(config),
        soft_vocab=SoftVocabModule(config),
        host=cast(PreTrainedModel, cast(object, build_lora_host(config))),
    )


__all__ = [
    'SoftTrunkModel',
    'SoftTrunkOutput',
    'TrunkExport',
    'build_lora_host',
    'build_soft_trunk',
    'mask_local_query',
    'masked_mean_pool',
]
