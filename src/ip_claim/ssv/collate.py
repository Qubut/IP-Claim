"""MLM collation with optional graph-guided mask mixing.

Tokenizes patent text and builds MLM labels via HuggingFace
``DataCollatorForLanguageModeling`` corruption (80/10/10). When span masking
is on, starts are Bernoulli-sampled from the mix rates divided by expected
span length, and each start ``i`` covers positions ``j`` with
``0 <= j-i < length_i``. When
``rho > 0``, those rates mix uniform MLM with soft-assignment mass (primary)
and an optional CPC-literal auxiliary boost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Never, cast

import torch
from patent_ate.nlp import TextWindow
from patent_ate.spec import AteSpec
from pydantic import BaseModel, ConfigDict, Field, model_validator
from returns.maybe import Maybe
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import HeteroData
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.graph_ingress import host_language
from ip_claim.ssv.host_tokenizer import batch_encoding_tensor
from ip_claim.ssv.soft_vocab import SoftVocabModule

JATE_DRAW_CHAR_CAP = 200_000  # Noun-phrase priming; host MLM sees 1024 tokens.


class SoftMlmExample(BaseModel):
    """One patent MLM row with independent claim and disclosure views."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str
    graph: HeteroData
    claim_text: str = ''
    disclosure_text: str = ''


class SoftMlmBatch(BaseModel):
    """Collated MLM tensors with per-row HeteroData graphs."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    input_ids: Tensor
    """Host token ids after optional MLM corruption, shape ``(B, L)``."""

    unmasked_input_ids: Tensor
    """Host token ids before MLM corruption, shape ``(B, L)``."""

    attention_mask: Tensor
    labels: Tensor
    graphs: tuple[HeteroData, ...]
    texts: tuple[str, ...] = ()
    claim_texts: tuple[str, ...] = ()
    disclosure_texts: tuple[str, ...] = ()
    term_spans: tuple[tuple[Any, ...], ...] = ()
    termhood_delta: Any | None = None
    rho: float = Field(default=0.0, ge=0.0, le=1.0)
    require_supervised: bool = Field(
        default=True,
        description='When true, the batch must contain at least one MLM target.',
    )

    def pin_memory(self) -> SoftMlmBatch:
        """Pin host tensors so the next device copy can overlap compute."""
        return self.model_copy(
            update={
                'input_ids': self.input_ids.pin_memory(),
                'unmasked_input_ids': self.unmasked_input_ids.pin_memory(),
                'attention_mask': self.attention_mask.pin_memory(),
                'labels': self.labels.pin_memory(),
                'graphs': tuple(
                    graph.pin_memory() if hasattr(graph, 'pin_memory') else graph
                    for graph in self.graphs
                ),
            }
        )

    @model_validator(mode='after')
    def _supervised_mlm_contract(self) -> SoftMlmBatch:
        if self.require_supervised and not self.labels.ge(0).any():
            msg = 'MLM batch must contain at least one supervised position'
            raise ValueError(msg)
        return self


@dataclass
class SoftMlmCollator(DataCollatorForLanguageModeling):
    """Tokenize patent text and apply ``(1-rho)`` uniform + ``rho`` graph MLM."""

    max_length: int = 256
    rho: float = 0.0
    cpc_boost: float = 4.0
    cpc_aux_weight: float = 0.0
    assignment_top_k: int = 1
    mlm_span_mask: bool = True
    mlm_span_geo_p: float = 0.2
    mlm_span_max_length: int = 10
    soft_vocab: SoftVocabModule | None = None
    token_embed: nn.Module | None = None
    _assignment_mass_override: Tensor | None = field(default=None, repr=False)
    _cpc_token_id_cache: dict[tuple[str, ...], Tensor] = field(
        default_factory=dict,
        repr=False,
    )
    generator: torch.Generator | None = None
    prime_jate_spans: bool = False
    spacy_model: str | None = None
    _jate_nlp: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate knobs, require MLM mask token, run HF collator setup."""
        DataCollatorForLanguageModeling.__post_init__(self)  # type: ignore[no-untyped-call]
        rho_value = float(self.rho)
        boost = float(self.cpc_boost)
        aux = float(self.cpc_aux_weight)
        top_k = int(self.assignment_top_k)
        geo_p = float(self.mlm_span_geo_p)
        span_max = int(self.mlm_span_max_length)
        failures = (
            (not 0.0 <= rho_value <= 1.0, f'rho must be in [0, 1]; got {rho_value}'),
            (boost <= 0.0, f'cpc_boost must be > 0; got {boost}'),
            (not 0.0 <= aux <= 1.0, f'cpc_aux_weight must be in [0, 1]; got {aux}'),
            (top_k < 1, f'assignment_top_k must be >= 1; got {top_k}'),
            (not 0.0 < geo_p < 1.0, f'mlm_span_geo_p must be in (0, 1); got {geo_p}'),
            (span_max < 1, f'mlm_span_max_length must be >= 1; got {span_max}'),
            (
                bool(self.mlm) and not self._maybe_mask_token_id(),
                'MLM collation requires tokenizer.mask_token_id; use a masked-LM host tokenizer',
            ),
        )
        for failed, msg in failures:
            if failed:
                raise ValueError(msg)
        self.rho = rho_value
        self.max_length = int(self.max_length)
        self.cpc_boost = boost
        self.cpc_aux_weight = aux
        self.assignment_top_k = top_k
        self.mlm_span_mask = bool(self.mlm_span_mask)
        self.mlm_span_geo_p = geo_p
        self.mlm_span_max_length = span_max

    def set_rho(self, rho: float) -> None:
        """Update the graph-guided mask mixture weight."""
        rho_value = float(rho)
        if rho_value < 0.0 or rho_value > 1.0:
            msg = f'rho must be in [0, 1]; got {rho_value}'
            raise ValueError(msg)
        self.rho = rho_value

    def bind_assignment_source(
        self,
        soft_vocab: SoftVocabModule,
        token_embed: nn.Module,
    ) -> None:
        """Attach live soft vocab and host input embeddings for mass scoring."""
        self.soft_vocab = soft_vocab
        self.token_embed = token_embed

    def set_assignment_mass(self, mass: Tensor | None) -> None:
        """Override assignment mass ``(B, L)`` until replaced or cleared (tests / cache)."""
        self._assignment_mass_override = mass

    def _maybe_mask_token_id(self) -> Maybe[int]:
        """Host MASK id when it is a single vocabulary integer."""
        token_id = self.tokenizer.mask_token_id
        return Maybe.from_optional(token_id if isinstance(token_id, int) else None)

    def _mlm_rate(self) -> float:
        """Uniform MLM draw probability; missing host value is zero."""
        return float(Maybe.from_optional(self.mlm_probability).value_or(0.0))

    def __call__(
        self,
        examples: Sequence[SoftMlmExample | Mapping[str, Any]],
        return_tensors: str | None = None,
    ) -> SoftMlmBatch:
        """Collate examples into a SoftMlmBatch under the current ``rho`` mix."""
        del return_tensors

        def as_example(item: SoftMlmExample | Mapping[str, Any]) -> SoftMlmExample:
            if isinstance(item, SoftMlmExample):
                return item
            return SoftMlmExample.model_validate(item)

        rows = tuple(as_example(item) for item in examples)
        encoded = self.tokenizer(
            [row.text for row in rows],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors='pt',
        )

        unmasked_input_ids = batch_encoding_tensor(encoded, 'input_ids').clone()
        work_ids = unmasked_input_ids.clone()
        attention_mask = batch_encoding_tensor(encoded, 'attention_mask')
        special_tokens_mask = batch_encoding_tensor(encoded, 'special_tokens_mask').bool() | (
            attention_mask == 0
        )

        if self.mlm_span_mask and self._mlm_rate() > 0.0:
            probability_matrix = self._graph_mix_probability(
                unmasked_input_ids,
                special_tokens_mask,
                rows,
            )
            masked_ids, labels = self._mask_from_span_rates(
                work_ids,
                special_tokens_mask,
                probability_matrix,
            )
        elif self.rho <= 0.0:
            masked_ids, labels = self.torch_mask_tokens(
                work_ids,
                special_tokens_mask=special_tokens_mask,
            )
        else:
            probability_matrix = self._graph_mix_probability(
                unmasked_input_ids,
                special_tokens_mask,
                rows,
            )
            masked_ids, labels = self._mask_with_probability(
                work_ids,
                special_tokens_mask,
                probability_matrix,
            )

        mlm_on = self._mlm_rate() > 0.0
        masked_ids, labels = self._ensure_supervised(
            masked_ids,
            labels,
            unmasked_input_ids,
            special_tokens_mask,
            mlm_on=mlm_on,
        )
        return self._with_jate_spans(
            SoftMlmBatch(
                input_ids=masked_ids,
                unmasked_input_ids=unmasked_input_ids,
                attention_mask=attention_mask,
                labels=labels,
                graphs=tuple(row.graph for row in rows),
                texts=tuple(row.text for row in rows),
                claim_texts=tuple(row.claim_text for row in rows),
                disclosure_texts=tuple(row.disclosure_text for row in rows),
                rho=self.rho,
                require_supervised=mlm_on,
            )
        )

    def _with_jate_spans(self, batch: SoftMlmBatch) -> SoftMlmBatch:
        """Attach worker-side noun-phrase spans when priming is on."""
        if not self.prime_jate_spans:
            return batch

        if self._jate_nlp is None:
            self._jate_nlp = host_language(
                self.spacy_model or AteSpec().spacy_model,
                gpu=False,
            )
        drawn = TextWindow(
            origin=0,
            texts=tuple(text[:JATE_DRAW_CHAR_CAP] for text in batch.texts),
        ).drawn(
            self._jate_nlp,
            keep_spans=True,
        )
        return batch.model_copy(
            update={
                'term_spans': drawn.docs,
            }
        )

    def _ensure_supervised(
        self,
        masked_ids: Tensor,
        labels: Tensor,
        unmasked_input_ids: Tensor,
        special_tokens_mask: Tensor,
        *,
        mlm_on: bool,
    ) -> tuple[Tensor, Tensor]:
        """Guarantee one MASK target when MLM is on and the draw selected nothing."""
        if not mlm_on or labels.ge(0).any():
            return masked_ids, labels
        maskable = ~special_tokens_mask
        if not maskable.any():
            msg = 'MLM batch has no maskable token'
            raise ValueError(msg)
        mask_id = self._maybe_mask_token_id().unwrap()
        flat_idx = torch.multinomial(maskable.reshape(-1).to(dtype=torch.float), 1)
        row = torch.div(flat_idx, labels.size(1), rounding_mode='floor')
        col = flat_idx - row * labels.size(1)
        labels = labels.clone()
        masked_ids = masked_ids.clone()
        labels[row, col] = unmasked_input_ids[row, col]
        masked_ids[row, col] = mask_id
        return masked_ids, labels

    def _graph_mix_probability(
        self,
        input_ids: Tensor,
        special_tokens_mask: Tensor,
        rows: Sequence[SoftMlmExample],
    ) -> Tensor:
        """Build ``(1-rho)`` uniform + ``rho`` assignment-mass Bernoulli rates."""
        mlm_p = self._mlm_rate()
        device = input_ids.device
        probability_matrix = torch.full(
            input_ids.shape,
            mlm_p,
            dtype=torch.float,
            device=device,
        )
        _ = probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        if mlm_p <= 0.0 or self.rho <= 0.0:
            return probability_matrix

        assign_mass = self._resolve_assignment_mass(input_ids, special_tokens_mask)
        p_graph = _mass_to_mask_rates(
            assign_mass,
            special_tokens_mask,
            mlm_p=mlm_p,
        )
        if self.cpc_aux_weight > 0.0:
            p_cpc = self._cpc_aux_rates(input_ids, special_tokens_mask, rows, mlm_p=mlm_p)
            aux = self.cpc_aux_weight
            p_graph = (1.0 - aux) * p_graph + aux * p_cpc
        mixed = (1.0 - self.rho) * probability_matrix + self.rho * p_graph
        return mixed.clamp(0.0, 1.0)

    def _resolve_assignment_mass(
        self,
        input_ids: Tensor,
        special_tokens_mask: Tensor,
    ) -> Tensor:
        """Prefer override mass; else soft-assign token embeddings; else fail."""

        def moved(override: Tensor) -> Tensor:
            if override.shape != input_ids.shape:
                msg = (
                    f'assignment mass shape {tuple(override.shape)} does not match '
                    f'input_ids {tuple(input_ids.shape)}'
                )
                raise ValueError(msg)
            return override.to(device=input_ids.device, dtype=torch.float)

        def live_mass() -> Tensor:
            def missing() -> Never:
                msg = (
                    'rho > 0 requires soft-assignment mass: call bind_assignment_source '
                    'or set_assignment_mass before collating'
                )
                raise ValueError(msg)

            vocab, embed = Maybe.do(
                (soft_vocab, token_embed)
                for soft_vocab in Maybe.from_optional(self.soft_vocab)
                for token_embed in Maybe.from_optional(self.token_embed)
            ).or_else_call(missing)
            with torch.no_grad():
                embed_device = next(embed.parameters()).device
                embeds = cast(Tensor, embed(input_ids.to(embed_device)))
                assignment, _ = vocab.soft_assign(embeds)
                mass = cast(
                    Tensor,
                    vocab.assignment_mass(assignment, top_k=self.assignment_top_k),
                )
            return mass.masked_fill(special_tokens_mask.to(mass.device), 0.0).to(input_ids.device)

        return (
            Maybe.from_optional(self._assignment_mass_override).map(moved).or_else_call(live_mass)
        )

    def _cpc_aux_rates(
        self,
        input_ids: Tensor,
        special_tokens_mask: Tensor,
        rows: Sequence[SoftMlmExample],
        *,
        mlm_p: float,
    ) -> Tensor:
        """Secondary CPC-literal boost rates (vectorized; token-id cache)."""
        device = input_ids.device
        cpc_ids = self._padded_cpc_token_ids(rows, device=device)
        valid = cpc_ids.unsqueeze(1) != -1
        cpc_hit = ((input_ids.unsqueeze(-1) == cpc_ids.unsqueeze(1)) & valid).any(dim=-1)
        boost = torch.ones(input_ids.shape, dtype=torch.float, device=device)
        boost = torch.where(cpc_hit, torch.full_like(boost, self.cpc_boost), boost)
        return _mass_to_mask_rates(boost, special_tokens_mask, mlm_p=mlm_p)

    def _padded_cpc_token_ids(
        self,
        rows: Sequence[SoftMlmExample],
        *,
        device: torch.device,
    ) -> Tensor:
        """Per-row CPC token ids padded to ``(batch, max_ids)`` with ``-1``."""

        def ids_for_graph(graph: HeteroData) -> Tensor:
            raw_labels = graph['cpc'].get('label', ()) if 'cpc' in graph.node_types else ()
            pieces = tuple(
                sorted({str(label).strip() for label in raw_labels if str(label).strip()})
            )
            if not pieces:
                return torch.empty(0, dtype=torch.long, device=device)

            def compute() -> Tensor:
                token_ids = self.tokenizer.encode(
                    ' '.join(pieces),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )
                tensor = (
                    Maybe
                    .from_optional(token_ids or None)
                    .map(lambda ids: torch.tensor(sorted(set(ids)), dtype=torch.long))
                    .value_or(torch.empty(0, dtype=torch.long))
                )
                self._cpc_token_id_cache[pieces] = tensor
                return tensor.to(device=device)

            return (
                Maybe
                .from_optional(self._cpc_token_id_cache.get(pieces))
                .map(lambda cached: cached.to(device=device))
                .or_else_call(compute)
            )

        per_row = tuple(ids_for_graph(row.graph) for row in rows)
        if not per_row:
            return torch.empty(0, 0, dtype=torch.long, device=device)
        if all(ids.numel() == 0 for ids in per_row):
            return torch.full((len(per_row), 0), -1, dtype=torch.long, device=device)
        return pad_sequence(list(per_row), batch_first=True, padding_value=-1)

    def _mask_from_span_rates(
        self,
        inputs: Tensor,
        special_tokens_mask: Tensor,
        probability_matrix: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample Geo-length spans from start rates, then apply HF 80/10/10."""
        geo_p = float(self.mlm_span_geo_p)
        span_max = int(self.mlm_span_max_length)
        expected_length = (1.0 - (1.0 - geo_p) ** span_max) / geo_p
        start_rates = (probability_matrix / expected_length).clamp(0.0, 1.0)
        start_rates = start_rates.masked_fill(special_tokens_mask, 0.0)
        starts = self._finite_bernoulli(start_rates)

        unit = torch.rand(
            start_rates.shape,
            generator=self.generator,
            device=start_rates.device,
            dtype=start_rates.dtype,
        ).clamp(min=1e-6, max=1.0 - 1e-6)
        lengths = (
            1
            + torch.div(
                unit.log(),
                torch.log(unit.new_tensor(1.0 - geo_p)),
                rounding_mode='floor',
            ).long()
        )
        lengths = lengths.clamp(1, span_max)
        position = torch.arange(starts.size(1), device=starts.device)
        reach = position.unsqueeze(0) - position.unsqueeze(1)
        cover = (
            (starts.unsqueeze(-1) & (reach >= 0) & (reach < lengths.unsqueeze(-1)))
            .any(dim=1)
            .masked_fill(special_tokens_mask, False)
        )
        return self._mask_with_probability(inputs, special_tokens_mask, cover.float())

    def _mask_with_probability(
        self,
        inputs: Tensor,
        special_tokens_mask: Tensor,
        probability_matrix: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """HF-style Bernoulli MLM using a caller-supplied probability matrix.

        Mirrors ``DataCollatorForLanguageModeling.torch_mask_tokens`` corruption
        (``mask_replace_prob`` / ``random_replace_prob``) after custom rates.
        """
        labels = inputs.clone()
        probability_matrix = probability_matrix.masked_fill(special_tokens_mask, 0.0)
        masked_indices = self._finite_bernoulli(probability_matrix)
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(
                torch.full(labels.shape, self.mask_replace_prob, device=inputs.device),
                generator=self.generator,
            ).bool()
            & masked_indices
        )
        mask_id = self._maybe_mask_token_id().unwrap()
        inputs = inputs.clone()
        inputs[indices_replaced] = mask_id

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1.0 - self.mask_replace_prob
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob
        indices_random = (
            torch.bernoulli(
                torch.full(
                    labels.shape,
                    random_replace_prob_scaled,
                    device=inputs.device,
                ),
                generator=self.generator,
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long,
            generator=self.generator,
            device=inputs.device,
        )
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def _finite_bernoulli(self, rates: Tensor) -> Tensor:
        """Draw Bernoulli masks; non-finite or out-of-range rates fail closed."""
        invalid = ~torch.isfinite(rates) | rates.lt(0) | rates.gt(1)
        if invalid.any():
            msg = 'MLM Bernoulli rates must be finite and inside [0, 1]'
            raise RuntimeError(msg)
        return torch.bernoulli(rates, generator=self.generator).bool()


def _mass_to_mask_rates(
    mass: Tensor,
    special_tokens_mask: Tensor,
    *,
    mlm_p: float,
) -> Tensor:
    """Normalize non-negative mass into mean-preserving Bernoulli rates."""
    rates = mass.to(dtype=torch.float).masked_fill(special_tokens_mask, 0.0)
    maskable = (~special_tokens_mask).float()
    mean_mass = (rates * maskable).sum(dim=-1, keepdim=True) / maskable.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    scaled = mlm_p * (rates / mean_mass.clamp_min(1e-8))
    return scaled.masked_fill(special_tokens_mask, 0.0).clamp(0.0, 1.0)


def build_eval_collator(
    config: SsvTrainConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> SoftMlmCollator:
    """Build a zero-MLM collator for trunk export during collision eval."""
    return SoftMlmCollator(
        tokenizer,
        mlm_probability=0.0,
        max_length=int(config.arch.max_length),
        rho=0.0,
        cpc_boost=float(config.mlm.cpc_boost),
        cpc_aux_weight=float(config.mlm.cpc_aux_weight),
        assignment_top_k=int(config.mlm.assignment_top_k),
        mlm_span_mask=bool(config.mlm.mlm_span_mask),
        mlm_span_geo_p=float(config.mlm.mlm_span_geo_p),
        mlm_span_max_length=int(config.mlm.mlm_span_max_length),
    )


__all__ = [
    'SoftMlmBatch',
    'SoftMlmCollator',
    'SoftMlmExample',
    'build_eval_collator',
]
