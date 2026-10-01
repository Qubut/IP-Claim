"""Saturation covering: unpaid claim demand against full-text supply.

Query intensity is claim-mask mass. Document intensity is attention-mask mass.
Presence is a per-slot saturation, not a simplex mean and not a count match.
Train-locked scales and optional keep rules live on the module as buffers,
not as learned weights. Default keep is dense.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn


class CoveringKnobs(BaseModel):
    """Saturation scales, relation mix, and optional inventory keep rules.

    Keep knobs are off by default (dense n). Slot and row keep stay unset
    on shipped eval YAML until they lock on train. Row keep sparsifies each
    token before the mask-weighted sum. Slot keep keeps the mass-carrying
    prefix of query demand. Document supply stays the full intensity so a
    weak disclosure is not dropped.
    """

    model_config = ConfigDict(frozen=True)

    sigma: float = Field(default=1.0, gt=0.0)
    sigma_edge: float = Field(default=1.0, gt=0.0)
    lambda_relation: float = Field(default=0.0, ge=0.0)
    row_top_k: int | None = Field(default=None, ge=1)
    row_mass_keep: float | None = Field(default=None, gt=0.0, le=1.0)
    slot_top_k: int | None = Field(default=None, ge=1)
    slot_mass_keep: float | None = Field(default=None, gt=0.0, le=1.0)


class CoveringScore(BaseModel):
    """Per-slot residual and paid mass plus the pair-level unpaid and covering."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    residual: Tensor
    paid: Tensor
    unpaid_mass: Tensor
    covering: Tensor
    demand_l1: Tensor


class CoveringTable(BaseModel):
    """Query-by-document unpaid and covering without a per-slot residual cube."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    unpaid_mass: Tensor
    covering: Tensor
    demand_l1: Tensor


class Covering(nn.Module):
    """Scores claim demand against full-text supply with locked saturation scales."""

    sigma: Tensor
    sigma_edge: Tensor
    lambda_relation: Tensor
    row_top_k: Tensor
    row_mass_keep: Tensor
    slot_top_k: Tensor
    slot_mass_keep: Tensor

    def __init__(self, knobs: CoveringKnobs) -> None:
        super().__init__()
        self.register_buffer('sigma', torch.tensor(knobs.sigma))
        self.register_buffer('sigma_edge', torch.tensor(knobs.sigma_edge))
        self.register_buffer('lambda_relation', torch.tensor(knobs.lambda_relation))
        self.register_buffer(
            'row_top_k',
            torch.tensor(-1 if knobs.row_top_k is None else knobs.row_top_k),
        )
        self.register_buffer(
            'row_mass_keep',
            torch.tensor(0.0 if knobs.row_mass_keep is None else knobs.row_mass_keep),
        )
        self.register_buffer(
            'slot_top_k',
            torch.tensor(-1 if knobs.slot_top_k is None else knobs.slot_top_k),
        )
        self.register_buffer(
            'slot_mass_keep',
            torch.tensor(0.0 if knobs.slot_mass_keep is None else knobs.slot_mass_keep),
        )

    def forward(self, n_query: Tensor, n_document: Tensor) -> CoveringScore:
        """Unpaid demand and covering from claim intensity versus full-text intensity."""
        return self._score(n_query, n_document, self.sigma)

    def relation_covering(self, n_query: Tensor, n_document: Tensor) -> CoveringScore:
        """Saturation covering on pair-graph intensities with the edge scale."""
        return self._score(n_query, n_document, self.sigma_edge)

    def pair_table(
        self,
        n_query: Tensor,
        n_document: Tensor,
        sigma: Tensor | None = None,
    ) -> CoveringTable:
        """Unpaid and covering as ``n_q @ (1 - f(n_d))^T``. No query-doc-slot cube."""
        queries = self.keep_slots(n_query)
        documents = n_document
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        if documents.ndim == 1:
            documents = documents.unsqueeze(0)
        placed = queries.to(
            device=documents.device,
            dtype=documents.dtype,
            non_blocking=documents.device.type == 'cuda',
        )
        gap = 1.0 - self.presence(documents, self.sigma if sigma is None else sigma)
        unpaid = placed @ gap.transpose(-2, -1)
        demand = placed.sum(dim=-1)
        covering = 1.0 - self._safe_ratio(unpaid, demand.unsqueeze(-1).expand_as(unpaid))
        return CoveringTable(unpaid_mass=unpaid, covering=covering, demand_l1=demand)

    def compose_windows(self, n_windows: Tensor, owners: Tensor, n_docs: int) -> Tensor:
        """One supply row per owner. Residual gaps multiply across windows.

        Presence is composed as a product of per-window gaps, then inverted
        through the hyperbolic map so ``pair_table`` sees the same presence.
        Empty owners stay unpaid. Window mass is not summed or max-pooled.
        """
        windows = n_windows if n_windows.ndim == 2 else n_windows.unsqueeze(0)
        slots = windows.size(-1)
        if windows.size(0) == 0 or n_docs < 1:
            return windows.new_zeros((max(n_docs, 0), slots))
        gap = (1.0 - self.presence(windows)).clamp(min=1e-12, max=1.0)
        composed = (
            windows
            .new_zeros((n_docs, slots))
            .index_add(
                0,
                owners.to(device=windows.device, dtype=torch.long),
                gap.log(),
            )
            .exp()
        )
        placed = self.sigma.to(device=windows.device, dtype=windows.dtype)
        return placed * (1.0 - composed) / composed

    def pair_tables(
        self,
        n_query: Tensor,
        document_shards: Sequence[Tensor],
        sigma: Tensor | None = None,
    ) -> CoveringTable:
        """Same pair table with the document axis split across devices."""
        shards = tuple(document_shards)
        if len(shards) == 1:
            return self.pair_table(n_query, shards[0], sigma)
        tables = tuple(self.pair_table(n_query, shard, sigma) for shard in shards)
        home = tables[0].unpaid_mass.device
        unpaid = torch.cat(tuple(table.unpaid_mass.to(home) for table in tables), dim=-1)
        demand = tables[0].demand_l1.to(home)
        covering = 1.0 - self._safe_ratio(unpaid, demand.unsqueeze(-1).expand_as(unpaid))
        return CoveringTable(unpaid_mass=unpaid, covering=covering, demand_l1=demand)

    def relation_table(self, n_query: Tensor, n_document: Tensor) -> CoveringTable:
        """Pair-graph covering table with the edge saturation scale."""
        return self.pair_table(n_query, n_document, self.sigma_edge)

    def relation_tables(
        self,
        n_query: Tensor,
        document_shards: Sequence[Tensor],
    ) -> CoveringTable:
        """Pair-graph table with the document axis split across devices."""
        return self.pair_tables(n_query, document_shards, self.sigma_edge)

    def unpaid_fraction(self, table: CoveringTable) -> Tensor:
        """Unpaid mass divided by demand. Empty demand is NaN, not a paid zero."""
        demand = table.demand_l1.unsqueeze(-1).expand_as(table.unpaid_mass)
        positive = demand > 0
        safe = torch.where(positive, demand, torch.ones_like(demand))
        return torch.where(
            positive,
            table.unpaid_mass / safe,
            torch.full_like(table.unpaid_mass, float('nan')),
        )

    def normalized_unpaid_gap(self, table: CoveringTable) -> Tensor:
        """Off-diagonal minus diagonal mean unpaid fraction on a square table.

        Positive when matching documents leave less unpaid demand than mismatches.
        Empty-demand rows do not count as paid; the gap is NaN when no finite
        diagonal remains.
        """
        unpaid = table.unpaid_mass
        if unpaid.ndim != 2 or unpaid.size(0) != unpaid.size(1):
            msg = 'normalized unpaid gap requires a square query-document table'
            raise ValueError(msg)
        frac = self.unpaid_fraction(table)
        order = unpaid.size(0)
        off = frac.masked_fill(
            torch.eye(order, dtype=torch.bool, device=frac.device),
            float('nan'),
        )
        return off.nanmean() - frac.diagonal().nanmean()

    def union_covering(
        self,
        n_query: Tensor,
        n_first: Tensor,
        n_second: Tensor,
    ) -> CoveringScore:
        """Covering when two documents pay the same demand together."""
        return self._score(n_query, n_first + n_second, self.sigma)

    def combination_residual(self, n_query: Tensor, partners: Tensor) -> Tensor:
        """Unpaid fraction of demand after partners join by residual-gap product.

        Partner rows stack on the first axis. Gaps multiply, then the
        hyperbolic map inverts that product so leftover unpaid matches
        owner-indexed window compose for a single owner. Empty demand is
        NaN.
        """
        stacked = partners if partners.ndim >= 2 else partners.unsqueeze(0)
        gap = (1.0 - self.presence(stacked)).clamp(min=1e-12, max=1.0)
        composed = gap.log().sum(dim=0).exp()
        scale = self.sigma.to(device=stacked.device, dtype=stacked.dtype)
        joined = scale * (1.0 - composed) / composed
        return self.unpaid_fraction(self.pair_table(n_query, joined))

    def reciprocal_score(self, n_query: Tensor, n_document: Tensor) -> CoveringScore:
        """Unpaid residual of forward covering times reverse covering.

        One-way presence can saturate every demanded slot on a long
        same-field disclosure. The product stays below one unless the claim
        also pays the document inventory.
        """
        forward = self._score(n_query, n_document, self.sigma)
        reverse = self._score(n_document, n_query, self.sigma)
        covering = forward.covering * reverse.covering
        unpaid = (1.0 - covering) * forward.demand_l1
        return CoveringScore(
            residual=forward.residual,
            paid=forward.paid,
            unpaid_mass=unpaid,
            covering=covering,
            demand_l1=forward.demand_l1,
        )

    def presence(self, intensity: Tensor, sigma: Tensor | None = None) -> Tensor:
        """Saturating presence ``t / (t + sigma)``. Zero at empty, one at infinity."""
        scale = self.sigma if sigma is None else sigma
        placed = scale.to(device=intensity.device, dtype=intensity.dtype)
        return intensity / (intensity + placed)

    def hard_presence(self, intensity: Tensor, epsilon: float) -> Tensor:
        """Ablation gate that pays a slot only when intensity meets ``epsilon``."""
        return (intensity >= epsilon).to(dtype=intensity.dtype)

    def keep_prefix(self, mass: Tensor, *, top_k: Tensor, mass_keep: Tensor) -> Tensor:
        """Zero coordinates outside the descending-mass prefix of the last axis."""
        k_cap = int(top_k.item())
        fraction = float(mass_keep.item())
        if k_cap < 1 and fraction <= 0.0:
            return mass
        values, order = mass.sort(dim=-1, descending=True)
        total = values.sum(dim=-1, keepdim=True)
        has_mass = total > 0
        nucleus = torch.ones_like(values, dtype=torch.bool)
        if fraction > 0.0:
            safe_total = torch.where(has_mass, total, torch.ones_like(total))
            previous = values.cumsum(dim=-1) - values
            nucleus = previous < fraction * safe_total
        if k_cap >= 1:
            ranks = torch.arange(mass.size(-1), device=mass.device, dtype=torch.long)
            nucleus &= ranks < k_cap
        kept = torch.where(has_mass & nucleus, values, torch.zeros_like(values))
        return torch.zeros_like(mass).scatter(-1, order, kept)

    def keep_rows(self, assignment: Tensor) -> Tensor:
        """Per-token keep on a late assignment matrix."""
        return self.keep_prefix(assignment, top_k=self.row_top_k, mass_keep=self.row_mass_keep)

    def keep_slots(self, intensity: Tensor) -> Tensor:
        """Per-filing keep on a summed intensity."""
        return self.keep_prefix(intensity, top_k=self.slot_top_k, mass_keep=self.slot_mass_keep)

    def masked_intensity(self, assignment: Tensor, token_mask: Tensor) -> Tensor:
        """Slot intensity as the mask-weighted assignment sum."""
        weights = token_mask.to(dtype=assignment.dtype)
        return torch.einsum('...l,...lk->...k', weights, self.keep_rows(assignment))

    def query_unpaid_field(
        self,
        assignment_query: Tensor,
        n_document: Tensor,
        n_query: Tensor | None = None,
    ) -> Tensor:
        """Token field whose mask-weighted sum is unpaid demand."""
        assignment = self.keep_rows(assignment_query)
        if n_query is not None:
            gate = (self.keep_slots(n_query) > 0).to(dtype=assignment.dtype)
            assignment = assignment.mul(gate.unsqueeze(-2))
        gap = 1.0 - self.presence(n_document)
        return torch.einsum('...lk,...k->...l', assignment, gap)

    def document_paying_field(
        self,
        assignment_document: Tensor,
        n_query: Tensor,
        n_document: Tensor,
    ) -> Tensor:
        """Token field whose mask-weighted sum is paid demand."""
        assignment = self.keep_rows(assignment_document)
        n_query = self.keep_slots(n_query)
        paid = n_query * self.presence(n_document)
        scale = self._safe_ratio(paid, n_document)
        return torch.einsum('...lk,...k->...l', assignment, scale)

    def alignment(
        self,
        assignment_query: Tensor,
        assignment_document: Tensor,
        paid: Tensor,
    ) -> Tensor:
        """Same-slot token pairing weighted by paid demand."""
        return torch.einsum(
            '...ik,...jk,...k->...ij',
            assignment_query,
            assignment_document,
            paid,
        )

    def paid_edges(self, query_edges: Tensor, document_edges: Tensor) -> Tensor:
        """Query pair-graph mass after document edge saturation."""
        return query_edges * self.presence(document_edges, self.sigma_edge)

    def edge_unpaid_fraction(self, query_edges: Tensor, document_edges: Tensor) -> Tensor:
        """Unpaid pair-graph mass after edge saturation, divided by query edge mass."""
        gap = 1.0 - self.presence(document_edges, self.sigma_edge)
        unpaid = (query_edges * gap).sum(dim=(-2, -1))
        demand = query_edges.sum(dim=(-2, -1))
        return self._safe_ratio(unpaid, demand)

    def _score(self, n_query: Tensor, n_document: Tensor, sigma: Tensor) -> CoveringScore:
        n_query = self.keep_slots(n_query)
        paid_fraction = self.presence(n_document, sigma)
        residual = n_query * (1.0 - paid_fraction)
        paid = n_query * paid_fraction
        unpaid_mass = residual.sum(dim=-1)
        demand_l1 = n_query.sum(dim=-1)
        covering = 1.0 - self._safe_ratio(unpaid_mass, demand_l1)
        return CoveringScore(
            residual=residual,
            paid=paid,
            unpaid_mass=unpaid_mass,
            covering=covering,
            demand_l1=demand_l1,
        )

    def _safe_ratio(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        """Quotient on positive denominators; zero elsewhere without a raw zero divide."""
        positive = denominator > 0
        safe_denominator = torch.where(positive, denominator, torch.ones_like(denominator))
        return torch.where(positive, numerator / safe_denominator, torch.zeros_like(numerator))
