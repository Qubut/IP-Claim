"""Soft-assign host states onto entity and relation banks and score diversity.

Owns entity/relation bank parameters, temperature softmax assignment, a
consumed refuse outcome on pairs, VQ codebook and commitment terms on the
entity side, DDP-synced usage EMA buffers, and dead-row restart from the
current batch.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from ip_claim.ssv.config import SsvTrainConfig


class DiversityTerms(BaseModel):
    """Unscaled diversity pieces. Dual weights are applied in the Lightning step."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    vq: Tensor
    """Entity codebook plus commitment (unit scale)."""

    inventory: Tensor
    """Mean Shannon entropy of per-filing assignment mass (nats)."""

    usage_kl: Tensor
    """KL of mean column usage to uniform."""

    gap: Tensor
    """MAGVIT token-vs-usage entropy gap, warmup-scaled."""

    rel_vq: Tensor
    """Relation reconstruction, or zero when none is scored."""


class SoftVocabOutput(BaseModel):
    """One forward pass of soft assignment, reconstruction, and diversity."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    assignment: Tensor
    """Row-stochastic entity soft assignment ``(..., K_e)``."""

    projected: Tensor
    """Host states projected into bank space ``(..., soft_dim)``."""

    soft_entities: Tensor
    """Soft reconstruction ``assignment @ entity_bank`` ``(..., soft_dim)``."""

    diversity_terms: DiversityTerms
    """Unscaled reconstruction and entropy pieces for this batch."""

    diversity_loss: Tensor
    """Reconstruction sum (entity VQ plus relation VQ)."""

    mean_row_entropy: Tensor
    """Mean Shannon entropy of assignment rows (nats)."""

    inventory_entropy: Tensor
    """Mean Shannon entropy of per-filing assignment mass (nats)."""

    batch_usage: Tensor
    """Mean column usage ``(K_e,)`` over the batch."""

    occupancy: Tensor
    """Fraction of entity slots whose usage EMA met ``utilization_eps`` before restart."""

    n_dead_before_restart: Tensor
    """Entity rows below ``utilization_eps`` before this step's restart."""

    usage_entropy: Tensor
    """Shannon entropy of batch-mean entity usage (nats)."""

    inverse_simpson: Tensor
    """Inverse Simpson of batch-mean entity usage."""

    relation_assignment: Tensor | None = None
    """Typed leftover relation mass ``(..., K_r)`` when pairs exist."""

    soft_relations: Tensor | None = None
    """Soft relation vectors ``(..., soft_dim)`` when pairs exist."""

    relation_batch_usage: Tensor | None = None
    """Mean relation column usage ``(K_r,)`` when pairs exist."""

    mean_relation_entropy: Tensor | None = None
    """Mean Shannon entropy of relation assignment rows when pairs exist."""

    relation_occupancy: Tensor | None = None
    """Fraction of relation slots whose usage EMA met ``utilization_eps`` before restart."""

    n_relation_dead_before_restart: Tensor | None = None
    """Relation rows below ``utilization_eps`` before this step's restart."""

    n_entity_restarts: Tensor
    """Entity bank rows replaced this step because usage EMA was below eps."""

    n_relation_restarts: Tensor
    """Relation bank rows replaced this step because usage EMA was below eps."""

    n_restarts: Tensor
    """``n_entity_restarts + n_relation_restarts``."""


class RelationAssignment(BaseModel):
    """Joint type-or-refuse softmax on one pair table.

    Typed leftover is the mass that may occupy a kept edge. Refuse mass is a
    consumed non-edge: it is not a relation-bank row and does not add pair
    mass. Soft while the bank is young; this envelope does not drop rows.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    typed: Tensor
    """Leftover typed mass ``(..., K_r)``. Excludes refuse."""

    consumed: Tensor
    """Refuse mass ``(...)``. Soft non-edge; not a relation-bank row."""

    projected: Tensor
    """Pair features ``(..., soft_dim)``."""

    def kept_pair_mass(self) -> Tensor:
        """Typed leftover per pair. Refuse mass does not add."""
        return self.typed.sum(dim=-1)

    def is_consumed(self) -> Tensor:
        """True where refuse is the unique winner of the joint softmax."""
        return self.consumed > self.typed.amax(dim=-1)


class SoftVocabModule(nn.Module):
    """Entity and relation soft banks, type-or-refuse assign, and ``L_div``."""

    entity_usage_ema: Tensor
    relation_usage_ema: Tensor
    _banks_seeded: Tensor

    def __init__(self, config: SsvTrainConfig) -> None:
        super().__init__()
        self._entity_bank_size = int(config.arch.entity_bank_size)
        self._relation_bank_size = int(config.arch.relation_bank_size)
        self._soft_dim = int(config.arch.soft_dim)
        self._temperature = float(config.arch.assign_temperature)
        self._relation_temperature = float(config.arch.relation_temperature)
        self._normalize_assignment_entropy = bool(config.dual.normalize_assignment_entropy)
        self._bank_init_from_batch = bool(config.bank.bank_init_from_batch)
        self._bank_init_kmeans_iters = int(config.bank.bank_init_kmeans_iters)
        self._entropy_scale = 1.0
        self._utilization_eps = float(config.bank.utilization_eps)
        self._log_eps = float(config.dual.log_eps)
        self._usage_ema_momentum = float(config.bank.usage_ema_momentum)

        self.project = nn.Linear(config.host.d_model, config.arch.soft_dim, bias=False)
        self.pair_project = nn.Linear(2 * config.arch.soft_dim, config.arch.soft_dim, bias=False)
        self.entity_bank = nn.Parameter(torch.empty(self._entity_bank_size, self._soft_dim))
        self.relation_bank = nn.Parameter(torch.empty(self._relation_bank_size, self._soft_dim))
        self.refuse_probe = nn.Parameter(torch.empty(self._soft_dim))
        self.entity_dmask = nn.Parameter(torch.empty(self._soft_dim))
        _ = nn.init.normal_(self.entity_bank, std=0.02)
        _ = nn.init.normal_(self.relation_bank, std=0.02)
        _ = nn.init.normal_(self.refuse_probe, std=0.02)
        _ = nn.init.normal_(self.entity_dmask, std=0.02)
        _ = nn.init.normal_(self.project.weight, std=0.02)
        _ = nn.init.normal_(self.pair_project.weight, std=0.02)

        self.register_buffer(
            'entity_usage_ema',
            torch.full((self._entity_bank_size,), 1.0 / self._entity_bank_size),
            persistent=True,
        )
        self.register_buffer(
            'relation_usage_ema',
            torch.full((self._relation_bank_size,), 1.0 / self._relation_bank_size),
            persistent=True,
        )
        self.register_buffer('_banks_seeded', torch.zeros((), dtype=torch.bool), persistent=True)

    @property
    def entity_bank_size(self) -> int:
        """Number of learnable entity soft-vocab slots."""
        return self._entity_bank_size

    @property
    def relation_bank_size(self) -> int:
        """Number of learnable relation soft-vocab slots."""
        return self._relation_bank_size

    def soft_assign(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Map host states to row-stochastic entity assignment and bank projection.

        Args:
            states: Token or span states ``(..., d_model)``.

        Returns:
            Assignment ``(..., K_e)`` and projected states ``(..., soft_dim)``.
        """
        projected = self.project(states)
        queries = F.normalize(projected, dim=-1)
        codes = F.normalize(self.entity_bank, dim=-1)
        logits = torch.einsum('...d,kd->...k', queries, codes)
        assignment = F.softmax(logits / self._temperature, dim=-1)
        return assignment, projected

    def soft_entities(self, assignment: Tensor) -> Tensor:
        """Soft entity embeddings as convex combinations of the entity bank."""
        return torch.einsum('...k,kd->...d', assignment, self.entity_bank)

    def assignment_mass(self, assignment: Tensor, top_k: int = 1) -> Tensor:
        """Peakiness score: sum of the top-``top_k`` entity assignment masses.

        Args:
            assignment: Row-stochastic entity assignment ``(..., K_e)``.
            top_k: Number of leading masses to sum; clamped to ``K_e``.

        Returns:
            Peakiness score with the assignment axis reduced away.
        """
        k = min(int(top_k), int(assignment.size(-1)))
        return assignment.topk(k, dim=-1).values.sum(dim=-1)

    def remask_entities(self, entities: Tensor, hide: Tensor) -> Tensor:
        """Replace hidden-token entity reconstructions with the learnable remask vector.

        ``hide`` is a boolean mask on the token axis. Visible tokens keep their
        assigned codes; hidden tokens become ``entity_dmask``.
        """
        return torch.where(hide.unsqueeze(-1), self.entity_dmask, entities)

    def pair_features(self, heads: Tensor, tails: Tensor) -> Tensor:
        """Project concatenated soft entity pairs into relation-bank space."""
        return cast(Tensor, self.pair_project(torch.cat([heads, tails], dim=-1)))

    def soft_assign_relations(self, pair_states: Tensor) -> RelationAssignment:
        """Map pair features to typed leftover and consumed refuse.

        The joint softmax is over the relation bank plus one refuse outcome.
        Refuse is not a bank row. Typed leftover is not renormalized, so a
        refused pair does not still occupy a type.

        Args:
            pair_states: Pair features ``(..., soft_dim)`` (already projected).

        Returns:
            Typed leftover, refuse mass, and the same pair states.
        """
        queries = F.normalize(pair_states, dim=-1)
        codes = F.normalize(self.relation_bank, dim=-1)
        refuse = F.normalize(self.refuse_probe, dim=-1)
        typed_logits = torch.einsum('...d,kd->...k', queries, codes)
        refuse_logit = torch.einsum('...d,d->...', queries, refuse).unsqueeze(-1)
        joint = F.softmax(
            torch.cat((typed_logits, refuse_logit), dim=-1) / self._relation_temperature,
            dim=-1,
        )
        return RelationAssignment(
            typed=joint[..., :-1],
            consumed=joint[..., -1],
            projected=pair_states,
        )

    def soft_relations(self, assignment: Tensor) -> Tensor:
        """Soft relation embeddings from typed leftover mass.

        ``assignment`` is leftover typed mass ``(..., K_r)``, not the joint
        including refuse. Refuse mass does not mix a bank row.
        """
        return torch.einsum('...k,kd->...d', assignment, self.relation_bank)

    @property
    def entropy_scale(self) -> float:
        """Live entropy-term scale in ``[0, 1]`` after the warmup pin or ramp."""
        return float(self._entropy_scale)

    def set_entropy_scale(self, scale: float) -> None:
        """Set the live entropy-term scale in ``[0, 1]`` (warmup schedule)."""
        self._entropy_scale = float(scale)

    def ensure_banks_seeded(self, states: Tensor) -> None:
        """Seed banks from projected ``states`` once, on the first training call."""
        if not self.training or not self._bank_init_from_batch or bool(self._banks_seeded.item()):
            return
        self.seed_banks_from_projected(self.project(states.detach()))
        _ = self._banks_seeded.fill_(True)

    def seed_banks_from_projected(self, projected: Tensor) -> None:
        """Replace bank rows with k-means centroids of the first projected batch.

        Entity rows come from flattened projected states. Relation rows come from
        ``pair_project`` features of distinct seeded entity-bank pairs. Rank 0
        computes centroids; other ranks receive a broadcast copy.
        """
        data = projected.detach()
        pool = data.reshape(-1, self._soft_dim)
        rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

        def centroids(data: Tensor, k: int) -> Tensor:
            if int(data.size(0)) == 0:
                return torch.zeros(k, self._soft_dim, device=data.device, dtype=data.dtype)
            rows = data
            if int(rows.size(0)) < k:
                need = k - int(rows.size(0))
                extra = rows[torch.randint(0, rows.size(0), (need,), device=rows.device)]
                rows = torch.cat((rows, extra), dim=0)
            km = KMeans(
                n_clusters=k,
                max_iter=self._bank_init_kmeans_iters,
                mode='cosine',
                verbose=0,
                init_method='kmeans++',
            )
            _ = km.fit_predict(rows)
            centroids = km.centroids
            if centroids is None:
                return torch.zeros(k, self._soft_dim, device=data.device, dtype=data.dtype)
            return F.normalize(centroids, dim=-1)

        if rank0:
            _ = self.entity_bank.data.copy_(centroids(pool, self._entity_bank_size))
            rows = self.entity_bank.data
            n_codes = int(rows.size(0))
            idx = torch.arange(n_codes, device=rows.device)
            src = idx.view(n_codes, 1).expand(n_codes, n_codes)
            dst = idx.view(1, n_codes).expand(n_codes, n_codes)
            keep = src != dst
            pair = self.pair_features(rows[src[keep]], rows[dst[keep]])
            _ = self.relation_bank.data.copy_(centroids(pair, self._relation_bank_size))
        if dist.is_available() and dist.is_initialized():
            _ = dist.broadcast(self.entity_bank.data, src=0)
            _ = dist.broadcast(self.relation_bank.data, src=0)

    def _token_usage_entropy_gap(
        self,
        mean_row_entropy: Tensor,
        batch_usage: Tensor,
        bank_size: int,
    ) -> Tensor:
        """Mean token entropy minus entropy of mean assignment (peaked rows, used bank)."""
        usage_entropy = -(batch_usage * batch_usage.clamp_min(self._log_eps).log()).sum()
        gap = mean_row_entropy - usage_entropy
        if self._normalize_assignment_entropy:
            return gap / math.log(bank_size)
        return gap

    def masked_intensity(self, assignment: Tensor, token_mask: Tensor | None = None) -> Tensor:
        """Slot intensity as the mask-weighted assignment sum."""
        weights = (
            assignment.new_ones(assignment.shape[:-1])
            if token_mask is None
            else token_mask.to(dtype=assignment.dtype)
        )
        return torch.einsum('...l,...lk->...k', weights, assignment)

    def claim_span_demand(self, assignment: Tensor, claim_mask: Tensor) -> Tensor:
        """Numbered-claim demand on top codes of claim tokens. Not occupancy."""
        weights = claim_mask.to(dtype=assignment.dtype)
        winners = assignment.argmax(dim=-1)
        mass = assignment.gather(-1, winners.unsqueeze(-1)).squeeze(-1) * weights
        demand = assignment.new_zeros(assignment.size(0), assignment.size(-1))
        return demand.scatter_add(-1, winners, mass)

    def diversity_loss(
        self,
        assignment: Tensor,
        projected: Tensor,
        token_mask: Tensor | None = None,
    ) -> tuple[DiversityTerms, Tensor, Tensor]:
        """Entity diversity pieces: VQ, filing inventory, usage KL, MAGVIT gap.

        Filing inventory is the Shannon entropy of mask-weighted assignment mass
        per document. Pad tokens do not enter that mass. Pieces carry no dual
        weights.

        Args:
            assignment: Row-stochastic ``(..., T, K_e)``.
            projected: Bank-space host states ``(..., T, soft_dim)``.
            token_mask: Live-token weights ``(..., T)``. Null treats every token
                as live.

        Returns:
            Unscaled pieces, mean row entropy, and batch mean usage ``(K_e,)``.
        """

        def filing_inventory_entropy() -> Tensor:
            intensity = self.masked_intensity(assignment, token_mask)
            mass = intensity.sum(dim=-1, keepdim=True)
            simplex = intensity / mass.clamp_min(self._log_eps)
            filing_h = torch.special.entr(simplex).sum(dim=-1)
            present = (mass.squeeze(-1) > 0).to(dtype=assignment.dtype)
            return cast(
                Tensor,
                (filing_h * present).sum() / present.sum().clamp_min(1),
            )

        log_a = assignment.clamp_min(self._log_eps).log()
        row_entropy = -(assignment * log_a).sum(dim=-1)
        mean_row_entropy = row_entropy.mean()
        inventory_entropy = filing_inventory_entropy()

        reduce_dims = tuple(range(assignment.ndim - 1))
        batch_usage = assignment.mean(dim=reduce_dims)
        log_uniform = -math.log(self._entity_bank_size)
        usage_kl = (batch_usage * (batch_usage.clamp_min(self._log_eps).log() - log_uniform)).sum()

        reconstructed = self.soft_entities(assignment)
        codebook = F.mse_loss(projected.detach(), reconstructed)
        commitment = F.mse_loss(projected, reconstructed.detach())

        entropy_term = self._token_usage_entropy_gap(
            mean_row_entropy,
            batch_usage,
            self._entity_bank_size,
        )
        terms = DiversityTerms(
            vq=codebook + commitment,
            inventory=inventory_entropy,
            usage_kl=usage_kl,
            gap=self._entropy_scale * entropy_term,
            rel_vq=assignment.new_zeros(()),
        )
        return terms, mean_row_entropy, batch_usage

    def relation_diversity_loss(self, assignment: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Relation reconstruction (zero) plus row entropy and usage for metrics.

        Args:
            assignment: Typed leftover ``(..., K_r)``. May sum below one when
                refuse mass is nonzero.

        Returns:
            Zero reconstruction, mean row entropy, and batch mean usage ``(K_r,)``.
        """
        log_a = assignment.clamp_min(self._log_eps).log()
        row_entropy = -(assignment * log_a).sum(dim=-1)
        mean_row_entropy = row_entropy.mean()

        reduce_dims = tuple(range(assignment.ndim - 1))
        batch_usage = assignment.mean(dim=reduce_dims)
        return assignment.new_zeros(()), mean_row_entropy, batch_usage

    def occupancy_and_dead(self, usage_ema: Tensor) -> tuple[Tensor, Tensor]:
        """Occupancy and dead-row count of ``usage_ema`` against ``utilization_eps``.

        Call on the EMA before restart. Restart writes a floor into every dead
        row, so the same fraction taken afterwards cannot fall below 1.0.
        """
        dead = usage_ema < self._utilization_eps
        occupancy = (~dead).to(dtype=torch.float32).mean()
        n_dead = dead.to(dtype=usage_ema.dtype).sum()
        return occupancy, n_dead

    def restart_dead_rows(
        self,
        bank: nn.Parameter,
        usage_ema: Tensor,
        projected: Tensor,
    ) -> Tensor:
        """Replace dead bank rows with L2-normalized tokens from the batch.

        A row is dead when its usage EMA is below ``utilization_eps``. Rank 0
        samples replacements from flattened ``projected``; other ranks receive
        the same slab via broadcast. Restarted EMA entries are set to
        ``max(1/K, utilization_eps)``. Runs only in train mode. An empty token
        pool leaves the bank unchanged.
        """

        def rank0_replacement_slab(n_dead: int) -> Tensor | None:
            if n_dead == 0:
                return None
            distributed = dist.is_available() and dist.is_initialized()
            rank0 = not distributed or dist.get_rank() == 0
            pool = projected.detach().reshape(-1, self._soft_dim)
            has_pool = torch.zeros((), device=bank.device, dtype=torch.int64)
            _ = has_pool.fill_(int(pool.size(0) > 0)) if rank0 else has_pool
            _ = dist.broadcast(has_pool, src=0) if distributed else has_pool
            if int(has_pool.item()) == 0:
                return None
            replacements = torch.empty(
                n_dead,
                self._soft_dim,
                device=bank.device,
                dtype=bank.dtype,
            )
            if rank0:
                idx = torch.randint(0, int(pool.size(0)), (n_dead,), device=pool.device)
                _ = replacements.copy_(F.normalize(pool.index_select(0, idx), dim=-1))
            _ = dist.broadcast(replacements, src=0) if distributed else replacements
            return replacements

        if not self.training:
            return projected.new_zeros(())
        dead = usage_ema < self._utilization_eps
        n_dead = int(dead.sum().item())
        replacements = rank0_replacement_slab(n_dead)
        if n_dead == 0 or replacements is None:
            return projected.new_zeros(())
        dead_idx = dead.nonzero(as_tuple=False).squeeze(-1)
        floor = max(1.0 / float(usage_ema.numel()), self._utilization_eps)
        with torch.no_grad():
            _ = bank.data.index_copy_(0, dead_idx, replacements)
            _ = usage_ema.index_fill_(0, dead_idx, floor)
        return projected.new_tensor(float(n_dead))

    def update_relation_usage(
        self,
        batch_usage: Tensor,
        projected: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """EMA-update relation usage, snapshot occupancy, then restart dead rows.

        Args:
            batch_usage: Mean relation assignment ``(K_r,)``.
            projected: Pair features in bank space used as the replacement pool.

        Returns:
            Restart count, pre-restart occupancy, and pre-restart dead count.
        """
        occupancy, n_dead = self.occupancy_and_dead(self.relation_usage_ema)
        if not self.training:
            zero = projected.new_zeros(())
            return zero, occupancy, n_dead
        synced_usage = _synced_batch_usage(batch_usage.detach())
        momentum = self._usage_ema_momentum
        _ = self.relation_usage_ema.copy_(
            self.relation_usage_ema * momentum + synced_usage * (1.0 - momentum)
        )
        occupancy, n_dead = self.occupancy_and_dead(self.relation_usage_ema)
        n_restarts = self.restart_dead_rows(
            self.relation_bank,
            self.relation_usage_ema,
            projected,
        )
        return n_restarts, occupancy, n_dead

    def soft_transe_loss(self, heads: Tensor, tails: Tensor, relations: Tensor) -> Tensor:
        """Mean soft TransE score ``||h + r - t||^2`` over relation-typed pairs."""
        return torch.mean(torch.sum((heads + relations - tails) ** 2, dim=-1))

    def bank_anchor(self) -> Tensor:
        """Zero scalar that still touches both banks and the remask vector."""
        return (
            self.entity_bank.sum() * 0.0
            + self.relation_bank.sum() * 0.0
            + self.entity_dmask.sum() * 0.0
        )

    def combine_diversity(self, terms: DiversityTerms) -> Tensor:
        """Sum reconstruction terms only; entropy pieces stay unweighted."""
        return terms.vq + terms.rel_vq + self.bank_anchor()

    def forward(
        self,
        states: Tensor,
        token_mask: Tensor | None = None,
    ) -> SoftVocabOutput:
        """Soft-assign ``states``, update entity usage EMA, return envelopes."""
        self.ensure_banks_seeded(states)
        assignment, projected = self.soft_assign(states)
        terms, mean_row_entropy, batch_usage = self.diversity_loss(
            assignment,
            projected,
            token_mask=token_mask,
        )
        soft_entities = self.soft_entities(assignment)

        n_entity_restarts = projected.new_zeros(())
        occupancy, n_dead = self.occupancy_and_dead(self.entity_usage_ema)
        if self.training:
            synced_usage = _synced_batch_usage(batch_usage.detach())
            momentum = self._usage_ema_momentum
            _ = self.entity_usage_ema.copy_(
                self.entity_usage_ema * momentum + synced_usage * (1.0 - momentum)
            )
            occupancy, n_dead = self.occupancy_and_dead(self.entity_usage_ema)
            n_entity_restarts = self.restart_dead_rows(
                self.entity_bank,
                self.entity_usage_ema,
                projected,
            )
        n_relation_restarts = projected.new_zeros(())
        usage_mass = batch_usage.clamp_min(0)
        usage_total = usage_mass.sum().clamp_min(1e-12)
        usage_p = usage_mass / usage_total
        usage_entropy = -(usage_p * usage_p.clamp_min(1e-12).log()).sum()
        inverse_simpson = usage_p.square().sum().reciprocal()

        return SoftVocabOutput(
            assignment=assignment,
            projected=projected,
            soft_entities=soft_entities,
            diversity_terms=terms,
            diversity_loss=self.combine_diversity(terms),
            mean_row_entropy=mean_row_entropy,
            inventory_entropy=terms.inventory,
            batch_usage=batch_usage,
            occupancy=occupancy,
            n_dead_before_restart=n_dead,
            usage_entropy=usage_entropy,
            inverse_simpson=inverse_simpson,
            n_entity_restarts=n_entity_restarts,
            n_relation_restarts=n_relation_restarts,
            n_restarts=n_entity_restarts + n_relation_restarts,
        )


def _synced_batch_usage(batch_usage: Tensor) -> Tensor:
    """Average usage across DDP ranks before EMA when distributed is active."""
    if not (dist.is_available() and dist.is_initialized()):
        return batch_usage
    synced = batch_usage.clone()
    _ = dist.all_reduce(synced, op=dist.ReduceOp.SUM)
    return synced / dist.get_world_size()


__all__ = [
    'DiversityTerms',
    'RelationAssignment',
    'SoftVocabModule',
    'SoftVocabOutput',
]
