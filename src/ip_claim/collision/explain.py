"""Covering contour and collision community on explained pairs.

Contour is the token pullback of unpaid and paid demand. Community is the same
mass on the shared slot graph. Assignment and pair-graph W are recomputed only
for explained pairs. Scores come from the covering module; this module does not
invent a second residual.
"""

from __future__ import annotations

from itertools import starmap
from typing import cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch_geometric.utils import get_laplacian, to_dense_adj

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.covering_trace import trace_tensor
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.soft_graph import SoftGraphOverlay, SoftRelationBundle


class CoveringContour(BaseModel):
    """Token fields, same-slot alignment, harmonic lift, and tau level-set runs."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    query_field: Tensor
    document_field: Tensor
    alignment: Tensor
    graph_field: Tensor
    spans: tuple[tuple[int, int], ...]


class CollisionCommunity(BaseModel):
    """Paid subgraph, unpaid remnant, edge unpaid fraction, and conductance leak."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    paid_mask: Tensor
    gap_mask: Tensor
    paid_edges: Tensor
    gap_edges: Tensor
    unpaid_edge: Tensor
    conductance: Tensor
    paid_demand: Tensor


def attribute_kept_residual(residual: Tensor, pair_mass: Tensor) -> Tensor:
    """Scatter leftover slot residual onto kept directed pairs.

    Zero pair mass, including consumed refuse and the identity, stays
    zero. Endpoint leftover is added so dest-axis roll of a kept pair
    moves the attributed cell.
    """
    slots = pair_mass.size(-1)
    if residual.size(-1) != slots:
        msg = 'leftover residual slot axis must match the pair table'
        raise ValueError(msg)
    identity = torch.eye(slots, dtype=torch.bool, device=pair_mass.device)
    kept = (pair_mass > 0) & ~identity
    ends = residual.to(dtype=pair_mass.dtype).unsqueeze(-1) + residual.to(
        dtype=pair_mass.dtype
    ).unsqueeze(-2)
    return (pair_mass * ends).masked_fill(~kept, 0)


class Explain(nn.Module):
    """Pull covering residuals onto tokens and the shared slot graph."""

    lift_lambda: Tensor

    def __init__(self, covering: Covering | None = None, *, lift_lambda: float = 1.0) -> None:
        super().__init__()
        self.covering = covering if covering is not None else Covering(CoveringKnobs())
        self.register_buffer('lift_lambda', torch.tensor(float(lift_lambda)))

    def slot_adjacency(
        self,
        pair_index: Tensor,
        pair_mass: Tensor,
        code_ids: Tensor,
        *,
        bank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Slot-to-slot mass from occupied-code pairs.

        ``pair_mass`` may be ``(E,)`` or ``(E, R)``. The trailing relation
        axis is kept when it is present.
        """
        adj = torch.zeros((bank, bank, *pair_mass.shape[1:]), device=device, dtype=dtype)
        if pair_index.size(1) == 0:
            return adj
        src = code_ids[pair_index[0]]
        dst = code_ids[pair_index[1]]
        return adj.index_put((src, dst), pair_mass, accumulate=True)

    def graphs_from_bundle(
        self,
        assignment: Tensor,
        token_mask: Tensor,
        bundle: SoftRelationBundle,
        *,
        keep_relation: bool = False,
        relation_bank: int | None = None,
    ) -> Tensor:
        """Per-row slot adjacency from a late pair bundle.

        The default sums the relation-code axis. ``keep_relation=True`` stores
        ``W[k, k', r]`` so labeled and collapsed tables share this builder.
        """
        assignment = self.covering.keep_rows(assignment)
        bank = int(assignment.size(-1))
        relation = bundle.relation_assignment
        n_rel = (
            int(relation_bank)
            if relation_bank is not None
            else 0
            if relation is None
            else int(relation.size(-1))
        )
        shape = (bank, bank, n_rel) if keep_relation else (bank, bank)
        zeros = torch.zeros(*shape, device=assignment.device, dtype=assignment.dtype)
        if relation is None or bundle.pair_count == 0:
            return zeros.expand(assignment.size(0), *shape).clone()
        counts = [int(overlay.edge_index.size(1)) for overlay in bundle.overlays]
        parts = relation.split(counts, dim=0)

        def one_row(
            assign_row: Tensor,
            _mask_row: Tensor,
            overlay: SoftGraphOverlay,
            rel_part: Tensor,
        ) -> Tensor:
            edge_index = overlay.edge_index
            if edge_index.size(1) == 0:
                return zeros
            return self.slot_adjacency(
                edge_index,
                rel_part if keep_relation else rel_part.sum(dim=-1),
                overlay.code_ids,
                bank=bank,
                device=assign_row.device,
                dtype=assign_row.dtype,
            )

        return torch.stack(
            tuple(
                starmap(one_row, zip(assignment, token_mask, bundle.overlays, parts, strict=True))
            )
        )

    def overlay_n(self, occupy: Tensor, pair_mass: Tensor) -> Tensor:
        """Occupy mass plus incidence of kept directed pairs."""
        overlay = PhiIntensity(int(occupy.size(-1))).to(device=occupy.device)
        return overlay(occupy, pair_mass, occupy > 0)

    def leftover_on_kept(
        self,
        demand: Tensor,
        occupy: Tensor,
        pair_mass: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Leftover unpaid of demand against Phi, attributed to kept pairs.

        Returns supply intensity, per-slot leftover, and the leftover
        scattered onto kept directed pairs. Consumed refuse stays zero.
        """
        supply = self.overlay_n(occupy, pair_mass)
        residual = self.covering(demand, supply).residual
        attributed = attribute_kept_residual(residual, pair_mass)
        names = ('src', 'dst') if attributed.ndim == 2 else ('batch', 'src', 'dst')
        if attributed.ndim in {2, 3}:
            _ = trace_tensor('kept_residual', attributed, *names)
        return supply, residual, attributed

    def pool_tokens(
        self,
        field: Tensor,
        weights: Tensor,
        token_node: Tensor,
        n_nodes: int,
    ) -> Tensor:
        """Sum token field mass onto graph nodes that own those tokens."""
        pooled = torch.zeros(n_nodes, device=field.device, dtype=field.dtype)
        valid = (token_node >= 0) & (token_node < n_nodes)
        if not valid.any():
            return pooled
        mass = weights.to(dtype=field.dtype) * field
        return pooled.scatter_add(0, token_node[valid], mass[valid])

    def harmonic_lift(
        self,
        pooled: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Quadratic attachment plus Dirichlet energy on the filing graph."""
        n_nodes = int(pooled.size(0))
        if n_nodes == 0:
            return pooled
        lap_index, lap_weight = get_laplacian(
            edge_index,
            edge_weight,
            normalization=None,
            dtype=pooled.dtype,
            num_nodes=n_nodes,
        )
        laplacian = cast(
            Tensor,
            to_dense_adj(
                lap_index,
                edge_attr=lap_weight,
                max_num_nodes=n_nodes,
            ).squeeze(0),
        )
        eye = torch.eye(n_nodes, device=pooled.device, dtype=pooled.dtype)
        return cast(Tensor, torch.linalg.solve(eye + self.lift_lambda * laplacian, pooled))

    def level_sets(self, field: Tensor, tau: Tensor | float) -> tuple[tuple[int, int], ...]:
        """Maximal half-open runs where the token field meets the display cut."""
        if field.ndim != 1:
            msg = 'level sets are defined on a single token field'
            raise ValueError(msg)
        floor = torch.as_tensor(tau, device=field.device, dtype=field.dtype)
        above = field >= floor
        if not above.any():
            return ()
        padded = F.pad(above.to(dtype=torch.int64), (1, 1))
        starts = ((padded[1:] == 1) & (padded[:-1] == 0)).nonzero(as_tuple=True)[0]
        ends = ((padded[1:] == 0) & (padded[:-1] == 1)).nonzero(as_tuple=True)[0]
        return tuple(
            (int(start), int(end))
            for start, end in zip(starts.tolist(), ends.tolist(), strict=True)
        )

    def conductance(self, adjacency: Tensor, member_mask: Tensor) -> Tensor:
        """Cut over the smaller volume of the paid set and its complement."""
        members = member_mask.to(dtype=adjacency.dtype)
        outsiders = 1.0 - members
        volume_in = (members * adjacency.sum(dim=-1)).sum()
        volume_out = (outsiders * adjacency.sum(dim=-1)).sum()
        cut = (adjacency * members.unsqueeze(-1) * outsiders.unsqueeze(-2)).sum()
        denom = torch.minimum(volume_in, volume_out)
        positive = denom > 0
        safe = torch.where(positive, denom, torch.ones_like(denom))
        return torch.where(positive, cut / safe, torch.zeros_like(cut))

    def contour(
        self,
        assignment_query: Tensor,
        assignment_document: Tensor,
        n_query: Tensor,
        n_document: Tensor,
        query_mask: Tensor,
        *,
        tau: Tensor | float,
        token_node: Tensor | None = None,
        graph_index: Tensor | None = None,
        graph_weight: Tensor | None = None,
    ) -> CoveringContour:
        """Query unpaid field, document paying field, alignment, and optional lift."""
        scored = self.covering(n_query, n_document)
        query_field = self.covering.query_unpaid_field(
            assignment_query,
            n_document,
            n_query,
        )
        document_field = self.covering.document_paying_field(
            assignment_document,
            n_query,
            n_document,
        )
        if token_node is None or graph_index is None:
            graph_field = torch.zeros(0, device=query_field.device, dtype=query_field.dtype)
        else:
            n_nodes = int(token_node.max().item()) + 1 if token_node.numel() else 0
            pooled = self.pool_tokens(query_field, query_mask, token_node, n_nodes)
            graph_field = self.harmonic_lift(pooled, graph_index, graph_weight)
        return CoveringContour(
            query_field=query_field,
            document_field=document_field,
            alignment=self.covering.alignment(
                self.covering.keep_rows(assignment_query),
                self.covering.keep_rows(assignment_document),
                scored.paid,
            ),
            graph_field=graph_field,
            spans=self.level_sets(query_field * query_mask.to(dtype=query_field.dtype), tau),
        )

    def community(
        self,
        n_query: Tensor,
        n_document: Tensor,
        query_edges: Tensor,
        document_edges: Tensor,
        *,
        tau: Tensor | float,
    ) -> CollisionCommunity:
        """Paid vertices and saturated pair lines at the display cut."""
        scored = self.covering(n_query, n_document)
        floor = torch.as_tensor(tau, device=scored.paid.device, dtype=scored.paid.dtype)
        paid_mask = scored.paid >= floor
        paid_edges = self.covering.paid_edges(query_edges, document_edges)
        gap = 1.0 - self.covering.presence(document_edges, self.covering.sigma_edge)
        return CollisionCommunity(
            paid_mask=paid_mask,
            gap_mask=scored.residual >= floor,
            paid_edges=paid_edges,
            gap_edges=query_edges * gap,
            unpaid_edge=self.covering.edge_unpaid_fraction(query_edges, document_edges),
            conductance=self.conductance(paid_edges, paid_mask),
            paid_demand=scored.paid[paid_mask].sum(),
        )

    def forward(
        self,
        assignment_query: Tensor,
        assignment_document: Tensor,
        n_query: Tensor,
        n_document: Tensor,
        query_edges: Tensor,
        document_edges: Tensor,
        query_mask: Tensor,
        *,
        tau: Tensor | float,
        token_node: Tensor | None = None,
        graph_index: Tensor | None = None,
        graph_weight: Tensor | None = None,
    ) -> tuple[CoveringContour, CollisionCommunity]:
        """Contour and community for one explained query-document pair."""
        return (
            self.contour(
                assignment_query,
                assignment_document,
                n_query,
                n_document,
                query_mask,
                tau=tau,
                token_node=token_node,
                graph_index=graph_index,
                graph_weight=graph_weight,
            ),
            self.community(
                n_query,
                n_document,
                query_edges,
                document_edges,
                tau=tau,
            ),
        )


__all__ = [
    'CollisionCommunity',
    'CoveringContour',
    'Explain',
    'attribute_kept_residual',
]
