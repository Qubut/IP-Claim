"""Encode patent HeteroData into a fixed-size soft-token sequence for the host LM.

Owns filing HGT, CompGCN-sub compose on soft pair edges, an RMS-matched
CPC identity residual on compose rows, and query readout.
Topology is declared on ``FoundationHeteroSchema``; PyG type strings appear only
when building metadata. Empty graphs yield query-only tokens. Host LoRA is out
of scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import starmap
from typing import cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from returns.maybe import Maybe, maybe
from returns.methods import partition
from returns.pipeline import flow
from returns.pointfree import bind_optional, map_
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HGTConv, Linear, MessagePassing
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch

from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.covering_trace import trace_tensor
from ip_claim.ssv.graph_batch import (
    CPC_CLASS_COUNT,
    CPC_IDENTITY_VOCAB_SIZE,
    CPC_SECTION_COUNT,
)


class FoundationHeteroSchema(BaseModel):
    """Node and edge type layout for the patent HeteroData encoder."""

    model_config = ConfigDict(frozen=True)

    node_types: tuple[str, ...] = ('cpc', 'claim', 'soft_entity')
    structural_edges: tuple[tuple[str, str, str], ...] = (
        ('cpc', 'parent_of', 'cpc'),
        ('claim', 'depends_on', 'claim'),
        ('soft_entity', 'soft_related', 'soft_entity'),
    )
    gifted_edge_types: tuple[tuple[str, str, str], ...] = (
        ('cpc', 'parent_of', 'cpc'),
        ('claim', 'depends_on', 'claim'),
    )
    self_loop_rel: str = 'self'
    in_channels: int = Field(default=1, ge=1)
    soft_entity_in_channels: int = Field(default=128, ge=1)

    def hgt_node_types(self) -> tuple[NodeType, ...]:
        """Filing node types that enter HGT (CPC and claim)."""
        compose = self.compose_node_type()
        return tuple(node_type for node_type in self.node_types if node_type != compose)

    def compose_node_type(self) -> NodeType:
        """Soft-entity node type consumed by the compose stack."""
        return 'soft_entity'

    def compose_edge_type(self) -> EdgeType:
        """Soft-related edge type that carries relation vectors."""
        return ('soft_entity', 'soft_related', 'soft_entity')

    def filing_edge_types(self) -> tuple[EdgeType, ...]:
        """CPC parent and claim-dependency edges for HGT and structural TransE."""
        return self.gifted_edge_types

    def hgt_metadata(self) -> tuple[list[NodeType], list[EdgeType]]:
        """Project filing types into PyG HGT ``(node_types, edge_types)`` metadata."""
        nodes: list[NodeType] = list(self.hgt_node_types())
        edges: list[EdgeType] = [
            *self.filing_edge_types(),
            *((node_type, self.self_loop_rel, node_type) for node_type in nodes),
        ]
        return nodes, edges

    def structural_edge_types(self) -> tuple[EdgeType, ...]:
        """Non-self-loop edge types from the schema (filing + soft-related)."""
        return self.structural_edges

    def gifted_ke_edge_types(self) -> tuple[EdgeType, ...]:
        """CPC/claim edges used for structural TransE scoring."""
        return self.gifted_edge_types

    def in_channels_for(self, node_type: NodeType) -> int:
        """Feature width expected for one node type."""
        if node_type == 'soft_entity':
            return int(self.soft_entity_in_channels)
        return int(self.in_channels)


class ComposeScoreReport(BaseModel):
    """Filtered TransE pair ranking on post-compose soft states.

    Protocol matches CompGCN / Bordes link prediction: mean reciprocal rank,
    mean rank, and Hits at 1, 3, and 10.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    compose_transe: Tensor
    """Mean ``||h + r - t||^2`` on compose-stack pair edges (scalar)."""

    compose_mrr: Tensor
    """Mean reciprocal rank of the true tail among same-filing entities."""

    compose_mr: Tensor
    """Mean rank of the true tail (1-based)."""

    compose_hits_at_1: Tensor
    compose_hits_at_3: Tensor
    compose_hits_at_10: Tensor


class SoftEncodeOutput(BaseModel):
    """Encoder output: soft tokens, structural KE, compose pair scores, and node states."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    soft_tokens: Tensor
    """Soft tokens ``(B, n_soft, gnn_hidden)``."""

    ke_loss: Tensor
    """Mean TransE score on structural edges (scalar)."""

    compose_scores: ComposeScoreReport
    """Filtered pair ranking on the CompGCN-sub stack."""

    node_states: Tensor
    """Post-compose soft-entity states padded to ``(B, L, gnn_hidden)``."""

    node_mask: Tensor
    """Boolean mask ``(B, L)`` of real nodes in ``node_states``."""

    batch_index: Tensor
    """Sparse graph assignment ``(N,)`` for each compose node."""


class HgtInputs(BaseModel):
    """Projected node features and edge indices for one HGT forward pass."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    x_dict: dict[str, Tensor]
    edge_index_dict: dict[EdgeType, Tensor]


class ComposeInputs(BaseModel):
    """Projected soft-entity states and relation-typed pair edges."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    x: Tensor
    """Soft-entity states ``(N, hidden)``."""

    edge_index: Tensor
    """Soft-related edges ``[2, E]``."""

    edge_attr: Tensor
    """Relation vectors ``(E, hidden)`` aligned with ``edge_index``."""


class HgtConvLayer(nn.Module):
    """One HGT hop plus ReLU on types that emitted states."""

    def __init__(
        self,
        hidden: int,
        metadata: tuple[list[NodeType], list[EdgeType]],
        heads: int,
    ) -> None:
        super().__init__()
        self.conv = HGTConv(hidden, hidden, metadata, heads=heads)

    def forward(self, inputs: HgtInputs) -> HgtInputs:
        """Update filing states; drop types HGT left as ``None``."""
        updated = self.conv(inputs.x_dict, inputs.edge_index_dict)
        return HgtInputs(
            x_dict={
                node_type: F.relu(states)
                for node_type, states in updated.items()
                if states is not None
            },
            edge_index_dict=inputs.edge_index_dict,
        )


class CompGcnSubConv(MessagePassing):  # type: ignore[misc]
    """One CompGCN subtraction layer: message is ``W(x_j + r)``, then residual ReLU.

    Relation vectors are transformed by ``W_rel`` for the next hop.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__(aggr='mean')
        self.message_lin = Linear(hidden, hidden)
        self.relation_lin = Linear(hidden, hidden)

    def forward(self, inputs: ComposeInputs) -> ComposeInputs:
        """Compose neighbor plus relation; identity when the pair set is empty."""
        unused = self.message_lin.weight.sum() * 0.0 + self.relation_lin.weight.sum() * 0.0
        x, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        if x.size(0) == 0 or edge_index.size(1) == 0:
            return ComposeInputs(x=x + unused, edge_index=edge_index, edge_attr=edge_attr + unused)
        aggregated = cast(Tensor, self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return ComposeInputs(
            x=F.relu(x + aggregated) + unused,
            edge_index=edge_index,
            edge_attr=cast(Tensor, self.relation_lin(edge_attr)),
        )

    def message(self, x_j: Tensor, *, edge_attr: Tensor | None = None) -> Tensor:
        """TransE composition: neighbor plus relation, then a learned map."""
        delta = x_j if edge_attr is None else x_j + edge_attr
        return cast(Tensor, self.message_lin(delta))


def relation_param_key(edge_type: EdgeType) -> str:
    """ParameterDict key for one structural edge type triple."""
    src, rel, dst = edge_type
    return f'{src}__{rel}__{dst}'


def store_tensor(store: object | None, name: str, *, missing: Tensor) -> Tensor:
    """Attribute tensor from a PyG store; ``missing`` when the store or field is absent."""
    return flow(
        Maybe.from_optional(store),
        bind_optional(lambda s: getattr(s, name, None)),
        map_(lambda raw: cast(Tensor, raw).to(device=missing.device, dtype=missing.dtype)),
    ).value_or(missing)


def row_rms(values: Tensor) -> Tensor:
    """Root-mean-square over the last axis.

    The mean square is floored so a zero row stays differentiable.
    """
    return (
        values
        .square()
        .mean(dim=-1, keepdim=True)
        .clamp_min(
            torch.finfo(values.dtype).eps,
        )
        .sqrt()
    )


def coalesce_hetero_batch(
    graphs: HeteroData | Sequence[HeteroData],
    *,
    seed_type: str,
) -> HeteroData:
    """Normalize a single graph or sequence into a PyG hetero batch."""
    if not isinstance(graphs, HeteroData):
        if not graphs:
            msg = 'SoftGraphEncoder requires at least one HeteroData graph'
            raise ValueError(msg)
        return cast(HeteroData, Batch.from_data_list(list(graphs)))
    if getattr(graphs[seed_type], 'batch', None) is not None:
        return graphs
    return cast(HeteroData, Batch.from_data_list([graphs]))


def node_count(batch: HeteroData, node_type: NodeType) -> int:
    """Number of nodes for one type; zero when absent."""
    if node_type not in batch.node_types:
        return 0
    return int(batch[node_type].num_nodes or 0)


def project_node_x_dict(
    batch: HeteroData,
    schema: FoundationHeteroSchema,
    lin_dict: nn.ModuleDict,
    device: torch.device,
) -> dict[str, Tensor]:
    """Linear projection and ReLU for each declared node type."""

    def node_feature_matrix(node_type: NodeType) -> Tensor:
        """Node features for one type; zeros when missing or empty."""
        width = schema.in_channels_for(node_type)
        n = node_count(batch, node_type)
        if n == 0:
            return torch.zeros((0, width), dtype=torch.float)
        feats = store_tensor(
            batch[node_type],
            'x',
            missing=torch.zeros((n, width), dtype=torch.float),
        )
        if feats.size(-1) == width:
            return feats
        padded = feats.new_zeros((n, width))
        copied = min(width, int(feats.size(-1)))
        padded[:, :copied] = feats[:, :copied]
        return padded

    return {
        node_type: lin_dict[node_type](node_feature_matrix(node_type).to(device)).relu()
        for node_type in schema.node_types
    }


def project_edge_index_dict(
    batch: HeteroData,
    schema: FoundationHeteroSchema,
    device: torch.device,
) -> dict[EdgeType, Tensor]:
    """Structural and self-loop edge indices for HGT."""

    def edge_index_or_empty(edge_type: EdgeType) -> Tensor:
        """Edge index for one type; empty ``[2, 0]`` when absent."""
        empty = torch.zeros((2, 0), dtype=torch.long)
        if edge_type not in batch.edge_types:
            return empty
        return store_tensor(batch[edge_type], 'edge_index', missing=empty)

    def self_loop_edge_index(n: int) -> Tensor:
        """Self-loop edges for ``n`` nodes; empty when ``n <= 0``."""
        if n <= 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        ids = torch.arange(n, dtype=torch.long, device=device)
        return torch.stack([ids, ids], dim=0)

    hgt_nodes = schema.hgt_node_types()
    return {
        **{
            edge_type: edge_index_or_empty(edge_type).to(device)
            for edge_type in schema.filing_edge_types()
        },
        **{
            (node_type, schema.self_loop_rel, node_type): self_loop_edge_index(
                node_count(batch, node_type),
            )
            for node_type in hgt_nodes
        },
    }


def project_hgt_inputs(
    batch: HeteroData,
    schema: FoundationHeteroSchema,
    x_all: dict[str, Tensor],
    device: torch.device,
) -> HgtInputs:
    """Select filing states and filing edges for one HGT pass."""
    return HgtInputs(
        x_dict={node_type: x_all[node_type] for node_type in schema.hgt_node_types()},
        edge_index_dict=project_edge_index_dict(batch, schema, device),
    )


def project_compose_inputs(
    batch: HeteroData,
    schema: FoundationHeteroSchema,
    soft_x: Tensor,
    relation_in: Linear,
    device: torch.device,
) -> ComposeInputs:
    """Lift pair relation vectors to hidden width for the compose stack."""
    edge_type = schema.compose_edge_type()
    unused = relation_in.weight.sum() * 0.0
    hidden = int(relation_in.out_channels)
    if edge_type not in batch.edge_types:
        empty_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_attr = torch.zeros((0, hidden), device=device)
        return ComposeInputs(x=soft_x, edge_index=empty_index, edge_attr=empty_attr + unused)
    store = batch[edge_type]
    index = store_tensor(
        store,
        'edge_index',
        missing=torch.zeros((2, 0), dtype=torch.long, device=device),
    )
    width = schema.in_channels_for(schema.compose_node_type())
    attr = store_tensor(
        store,
        'edge_attr',
        missing=torch.zeros((int(index.size(1)), width), device=device),
    )
    return ComposeInputs(
        x=soft_x,
        edge_index=index,
        edge_attr=relation_in(attr.float()) + unused,
    )


def apply_compose_convs(
    inputs: ComposeInputs,
    convs: nn.Sequential,
) -> ComposeInputs:
    """Run the CompGCN-sub stack; returns updated states and hop-final ``r``."""
    return cast(ComposeInputs, convs(inputs))


def score_compose_pairs(
    states: Tensor,
    edge_index: Tensor,
    relations: Tensor,
    batch_index: Tensor,
    occupied: Tensor | None = None,
) -> ComposeScoreReport:
    """Filtered TransE ranking of each pair tail among same-filing soft entities.

    Distances are scored one document at a time so ranking tensors stay
    ``O(n_doc^2)``, not ``O((B n_doc)^2)``. Ranking considers occupied nodes
    only when an occupancy mask is supplied.
    """

    def empty_report() -> ComposeScoreReport:
        zero = torch.zeros((), device=states.device)
        return ComposeScoreReport(
            compose_transe=zero,
            compose_mrr=zero,
            compose_mr=zero,
            compose_hits_at_1=zero,
            compose_hits_at_3=zero,
            compose_hits_at_10=zero,
        )

    def ranks_in_graph(graph_id: Tensor) -> Tensor:
        node_mask = batch_index == graph_id
        if occupied is not None:
            node_mask &= occupied
        global_nodes = node_mask.nonzero(as_tuple=True)[0]
        edge_mask = node_mask.index_select(0, src) & node_mask.index_select(0, dst)
        if global_nodes.numel() == 0 or not bool(edge_mask.any()):
            return states.new_zeros((0,))
        local_of = torch.full((states.size(0),), -1, dtype=torch.long, device=states.device)
        local_of[global_nodes] = torch.arange(global_nodes.size(0), device=states.device)
        local_src = local_of.index_select(0, src[edge_mask])
        local_dst = local_of.index_select(0, dst[edge_mask])
        local_states = states.index_select(0, global_nodes)
        local_rels = relations.index_select(0, edge_mask.nonzero(as_tuple=True)[0])
        n_local = int(local_states.size(0))
        adjacency = torch.zeros((n_local, n_local), dtype=torch.bool, device=states.device)
        adjacency[local_src, local_dst] = True
        other_true = adjacency.index_select(0, local_src)
        other_true.scatter_(1, local_dst.unsqueeze(1), False)
        translated = local_states.index_select(0, local_src) + local_rels
        dist = torch.sum((translated.unsqueeze(1) - local_states.unsqueeze(0)) ** 2, dim=-1)
        dist = dist.masked_fill(other_true, torch.finfo(dist.dtype).max)
        true_dist = dist.gather(1, local_dst.unsqueeze(1)).squeeze(1)
        return 1 + (dist < true_dist.unsqueeze(1)).sum(dim=1).to(dtype=dist.dtype)

    if states.size(0) == 0 or edge_index.size(1) == 0:
        return empty_report()
    src = edge_index[0]
    dst = edge_index[1]
    residual = states.index_select(0, src) + relations - states.index_select(0, dst)
    compose_transe = torch.mean(torch.sum(residual**2, dim=-1))
    graph_ids = cast(Tensor, batch_index.unique(sorted=True))  # type: ignore[no-untyped-call]
    rank = torch.cat(tuple(ranks_in_graph(graph_id) for graph_id in graph_ids))
    if rank.numel() == 0:
        return empty_report()
    return ComposeScoreReport(
        compose_transe=compose_transe,
        compose_mrr=torch.mean(rank.reciprocal()),
        compose_mr=torch.mean(rank),
        compose_hits_at_1=torch.mean((rank <= 1).to(dtype=rank.dtype)),
        compose_hits_at_3=torch.mean((rank <= 3).to(dtype=rank.dtype)),
        compose_hits_at_10=torch.mean((rank <= 10).to(dtype=rank.dtype)),
    )


def apply_hgt_convs(
    inputs: HgtInputs,
    convs: nn.Sequential,
) -> dict[str, Tensor]:
    """Run the declared HGT stack; ReLU after each layer."""
    return cast(HgtInputs, convs(inputs)).x_dict


def mean_structural_ke_loss(
    h_dict: dict[str, Tensor],
    edge_index_dict: dict[EdgeType, Tensor],
    relation: nn.ParameterDict,
    schema: FoundationHeteroSchema,
    device: torch.device,
) -> Tensor:
    """Mean TransE score over schema structural edges; zero when none exist."""

    @maybe
    def trans_e_mean_for_edge(edge_index: Tensor, edge_type: EdgeType) -> Tensor | None:
        """Mean TransE distance for one edge type, or None when empty or unknown."""
        src_type, _, dst_type = edge_type
        if edge_index.numel() == 0 or src_type not in h_dict or dst_type not in h_dict:
            return None
        key = relation_param_key(edge_type)
        if key not in relation:
            return None
        src = h_dict[src_type].index_select(0, edge_index[0])
        dst = h_dict[dst_type].index_select(0, edge_index[1])
        rel = relation[key]
        # Scale by 1/sqrt(d) so the TransE loss stays O(1) without mutating the
        # leaf relation Parameter in place.
        scale = src.size(-1) ** -0.5
        return torch.mean(torch.sum(((src + rel - dst) * scale) ** 2, dim=-1))

    empty = torch.zeros((2, 0), dtype=torch.long, device=device)
    scores, _ = partition(
        trans_e_mean_for_edge(edge_index_dict.get(edge_type, empty), edge_type)
        for edge_type in schema.gifted_ke_edge_types()
    )
    if not scores:
        zero = torch.zeros((), device=device)
        return zero + sum(param.sum() * 0.0 for param in relation.parameters())
    return torch.stack(scores).mean()


def attend_soft_tokens(
    h_dict: dict[str, Tensor],
    batch: HeteroData,
    schema: FoundationHeteroSchema,
    queries: Tensor,
    readout: nn.MultiheadAttention,
    device: torch.device,
    text_query: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Attend learnable queries, FiLM-conditioned by ``text_query`` when supplied.

    ``text_query`` is ``None`` only from the retrieval-index export path, which pools no
    masked/unmasked pair; every MLM training call supplies it.
    """

    def node_batch_vector(node_type: NodeType, n: int) -> Tensor:
        return store_tensor(
            batch[node_type],
            'batch',
            missing=torch.zeros(n, dtype=torch.long, device=device),
        )

    batch_size = int(batch.num_graphs) if hasattr(batch, 'num_graphs') else 1
    query_seq = queries.unsqueeze(0).expand(batch_size, -1, -1)
    if text_query is not None:
        gamma, beta = text_query
        query_seq = gamma.unsqueeze(1) * query_seq + beta.unsqueeze(1)

    def dense_keep(
        states: Tensor,
        batch_vec: Tensor,
        node_type: NodeType,
    ) -> tuple[Tensor, Tensor]:
        dense, mask = to_dense_batch(states, batch_vec, batch_size=batch_size)
        occupied = store_tensor(
            batch[node_type],
            'occupied',
            missing=torch.ones(states.size(0), dtype=torch.bool, device=device),
        )
        keep, _ = to_dense_batch(
            occupied.to(dtype=states.dtype).unsqueeze(-1),
            batch_vec,
            batch_size=batch_size,
        )
        return dense, mask & keep.squeeze(-1).bool()

    type_slices = tuple(
        (states, node_batch_vector(node_type, states.size(0)), node_type)
        for node_type in schema.node_types
        if (states := h_dict.get(node_type)) is not None and states.size(0) > 0
    )
    if not type_slices:
        return query_seq

    dense_parts, mask_parts = map(
        list,
        zip(*starmap(dense_keep, type_slices), strict=False),
    )
    dense = torch.cat(dense_parts, dim=1)
    mask = torch.cat(mask_parts, dim=1)
    attended, _ = readout(
        query_seq,
        dense,
        dense,
        key_padding_mask=~mask,
        need_weights=False,
    )
    return cast(Tensor, attended)


class SoftGraphEncoder(nn.Module):
    """Map HeteroData to ``(B, n_soft, gnn_hidden)`` via filing HGT, mixed compose, and readout."""

    def __init__(
        self,
        config: SsvTrainConfig,
        *,
        schema: FoundationHeteroSchema | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema or FoundationHeteroSchema(
            in_channels=int(config.arch.cpc_identity_dim),
            soft_entity_in_channels=int(config.arch.soft_dim),
        )
        self.cpc_identity = nn.Embedding(
            CPC_IDENTITY_VOCAB_SIZE,
            int(self.schema.in_channels),
        )
        self.cpc_feature_source = 'identity'
        hidden = int(config.arch.gnn_hidden)
        heads = int(config.arch.gnn_heads)
        layers = int(config.arch.gnn_layers)
        n_soft = int(config.arch.n_soft_tokens)
        if hidden % heads != 0:
            msg = f'gnn_hidden ({hidden}) must be divisible by gnn_heads ({heads})'
            raise ValueError(msg)

        self._n_soft = n_soft
        self._hidden = hidden
        hgt_nodes, hgt_edges = self.schema.hgt_metadata()
        metadata = (hgt_nodes, hgt_edges)

        self.lin_dict = nn.ModuleDict({
            node_type: Linear(self.schema.in_channels_for(node_type), hidden)
            for node_type in self.schema.node_types
        })
        compose_in = self.schema.in_channels_for(self.schema.compose_node_type())
        self.relation_in = Linear(compose_in, hidden)
        self.convs = nn.Sequential(
            *(HgtConvLayer(hidden, metadata, heads) for _ in range(layers)),
        )
        self.compose_convs = nn.Sequential(*(CompGcnSubConv(hidden) for _ in range(layers)))
        self.cpc_to_compose = nn.Linear(hidden, hidden)
        self.queries = nn.Parameter(torch.empty(n_soft, hidden))
        _ = nn.init.normal_(self.queries, std=0.02)
        self.readout = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.film_gamma = nn.Linear(int(config.host.d_model), hidden)
        self.film_beta = nn.Linear(int(config.host.d_model), hidden)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)
        self.relation = nn.ParameterDict({
            relation_param_key(edge_type): nn.Parameter(F.normalize(torch.randn(hidden), dim=0))
            for edge_type in self.schema.gifted_ke_edge_types()
        })

    @property
    def n_soft_tokens(self) -> int:
        """Soft tokens emitted per graph."""
        return self._n_soft

    @property
    def hidden_dim(self) -> int:
        """Hidden width of each soft token."""
        return self._hidden

    def fill_filing_features(self, batch: HeteroData, device: torch.device) -> None:
        """Write CPC identity (or depth control) into filing node features."""
        width = int(self.schema.in_channels)
        n_cpc = node_count(batch, 'cpc')
        ids = Maybe.do(
            cast(Tensor, raw).to(device=device, dtype=torch.long)
            for store in Maybe.from_optional(batch['cpc'] if 'cpc' in batch.node_types else None)
            for raw in Maybe.from_optional(getattr(store, 'cpc_id', None))
        ).value_or(torch.zeros(n_cpc, dtype=torch.long, device=device))
        if self.cpc_feature_source == 'depth':
            depths = torch.zeros(ids.shape, dtype=torch.float, device=device)
            depths = depths.masked_fill(ids > CPC_SECTION_COUNT, 1.0)
            depths = depths.masked_fill(ids > CPC_SECTION_COUNT + CPC_CLASS_COUNT, 2.0)
            features = torch.zeros((n_cpc, width), dtype=torch.float, device=device)
            if n_cpc:
                features[:, 0] = depths
            batch['cpc'].x = features
        else:
            batch['cpc'].x = self.cpc_identity(ids)
        n_claim = node_count(batch, 'claim')
        batch['claim'].x = torch.zeros((n_claim, width), dtype=torch.float, device=device)

    def forward(
        self,
        graphs: HeteroData | Sequence[HeteroData],
        text_query: Tensor | None = None,
    ) -> SoftEncodeOutput:
        """Encode one HeteroData graph or a batch sequence."""

        def add_filing_identity(compose_x: Tensor, compose_batch: Tensor, n_graphs: int) -> Tensor:
            """Mean-pool CPC HGT states per filing, map, RMS-match, and add.

            The residual is scaled so its RMS matches the compose row. An
            unmatched add would dominate CompGCN assignment states. Graphs
            with no CPC nodes receive a zero residual.
            """
            unused = self.cpc_to_compose.weight.sum() * 0.0
            if self.cpc_to_compose.bias is not None:
                unused += self.cpc_to_compose.bias.sum() * 0.0
            cpc_states = filing_states.get('cpc')
            if compose_x.size(0) == 0 or cpc_states is None or cpc_states.size(0) == 0:
                return compose_x + unused
            cpc_batch = store_tensor(
                batch['cpc'],
                'batch',
                missing=torch.zeros(cpc_states.size(0), dtype=torch.long, device=device),
            )
            dense, mask = to_dense_batch(cpc_states, cpc_batch, batch_size=n_graphs)
            present = mask.to(dtype=dense.dtype).unsqueeze(-1)
            pooled = (dense * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)
            identity = self.cpc_to_compose(pooled)[compose_batch]
            has_cpc = present.sum(dim=1).squeeze(-1)[compose_batch].gt(0).unsqueeze(-1)
            aligned = identity * (row_rms(compose_x) / row_rms(identity))
            return compose_x + torch.where(has_cpc, aligned, unused)

        device = next(self.parameters()).device
        batch = coalesce_hetero_batch(graphs, seed_type=self.schema.node_types[0]).to(device)
        self.fill_filing_features(batch, device)
        x_all = project_node_x_dict(batch, self.schema, self.lin_dict, device)
        inputs = project_hgt_inputs(batch, self.schema, x_all, device)
        filing_states = apply_hgt_convs(inputs, self.convs)
        compose = apply_compose_convs(
            project_compose_inputs(
                batch,
                self.schema,
                x_all[self.schema.compose_node_type()],
                self.relation_in,
                device,
            ),
            self.compose_convs,
        )
        compose_type = self.schema.compose_node_type()
        n_soft = int(compose.x.size(0))
        batch_index = store_tensor(
            batch[compose_type],
            'batch',
            missing=torch.zeros(n_soft, dtype=torch.long, device=device),
        )
        occupied = store_tensor(
            batch[compose_type],
            'occupied',
            missing=torch.ones(n_soft, dtype=torch.bool, device=device),
        )
        # Isolated slots have no neighbors; mean aggregation is zero there.
        compose_x = trace_tensor(
            'compose_state',
            torch.nan_to_num(compose.x),
            'node',
            'hidden',
        )
        compose_scores = score_compose_pairs(
            compose_x,
            compose.edge_index,
            compose.edge_attr,
            batch_index,
            occupied,
        )
        graph_count = int(batch.num_graphs) if hasattr(batch, 'num_graphs') else 1
        mixed_compose = trace_tensor(
            'mixed_compose',
            add_filing_identity(compose_x, batch_index, graph_count),
            'node',
            'hidden',
        )
        ke_states = {**filing_states, compose_type: compose_x}
        attend_states = {**filing_states, compose_type: mixed_compose}
        film = (
            (1.0 + self.film_gamma(text_query), self.film_beta(text_query))
            if text_query is not None
            else None
        )
        soft_tokens = trace_tensor(
            'graph_readout',
            attend_soft_tokens(
                attend_states,
                batch,
                self.schema,
                self.queries,
                self.readout,
                device,
                film,
            ),
            'batch',
            'slot',
            'hidden',
        )
        ke_loss = mean_structural_ke_loss(
            ke_states,
            inputs.edge_index_dict,
            self.relation,
            self.schema,
            device,
        )
        node_states, node_mask = to_dense_batch(
            mixed_compose,
            batch_index,
            batch_size=graph_count,
        )
        occupied_dense, _ = to_dense_batch(
            occupied.to(dtype=mixed_compose.dtype).unsqueeze(-1),
            batch_index,
            batch_size=graph_count,
        )
        node_mask &= occupied_dense.squeeze(-1).bool()
        return SoftEncodeOutput(
            soft_tokens=soft_tokens,
            ke_loss=ke_loss,
            compose_scores=compose_scores,
            node_states=node_states,
            node_mask=node_mask,
            batch_index=batch_index,
        )


__all__ = [
    'CompGcnSubConv',
    'ComposeInputs',
    'ComposeScoreReport',
    'FoundationHeteroSchema',
    'HgtConvLayer',
    'SoftEncodeOutput',
    'SoftGraphEncoder',
    'row_rms',
    'score_compose_pairs',
    'store_tensor',
]
