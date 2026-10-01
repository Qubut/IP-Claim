"""Why matching vs foreign graphs leave ST.14 unpaid unchanged.

Graph-off is a rolled HeteroData swap on the same disclosure text. Overlay
mass is occupied codes from token assignment, so a nonzero overlay does not
mean the filing graph moved. Covering leftover unpaid is of covering n (occupy plus kept-edge
addends) after the projected prefix. Prefix readout keys mixed compose
rows plus filing HGT.
Covering last-layer uses the exported host text states after the RMS-matched
residual on every text token. ``GT.data_color`` paints the product
``Covering`` paid and residual slot tables.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from itertools import batched, starmap
from pathlib import Path

import polars as pl
import pytest
import torch
import torch.nn.functional as F
from great_tables import GT
from torch import Tensor
from torch_geometric.data import HeteroData

from experiments.covering_objective.corpus import HupdDraw, HupdLoader
from experiments.covering_objective.encode_pool import (
    EncodeView,
    covering_pair_devices,
    unnamed_tensor,
)
from experiments.covering_objective.ledgers import GateRecorder, ledger_dir
from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.relational_arms import (
    CLAIM_MASK,
    DECLARED_EDGE_SWEEP,
    FULL_MASK,
    RELATIONAL_CONDITIONS,
    SEAM_RETENTION,
    RelationalArms,
    endpoint_shuffle,
    owner_mass_rows,
    relation_label_shuffle,
    relational_arms,
    retain_seams,
    retention_budget_bytes,
    write_frozen_campaign,
)
from ip_claim.collision.artefacts import write_json
from ip_claim.collision.collide import PatentEmbeddingRecord
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.cover import Covering, CoveringScore
from ip_claim.collision.data.citation_pairs import CitationPair, split_pairs_by_query
from ip_claim.collision.diagnose import (
    IntensityIndex,
    citation_letter,
    intensity_index,
    query_macro_unpaid_fraction_xa,
    score_cited_pairs,
)
from ip_claim.collision.encode_job import CollisionEncodeRow
from ip_claim.collision.explain import attribute_kept_residual
from ip_claim.ssv.collate import SoftMlmCollator, SoftMlmExample
from ip_claim.ssv.covering_trace import (
    CoveringTrace,
    covering_adapters_off,
    covering_patches,
    disable_covering_adapters,
    keep_covering_seams,
    patch_covering,
    record_covering,
    shift_overlay_edges,
)
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = pytest.mark.experiment

_HEATMAP_CELLS = ('graph_on', 'graph_off', 'prefix_off')
_SLOT_TOP_K = 32


def _cosine(left: Tensor, right: Tensor) -> float:
    flat_left = unnamed_tensor(left).flatten(start_dim=1)
    flat_right = unnamed_tensor(right).flatten(start_dim=1)
    return float(F.cosine_similarity(flat_left, flat_right, dim=-1).mean().item())


def _relative_l2(left: Tensor, right: Tensor) -> float:
    base = unnamed_tensor(left)
    delta = (base - unnamed_tensor(right)).norm()
    return float((delta / base.norm().clamp_min(1e-12)).item())


def _energy(tensor: Tensor) -> float:
    return float(unnamed_tensor(tensor).norm(dim=-1).mean().item())


def _optional_cosine(left: Tensor | None, right: Tensor | None) -> float | None:
    if left is None or right is None:
        return None
    if unnamed_tensor(left).shape != unnamed_tensor(right).shape:
        return None
    return _cosine(left, right)


def _optional_energy(tensor: Tensor | None) -> float | None:
    return None if tensor is None else _energy(tensor)


def _node_counts(graphs: Sequence[HeteroData], node_type: str) -> tuple[int, ...]:
    return tuple(
        int(graph[node_type].num_nodes) if node_type in graph.node_types else 0 for graph in graphs
    )


def filing_node_shift(
    match: Sequence[HeteroData],
    rolled: Sequence[HeteroData],
) -> tuple[bool, float, float]:
    """Whether rolled filings change CPC or claim node counts, plus CPC means."""
    cpc_on = _node_counts(match, 'cpc')
    cpc_off = _node_counts(rolled, 'cpc')
    claim_on = _node_counts(match, 'claim')
    claim_off = _node_counts(rolled, 'claim')
    return (
        cpc_on != cpc_off or claim_on != claim_off,
        sum(cpc_on) / max(len(cpc_on), 1),
        sum(cpc_off) / max(len(cpc_off), 1),
    )


def supply_index(
    rows: Sequence[CollisionEncodeRow],
    claims: Tensor,
    supply: Tensor,
) -> IntensityIndex:
    """Pair claim demand with one disclosure supply for covering."""
    records = tuple(
        PatentEmbeddingRecord(
            application_number=row.application_number,
            z_d=torch.zeros(1),
            n_entity_claim=unnamed_tensor(claim),
            n_entity_full=unnamed_tensor(full),
            n_relation_claim=torch.zeros(1),
            n_relation_full=torch.zeros(1),
        )
        for row, claim, full in zip(rows, claims, supply, strict=True)
    )
    return intensity_index(records)


def letter_xa_margin(
    pairs: Sequence[CitationPair],
    index: IntensityIndex,
    covering: Covering,
) -> float | None:
    """Query-macro unpaid fraction of A minus X. Pair-micro raw U is not the letter."""
    return query_macro_unpaid_fraction_xa(score_cited_pairs(pairs, index, covering))


def heatmap_cell(
    pairs: Sequence[CitationPair],
    rows: Sequence[CollisionEncodeRow],
    claims: Tensor,
    supply: Tensor,
) -> tuple[Tensor, Tensor, tuple[str, ...]] | None:
    """Stack cited X/A query and document intensities for one ablation cell."""
    packed = _letter_stack(pairs, supply_index(rows, claims, supply))
    return None if packed is None else (packed[1], packed[2], packed[0])


def graph_contribution_seam(
    *,
    claims: Tensor,
    match_n: Tensor,
    off_n: Tensor,
    text_n: Tensor,
    match_trace: CoveringTrace,
    off_trace: CoveringTrace,
    text_trace: CoveringTrace,
    covering: Covering,
    rows: Sequence[CollisionEncodeRow],
    scored_pairs: Sequence[CitationPair],
    noise: float,
    cpc_nodes_changed: bool,
    cpc_nodes_on_mean: float,
    cpc_nodes_off_mean: float,
    graph_swapped: bool,
    inject_scale: float = 0.0,
) -> dict[str, object]:
    """Name the encode seam that dropped rolled-graph identity before unpaid."""

    def present_cosine(left: Tensor | None, right: Tensor | None) -> float | None:
        if left is None or right is None:
            return None
        if unnamed_tensor(left).shape != unnamed_tensor(right).shape:
            return 0.0
        return _cosine(left, right)

    prefix_on = match_trace.tensors.get('projected_prefix')
    prefix_off_t = off_trace.tensors.get('projected_prefix')
    prefix_text = text_trace.tensors.get('projected_prefix')
    overlay_cos_off = _optional_cosine(
        match_trace.tensors.get('overlay_state'),
        off_trace.tensors.get('overlay_state'),
    )
    prefix_cos_off = _optional_cosine(prefix_on, prefix_off_t)
    graph_readout_cos_off = _optional_cosine(
        match_trace.tensors.get('graph_readout'),
        off_trace.tensors.get('graph_readout'),
    )
    host_text_cos_off = _optional_cosine(
        match_trace.tensors.get('host_text_state'),
        off_trace.tensors.get('host_text_state'),
    )
    late_assign_cos_off = _optional_cosine(
        match_trace.tensors.get('late_assignment'),
        off_trace.tensors.get('late_assignment'),
    )
    mixed_compose_cos_off = present_cosine(
        match_trace.tensors.get('mixed_compose'),
        off_trace.tensors.get('mixed_compose'),
    )
    n_rel_l2_off = _relative_l2(match_n, off_n)
    claim_mass = unnamed_tensor(claims)
    match_mass = unnamed_tensor(match_n)
    off_mass = unnamed_tensor(off_n)
    demand_support = claim_mass.sum(dim=0) > 0
    support_w = demand_support.to(dtype=match_mass.dtype)
    silent_w = (~demand_support).to(dtype=match_mass.dtype)
    n_delta = (match_mass - off_mass).abs()
    support_delta_l1 = float((n_delta * support_w).sum().item())
    silent_delta_l1 = float((n_delta * silent_w).sum().item())
    support_match = match_mass * support_w
    n_support_rel_l2_off = float(
        (
            (support_match - off_mass * support_w).norm() / support_match.norm().clamp_min(1e-12)
        ).item()
    )
    graph_on_xa = letter_xa_margin(scored_pairs, supply_index(rows, claims, match_n), covering)
    graph_off_xa = letter_xa_margin(scored_pairs, supply_index(rows, claims, off_n), covering)
    text_only_xa = letter_xa_margin(scored_pairs, supply_index(rows, claims, text_n), covering)
    unpaid_moved = (
        graph_on_xa is not None
        and graph_off_xa is not None
        and abs(graph_on_xa - graph_off_xa) > noise
    )
    culprit, locus = name_graph_seam(
        n_rel_l2_off=n_rel_l2_off,
        prefix_cos_off=prefix_cos_off,
        unpaid_moved=unpaid_moved,
        graph_readout_cos_off=graph_readout_cos_off,
        host_text_cos_off=host_text_cos_off,
        overlay_cos_off=overlay_cos_off,
        mixed_compose_cos_off=mixed_compose_cos_off,
        cpc_nodes_changed=cpc_nodes_changed,
        inject_scale=inject_scale,
        silent_delta_l1=silent_delta_l1,
        support_delta_l1=support_delta_l1,
    )
    return {
        'named_seam': locus,
        'culprit': culprit or 'none',
        'inject_scale': inject_scale,
        'mixed_compose_cos_off': mixed_compose_cos_off,
        'n_rel_l2_off': n_rel_l2_off,
        'n_support_rel_l2_off': n_support_rel_l2_off,
        'support_delta_l1': support_delta_l1,
        'silent_delta_l1': silent_delta_l1,
        'n_rel_l2_text': _relative_l2(match_n, text_n),
        'n_cos_off': _cosine(match_n, off_n),
        'n_cos_text': _cosine(match_n, text_n),
        'prefix_cos_off': prefix_cos_off,
        'prefix_energy_on': _optional_energy(prefix_on),
        'prefix_energy_off': _optional_energy(prefix_off_t),
        'prefix_energy_text': _optional_energy(prefix_text),
        'overlay_cos_off': overlay_cos_off,
        'graph_readout_cos_off': graph_readout_cos_off,
        'host_text_cos_off': host_text_cos_off,
        'late_assign_cos_off': late_assign_cos_off,
        'pair_unpaid_rel_l2': _relative_l2(
            covering.pair_table(claims, match_n).unpaid_mass,
            covering.pair_table(claims, off_n).unpaid_mass,
        ),
        'cpc_nodes_changed': cpc_nodes_changed,
        'cpc_nodes_on_mean': cpc_nodes_on_mean,
        'cpc_nodes_off_mean': cpc_nodes_off_mean,
        'graph_swapped': graph_swapped,
        'graph_on_xa': graph_on_xa,
        'graph_off_xa': graph_off_xa,
        'text_only_xa': text_only_xa,
        'unpaid_moved': unpaid_moved,
    }


def name_graph_seam(
    *,
    n_rel_l2_off: float,
    prefix_cos_off: float | None,
    unpaid_moved: bool,
    graph_readout_cos_off: float | None = None,
    host_text_cos_off: float | None = None,
    overlay_cos_off: float | None = None,
    mixed_compose_cos_off: float | None = None,
    cpc_nodes_changed: bool = True,
    inject_scale: float = 0.0,
    silent_delta_l1: float = 0.0,
    support_delta_l1: float = 0.0,
) -> tuple[str, str]:
    """Name where rolled-graph identity is lost before covering unpaid.

    A missing cosine is unmeasured, not tied. Covering last-layer export mixes
    the RMS-matched residual onto text tokens. Prefix readout keys mixed
    compose rows plus filing HGT.
    """

    def tied(value: float | None) -> bool | None:
        if value is None:
            return None
        return value >= 0.95

    n_tied = n_rel_l2_off <= 1e-3
    prefix_tied = tied(prefix_cos_off)
    readout_tied = tied(graph_readout_cos_off)
    host_tied = tied(host_text_cos_off)
    overlay_tied = tied(overlay_cos_off)
    mixed_tied = tied(mixed_compose_cos_off)
    unmeasured = None in {prefix_tied, readout_tied, host_tied, overlay_tied}
    named = (
        ('DATA', 'IDENTICAL_FILING', n_tied and not cpc_nodes_changed),
        (
            'FORWARD_SEAM',
            'COVERING_NEVER_INJECTS',
            n_tied
            and inject_scale <= 0.0
            and mixed_tied is False
            and prefix_tied is True
            and readout_tied is True,
        ),
        (
            'FORWARD_SEAM',
            'READOUT_FILING_INVARIANT',
            n_tied and overlay_tied is True and readout_tied is True,
        ),
        (
            'FORWARD_SEAM',
            'PREFIX_PROJECTOR',
            n_tied and prefix_tied is True and readout_tied is False,
        ),
        (
            'FORWARD_SEAM',
            'INVENTORY_READOUT',
            n_tied and prefix_tied is False and host_tied is True,
        ),
        ('FORWARD_SEAM', 'ASSIGNMENT_BANK', n_tied and host_tied is False),
        ('FORWARD_SEAM', 'UNMEASURED', n_tied and unmeasured),
        (
            'SCORE',
            'SILENT_DELTA',
            not unpaid_moved
            and not n_tied
            and silent_delta_l1 > support_delta_l1
            and silent_delta_l1 > 0.0,
        ),
        ('SCORE', 'SATURATION', not unpaid_moved),
        ('', 'none', True),
    )
    return next((culprit, locus) for culprit, locus, hit in named if hit)


def _letter_stack(
    pairs: Sequence[CitationPair],
    index: IntensityIndex,
) -> tuple[tuple[str, ...], Tensor, Tensor] | None:
    live = tuple(
        (
            mark,
            index.claim[pair.query_application_number],
            index.full[pair.partner_application_number],
        )
        for pair in pairs
        if (mark := citation_letter(pair)) is not None
        and pair.query_application_number in index.claim
        and pair.partner_application_number in index.full
    )
    if not live:
        return None
    return (
        tuple(mark for mark, _query, _doc in live),
        torch.stack(tuple(query for _mark, query, _doc in live)),
        torch.stack(tuple(document for _mark, _query, document in live)),
    )


def _letter_mean(values: Tensor, marks: Sequence[str], letter: str) -> Tensor:
    mask = torch.tensor([mark == letter for mark in marks], device=values.device)
    chosen = values[mask]
    return chosen.mean(dim=0) if chosen.size(0) else values.new_zeros(values.size(-1))


def matching_heatmap(
    covering: Covering,
    cells: Mapping[str, tuple[Tensor, Tensor, tuple[str, ...]]],
) -> dict[str, CoveringScore]:
    """Product ``Covering.forward`` residual and paid tables for each ablation cell."""
    return {name: covering(query, document) for name, (query, document, _marks) in cells.items()}


def write_matching_heatmap(
    output_dir: Path,
    *,
    scores: Mapping[str, CoveringScore],
    cells: Mapping[str, tuple[Tensor, Tensor, tuple[str, ...]]],
    seam: Mapping[str, object],
    arms: Mapping[str, RelationalArms] | None = None,
) -> Path:
    """Great Tables heatmap of product paid and residual slots plus seam scalars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first = next(iter(cells.values()))
    demand = unnamed_tensor(first[0].mean(dim=0))
    keep = min(_SLOT_TOP_K, int(demand.numel()))
    residual = unnamed_tensor(scores['graph_on'].residual).mean(dim=0)
    paid_on = unnamed_tensor(scores['graph_on'].paid).mean(dim=0)
    off = scores.get('graph_off')
    delta = (
        residual.new_zeros(residual.shape)
        if off is None
        else (paid_on - unnamed_tensor(off.paid).mean(dim=0)).abs()
        + (residual - unnamed_tensor(off.residual).mean(dim=0)).abs()
    )
    slots = torch.unique(
        torch.cat((
            demand.topk(keep).indices,
            residual.topk(keep).indices,
            delta.topk(keep).indices,
        )),
        sorted=True,
    )
    letters = ('X', 'A')
    on_arms = None if arms is None else arms.get('graph_on')
    vertex_windows = (
        None
        if on_arms is None or on_arms.strongest_vertex_window is None
        else unnamed_tensor(on_arms.strongest_vertex_window)
    )

    def mean_cell(tensor: Tensor) -> Tensor:
        values = unnamed_tensor(tensor)
        return values.mean(dim=0) if values.ndim > 1 else values

    slot_rows = tuple(
        {
            'slot': int(slot),
            'demand': round(float(demand[slot].item()), 4),
            **{
                f'{cell}_{stat}_{letter}': round(
                    float(
                        _letter_mean(
                            unnamed_tensor(getattr(scores[cell], stat)),
                            cells[cell][2],
                            letter,
                        )[slot].item()
                    ),
                    4,
                )
                for cell in _HEATMAP_CELLS
                if cell in scores
                for stat in ('paid', 'residual')
                for letter in letters
            },
            **(
                {}
                if vertex_windows is None
                else {
                    'strongest_window': int(
                        vertex_windows.reshape(-1, vertex_windows.size(-1))[0, slot].item()
                    )
                }
            ),
        }
        for slot in slots.tolist()
    )
    color_cols = tuple(
        name
        for name in (slot_rows[0] if slot_rows else {})
        if name not in {'slot', 'demand', 'strongest_window'}
    )
    slot_table = GT(pl.DataFrame(slot_rows or ({'note': 'no cited X/A slots'},))).tab_header(
        title='Covering paid and residual slots (product matching heatmap)',
        subtitle='graph-on vs rolled graph-off vs zero prefix; X vs A cited partners',
    )
    painted = (
        slot_table.data_color(columns=list(color_cols), palette='YlOrRd')
        if color_cols
        else slot_table
    )
    seam_table = GT(pl.DataFrame([dict(seam)])).tab_header(
        title='Graph contribution seams into pair_table',
    )
    arm_rows = {} if arms is None else dict(arms)
    rel_demand = (
        None
        if not arm_rows
        else mean_cell(
            next(iter(arm_rows.values())).relation_type.paid
            + next(iter(arm_rows.values())).relation_type.residual
        )
    )
    rel_payload = (
        ()
        if rel_demand is None
        else tuple(
            {
                'relation': int(slot),
                **{
                    f'{cell}_rel_{stat}': round(
                        float(mean_cell(getattr(payload.relation_type, stat))[slot].item()),
                        4,
                    )
                    for cell, payload in arm_rows.items()
                    for stat in ('paid', 'residual')
                },
            }
            for slot in rel_demand.topk(min(_SLOT_TOP_K, int(rel_demand.numel()))).indices.tolist()
        )
    )
    kept_pairs = (
        None if on_arms is None else mean_cell(on_arms.collapsed_paid + on_arms.collapsed_residual)
    )
    leftover_by_cell = {
        cell: attribute_kept_residual(
            mean_cell(unnamed_tensor(scores[cell].residual)),
            kept_pairs,
        )
        for cell in _HEATMAP_CELLS
        if cell in scores and kept_pairs is not None
    }
    query_edges = leftover_by_cell.get('graph_on')
    edge_idx = (
        torch.zeros(0, dtype=torch.long)
        if query_edges is None or query_edges.ndim != 2
        else query_edges
        .reshape(-1)
        .topk(
            min(
                _SLOT_TOP_K,
                max(int((query_edges.reshape(-1) > 0).sum().item()), 1),
                int(query_edges.numel()),
            )
        )
        .indices
    )
    edge_src = (
        torch.div(edge_idx, query_edges.size(-1), rounding_mode='floor')
        if query_edges is not None and query_edges.ndim == 2
        else edge_idx
    )
    edge_dst = (
        torch.remainder(edge_idx, query_edges.size(-1))
        if query_edges is not None and query_edges.ndim == 2
        else edge_idx
    )
    edge_windows = (
        None
        if on_arms is None or on_arms.strongest_edge_window is None
        else unnamed_tensor(on_arms.strongest_edge_window)
    )
    edge_payload = (
        ()
        if query_edges is None or query_edges.ndim != 2 or kept_pairs is None
        else tuple(
            {
                'src': int(edge_src[row].item()),
                'dst': int(edge_dst[row].item()),
                'query_edge': round(
                    float(kept_pairs[edge_src[row], edge_dst[row]].item()),
                    4,
                ),
                **{
                    f'{cell}_payment': round(
                        float(
                            mean_cell(payload.collapsed_paid)[edge_src[row], edge_dst[row]].item()
                        ),
                        4,
                    )
                    for cell, payload in arm_rows.items()
                },
                **{
                    f'{cell}_leftover': round(
                        float(leftover_by_cell[cell][edge_src[row], edge_dst[row]].item()),
                        4,
                    )
                    for cell in leftover_by_cell
                },
                **(
                    {}
                    if edge_windows is None
                    else {
                        'strongest_window': int(
                            edge_windows.reshape(-1, *kept_pairs.shape)[
                                0, edge_src[row], edge_dst[row]
                            ].item()
                        )
                    }
                ),
            }
            for row in range(int(edge_idx.numel()))
        )
    )
    extras = ''.join(
        GT(pl.DataFrame(list(rows))).tab_header(title=title).as_raw_html()
        for rows, title in (
            (rel_payload, 'Relation-type paid and residual cells'),
            (edge_payload, 'Kept-edge leftover residual'),
        )
        if rows
    )
    html_path = output_dir / 'matching_heatmap.html'
    _ = html_path.write_text(
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        '<title>Covering matching heatmap</title></head><body>'
        f'{seam_table.as_raw_html()}{painted.as_raw_html()}{extras}</body></html>',
        encoding='utf-8',
    )
    write_json(
        output_dir / 'matching_heatmap.json',
        {
            'seam': dict(seam),
            'slots': list(slot_rows),
            'relations': list(rel_payload),
            'edges': list(edge_payload),
            'retention_bytes': retention_budget_bytes(),
            'conditions': list(RELATIONAL_CONDITIONS),
        },
    )
    return html_path


class TestLetterGraphContribution:
    """Host-free graph-off ablation and product covering heatmap."""

    def test_rolled_views_swap_graph_and_keep_disclosure_text(self) -> None:
        """Graph-off is a foreign HeteroData on the same disclosure string."""

        def row(app: str, graph: HeteroData, text: str) -> CollisionEncodeRow:
            return CollisionEncodeRow(
                application_number=app,
                example=SoftMlmExample(text=text, graph=graph, disclosure_text=text),
                claim_blob=f'claim-{app}',
                disclosure=text,
            )

        graphs = (HeteroData(), HeteroData(), HeteroData())
        rows = tuple(row(str(index), graph, f'disc-{index}') for index, graph in enumerate(graphs))
        draw = HupdDraw(rows=rows, n_pool=3, claim_rows=rows, disc_rows=rows)
        foreign = draw.rolled()
        swapped = draw.views(draw.disc_texts(), tuple(draw.graphs()[index] for index in foreign))
        assert foreign != tuple(range(3))
        assert tuple(row.disclosure for row in swapped) == draw.disc_texts()
        assert tuple(row.example.graph for row in swapped) != draw.graphs()
        assert swapped[0].example.graph is graphs[1]
        assert swapped[0].disclosure == 'disc-0'

    def test_product_heatmap_shows_a_saturation_versus_x_gap(
        self,
        covering: Covering,
        tmp_path: Path,
    ) -> None:
        """A document that fills every slot paints paid-A dark and residual-X live."""
        query = torch.tensor([[8.0, 0.0], [8.0, 0.0]])
        x_doc = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
        a_doc = torch.tensor([[80.0, 80.0], [80.0, 80.0]])
        marks = ('X', 'A')
        cells = {
            'graph_on': (query, torch.stack((x_doc[0], a_doc[0])), marks),
            'graph_off': (query, torch.stack((x_doc[1], a_doc[1])), marks),
            'prefix_off': (query, torch.stack((x_doc[0], a_doc[0])), marks),
        }
        scores = matching_heatmap(covering, cells)
        path = write_matching_heatmap(
            tmp_path,
            scores=scores,
            cells=cells,
            seam={'named_seam': 'PREFIX_IDENTITY', 'n_rel_l2_off': 0.0},
        )
        on = scores['graph_on']
        paid_x = float(_letter_mean(on.paid, marks, 'X').sum().item())
        paid_a = float(_letter_mean(on.paid, marks, 'A').sum().item())
        residual_x = float(_letter_mean(on.residual, marks, 'X').sum().item())
        residual_a = float(_letter_mean(on.residual, marks, 'A').sum().item())
        html = path.read_text(encoding='utf-8')
        assert path.is_file()
        assert 'graph_on_paid_X' in html
        assert paid_a > paid_x
        assert residual_x > residual_a

    def test_five_arm_heatmap_records_relation_edges_and_windows(
        self,
        covering: Covering,
        tmp_path: Path,
    ) -> None:
        """Relation-type and endpoint tables sit beside the vertex heatmap."""
        query = torch.tensor([[8.0, 2.0], [8.0, 2.0]])
        document = torch.tensor([[4.0, 1.0], [80.0, 80.0]])
        marks = ('X', 'A')
        cells = {
            'graph_on': (query, document, marks),
            'graph_off': (query, document, marks),
            'prefix_off': (query, document, marks),
        }
        labeled_query = torch.zeros(2, 2, 2, 2)
        labeled_query[:, 0, 1, 0] = 3.0
        labeled_document = torch.zeros(2, 2, 2, 2)
        labeled_document[0, 0, 1, 0] = 1.0
        labeled_document[1, 0, 1, 1] = 4.0
        collapsed_query = labeled_query.sum(dim=-1)
        collapsed_document = labeled_document.sum(dim=-1)
        report = relational_arms(
            n_query=query,
            n_document=document,
            n_rel_query=labeled_query.sum(dim=(-3, -2)),
            n_rel_document=labeled_document.sum(dim=(-3, -2)),
            labeled_query=labeled_query,
            labeled_document=labeled_document,
            collapsed_query=collapsed_query,
            collapsed_document=collapsed_document,
            covering=covering,
            window_supply=torch.stack((document, document * 0.25)),
            window_edges=torch.stack((collapsed_document, collapsed_document * 0.25)),
            passage_ids=('w0', 'w1'),
        )
        shuffled = endpoint_shuffle(labeled_document)
        relabeled = relation_label_shuffle(labeled_document)
        path = write_matching_heatmap(
            tmp_path,
            scores=matching_heatmap(covering, cells),
            cells=cells,
            seam={'named_seam': 'PREFIX_IDENTITY', 'n_rel_l2_off': 0.0},
            arms={'graph_on': report, 'graph_off': report, 'prefix_off': report},
        )
        html = path.read_text(encoding='utf-8')
        assert path.is_file()
        assert 'graph_on_paid_X' in html
        assert 'Relation-type paid and residual cells' in html
        assert 'Kept-edge leftover residual' in html
        assert 'strongest_window' in html
        assert not torch.equal(shuffled, labeled_document)
        assert torch.equal(relabeled.sum(dim=-1), labeled_document.sum(dim=-1))
        assert not torch.equal(report.labeled_unpaid, report.collapsed_unpaid)
        assert report.joint_covering.shape == report.vertex.covering.shape

    def test_heatmap_leftover_follows_kept_edge_not_refuse(
        self,
        covering: Covering,
        tmp_path: Path,
    ) -> None:
        """Heatmap leftover sits on a kept pair; dest-roll of zeros leaves it put."""
        query = torch.tensor([[8.0, 2.0], [8.0, 2.0]])
        document = torch.tensor([[4.0, 1.0], [80.0, 80.0]])
        marks = ('X', 'A')
        cells = {
            'graph_on': (query, document, marks),
            'graph_off': (query, document, marks),
            'prefix_off': (query, document, marks),
        }
        labeled_query = torch.zeros(2, 2, 2, 2)
        labeled_query[:, 0, 1, 0] = 3.0
        labeled_document = torch.zeros(2, 2, 2, 2)
        labeled_document[:, 0, 1, 0] = 1.0
        collapsed_query = labeled_query.sum(dim=-1)
        collapsed_document = labeled_document.sum(dim=-1)
        report = relational_arms(
            n_query=query,
            n_document=document,
            n_rel_query=labeled_query.sum(dim=(-3, -2)),
            n_rel_document=labeled_document.sum(dim=(-3, -2)),
            labeled_query=labeled_query,
            labeled_document=labeled_document,
            collapsed_query=collapsed_query,
            collapsed_document=collapsed_document,
            covering=covering,
        )
        scores = matching_heatmap(covering, cells)
        path = write_matching_heatmap(
            tmp_path,
            scores=scores,
            cells=cells,
            seam={'named_seam': 'PREFIX_IDENTITY', 'n_rel_l2_off': 0.0},
            arms={'graph_on': report, 'graph_off': report, 'prefix_off': report},
        )
        payload = json.loads((path.parent / 'matching_heatmap.json').read_text(encoding='utf-8'))
        pairs = {(int(row['src']), int(row['dst'])) for row in payload['edges']}
        leftover = unnamed_tensor(scores['graph_on'].residual).mean(dim=0)
        kept = unnamed_tensor(collapsed_query).mean(dim=0)
        paid = attribute_kept_residual(leftover, kept)
        rolled = attribute_kept_residual(leftover, kept.roll(1, dims=-1))
        zeros = torch.zeros_like(kept)
        refused = attribute_kept_residual(leftover, zeros)
        rolled_refuse = attribute_kept_residual(leftover, zeros.roll(1, dims=-1))
        assert (0, 1) in pairs
        assert (0, 0) not in pairs
        assert paid[0, 1].item() > 0.0
        assert paid[0, 1].item() != rolled[0, 1].item()
        assert torch.allclose(refused, rolled_refuse)

    def test_name_graph_seam_filing_invariant_versus_saturation(self) -> None:
        """Rolled CPC with a frozen readout is a forward seam, not saturation."""
        invariant = name_graph_seam(
            n_rel_l2_off=1e-6,
            prefix_cos_off=0.99,
            unpaid_moved=False,
            graph_readout_cos_off=0.99,
            overlay_cos_off=1.0,
            cpc_nodes_changed=True,
        )
        saturated = name_graph_seam(
            n_rel_l2_off=0.2,
            prefix_cos_off=0.5,
            unpaid_moved=False,
            graph_readout_cos_off=0.5,
            overlay_cos_off=0.5,
            cpc_nodes_changed=True,
        )
        projector = name_graph_seam(
            n_rel_l2_off=1e-6,
            prefix_cos_off=0.99,
            unpaid_moved=False,
            graph_readout_cos_off=0.2,
            overlay_cos_off=1.0,
            cpc_nodes_changed=True,
        )
        never_injects = name_graph_seam(
            n_rel_l2_off=1e-6,
            prefix_cos_off=0.99,
            unpaid_moved=False,
            graph_readout_cos_off=0.99,
            overlay_cos_off=1.0,
            mixed_compose_cos_off=0.1,
            cpc_nodes_changed=True,
            inject_scale=0.0,
        )
        missing_cosines = name_graph_seam(
            n_rel_l2_off=1e-6,
            prefix_cos_off=None,
            unpaid_moved=False,
            graph_readout_cos_off=None,
            overlay_cos_off=None,
            cpc_nodes_changed=True,
        )
        silent = name_graph_seam(
            n_rel_l2_off=0.2,
            prefix_cos_off=0.5,
            unpaid_moved=False,
            graph_readout_cos_off=0.5,
            overlay_cos_off=0.5,
            cpc_nodes_changed=True,
            silent_delta_l1=8.0,
            support_delta_l1=0.1,
        )
        assert invariant == ('FORWARD_SEAM', 'READOUT_FILING_INVARIANT')
        assert saturated == ('SCORE', 'SATURATION')
        assert silent == ('SCORE', 'SILENT_DELTA')
        assert projector == ('FORWARD_SEAM', 'PREFIX_PROJECTOR')
        assert never_injects == ('FORWARD_SEAM', 'COVERING_NEVER_INJECTS')
        assert missing_cosines == ('FORWARD_SEAM', 'UNMEASURED')


@pytest.mark.live
class TestLetterGraphContributionLive:
    """Matching vs rolled vs zero-prefix intensities on the ST.14 letter draw."""

    def test_graph_identity_into_covering_heatmap(
        self,
        covering_pilot: CoveringPilotSpec,
        ssv_model: SoftTrunkModel,
        load_hupd_draw: HupdLoader,
        encode_view: EncodeView,
        covering: Covering,
        covering_collator: SoftMlmCollator,
        citation_pairs: tuple[CitationPair, ...],
        attach_termhood: Callable[[tuple[str, ...]], int],
        announce_gate: GateRecorder,
        request: pytest.FixtureRequest,
    ) -> None:
        """Trace rolled graphs into n, then paint product paid/residual heatmaps."""
        eval_config = CollisionEvalConfig.from_yaml()
        _train, development, _held = split_pairs_by_query(
            citation_pairs,
            train=eval_config.split.train,
            eval_fraction=eval_config.split.eval,
            test=eval_config.split.test,
            seed=eval_config.split.seed,
            query_limit=eval_config.query_limit,
        )
        scored_pairs = covering_pilot.letter_scored_pairs(development)
        stems = tuple(
            dict.fromkeys(
                stem
                for pair in scored_pairs
                for stem in (pair.query_application_number, pair.partner_application_number)
            )
        )
        draw = load_hupd_draw(stems=stems)
        termhood_n = attach_termhood(draw.lemma_keys(ssv_model))
        graphs = draw.graphs()
        foreign = draw.rolled()
        off_graphs = tuple(graphs[index] for index in foreign)
        graph_swapped = tuple(
            match is not rolled for match, rolled in zip(graphs, off_graphs, strict=True)
        )
        cpc_changed, cpc_on_mean, cpc_off_mean = filing_node_shift(graphs, off_graphs)
        output = ledger_dir(Path(request.config.rootpath), 'letter_graph_contribution')
        collected: dict[str, tuple[dict[float, RelationalArms], dict[str, Tensor]]] = {}
        campaign_seams = frozenset({
            'labeled_endpoint',
            'collapsed_endpoint',
            'projected_prefix',
            'token_residual',
            'disclosure_intensity',
            'claim_intensity',
            'adapter_delta',
            'host_text_state',
            'late_assignment',
        })

        def composed_supply(
            filing: Sequence[HeteroData],
        ) -> tuple[Tensor, CoveringTrace, Tensor, tuple[str, ...]]:
            windows, owners = draw.window_rows(
                covering_collator.tokenizer,
                max_length=int(covering_collator.max_length),
                max_chunks=covering_pilot.disclosure_max_chunks,
                graphs=filing,
            )
            listed = tuple(windows)
            unique = tuple(
                dict.fromkeys(row.example.text for row in listed if row.example.text.strip())
            )
            if unique:
                _ = ssv_model.graph_ingress.candidates(unique)
            pair_n = max(len(covering_pair_devices()), 1)
            width = max(int(covering_pilot.batch_size) * pair_n, 1)
            chunks = tuple(tuple(chunk) for chunk in batched(listed, width))
            active = {name: unnamed_tensor(value) for name, value in covering_patches().items()}
            adapters = covering_adapters_off()

            def encode_chunk(
                chunk: tuple[CollisionEncodeRow, ...],
                start: int,
            ) -> tuple[Tensor, CoveringTrace]:
                sliced = {
                    name: value[start : start + len(chunk)]
                    if value.size(0) >= start + len(chunk)
                    else value
                    for name, value in active.items()
                }
                with (
                    keep_covering_seams(campaign_seams),
                    disable_covering_adapters() if adapters else nullcontext(),
                    patch_covering(sliced) if sliced else nullcontext(),
                ):
                    return encode_view(chunk, as_claim=False, full_trace=True)

            starts = tuple(
                sum(len(item) for item in chunks[:index]) for index in range(len(chunks))
            )
            parts = tuple(starmap(encode_chunk, zip(chunks, starts, strict=True)))
            encoded = torch.cat(tuple(item[0] for item in parts), dim=0)
            names = {name for _intensity, trace in parts for name in trace.tensors}
            with record_covering() as trace:
                _ = tuple(
                    trace.record(
                        name,
                        torch.cat(
                            tuple(
                                unnamed_tensor(item[1].tensors[name])
                                for item in parts
                                if name in item[1].tensors
                            ),
                            dim=0,
                        ),
                    )
                    for name in names
                )
            owner_ids = torch.tensor(owners, device=encoded.device, dtype=torch.long)
            return (
                covering.compose_windows(encoded, owner_ids, len(draw.rows)),
                trace,
                owner_ids.cpu(),
                tuple(f'{row.application_number}:{index}' for index, row in enumerate(listed)),
            )

        with keep_covering_seams(campaign_seams):
            claims, claim_trace = encode_view(draw.claim_rows, as_claim=True, full_trace=True)

        match_n, match_trace, owner_ids, passages = composed_supply(graphs)
        prefix = match_trace.tensors.get('projected_prefix')
        prefix_zeros = None if prefix is None else torch.zeros_like(unnamed_tensor(prefix))
        prefix_patch = {} if prefix_zeros is None else {'projected_prefix': prefix_zeros}
        residual_patch = {'token_residual': torch.zeros(1)}

        def arms_for(
            supply: Tensor,
            trace: CoveringTrace,
            labeled_document: Tensor | None = None,
            sigma_edge: Tensor | None = None,
        ) -> RelationalArms | None:
            claim_labeled = claim_trace.tensors.get('labeled_endpoint')
            window_labeled = trace.tensors.get('labeled_endpoint')
            window_collapsed = trace.tensors.get('collapsed_endpoint')
            claim_collapsed = claim_trace.tensors.get('collapsed_endpoint')
            if (
                claim_labeled is None
                or window_labeled is None
                or window_collapsed is None
                or claim_collapsed is None
            ):
                return None
            claim_w = unnamed_tensor(claim_labeled)[:, CLAIM_MASK]
            claim_c = unnamed_tensor(claim_collapsed)[:, CLAIM_MASK]
            window_w = unnamed_tensor(window_labeled)[:, FULL_MASK]
            doc_w = owner_mass_rows(
                window_w if labeled_document is None else labeled_document,
                owner_ids,
                len(draw.rows),
            )
            doc_c = owner_mass_rows(
                unnamed_tensor(window_collapsed)[:, FULL_MASK]
                if labeled_document is None
                else labeled_document.sum(dim=-1),
                owner_ids,
                len(draw.rows),
            )
            return relational_arms(
                n_query=claims,
                n_document=supply,
                n_rel_query=claim_w.sum(dim=(-3, -2)),
                n_rel_document=doc_w.sum(dim=(-3, -2)),
                labeled_query=claim_w,
                labeled_document=doc_w,
                collapsed_query=claim_c,
                collapsed_document=doc_c,
                covering=covering,
                window_supply=unnamed_tensor(trace.tensors['disclosure_intensity'])
                if 'disclosure_intensity' in trace.tensors
                else None,
                window_edges=unnamed_tensor(window_collapsed)[:, FULL_MASK],
                window_owners=owner_ids,
                passage_ids=passages,
                sigma_edge=sigma_edge,
            )

        def stash(
            name: str,
            supply: Tensor,
            trace: CoveringTrace,
            labeled: Tensor | None = None,
        ) -> None:
            collected[name] = (
                {
                    sigma: report
                    for sigma in DECLARED_EDGE_SWEEP
                    if (
                        report := arms_for(
                            supply,
                            trace,
                            labeled,
                            sigma_edge=supply.new_tensor(sigma),
                        )
                    )
                    is not None
                },
                retain_seams(trace, name, labeled_composed=labeled),
            )

        with shift_overlay_edges(dest=1):
            shuffle_n, shuffle_trace, _shuffle_owners, _shuffle_passages = composed_supply(graphs)
        with shift_overlay_edges(attr=1):
            relabel_n, relabel_trace, _relabel_owners, _relabel_passages = composed_supply(graphs)
        off_n, off_trace, _off_owners, _off_passages = composed_supply(off_graphs)
        with patch_covering(prefix_patch):
            text_n, text_trace, _text_owners, _text_passages = composed_supply(graphs)
        with patch_covering(residual_patch):
            residual_n, residual_trace, _residual_owners, _residual_passages = composed_supply(
                graphs
            )
        with patch_covering({**prefix_patch, **residual_patch}):
            both_n, both_trace, _both_owners, _both_passages = composed_supply(graphs)
        with disable_covering_adapters():
            adapter_n, adapter_trace, _adapter_owners, _adapter_passages = composed_supply(graphs)
        stash('matching', match_n, match_trace)
        stash('endpoint_shuffle', shuffle_n, shuffle_trace)
        stash('relation_label_shuffle', relabel_n, relabel_trace)
        stash('rolled_filing', off_n, off_trace)
        stash('prefix_off', text_n, text_trace)
        stash('residual_off', residual_n, residual_trace)
        stash('both_off', both_n, both_trace)
        stash('adapter_off', adapter_n, adapter_trace)
        write_frozen_campaign(output, conditions=collected)
        seam = graph_contribution_seam(
            claims=claims,
            match_n=match_n,
            off_n=off_n,
            text_n=text_n,
            match_trace=match_trace,
            off_trace=off_trace,
            text_trace=text_trace,
            covering=covering,
            rows=draw.rows,
            scored_pairs=scored_pairs,
            noise=covering_pilot.letter.noise,
            cpc_nodes_changed=cpc_changed,
            cpc_nodes_on_mean=cpc_on_mean,
            cpc_nodes_off_mean=cpc_off_mean,
            graph_swapped=all(graph_swapped),
            inject_scale=ssv_model.covering_inject_scale,
        )
        packed_cells = {
            name: heatmap_cell(scored_pairs, draw.rows, claims, supply)
            for name, supply in (
                ('graph_on', match_n),
                ('graph_off', off_n),
                ('prefix_off', text_n),
            )
        }
        live_cells = {name: cell for name, cell in packed_cells.items() if cell is not None}
        scores = matching_heatmap(covering, live_cells)
        live_arms = {
            name: report
            for name, report in (
                ('graph_on', arms_for(match_n, match_trace)),
                ('graph_off', arms_for(off_n, off_trace)),
                *(
                    (name, arms_for(supply, trace, labeled))
                    for name, (supply, trace, labeled) in (
                        ('matching', (match_n, match_trace, None)),
                        ('rolled_filing', (off_n, off_trace, None)),
                        ('endpoint_shuffle', (shuffle_n, shuffle_trace, None)),
                        ('relation_label_shuffle', (relabel_n, relabel_trace, None)),
                        ('prefix_off', (text_n, text_trace, None)),
                    )
                ),
            )
            if report is not None
        }
        heatmap = write_matching_heatmap(
            output,
            scores=scores,
            cells=live_cells,
            seam=seam,
            arms=live_arms or None,
        )
        campaign = output / 'frozen_campaign'
        complete = tuple(
            (campaign / name / 'complete.json').is_file()
            and (campaign / name / 'arms.pt').is_file()
            and (campaign / name / 'seams.pt').is_file()
            for name in RELATIONAL_CONDITIONS
        )
        announce_gate(
            'letter_graph_contribution',
            {
                'n_patents': len(draw.rows),
                'n_scored_pairs': len(scored_pairs),
                'termhood_n': termhood_n,
                **seam,
                'heatmap': str(heatmap),
                'campaign': str(campaign),
                'declared_conditions': list(RELATIONAL_CONDITIONS),
                'declared_count': len(RELATIONAL_CONDITIONS),
                'complete_conditions': int(sum(complete)),
                'sweep': list(DECLARED_EDGE_SWEEP),
                'seams': tuple(match_trace.tensors),
                'arm_conditions': tuple(live_arms),
                'retention': SEAM_RETENTION.to_dicts(),
                'retention_bytes': retention_budget_bytes(),
            },
            culprit=str(seam['culprit']),
            culprit_name=str(seam['named_seam']),
        )
        assert all(graph_swapped)
        assert heatmap.is_file()
        assert all(complete)
        assert float(unnamed_tensor(match_n).abs().sum().item()) > 0.0
