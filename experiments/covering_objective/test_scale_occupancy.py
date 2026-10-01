"""Edge-scale quantiles and claim endpoint occupancy. No letter scores."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import batched

import pytest
import torch
from torch import Tensor

from experiments.covering_objective.corpus import HupdDraw, HupdLoader
from experiments.covering_objective.encode_pool import EncodeView, unnamed_tensor
from experiments.covering_objective.ledgers import GateRecorder
from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.relational_arms import (
    CLAIM_MASK,
    FULL_MASK,
    ClaimOccupancyReport,
    EdgeScaleReport,
    claim_endpoint_occupancy,
    edge_scale_from_labeled,
    gate_a,
)
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.data.citation_pairs import CitationPair, split_pairs_by_query
from ip_claim.collision.diagnose import citation_letter
from ip_claim.collision.encode_job import CollisionEncodeRow
from ip_claim.ssv.covering_trace import CoveringTrace, keep_covering_seams
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = pytest.mark.experiment


def _toy_labeled() -> Tensor:
    table = torch.zeros(2, 4, 4, 3)
    table[0, 0, 1, 0] = 2.0
    table[0, 0, 2, 1] = 1.0
    table[0, 1, 3, 2] = 3.0
    table[1, 2, 0, 0] = 4.0
    table[1, 3, 1, 1] = 5.0
    return table


def _labeled_from_trace(trace: CoveringTrace, mask: int) -> Tensor | None:
    raw = trace.tensors.get('labeled_endpoint')
    if raw is None:
        return None
    table = unnamed_tensor(raw)
    if table.ndim < 2 or table.size(1) <= mask:
        return None
    return table[:, mask]


def _letter_stems(pairs: Sequence[CitationPair]) -> frozenset[str]:
    return frozenset(
        stem
        for pair in pairs
        if citation_letter(pair) is not None
        for stem in (pair.query_application_number, pair.partner_application_number)
    )


def _without_stems(draw: HupdDraw, banned: frozenset[str], limit: int) -> HupdDraw:
    kept = tuple(
        (row, claim, disc)
        for row, claim, disc in zip(draw.rows, draw.claim_rows, draw.disc_rows, strict=True)
        if row.application_number not in banned
    )[:limit]
    return HupdDraw(
        rows=tuple(item[0] for item in kept),
        n_pool=draw.n_pool,
        claim_rows=tuple(item[1] for item in kept),
        disc_rows=tuple(item[2] for item in kept),
    )


class TestScaleOccupancyHost:
    """CPU occupancy, quantile scale, and Gate A. No encode."""

    def test_occupancy_counts_nonzero_pairs_and_top_mass(self) -> None:
        """Two claims, five distinct pairs, concentrated top mass."""
        report = claim_endpoint_occupancy(_toy_labeled(), top_n=1)
        assert report.n_claims == 2
        assert report.n_nonzero == 2
        assert report.n_distinct_pairs_union == 5
        assert report.mean_occupied_pairs == pytest.approx(2.5)
        assert report.total_mass == pytest.approx(15.0)
        assert report.mean_top_1_fraction == pytest.approx((0.5 + 5.0 / 9.0) / 2.0)
        assert report.bank == 4

    def test_empty_labeled_tensor_is_inventory_empty(self) -> None:
        """Zero mass cannot declare a scale or pass Gate A."""
        labeled = torch.zeros(3, 8, 8, 4)
        occupancy = claim_endpoint_occupancy(labeled)
        scale = edge_scale_from_labeled(labeled)
        verdict = gate_a(occupancy, scale)
        assert occupancy.n_nonzero == 0
        assert occupancy.total_mass == pytest.approx(0.0)
        assert scale.sigma_edge is None
        assert scale.n_occupied_cells == 0
        assert verdict.passed is False
        assert verdict.locus == 'RELATION_INVENTORY_EMPTY'

    def test_flat_dense_demand_is_code_diffuse(self) -> None:
        """Near-complete equal mass has no usable pair concentration."""
        labeled = torch.ones(1, 32, 32, 2)
        occupancy = claim_endpoint_occupancy(labeled, top_n=32)
        scale = EdgeScaleReport(
            sigma_edge=1.0,
            sweep_low=1.0,
            sweep_high=1.0,
            n_documents=4,
            n_documents_nonzero=4,
            n_occupied_cells=8,
            quantile_p25=1.0,
            quantile_p50=1.0,
            quantile_p75=1.0,
        )
        verdict = gate_a(occupancy, scale)
        assert occupancy.n_nonzero == 1
        assert occupancy.mean_occupied_pairs == pytest.approx(1024.0)
        assert occupancy.mean_top_n_fraction == pytest.approx(32.0 / 1024.0)
        assert verdict.passed is False
        assert verdict.locus == 'RELATION_CODE_DIFFUSE'

    def test_sigma_edge_is_occupied_cell_median_not_document_l1(self) -> None:
        """Presence is per-cell. Document L1 is recorded and is not sigma."""
        labeled = _toy_labeled()
        scale = edge_scale_from_labeled(labeled)
        cells = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = torch.quantile(cells, torch.tensor([0.25, 0.50, 0.75]))
        assert scale.n_occupied_cells == 5
        assert scale.sigma_edge == pytest.approx(float(expected[1].item()))
        assert scale.sweep_low == pytest.approx(float(expected[0].item()))
        assert scale.sweep_high == pytest.approx(float(expected[2].item()))
        assert scale.document_l1_median is not None
        assert scale.document_l1_median != scale.sigma_edge
        occupancy = claim_endpoint_occupancy(labeled)
        verdict = gate_a(occupancy, scale)
        assert verdict.passed is True
        assert verdict.locus is None

    def test_gate_a_does_not_consume_letter_scores(self) -> None:
        """Gate A reads occupancy and scale only."""
        occupancy = ClaimOccupancyReport(
            n_claims=4,
            n_nonzero=3,
            n_distinct_pairs_union=6,
            mean_occupied_pairs=2.0,
            top_n=32,
            mean_top_n_fraction=0.8,
            mean_top_1_fraction=0.4,
            mean_top_8_fraction=0.8,
            total_mass=12.0,
            bank=256,
        )
        scale = EdgeScaleReport(
            sigma_edge=0.5,
            sweep_low=0.2,
            sweep_high=0.9,
            n_documents=16,
            n_documents_nonzero=16,
            n_occupied_cells=40,
        )
        verdict = gate_a(occupancy, scale)
        dumped = verdict.model_dump()
        assert dumped.keys() == {'passed', 'locus', 'occupancy', 'scale'}
        assert 'unpaid' not in dumped
        assert 'letters' not in dumped
        assert 'unpaid' not in dumped['occupancy']
        assert 'unpaid' not in dumped['scale']


@pytest.mark.live
class TestScaleOccupancyLive:
    """Encode cited-pair claims and held-out non-letter filings. Do not score letters."""

    def test_edge_scale_and_claim_occupancy(
        self,
        covering_pilot: CoveringPilotSpec,
        ssv_model: SoftTrunkModel,
        load_hupd_draw: HupdLoader,
        encode_view: EncodeView,
        citation_pairs: tuple[CitationPair, ...],
        attach_termhood: Callable[[tuple[str, ...]], int],
        announce_gate: GateRecorder,
    ) -> None:
        """Fix sigma_edge from held-out cells; read claim W occupancy; judge Gate A."""
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
        claim_stems = tuple(
            dict.fromkeys(
                stem
                for pair in scored_pairs
                for stem in (pair.query_application_number, pair.partner_application_number)
            )
        )
        claims_draw = load_hupd_draw(stems=claim_stems, occupy=False)
        banned = _letter_stems(citation_pairs)
        held_draw = _without_stems(
            load_hupd_draw(
                covering_pilot.hygiene_n * 4,
                seed=covering_pilot.seed + 17,
                occupy=False,
            ),
            banned,
            covering_pilot.hygiene_n,
        )
        if not claims_draw.claim_rows:
            announce_gate(
                'scale_occupancy',
                {'n_claim_stems': len(claim_stems), 'n_held': len(held_draw.disc_rows)},
                culprit='DATA',
                culprit_name='CITED_PAIR_DRAW_EMPTY',
            )
            pytest.fail('cited-pair claim draw is empty')
        if not held_draw.disc_rows:
            announce_gate(
                'scale_occupancy',
                {
                    'n_claim_stems': len(claim_stems),
                    'n_held': 0,
                    'n_letter_stems': len(banned),
                },
                culprit='DATA',
                culprit_name='HELD_OUT_NON_LETTER_EMPTY',
            )
            pytest.fail('held-out non-letter draw is empty')
        termhood_n = attach_termhood(
            claims_draw.lemma_keys(ssv_model, claims_draw.claim_rows)
            + held_draw.lemma_keys(ssv_model, held_draw.disc_rows)
        )
        width = covering_pilot.disclosure_max_chunks

        def labeled_encode(
            rows: Sequence[CollisionEncodeRow],
            *,
            as_claim: bool,
            mask: int,
        ) -> tuple[Tensor | None, CoveringTrace]:
            chunks = tuple(tuple(chunk) for chunk in batched(tuple(rows), width))
            with keep_covering_seams(frozenset({'labeled_endpoint'})):
                traces = tuple(
                    encode_view(chunk, as_claim=as_claim, full_trace=True)[1] for chunk in chunks
                )
            tables = tuple(_labeled_from_trace(trace, mask) for trace in traces)
            empty = CoveringTrace()
            if not traces or any(table is None for table in tables):
                return None, traces[0] if traces else empty
            stacked = tuple(table for table in tables if isinstance(table, Tensor))
            return torch.cat(stacked, dim=0), traces[0]

        claim_w, claim_trace = labeled_encode(
            claims_draw.claim_rows, as_claim=True, mask=CLAIM_MASK
        )
        doc_w, doc_trace = labeled_encode(held_draw.disc_rows, as_claim=False, mask=FULL_MASK)
        if claim_w is None or doc_w is None:
            announce_gate(
                'scale_occupancy',
                {
                    'claim_seams': tuple(claim_trace.tensors),
                    'doc_seams': tuple(doc_trace.tensors),
                    'termhood_n': termhood_n,
                },
                culprit='LABELED_ENDPOINT_SEAM_MISSING',
                culprit_name='LABELED_ENDPOINT_SEAM_MISSING',
            )
            pytest.fail('labeled_endpoint seam missing from the live encode trace')
        occupancy = claim_endpoint_occupancy(claim_w)
        scale = edge_scale_from_labeled(doc_w)
        verdict = gate_a(occupancy, scale)
        payload = {
            'n_claim_rows': len(claims_draw.claim_rows),
            'n_held_docs': len(held_draw.disc_rows),
            'n_letter_stems_excluded': len(banned),
            'termhood_n': termhood_n,
            'claim_seams': tuple(claim_trace.tensors),
            **verdict.model_dump(mode='json'),
        }
        announce_gate(
            'scale_occupancy',
            payload,
            culprit='none' if verdict.passed else str(verdict.locus),
            culprit_name='GATE_A_PASSED' if verdict.passed else str(verdict.locus),
        )
        if not verdict.passed:
            pytest.fail(f'Gate A failed: {verdict.locus}')
        assert scale.sigma_edge is not None
        assert scale.sweep_low is not None
        assert scale.sweep_high is not None
        assert occupancy.n_nonzero >= 1
