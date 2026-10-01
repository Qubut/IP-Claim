"""Host-free mechanism, fusion, and letter judgment on a toy frozen campaign."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.covering_objective.frozen_judgment import (
    THIRD_OUTCOME,
    document_stems,
    gate_b,
    gate_c,
    judge_frozen_campaign,
    load_campaign,
    query_mean_interval,
)
from experiments.covering_objective.relational_arms import (
    DECLARED_EDGE_SWEEP,
    RELATIONAL_CONDITIONS,
    RelationalArms,
    endpoint_shuffle,
    owner_mass_rows,
    relational_arms,
    retain_seams,
    write_frozen_campaign,
)
from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.data.citation_pairs import CitationPair, St14Mark
from ip_claim.ssv.covering_trace import CoveringTrace

pytestmark = pytest.mark.experiment


def _pair(query: str, partner: str, mark: St14Mark) -> CitationPair:
    return CitationPair(
        query_application_number=query,
        partner_application_number=partner,
        marks=(mark,),
    )


def _labeled(n_docs: int) -> torch.Tensor:
    table = torch.zeros(n_docs, 4, 4, 3)
    src = torch.zeros(n_docs, dtype=torch.long)
    dest = torch.ones(n_docs, dtype=torch.long)
    rel = torch.zeros(n_docs, dtype=torch.long)
    table[torch.arange(n_docs), src, dest, rel] = 4.0
    return table


def _campaign(
    tmp_path: Path,
    *,
    shuffle: bool,
    n_docs: int = 8,
    scramble_source: bool = False,
) -> Path:
    covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=0.5))
    labeled = _labeled(n_docs)
    shuffled = endpoint_shuffle(labeled, shift=1) if shuffle else labeled.clone()
    if scramble_source:
        moved = shuffled.clone()
        moved[:, 1] = shuffled[:, 0]
        moved[:, 0] = 0
        shuffled = moved
    query = torch.ones(n_docs, 4)
    document = torch.ones(n_docs, 4)
    passages = tuple(f'{10_000_000 + index}:0' for index in range(n_docs))
    owners = tuple(range(n_docs))

    def arms_of(document_labeled: torch.Tensor) -> RelationalArms:
        return relational_arms(
            n_query=query,
            n_document=document,
            n_rel_query=labeled.sum(dim=(-3, -2)),
            n_rel_document=document_labeled.sum(dim=(-3, -2)),
            labeled_query=labeled,
            labeled_document=document_labeled,
            collapsed_query=labeled.sum(dim=-1),
            collapsed_document=document_labeled.sum(dim=-1),
            covering=covering,
            passage_ids=passages,
            window_owners=torch.tensor(owners),
        )

    match_arms = arms_of(labeled)
    shuffle_arms = arms_of(shuffled)
    rolled = labeled.clone()
    rolled[:, 0, 2, 0] = 9.0
    rolled_arms = arms_of(rolled)
    prefix = torch.ones(n_docs, 2, 8)
    residual = torch.ones(n_docs, 8)

    def trace_of(labeled_table: torch.Tensor) -> CoveringTrace:
        dest = labeled_table.sum(dim=(-3, -1)).argmax(dim=-1)
        assignment = torch.nn.functional.one_hot(dest, num_classes=labeled_table.size(2)).to(
            dtype=torch.float32
        )
        return CoveringTrace(
            tensors={
                'labeled_endpoint': labeled_table,
                'token_residual': residual,
                'projected_prefix': prefix,
                'adapter_delta': torch.zeros(n_docs, 2, 8),
                'claim_intensity': query,
                'disclosure_intensity': document,
                'late_assignment': assignment,
            }
        )

    payload = {
        'matching': (
            dict.fromkeys(DECLARED_EDGE_SWEEP, match_arms),
            retain_seams(trace_of(labeled), 'matching', labeled_composed=labeled),
        ),
        'endpoint_shuffle': (
            dict.fromkeys(DECLARED_EDGE_SWEEP, shuffle_arms),
            retain_seams(trace_of(shuffled), 'endpoint_shuffle', labeled_composed=shuffled),
        ),
        'rolled_filing': (
            dict.fromkeys(DECLARED_EDGE_SWEEP, rolled_arms),
            retain_seams(trace_of(rolled), 'rolled_filing', labeled_composed=rolled),
        ),
    }
    extra = {name: payload['matching'] for name in RELATIONAL_CONDITIONS if name not in payload}
    return write_frozen_campaign(tmp_path, conditions={**payload, **extra})


class TestFrozenJudgmentHost:
    """CPU mechanism gate, fusion locus, and letter report. No encode."""

    def test_owner_mass_rows_keeps_heaviest_window(self) -> None:
        """Tied mass keeps the earliest window; empty owners stay zero."""
        values = torch.tensor([[1.0, 0.0], [3.0, 0.0], [3.0, 1.0]])
        owners = torch.tensor([0, 0, 1])
        rows = owner_mass_rows(values, owners, 2)
        torch.testing.assert_close(rows[0], values[1])
        torch.testing.assert_close(rows[1], values[2])

    def test_document_stems_take_first_window_per_owner(self) -> None:
        """Passage ids map onto composed documents without a Python loop body."""
        stems = document_stems(('11:0', '11:1', '22:0'), (0, 0, 1), 2)
        assert stems == ('11', '22')

    def test_query_bootstrap_excludes_zero_on_one_sided_means(self) -> None:
        """Resampling query rows, not pairs, yields a CI that misses zero."""
        values = torch.full((12,), 0.4)
        interval = query_mean_interval(values)
        assert interval.n_queries == 12
        assert interval.excludes_zero
        assert interval.low is not None
        assert interval.low > 0.0

    def test_endpoint_shuffle_opens_the_mechanism_gate(self, tmp_path: Path) -> None:
        """Dest-axis roll changes C_W, keeps marginals, and is not a vertex transform."""
        root = _campaign(tmp_path, shuffle=True)
        campaign = load_campaign(root)
        covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=0.5))
        pairs = tuple(
            _pair(f'{10_000_000 + index}', f'{10_000_000 + ((index + 1) % 8)}', 'X')
            for index in range(8)
        )
        verdict = gate_b(campaign, covering, pairs)
        assert all(item.marginals_preserved for item in verdict.scales)
        assert all(item.delta_interval.excludes_zero for item in verdict.scales)
        assert verdict.passed
        assert verdict.locus is None

    def test_identity_shuffle_is_endpoint_binding_invariant(self, tmp_path: Path) -> None:
        """A no-op dest roll cannot move C_W."""
        root = _campaign(tmp_path, shuffle=False)
        campaign = load_campaign(root)
        covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=0.5))
        verdict = gate_b(campaign, covering, ())
        assert verdict.passed is False
        assert verdict.locus == 'ENDPOINT_BINDING_INVARIANT'

    def test_moved_cw_with_broken_marginals_is_not_invariant(self, tmp_path: Path) -> None:
        """Dest shuffle that moves C_W but breaks source mass cannot be judged as invariant."""
        root = _campaign(tmp_path, shuffle=True, scramble_source=True)
        campaign = load_campaign(root)
        covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=0.5))
        pairs = tuple(
            _pair(f'{10_000_000 + index}', f'{10_000_000 + ((index + 1) % 8)}', 'X')
            for index in range(8)
        )
        verdict = gate_b(campaign, covering, pairs)
        assert all(item.delta_interval.excludes_zero for item in verdict.scales)
        assert all(item.marginals_preserved is False for item in verdict.scales)
        assert verdict.passed is False
        assert verdict.locus == 'SHUFFLE_MARGINAL_UNPRESERVED'

    def test_fusion_names_missing_train_export_or_undecodable_endpoint(
        self,
        tmp_path: Path,
    ) -> None:
        """Covering-export artifacts never stored a training-conditioned readout."""
        root = _campaign(tmp_path, shuffle=True)
        campaign = load_campaign(root)
        fusion = gate_c(campaign)
        assert fusion.passed is False
        assert isinstance(fusion.locus, str)
        assert fusion.locus in {'ENDPOINT_NOT_DECODABLE', 'TRAIN_EXPORT_CONDITION_MISMATCH'}
        assert fusion.train_export_present is False
        assert fusion.gradient_cosines_present is False
        adapter = next(item for item in fusion.seam_deltas if item.name == 'adapter_delta')
        assert adapter.separates is False
        if fusion.probe.n_test:
            assert fusion.probe.accuracy is not None
            assert fusion.probe.permutation_accuracy is not None

    def test_letter_report_and_third_outcome_when_xa_does_not_improve(
        self,
        tmp_path: Path,
    ) -> None:
        """Letters are written whatever the fusion locus. X versus A is the contrast."""
        root = _campaign(tmp_path, shuffle=True)
        covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=0.5))
        pairs = (
            _pair('10000000', '10000001', 'X'),
            _pair('10000000', '10000002', 'A'),
            _pair('10000003', '10000004', 'Y'),
            _pair('10000003', '10000005', 'A'),
        )
        report = judge_frozen_campaign(root, pairs, covering)
        assert report.letters.n_pairs >= 1
        assert 'X' in report.letters.arms['matching'].endpoint
        assert 'A' in report.letters.arms['matching'].endpoint
        assert report.gate_c.passed is False
        if report.gate_b.passed and not report.letters.xa_improved:
            assert report.third_outcome == THIRD_OUTCOME
        if not report.gate_b.passed:
            assert report.third_outcome is None
