"""Leftover unpaid of demand against Phi is attributed to kept typed edges.

Dest-axis roll of a kept pair moves residual attribution. Consumed refuse
does not enter the pair table, so rolling or including it leaves attribution
put. Occupy-only intensity is not the supply when kept addends exist.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ip_claim.collision.artefacts import ExplainState, artefact_from_states
from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.explain import Explain, attribute_kept_residual
from ip_claim.ssv.covering_trace import record_covering
from ip_claim.ssv.inspect import (
    DocumentInspection,
    EntitySpan,
    GraphInspectionReport,
    RelationArc,
    SnapshotInspection,
)
from ip_claim.ssv.phi import PhiIntensity

_SLOTS = 3
_OCCUPY = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64)
_DEMAND = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
_KEPT = (0, 1)
_REFUSE = (0, 2)
_EDGE_MASS = 0.5
_DEST_SHIFT = 1


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def explain(covering: Covering) -> Explain:
    return Explain(covering)


def _pair_table(*, kept: float = _EDGE_MASS, refuse: float = 0.0) -> Tensor:
    table = torch.zeros(_SLOTS, _SLOTS, dtype=torch.float64)
    table[_KEPT] = kept
    table[_REFUSE] = refuse
    return table


def _state(
    app: str,
    assignment: Tensor,
    edges: Tensor,
    *,
    claim_mask: Tensor | None = None,
) -> ExplainState:
    tokens = assignment.size(0)
    return ExplainState(
        application_number=app,
        assignment=assignment,
        attention_mask=torch.ones(tokens),
        claim_mask=claim_mask if claim_mask is not None else torch.ones(tokens),
        claim_edges=edges,
        full_edges=edges,
    )


class TestKeptResidualAttribution:
    """Kept-edge roll moves leftover attribution; refuse does not."""

    def test_kept_edge_roll_moves_attribution(self, explain: Explain) -> None:
        """Dest-axis roll of a kept pair moves Phi leftover on that cell."""
        pairs = _pair_table()
        rolled = pairs.roll(_DEST_SHIFT, dims=-1)
        _paid_n, paid_residual, paid = explain.leftover_on_kept(_DEMAND, _OCCUPY, pairs)
        _moved_n, moved_residual, moved = explain.leftover_on_kept(_DEMAND, _OCCUPY, rolled)
        assert not torch.allclose(paid_residual, moved_residual)
        assert paid[_KEPT].item() != moved[_KEPT].item()
        assert paid[_KEPT].item() > 0.0
        assert torch.equal(moved[_KEPT], moved.new_zeros(()))
        assert moved[0, 2].item() > 0.0

    def test_refuse_does_not_appear_or_move_attribution(self, explain: Explain) -> None:
        """A consumed refuse cell stays zero; dest-roll of zeros leaves attribution put."""
        kept = _pair_table(refuse=0.0)
        zeros = torch.zeros(_SLOTS, _SLOTS, dtype=torch.float64)
        paid = explain.leftover_on_kept(_DEMAND, _OCCUPY, kept)[2]
        empty = explain.leftover_on_kept(_DEMAND, _OCCUPY, zeros)[2]
        rolled_zeros = zeros.roll(_DEST_SHIFT, dims=-1)
        rolled_refuse = explain.leftover_on_kept(_DEMAND, _OCCUPY, rolled_zeros)[2]
        assert torch.equal(paid[_REFUSE], paid.new_zeros(()))
        assert torch.allclose(empty, rolled_refuse)
        assert torch.equal(
            attribute_kept_residual(_DEMAND.new_ones(_SLOTS), kept)[_REFUSE],
            kept.new_zeros(()),
        )

    def test_occupy_only_is_not_supply_when_kept_addends_exist(self, explain: Explain) -> None:
        """Phi supply exceeds occupy; leftover against occupy-only differs."""
        pairs = _pair_table()
        supply, residual, attributed = explain.leftover_on_kept(_DEMAND, _OCCUPY, pairs)
        occupy_only = explain.covering(_DEMAND, _OCCUPY).residual
        phi = PhiIntensity(_SLOTS)
        assert torch.allclose(supply, phi(_OCCUPY, pairs, _OCCUPY > 0))
        assert not torch.allclose(supply, _OCCUPY)
        assert not torch.allclose(residual, occupy_only)
        assert attributed[_KEPT].item() > 0.0


class TestExplainArtefactReadsPhi:
    """Explain residual uses occupy plus kept-edge addends."""

    def test_kept_addends_change_artefact_unpaid(self, explain: Explain) -> None:
        """A kept pair changes leftover unpaid versus occupy-only edges."""
        assignment = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        zeros = torch.zeros(3, 3)
        kept = torch.zeros(3, 3)
        kept[0, 1] = 3.0
        query = _state('100', assignment, zeros, claim_mask=torch.tensor([1.0, 1.0, 0.0]))
        occupy_only = artefact_from_states(
            query,
            _state('200', assignment, zeros),
            explain,
            tau=0.1,
        )
        with_kept = artefact_from_states(
            query,
            _state('200', assignment, kept),
            explain,
            tau=0.1,
        )
        rolled = artefact_from_states(
            query,
            _state('200', assignment, kept.roll(_DEST_SHIFT, dims=-1)),
            explain,
            tau=0.1,
        )
        refused = artefact_from_states(
            query,
            _state('200', assignment, zeros.roll(_DEST_SHIFT, dims=-1)),
            explain,
            tau=0.1,
        )
        assert occupy_only.unpaid != with_kept.unpaid
        assert with_kept.unpaid != rolled.unpaid
        assert occupy_only.unpaid == pytest.approx(refused.unpaid)


class TestInspectResidualPane:
    """Inspect residual rows list leftover on kept arcs, never refuse."""

    def test_residual_rows_omit_zero_pair_and_show_kept_leftover(self) -> None:
        """The leftover pane lists kept arcs; a refuse cell is not a row."""
        kept = RelationArc(
            application_number='1',
            src_start=0,
            src_end=1,
            dst_start=1,
            dst_end=2,
            src_code=0,
            dst_code=1,
            rel_code=0,
            mass=0.5,
            residual=0.8,
            src_text='pump',
            dst_text='housing',
        )
        snapshot = SnapshotInspection(
            checkpoint='a.ckpt',
            global_step=1,
            documents=(
                DocumentInspection(
                    application_number='1',
                    text='pump housing',
                    spans=(
                        EntitySpan(
                            application_number='1',
                            start=0,
                            end=1,
                            code=0,
                            mass=0.9,
                            text='pump',
                        ),
                    ),
                    relations=(kept,),
                ),
            ),
            defects=(),
            entity_occupancy=1.0,
            relation_occupancy=1.0,
            mean_row_entropy=1.0,
            n_dead_entity=0,
            n_dead_relation=0,
        )
        rows = snapshot.residual_rows()
        assert len(rows) == 1
        assert rows[0]['src_code'] == 0
        assert rows[0]['dst_code'] == 1
        assert rows[0]['leftover'] == pytest.approx(0.8)
        html = GraphInspectionReport(earlier=snapshot).residual_table().as_html()
        assert 'Kept-edge leftover residual' in html
        assert '0.8' in html
        assert (2, 0) not in {(row['src_code'], row['dst_code']) for row in rows}


class TestCoveringTraceKeptResidual:
    """Active covering capture stores leftover on kept pairs."""

    def test_leftover_on_kept_records_kept_residual(self, explain: Explain) -> None:
        """A live covering trace stores leftover attribution under kept_residual."""
        pairs = _pair_table()
        with record_covering() as trace:
            attributed = explain.leftover_on_kept(_DEMAND, _OCCUPY, pairs)[2]
        assert 'kept_residual' in trace.tensors
        assert torch.allclose(trace.tensors['kept_residual'].rename(None), attributed)
        assert trace.tensors['kept_residual'].names == ('src', 'dst')
