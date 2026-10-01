"""Letter protocol: leftover unpaid of numbered demand, combination residual, high-U A.

X is single-document leftover unpaid of numbered-claim demand against Phi.
Y is the combination residual (product of gaps) outside Phi. A is high leftover
unpaid on that demand, including when shared-field occupancy is high. CLEF-IP
grade 2 is not Y. Date and evidence letters are not occupancy bars. Missing
locators are unmapped, not a letter win. Category letters are not overlay
edges. These tests do not train a trunk.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.data.citation_pairs import AnalyzeCited
from ip_claim.ssv.phi import PhiIntensity

_SLOTS = 4
_DISTINCTIVE = 0
_COMPLEMENT = 1
_SHARED = 2
_SILENT = 3
_RELEVANCE_MARKS = frozenset({'A', 'X', 'Y'})
_DATE_EVIDENCE = ('D', 'E', 'L', 'O', 'P', 'T')
_NUMBERED_X = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
_NUMBERED_Y = torch.tensor((1.0, 1.0, 0.0, 0.0), dtype=torch.float64)
_SHIPPED_X = {
    'cited_id': '14111139',
    'categories': ['X'],
    'paths': ['14111139.json'],
}


def _leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


def _relevance_marks(cited: AnalyzeCited) -> tuple[str, ...]:
    return tuple(mark for mark in cited.marks if mark in _RELEVANCE_MARKS)


def _phi_of(
    occupy: Tensor,
    source: int | None = None,
    destination: int | None = None,
    mass: float = 0.0,
) -> Tensor:
    overlay = PhiIntensity(int(occupy.size(-1)))
    empty = occupy.new_zeros(occupy.size(-1), occupy.size(-1))
    if source is None or destination is None:
        return overlay(occupy, empty, occupy > 0)
    addends = overlay.kept_pair_addends(
        torch.tensor((source,)),
        torch.tensor((destination,)),
        torch.tensor((mass,), dtype=occupy.dtype),
        torch.tensor((True,)),
    )
    return overlay(occupy, addends, occupy > 0)


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def partner_phi() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    occupy_x = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy_x[_DISTINCTIVE] = 1.0
    occupy_x[_COMPLEMENT] = 1.0
    occupy_a = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy_a[_SHARED] = 1.0
    occupy_a[_SILENT] = 1.0
    occupy_y1 = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy_y1[_DISTINCTIVE] = 3.0
    occupy_y2 = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy_y2[_COMPLEMENT] = 3.0
    return (
        _phi_of(occupy_x, _DISTINCTIVE, _COMPLEMENT, 1.0),
        _phi_of(occupy_a, _SHARED, _SILENT, 99.0),
        _phi_of(occupy_y1),
        _phi_of(occupy_y2),
    )


def test_x_is_single_document_leftover_unpaid_of_numbered_demand_against_phi(
    covering: Covering,
    partner_phi: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """X residual is leftover unpaid of one numbered claim against one Phi."""
    phi_x, phi_a, _phi_y1, _phi_y2 = partner_phi
    stamp = AnalyzeCited(categories=('X',), claims=(1,))
    unpaid_x = _leftover(covering, _NUMBERED_X, phi_x)
    unpaid_a = _leftover(covering, _NUMBERED_X, phi_a)
    assert stamp.marks == ('X',)
    assert stamp.mapped_claims() == (1,)
    assert torch.equal(phi_x[:2], phi_x.new_tensor((2.0, 2.0)))
    assert float(covering.sigma.item()) == pytest.approx(1.0)
    assert unpaid_x.item() == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert unpaid_a.item() == pytest.approx(1.0, abs=1e-12)
    assert unpaid_x.item() < unpaid_a.item()
    assert unpaid_x.ndim == 0


def test_y_is_combination_residual_outside_phi(
    covering: Covering,
    partner_phi: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Y residual is the product of gaps. Neither partner is X taken alone."""
    phi_x, _phi_a, phi_y1, phi_y2 = partner_phi
    stamp_y1 = AnalyzeCited(categories=('Y',), claims=(1,))
    stamp_y2 = AnalyzeCited(categories=('Y',), claims=(1,))
    before = phi_y1.clone()
    unpaid_first = _leftover(covering, _NUMBERED_Y, phi_y1)
    unpaid_second = _leftover(covering, _NUMBERED_Y, phi_y2)
    unpaid_x = _leftover(covering, _NUMBERED_X, phi_x)
    cheaper = torch.minimum(unpaid_first, unpaid_second)
    combined = covering.combination_residual(
        _NUMBERED_Y,
        torch.stack((phi_y1, phi_y2)),
    ).squeeze()
    owners = torch.zeros(2, dtype=torch.long)
    composed = covering.compose_windows(torch.stack((phi_y1, phi_y2)), owners, 1)
    from_windows = covering.unpaid_fraction(covering.pair_table(_NUMBERED_Y, composed)).squeeze()
    shared = torch.zeros(_SLOTS, dtype=torch.float64)
    shared[_COMPLEMENT] = 2.0
    shared_phi = _phi_of(shared)
    shared_pair = covering.combination_residual(
        _NUMBERED_Y,
        torch.stack((shared_phi, shared_phi)),
    ).squeeze()
    shared_cheaper = _leftover(covering, _NUMBERED_Y, shared_phi)
    summed = covering.union_covering(_NUMBERED_Y, shared_phi, shared_phi)
    union_unpaid = _leftover(covering, _NUMBERED_Y, shared_phi + shared_phi)
    assert stamp_y1.marks == ('Y',)
    assert stamp_y2.mapped_claims() == (1,)
    assert torch.equal(phi_y1, before)
    assert torch.equal(phi_y1, torch.tensor((3.0, 0.0, 0.0, 0.0), dtype=torch.float64))
    assert torch.equal(phi_y2, torch.tensor((0.0, 3.0, 0.0, 0.0), dtype=torch.float64))
    assert unpaid_first.item() == pytest.approx(0.625, abs=1e-12)
    assert unpaid_second.item() == pytest.approx(0.625, abs=1e-12)
    assert unpaid_first.item() > unpaid_x.item()
    assert unpaid_second.item() > unpaid_x.item()
    assert combined.item() == pytest.approx(0.25, abs=1e-12)
    assert torch.allclose(combined, from_windows)
    assert cheaper.item() > combined.item()
    assert combined.item() != pytest.approx(cheaper.item(), abs=1e-8)
    assert union_unpaid.item() == pytest.approx(
        (summed.unpaid_mass / summed.demand_l1).item(),
        abs=1e-12,
    )
    assert shared_pair.item() != pytest.approx(union_unpaid.item(), abs=1e-8)
    assert shared_cheaper.item() > shared_pair.item()


def test_clef_ip_grade_two_is_not_combination_residual(
    covering: Covering,
    partner_phi: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """X and Y both map to CLEF-IP grade 2. That grade is not Y residual."""
    _phi_x, _phi_a, phi_y1, phi_y2 = partner_phi
    cited_x = AnalyzeCited(categories=('X',), claims=(1,))
    cited_y = AnalyzeCited(categories=('Y',), claims=(1,))
    cited_xy = AnalyzeCited(categories=('XY',), claims=(1,))
    unpaid_y = _leftover(covering, _NUMBERED_Y, phi_y1)
    combined = covering.combination_residual(
        _NUMBERED_Y,
        torch.stack((phi_y1, phi_y2)),
    ).squeeze()
    assert cited_x.grade == 2
    assert cited_y.grade == 2
    assert cited_xy.grade == 2
    assert cited_x.grade == cited_y.grade
    assert unpaid_y.item() == pytest.approx(0.625, abs=1e-12)
    assert combined.item() == pytest.approx(0.25, abs=1e-12)
    assert unpaid_y.item() > combined.item()


def test_a_stays_high_unpaid_when_shared_field_occupancy_is_high(
    covering: Covering,
    partner_phi: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """A is high leftover unpaid on numbered demand, not low occupancy."""
    phi_x, phi_a, _phi_y1, _phi_y2 = partner_phi
    stamp = AnalyzeCited(categories=('A',), claims=(1,))
    unpaid_x = _leftover(covering, _NUMBERED_X, phi_x)
    unpaid_a = _leftover(covering, _NUMBERED_X, phi_a)
    assert stamp.marks == ('A',)
    assert stamp.grade == 1
    assert float(phi_a.sum().item()) == pytest.approx(200.0, abs=1e-12)
    assert float(phi_x.sum().item()) == pytest.approx(4.0, abs=1e-12)
    assert float(phi_a.sum().item()) > float(phi_x.sum().item())
    assert unpaid_a.item() == pytest.approx(1.0, abs=1e-12)
    assert unpaid_a.item() > unpaid_x.item()


@pytest.mark.parametrize('letter', _DATE_EVIDENCE)
def test_date_and_evidence_letters_are_not_occupancy_bars(letter: str) -> None:
    """P, E, T, L, O, and D do not enter leftover-unpaid letter bars."""
    cited = AnalyzeCited(categories=(letter,))
    assert cited.marks == (letter,)
    assert cited.grade == 0
    assert _relevance_marks(cited) == ()
    assert cited.locator_status == 'unmapped'


def test_px_stamp_keeps_x_residual_and_drops_the_date_flag(
    covering: Covering,
    partner_phi: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """A P,X row scores leftover unpaid as X. P is not an occupancy grade."""
    phi_x, phi_a, _phi_y1, _phi_y2 = partner_phi
    cited = AnalyzeCited(categories=('PX',), claims=(1,))
    unpaid_x = _leftover(covering, _NUMBERED_X, phi_x)
    unpaid_a = _leftover(covering, _NUMBERED_X, phi_a)
    assert cited.marks == ('P', 'X')
    assert cited.grade == 2
    assert _relevance_marks(cited) == ('X',)
    assert unpaid_x.item() < unpaid_a.item()


def test_unmapped_locators_are_not_a_letter_win(
    covering: Covering,
    partner_phi: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """A leftover gap without locators stays unmapped, not an X assignment."""
    phi_x, phi_a, _phi_y1, _phi_y2 = partner_phi
    cited = AnalyzeCited.model_validate(_SHIPPED_X)
    unpaid_x = _leftover(covering, _NUMBERED_X, phi_x)
    unpaid_a = _leftover(covering, _NUMBERED_X, phi_a)
    assert cited.marks == ('X',)
    assert cited.grade == 2
    assert cited.locator_status == 'unmapped'
    assert cited.claims == ()
    assert unpaid_x.item() < unpaid_a.item()
    with pytest.raises(ValueError, match='claim locators are unmapped'):
        cited.mapped_claims()


def test_category_letter_is_not_a_passage_or_overlay_edge(
    partner_phi: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Stamps stay eval labels. A category letter is not a kept pair or passage."""
    phi_x, _phi_a, _phi_y1, _phi_y2 = partner_phi
    cited = AnalyzeCited(categories=('X',))
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_DISTINCTIVE] = 1.0
    occupy[_COMPLEMENT] = 1.0
    rebuilt = _phi_of(occupy, _DISTINCTIVE, _COMPLEMENT, 1.0)
    assert cited.marks == ('X',)
    assert cited.passages == ()
    assert cited.locator_status == 'unmapped'
    assert torch.equal(phi_x, rebuilt)
    with pytest.raises(ValidationError):
        AnalyzeCited(categories=('X',), passages=('X',))
    with pytest.raises(ValidationError):
        AnalyzeCited(categories=('Y',), claims=('Y',))


def test_leftover_unpaid_is_not_pair_table_edge_unpaid(
    covering: Covering,
    partner_phi: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """X residual is leftover unpaid of demand against Phi, not edge unpaid."""
    phi_x, _phi_a, _phi_y1, _phi_y2 = partner_phi
    leftover = _leftover(covering, _NUMBERED_X, phi_x)
    query_edges = torch.ones(1, _SLOTS, _SLOTS, dtype=torch.float64)
    edge = covering.edge_unpaid_fraction(query_edges, query_edges).squeeze()
    assert leftover.item() == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert edge.item() == pytest.approx(0.5, abs=1e-12)
    assert leftover.item() != pytest.approx(edge.item(), abs=1e-8)
