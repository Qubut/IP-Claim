"""Saturating shared-field Phi can invert leftover unpaid of the joined blob.

Numbered-claim leftover unpaid against the same Phi does not invert on an
unpaid independent. When demand support is a proper subset of the scored
vector, total leftover unpaid can invert. These tests score Covering leftover
unpaid of demand against Phi. They do not train a trunk and they do not
assign citation letters.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ingestion.adapters.hupd_json.claim_parser import parse_claims
from ip_claim.ingestion.adapters.hupd_json.loader import patent_from_hupd_dict
from ip_claim.ingestion.models import Patent
from ip_claim.ssv.graph_batch import (
    dest_claim_texts,
    graph_batch_from_patent,
    patent_claim_blob,
)
from ip_claim.ssv.phi import PhiIntensity

_CLAIMS_BLOB = (
    '1. A widget comprising a latchbolt. '
    '2. A gadget comprising a photodiode. '
    '3. The widget of claim 1, wherein the apparatus is an apparatus '
    'and the apparatus includes an apparatus.'
)
_HUPD_ROW = {
    'application_number': '99000001',
    'decision': 'PENDING',
    'claims': _CLAIMS_BLOB,
}
_SLOTS = 4
_LATCHBOLT = 0
_PHOTODIODE = 1
_APPARATUS = 2
_SILENT = 3
_INDEPENDENT_MASS = 1.0
_DEPENDENT_MASS = 3.0
_X_OCCUPY = 1.0
_X_EDGE = 1.0
_A_OCCUPY = 1.0
_A_EDGE = 99.0
_SUBSET_DEMAND = torch.tensor((1.0, 1.0), dtype=torch.float64)
_SUBSET_OCCUPY_X = torch.tensor((1.0, 0.0), dtype=torch.float64)
_SUBSET_OCCUPY_A = torch.tensor((0.0, 100.0), dtype=torch.float64)


def _leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def patent() -> Patent:
    return patent_from_hupd_dict(_HUPD_ROW)


@pytest.fixture
def numbered_demand() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    independent_first = torch.zeros(_SLOTS, dtype=torch.float64)
    independent_first[_LATCHBOLT] = _INDEPENDENT_MASS
    independent_second = torch.zeros(_SLOTS, dtype=torch.float64)
    independent_second[_PHOTODIODE] = _INDEPENDENT_MASS
    dependent = torch.zeros(_SLOTS, dtype=torch.float64)
    dependent[_APPARATUS] = _DEPENDENT_MASS
    blob = independent_first + independent_second + dependent
    return independent_first, independent_second, dependent, blob


@pytest.fixture
def partner_phi() -> tuple[Tensor, Tensor]:
    def intensity(occupy: Tensor, source: int, destination: int, mass: float) -> Tensor:
        overlay = PhiIntensity(int(occupy.size(-1)))
        addends = overlay.kept_pair_addends(
            torch.tensor((source,)),
            torch.tensor((destination,)),
            torch.tensor((mass,), dtype=occupy.dtype),
            torch.tensor((True,)),
        )
        return overlay(occupy, addends, occupy > 0)

    occupy_x = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy_x[_LATCHBOLT] = _X_OCCUPY
    occupy_x[_PHOTODIODE] = _X_OCCUPY
    occupy_a = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy_a[_APPARATUS] = _A_OCCUPY
    occupy_a[_SILENT] = _A_OCCUPY
    return (
        intensity(occupy_x, _LATCHBOLT, _PHOTODIODE, _X_EDGE),
        intensity(occupy_a, _APPARATUS, _SILENT, _A_EDGE),
    )


def test_filing_keeps_unpaid_independents_and_a_shared_field_dependent(
    patent: Patent,
) -> None:
    """Dest demand is the independents. The dependent carries shared-field text."""
    claims = parse_claims(_CLAIMS_BLOB)
    independents = tuple(claim for claim in claims if claim.is_independent)
    dependents = tuple(claim for claim in claims if not claim.is_independent)
    assert tuple(claim.number for claim in independents) == (1, 2)
    assert tuple(claim.number for claim in dependents) == (3,)
    assert dependents[0].parent_number == 1
    assert 'apparatus' in dependents[0].text.lower()
    assert 'apparatus' not in independents[0].text.lower()
    assert 'apparatus' not in independents[1].text.lower()
    batch = graph_batch_from_patent(patent)
    blob = patent_claim_blob(patent)
    assert dest_claim_texts(batch) == (independents[0].text, independents[1].text)
    assert blob == ' '.join(claim.text for claim in claims)
    assert blob not in dest_claim_texts(batch)


def test_blob_leftover_inverts_while_numbered_claim_leftover_does_not(
    covering: Covering,
    numbered_demand: tuple[Tensor, Tensor, Tensor, Tensor],
    partner_phi: tuple[Tensor, Tensor],
) -> None:
    """Joined-blob leftover prefers saturating Phi; independents stay unpaid there."""
    independent_first, independent_second, dependent, blob = numbered_demand
    phi_x, phi_a = partner_phi
    assert torch.equal(phi_x[:3], phi_x.new_tensor((_X_OCCUPY + _X_EDGE, _X_OCCUPY + _X_EDGE, 0.0)))
    assert torch.equal(phi_a[:3], phi_a.new_tensor((0.0, 0.0, _A_OCCUPY + _A_EDGE)))
    assert float(covering.sigma.item()) == pytest.approx(1.0)

    unpaid_x = tuple(
        _leftover(covering, demand, phi_x)
        for demand in (independent_first, independent_second, dependent, blob)
    )
    unpaid_a = tuple(
        _leftover(covering, demand, phi_a)
        for demand in (independent_first, independent_second, dependent, blob)
    )
    assert unpaid_x[0].item() == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert unpaid_x[1].item() == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert unpaid_a[0].item() == pytest.approx(1.0, abs=1e-12)
    assert unpaid_a[1].item() == pytest.approx(1.0, abs=1e-12)
    assert unpaid_x[0].item() < unpaid_a[0].item()
    assert unpaid_x[1].item() < unpaid_a[1].item()
    assert unpaid_a[2].item() == pytest.approx(1.0 / 101.0, abs=1e-12)
    assert unpaid_x[2].item() == pytest.approx(1.0, abs=1e-12)
    assert unpaid_a[2].item() < unpaid_x[2].item()
    assert unpaid_x[3].item() == pytest.approx(11.0 / 15.0, abs=1e-12)
    assert unpaid_a[3].item() == pytest.approx(41.0 / 101.0, abs=1e-12)
    assert unpaid_a[3].item() < unpaid_x[3].item()
    row_mean_x = torch.stack(unpaid_x[:3]).mean()
    row_mean_a = torch.stack(unpaid_a[:3]).mean()
    assert row_mean_x.item() < row_mean_a.item()
    assert not torch.allclose(unpaid_x[3], row_mean_x)
    assert not torch.allclose(unpaid_a[3], row_mean_a)


def test_proper_subset_support_inverts_total_leftover_against_phi(
    covering: Covering,
) -> None:
    """Full two-slot demand inverts; the X-paid numbered claim does not."""
    overlay = PhiIntensity(2)
    empty = torch.zeros(2, 2, dtype=torch.float64)
    phi_x = overlay(_SUBSET_OCCUPY_X, empty, _SUBSET_OCCUPY_X > 0)
    phi_a = overlay(_SUBSET_OCCUPY_A, empty, _SUBSET_OCCUPY_A > 0)
    assert torch.equal(phi_x, _SUBSET_OCCUPY_X)
    assert torch.equal(phi_a, _SUBSET_OCCUPY_A)
    unpaid_full_x = _leftover(covering, _SUBSET_DEMAND, phi_x)
    unpaid_full_a = _leftover(covering, _SUBSET_DEMAND, phi_a)
    numbered_paid = _SUBSET_DEMAND * torch.tensor((1.0, 0.0), dtype=torch.float64)
    unpaid_claim_x = _leftover(covering, numbered_paid, phi_x)
    unpaid_claim_a = _leftover(covering, numbered_paid, phi_a)
    assert float(covering.sigma.item()) == pytest.approx(1.0)
    assert unpaid_full_x.item() == pytest.approx(0.75, abs=1e-12)
    assert unpaid_full_a.item() == pytest.approx(0.5 + 1.0 / 202.0, abs=1e-12)
    assert unpaid_full_x.item() > unpaid_full_a.item()
    assert unpaid_claim_x.item() == pytest.approx(0.5, abs=1e-12)
    assert unpaid_claim_a.item() == pytest.approx(1.0, abs=1e-12)
    assert unpaid_claim_x.item() < unpaid_claim_a.item()
