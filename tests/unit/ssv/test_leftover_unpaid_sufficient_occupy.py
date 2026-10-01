"""Leftover-unpaid-sufficient occupy is a smallest reconstructible support.

Two planted filings can need different support sizes. Dropping a leftover-unpaid-
relevant slot moves leftover unpaid. Dropping a leftover-unpaid-irrelevant slot
does not. Covering leftover unpaid of numbered-claim demand against Phi is the
destination type. These tests do not rewire production occupy select.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import NamedTuple

import pytest
import torch
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.phi import PhiIntensity

_SLOTS = 4
_DEMANDED = 0
_SECOND = 1
_PAYING_END = 2
_SILENT = 3
_OCCUPY_MASS = 1.0
_EDGE_MASS = 0.5
_SILENT_MASS = 0.4
_CLAIM_MASS = 2.0


class PlantedFiling(NamedTuple):
    """Ingress-legal occupy, kept pair table, and numbered-claim demands."""

    occupy: Tensor
    pair_mass: Tensor
    ingress: Tensor
    demands: tuple[Tensor, ...]


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def overlay() -> PhiIntensity:
    return PhiIntensity(_SLOTS)


def _leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


def _leftover_on_claims(
    covering: Covering,
    demands: Sequence[Tensor],
    supply: Tensor,
) -> Tensor:
    return torch.stack(tuple(_leftover(covering, demand, supply) for demand in demands))


def _phi_on(overlay: PhiIntensity, occupy: Tensor, pair_mass: Tensor, live: Tensor) -> Tensor:
    return overlay(occupy, pair_mass, live)


def _mask(chosen: frozenset[int]) -> Tensor:
    live = torch.zeros(_SLOTS, dtype=torch.bool)
    if chosen:
        live[torch.tensor(tuple(chosen), dtype=torch.long)] = True
    return live


def _ingress_slots(ingress: Tensor) -> tuple[int, ...]:
    return tuple(int(index) for index in ingress.nonzero(as_tuple=False).flatten())


def _subsets(slots: tuple[int, ...]) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(combo) for width in range(len(slots) + 1) for combo in combinations(slots, width)
    )


def _is_leftover_unpaid_sufficient(
    covering: Covering,
    overlay: PhiIntensity,
    filing: PlantedFiling,
    live: Tensor,
) -> bool:
    ambient = _leftover_on_claims(
        covering,
        filing.demands,
        _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress),
    )
    restricted = _leftover_on_claims(
        covering,
        filing.demands,
        _phi_on(overlay, filing.occupy, filing.pair_mass, live),
    )
    return bool(torch.allclose(ambient, restricted))


def _smallest_leftover_unpaid_sufficient(
    covering: Covering,
    overlay: PhiIntensity,
    filing: PlantedFiling,
) -> frozenset[int]:
    sufficient = tuple(
        subset
        for subset in _subsets(_ingress_slots(filing.ingress))
        if _is_leftover_unpaid_sufficient(covering, overlay, filing, _mask(subset))
    )
    width = min(map(len, sufficient))
    stars = tuple(subset for subset in sufficient if len(subset) == width)
    assert len(stars) == 1
    return stars[0]


def _narrow_filing() -> PlantedFiling:
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_DEMANDED] = _OCCUPY_MASS
    occupy[_SECOND] = _SILENT_MASS
    occupy[_PAYING_END] = _SILENT_MASS
    demand = torch.zeros(_SLOTS, dtype=torch.float64)
    demand[_DEMANDED] = _CLAIM_MASS
    ingress = occupy > 0
    return PlantedFiling(
        occupy=occupy,
        pair_mass=occupy.new_zeros(_SLOTS, _SLOTS),
        ingress=ingress,
        demands=(demand,),
    )


def _wide_filing() -> PlantedFiling:
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_DEMANDED] = _OCCUPY_MASS
    occupy[_SECOND] = _OCCUPY_MASS
    occupy[_PAYING_END] = _OCCUPY_MASS
    occupy[_SILENT] = _SILENT_MASS
    first = torch.zeros(_SLOTS, dtype=torch.float64)
    first[_DEMANDED] = _CLAIM_MASS
    second = torch.zeros(_SLOTS, dtype=torch.float64)
    second[_SECOND] = _CLAIM_MASS
    overlay = PhiIntensity(_SLOTS)
    pair_mass = overlay.kept_pair_addends(
        torch.tensor((_PAYING_END,)),
        torch.tensor((_DEMANDED,)),
        torch.tensor((_EDGE_MASS,), dtype=torch.float64),
        torch.tensor((True,)),
    )
    return PlantedFiling(
        occupy=occupy,
        pair_mass=pair_mass,
        ingress=occupy > 0,
        demands=(first, second),
    )


class TestLeftoverUnpaidSufficientOccupy:
    """Smallest occupy support that reconstructs leftover unpaid of every numbered claim."""

    def test_two_filings_have_different_support_cardinality(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """M(D) is leftover-unpaid-sufficient width. Two filings need different widths."""
        narrow = _smallest_leftover_unpaid_sufficient(covering, overlay, _narrow_filing())
        wide = _smallest_leftover_unpaid_sufficient(covering, overlay, _wide_filing())
        assert narrow == frozenset((_DEMANDED,))
        assert wide == frozenset((_DEMANDED, _SECOND, _PAYING_END))
        assert len(narrow) != len(wide)
        assert len(_wide_filing().demands) != len(wide)

    def test_smallest_support_reconstructs_ambient_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Phi on S-star leaves leftover unpaid of every numbered claim put."""
        filing = _wide_filing()
        star = _smallest_leftover_unpaid_sufficient(covering, overlay, filing)
        ambient = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress),
        )
        reconstructed = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, _mask(star)),
        )
        assert torch.allclose(ambient, reconstructed)
        assert torch.all(torch.isfinite(ambient))
        assert not torch.allclose(ambient, ambient.new_zeros(ambient.shape))

    def test_dropping_leftover_unpaid_relevant_slot_moves_leftover(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """A paying-edge endpoint sits in every minimal support. Dropping it moves U."""
        filing = _wide_filing()
        ingress = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress),
        )
        dropped = filing.ingress.clone()
        dropped[_PAYING_END] = False
        without = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, dropped),
        )
        assert not torch.allclose(ingress, without)
        assert not _is_leftover_unpaid_sufficient(covering, overlay, filing, dropped)

    def test_restoring_leftover_unpaid_relevant_slot_moves_leftover(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Putting a dropped leftover-unpaid-relevant slot back moves leftover unpaid."""
        filing = _wide_filing()
        dropped = filing.ingress.clone()
        dropped[_PAYING_END] = False
        without = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, dropped),
        )
        restored = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress),
        )
        assert not torch.allclose(without, restored)

    def test_dropping_leftover_unpaid_irrelevant_slot_leaves_leftover(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Occupy off demanded support, with no paying edge, is not in S-star."""
        filing = _wide_filing()
        ingress = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress),
        )
        dropped = filing.ingress.clone()
        dropped[_SILENT] = False
        without = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, dropped),
        )
        star = _smallest_leftover_unpaid_sufficient(covering, overlay, filing)
        assert torch.allclose(ingress, without)
        assert _SILENT not in star
        assert _is_leftover_unpaid_sufficient(covering, overlay, filing, dropped)

    def test_support_is_not_the_ingress_floor_alone(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Ingress-legal occupy is the ambient source. It is larger than S-star here."""
        filing = _wide_filing()
        star = _smallest_leftover_unpaid_sufficient(covering, overlay, filing)
        ingress = frozenset(_ingress_slots(filing.ingress))
        assert star < ingress
        assert _SILENT in ingress
        assert _SILENT not in star

    def test_destination_type_is_covering_leftover_unpaid_of_phi(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Sufficiency scores unpaid_fraction of pair_table(L, Phi), not a second plane."""
        filing = _wide_filing()
        supply = _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress)
        table = covering.pair_table(filing.demands[0], supply)
        leftover = covering.unpaid_fraction(table).squeeze()
        assert leftover.ndim == 0
        assert torch.isfinite(leftover)
        assert 'cited' not in PlantedFiling._fields
        assert 'category' not in PlantedFiling._fields
