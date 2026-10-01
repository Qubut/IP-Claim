"""Host ceiling is compute. Scientific occupy width is leftover-unpaid support.

A planted filing has leftover-unpaid-sufficient width three. A host cap of
two cannot hold that support. Silent top-two by occupancy drops a demanded
slot and a paying-edge endpoint, so leftover unpaid of numbered-claim demand
against Phi moves. Declared truncation names the overflow and leaves
scientific support put. Sixteen, thirty-two, and sixty-four are host
integers, not that width. Covering leftover unpaid is the train head.
These tests do not rewire production occupy select.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.config import ArchSpec
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.soft_graph import OccupiedCodes

_SLOTS = 4
_PAID = 0
_LIGHT = 1
_EDGE = 2
_AMBIENT = 3
_OCCUPY = (1.0, 0.25, 0.20, 0.85)
_CLAIM_MASS = 2.0
_EDGE_MASS = 0.5
_SCIENTIFIC_WIDTH = 3
_TIGHT_HOST = 2
_NAMED_FAIL_CAPS = (16, 32, 64)


class PlantedFiling(NamedTuple):
    """Ingress occupy, kept pair table, and numbered-claim demands."""

    occupy: Tensor
    pair_mass: Tensor
    ingress: Tensor
    demands: tuple[Tensor, ...]
    scientific: frozenset[int]


class ScientificSupport(NamedTuple):
    """Leftover-unpaid-sufficient occupy. Width is scientific M(D)."""

    slots: frozenset[int]

    @property
    def width(self) -> int:
        return len(self.slots)


class HostCeiling(NamedTuple):
    """Compute bound on occupied rows. Not scientific support."""

    cap: int


class CeilingReport(NamedTuple):
    """Host bound against leftover-unpaid-sufficient width.

    Truncation is named when scientific width exceeds the cap. Scientific
    support is not rewritten as the host bag.
    """

    scientific: ScientificSupport
    host: HostCeiling
    compute_slots: frozenset[int]
    truncated: bool
    declared: bool


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
    demands: tuple[Tensor, ...],
    supply: Tensor,
) -> Tensor:
    return torch.stack(tuple(_leftover(covering, demand, supply) for demand in demands))


def _mask(chosen: frozenset[int]) -> Tensor:
    live = torch.zeros(_SLOTS, dtype=torch.bool)
    if chosen:
        live[torch.tensor(tuple(chosen), dtype=torch.long)] = True
    return live


def _phi_on(overlay: PhiIntensity, filing: PlantedFiling, live: Tensor) -> Tensor:
    return overlay(filing.occupy, filing.pair_mass, live)


def declare_host_ceiling(
    scientific: ScientificSupport,
    host: HostCeiling,
) -> CeilingReport:
    """Name overflow when leftover-unpaid-sufficient width exceeds the cap."""
    truncated = scientific.width > host.cap
    return CeilingReport(
        scientific=scientific,
        host=host,
        compute_slots=scientific.slots,
        truncated=truncated,
        declared=truncated,
    )


def _silent_top_mass(occupy: Tensor, host: HostCeiling) -> OccupiedCodes:
    """Rejected top-M bag used only to name the fail. Not a production selector."""
    bank = torch.eye(_SLOTS, dtype=occupy.dtype)
    _mass, code_ids = occupy.unsqueeze(0).topk(host.cap, dim=-1)
    keep = torch.ones_like(code_ids, dtype=torch.bool)
    return OccupiedCodes(
        features=bank[code_ids],
        code_ids=code_ids,
        keep=keep,
        lengths=keep.sum(dim=-1),
    )


def _kept_codes(occupied: OccupiedCodes) -> frozenset[int]:
    return frozenset(int(index) for index in occupied.code_ids[0][occupied.keep[0]].tolist())


def _planted() -> PlantedFiling:
    occupy = torch.tensor(_OCCUPY, dtype=torch.float64)
    first = occupy.new_zeros(_SLOTS)
    first[_PAID] = _CLAIM_MASS
    second = occupy.new_zeros(_SLOTS)
    second[_LIGHT] = _CLAIM_MASS
    overlay = PhiIntensity(_SLOTS)
    pair_mass = overlay.kept_pair_addends(
        torch.tensor((_EDGE,)),
        torch.tensor((_PAID,)),
        torch.tensor((_EDGE_MASS,), dtype=torch.float64),
        torch.tensor((True,)),
    )
    return PlantedFiling(
        occupy=occupy,
        pair_mass=pair_mass,
        ingress=occupy > 0,
        demands=(first, second),
        scientific=frozenset((_PAID, _LIGHT, _EDGE)),
    )


class TestHostCeilingVersusScientificSupport:
    """Compute cap versus leftover-unpaid-sufficient width. Silent top-M fails."""

    def test_scientific_width_is_independent_of_host_cap(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """M(D) stays leftover-unpaid-sufficient width under two host caps."""
        filing = _planted()
        scientific = ScientificSupport(filing.scientific)
        tight = declare_host_ceiling(scientific, HostCeiling(_TIGHT_HOST))
        roomy = declare_host_ceiling(scientific, HostCeiling(_SLOTS))
        ambient = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing, filing.ingress),
        )
        reconstructed = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing, _mask(scientific.slots)),
        )
        assert scientific.width == _SCIENTIFIC_WIDTH
        assert tight.scientific.width == roomy.scientific.width
        assert tight.scientific.slots == roomy.scientific.slots
        assert tight.host.cap != roomy.host.cap
        assert torch.allclose(ambient, reconstructed)

    def test_silent_top_mass_drops_leftover_unpaid_relevant_slots(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Top-two by occupancy keeps the heavy ambient row and drops paying slots."""
        filing = _planted()
        occupied = _silent_top_mass(filing.occupy, HostCeiling(_TIGHT_HOST))
        kept = _kept_codes(occupied)
        ambient = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing, filing.ingress),
        )
        silent = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing, _mask(kept)),
        )
        assert kept == frozenset((_PAID, _AMBIENT))
        assert _LIGHT not in kept
        assert _EDGE not in kept
        assert _AMBIENT in kept
        assert kept != filing.scientific
        assert not torch.allclose(ambient, silent)
        assert OccupiedCodes.model_fields.keys() == frozenset((
            'features',
            'code_ids',
            'keep',
            'lengths',
        ))

    def test_overflow_is_declared_truncate_not_silent_top_mass(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Width above the host cap is named. Scientific support is not the host bag."""
        filing = _planted()
        scientific = ScientificSupport(filing.scientific)
        host = HostCeiling(_TIGHT_HOST)
        report = declare_host_ceiling(scientific, host)
        silent = _kept_codes(_silent_top_mass(filing.occupy, host))
        ambient = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing, filing.ingress),
        )
        named = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing, _mask(report.scientific.slots)),
        )
        assert report.truncated
        assert report.declared
        assert report.scientific.width == _SCIENTIFIC_WIDTH
        assert report.scientific.width > report.host.cap
        assert report.scientific.slots == filing.scientific
        assert report.compute_slots == filing.scientific
        assert report.compute_slots != silent
        assert torch.allclose(ambient, named)

    def test_host_that_holds_support_does_not_truncate(self) -> None:
        """A cap at or above leftover-unpaid-sufficient width is not overflow."""
        filing = _planted()
        report = declare_host_ceiling(
            ScientificSupport(filing.scientific),
            HostCeiling(_SLOTS),
        )
        assert not report.truncated
        assert not report.declared
        assert report.scientific.width <= report.host.cap
        assert report.compute_slots == filing.scientific

    @pytest.mark.parametrize('pretend', _NAMED_FAIL_CAPS)
    def test_global_occupy_integer_is_not_scientific_width(self, pretend: int) -> None:
        """A host integer is compute. It is not leftover-unpaid-sufficient width."""
        filing = _planted()
        scientific = ScientificSupport(filing.scientific)
        report = declare_host_ceiling(scientific, HostCeiling(pretend))
        assert scientific.width == _SCIENTIFIC_WIDTH
        assert scientific.width != pretend
        assert report.scientific.width != pretend
        assert report.host.cap == pretend
        assert not report.truncated
        assert 'soft_occupied_max' not in ArchSpec.model_fields
        assert pretend != scientific.width

    def test_destination_type_is_covering_leftover_unpaid_of_phi(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """The covering head reads unpaid_fraction of pair_table(L, Phi)."""
        filing = _planted()
        supply = _phi_on(overlay, filing, filing.ingress)
        leftover = covering.unpaid_fraction(
            covering.pair_table(filing.demands[0], supply)
        ).squeeze()
        assert leftover.ndim == 0
        assert torch.isfinite(leftover)
        assert 'cited' not in PlantedFiling._fields
        assert 'category' not in PlantedFiling._fields
