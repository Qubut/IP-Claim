"""Production occupy select is leftover-unpaid-sufficient support.

The smallest leftover-unpaid-sufficient occupy set is the occupy bag.
Silent top-k by occupancy that moves leftover unpaid of numbered-claim
demand against Phi is a named fail. There is no occupy integer cap.
Destination leftover unpaid stays unpaid_fraction of pair_table.
"""

from __future__ import annotations

import pytest
import torch
from tests._ssv_fixtures import ssv_tiny_vocab_config
from tests.unit.ssv.test_host_ceiling_scientific_support import (
    _NAMED_FAIL_CAPS,
    _TIGHT_HOST,
    HostCeiling,
    _kept_codes,
    _silent_top_mass,
)
from tests.unit.ssv.test_host_ceiling_scientific_support import (
    _planted as _ceiling_planted,
)
from tests.unit.ssv.test_leftover_unpaid_sufficient_occupy import (
    _leftover,
    _leftover_on_claims,
    _mask,
    _narrow_filing,
    _phi_on,
    _smallest_leftover_unpaid_sufficient,
    _wide_filing,
)
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.config import ArchSpec
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.phi import PhiIntensity
from ip_claim.ssv.soft_graph import (
    OccupiedCodes,
    build_soft_relation_bundle,
    leftover_unpaid_occupy_mask,
    select_leftover_unpaid_occupy,
)
from ip_claim.ssv.soft_vocab import SoftVocabModule

_SLOTS = 4
_PAID = 0
_LIGHT = 1
_EDGE = 2
_AMBIENT = 3


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def overlay() -> PhiIntensity:
    return PhiIntensity(_SLOTS)


def _demand_table(demands: tuple[Tensor, ...]) -> Tensor:
    return torch.stack(demands)


def _kept_from_select(select) -> frozenset[int]:
    return frozenset(
        int(index) for index in select.occupied.code_ids[0][select.occupied.keep[0]].tolist()
    )


def _assignment_from_occupy(occupy: Tensor) -> tuple[Tensor, Tensor]:
    assignment = torch.zeros(1, _SLOTS, _SLOTS, dtype=occupy.dtype)
    assignment[0, torch.arange(_SLOTS), torch.arange(_SLOTS)] = occupy
    return assignment, (occupy > 0).to(dtype=occupy.dtype).unsqueeze(0)


class TestLeftoverUnpaidOccupySelect:
    """Production occupy is leftover-unpaid-sufficient. Width is |S-star|."""

    def test_two_filings_select_different_leftover_unpaid_width(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Leftover-unpaid-sufficient width follows the filing, not a host integer."""
        narrow = _narrow_filing()
        wide = _wide_filing()
        bank = torch.eye(_SLOTS, dtype=narrow.occupy.dtype)
        picked_narrow = select_leftover_unpaid_occupy(
            bank,
            narrow.occupy.unsqueeze(0),
            demand=_demand_table(narrow.demands),
            pair_mass=narrow.pair_mass,
            mass_floor=0.0,
        )
        picked_wide = select_leftover_unpaid_occupy(
            bank,
            wide.occupy.unsqueeze(0),
            demand=_demand_table(wide.demands),
            pair_mass=wide.pair_mass,
            mass_floor=0.0,
        )
        assert _kept_from_select(picked_narrow) == _smallest_leftover_unpaid_sufficient(
            covering, overlay, narrow
        )
        assert _kept_from_select(picked_wide) == _smallest_leftover_unpaid_sufficient(
            covering, overlay, wide
        )
        assert int(picked_narrow.scientific_width.item()) != int(
            picked_wide.scientific_width.item()
        )

    def test_selected_support_reconstructs_ambient_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Phi on the selected bag leaves leftover unpaid of every numbered claim put."""
        filing = _wide_filing()
        bank = torch.eye(_SLOTS, dtype=filing.occupy.dtype)
        picked = select_leftover_unpaid_occupy(
            bank,
            filing.occupy.unsqueeze(0),
            demand=_demand_table(filing.demands),
            pair_mass=filing.pair_mass,
            mass_floor=0.0,
        )
        star = leftover_unpaid_occupy_mask(
            filing.occupy.unsqueeze(0),
            _demand_table(filing.demands),
            filing.pair_mass,
            0.0,
        )
        ambient = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress),
        )
        selected = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, star.squeeze(0)),
        )
        assert torch.allclose(ambient, selected)
        assert _kept_from_select(picked) == frozenset((_PAID, _LIGHT, _EDGE))
        assert _AMBIENT not in _kept_from_select(picked)

    def test_silent_top_mass_still_moves_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Floor then top-M remains the named fail. Production select is not that bag."""
        filing = _ceiling_planted()
        bank = torch.eye(_SLOTS, dtype=filing.occupy.dtype)
        silent = _kept_codes(_silent_top_mass(filing.occupy, HostCeiling(_TIGHT_HOST)))
        picked = select_leftover_unpaid_occupy(
            bank,
            filing.occupy.unsqueeze(0),
            demand=_demand_table(filing.demands),
            pair_mass=filing.pair_mass,
            mass_floor=0.0,
        )
        ambient = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress),
        )
        silent_u = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, _mask(silent)),
        )
        selected_u = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(
                overlay,
                filing.occupy,
                filing.pair_mass,
                leftover_unpaid_occupy_mask(
                    filing.occupy.unsqueeze(0),
                    _demand_table(filing.demands),
                    filing.pair_mass,
                    0.0,
                ).squeeze(0),
            ),
        )
        assert silent == frozenset((_PAID, _AMBIENT))
        assert _kept_from_select(picked) == filing.scientific
        assert not torch.allclose(ambient, silent_u)
        assert torch.allclose(ambient, selected_u)
        assert OccupiedCodes.model_fields.keys() == frozenset((
            'features',
            'code_ids',
            'keep',
            'lengths',
        ))

    def test_selected_width_is_leftover_unpaid_sufficient_not_top_m(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Production occupy packs S-star. Rejected top-M is a different bag."""
        filing = _ceiling_planted()
        bank = torch.eye(_SLOTS, dtype=filing.occupy.dtype)
        picked = select_leftover_unpaid_occupy(
            bank,
            filing.occupy.unsqueeze(0),
            demand=_demand_table(filing.demands),
            pair_mass=filing.pair_mass,
            mass_floor=0.0,
        )
        silent = frozenset(
            int(index)
            for index in filing.occupy.topk(_TIGHT_HOST).indices.tolist()
        )
        ambient = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress),
        )
        named = _leftover_on_claims(
            covering,
            filing.demands,
            _phi_on(
                overlay,
                filing.occupy,
                filing.pair_mass,
                leftover_unpaid_occupy_mask(
                    filing.occupy.unsqueeze(0),
                    _demand_table(filing.demands),
                    filing.pair_mass,
                    0.0,
                ).squeeze(0),
            ),
        )
        assert int(picked.scientific_width.item()) == 3
        assert _kept_from_select(picked) == filing.scientific
        assert _kept_from_select(picked) != silent
        assert torch.allclose(ambient, named)

    def test_scientific_width_is_per_filing(self) -> None:
        """Leftover-unpaid-sufficient width is |S-star|, not a bank integer."""
        filing = _wide_filing()
        bank = torch.eye(_SLOTS, dtype=filing.occupy.dtype)
        picked = select_leftover_unpaid_occupy(
            bank,
            filing.occupy.unsqueeze(0),
            demand=_demand_table(filing.demands),
            pair_mass=filing.pair_mass,
            mass_floor=0.0,
        )
        assert int(picked.scientific_width.item()) == 3
        assert int(picked.scientific_width.item()) <= _SLOTS

    @pytest.mark.parametrize('pretend', _NAMED_FAIL_CAPS)
    def test_global_occupy_integer_is_not_scientific_width(self, pretend: int) -> None:
        """A host integer is not leftover-unpaid-sufficient width."""
        filing = _wide_filing()
        bank = torch.eye(_SLOTS, dtype=filing.occupy.dtype)
        picked = select_leftover_unpaid_occupy(
            bank,
            filing.occupy.unsqueeze(0),
            demand=_demand_table(filing.demands),
            pair_mass=filing.pair_mass,
            mass_floor=0.0,
        )
        assert int(picked.scientific_width.item()) == 3
        assert int(picked.scientific_width.item()) != pretend
        assert 'soft_occupied_max' not in ArchSpec.model_fields

    def test_destination_type_is_covering_leftover_unpaid_of_phi(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """The covering head reads unpaid_fraction of pair_table(L, Phi)."""
        filing = _wide_filing()
        supply = _phi_on(
            overlay,
            filing.occupy,
            filing.pair_mass,
            leftover_unpaid_occupy_mask(
                filing.occupy.unsqueeze(0),
                _demand_table(filing.demands),
                filing.pair_mass,
                0.0,
            ).squeeze(0),
        )
        leftover = covering.unpaid_fraction(
            covering.pair_table(filing.demands[0], supply)
        ).squeeze()
        assert leftover.ndim == 0
        assert torch.isfinite(leftover)
        assert torch.allclose(leftover, _leftover(covering, filing.demands[0], supply))

    def test_bundle_with_demand_keeps_paying_endpoint(self) -> None:
        """Production bundle occupy is leftover-unpaid-sufficient, not silent top-two."""
        filing = _wide_filing()
        vocab = SoftVocabModule(
            ssv_tiny_vocab_config(entity_bank_size=_SLOTS, relation_bank_size=2, soft_dim=8)
        )
        assignment, mask = _assignment_from_occupy(filing.occupy)
        bundle = build_soft_relation_bundle(
            vocab,
            assignment,
            mask,
            mass_floor=0.0,
            demand=_demand_table(filing.demands),
            pair_mass=filing.pair_mass.unsqueeze(0),
        )
        silent = frozenset(
            int(index) for index in filing.occupy.topk(_TIGHT_HOST).indices.tolist()
        )
        kept = frozenset(int(code) for code in bundle.overlays[0].code_ids.tolist())
        assert bundle.scientific_width is not None
        assert int(bundle.scientific_width.item()) == 3
        assert kept == frozenset((_PAID, _LIGHT, _EDGE))
        assert kept != silent
        assert _AMBIENT not in kept

    def test_no_demand_bundle_is_empty_occupy_not_top_m(self) -> None:
        """Missing demand is empty occupy. It is not a top-M occupancy bag."""
        filing = _wide_filing()
        vocab = SoftVocabModule(
            ssv_tiny_vocab_config(entity_bank_size=_SLOTS, relation_bank_size=2, soft_dim=8)
        )
        assignment, mask = _assignment_from_occupy(filing.occupy)
        bundle = build_soft_relation_bundle(
            vocab,
            assignment,
            mask,
            mass_floor=0.0,
        )
        top = frozenset(
            int(index) for index in filing.occupy.topk(_TIGHT_HOST).indices.tolist()
        )
        kept = frozenset(int(code) for code in bundle.overlays[0].code_ids.tolist())
        assert kept == frozenset()
        assert bundle.scientific_width is not None
        assert int(bundle.scientific_width.item()) == 0
        assert kept != top

    def test_inventory_overlay_uses_leftover_unpaid_occupy(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Inventory overlay intensity on leftover-unpaid occupy leaves leftover unpaid put."""
        filing = _wide_filing()
        inventory = Inventory(occupied_floor=0.0)
        labeled = filing.pair_mass.unsqueeze(0).unsqueeze(-1)
        occupy = filing.occupy.unsqueeze(0)
        demand = _demand_table(filing.demands)
        supply = inventory.overlay_intensity(occupy, labeled, demand=demand).squeeze(0)
        ambient = _phi_on(overlay, filing.occupy, filing.pair_mass, filing.ingress)
        leftover_selected = covering.unpaid_fraction(
            covering.pair_table(filing.demands[0], supply)
        ).squeeze()
        leftover_ambient = covering.unpaid_fraction(
            covering.pair_table(filing.demands[0], ambient)
        ).squeeze()
        assert torch.allclose(leftover_selected, leftover_ambient)
        assert 'cited' not in type(filing)._fields
        assert 'category' not in type(filing)._fields
