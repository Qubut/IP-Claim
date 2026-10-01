"""TES-SAC lock, undershoot snap, and target ratchet."""

from __future__ import annotations

import pytest

from ip_claim.ssv.config import TesSacSpec
from ip_claim.ssv.steer.tes_sac import (
    finch_update,
    tes_sac_inventory_locked,
    tes_sac_ratchet,
    tes_sac_snap_undershoot,
)


def test_finch_update_matches_incremental_identity() -> None:
    mean, var = finch_update(mean=4.0, var=0.0, live=5.0, decay=0.9)
    assert mean == pytest.approx(4.1)
    assert var == pytest.approx(0.09)


def test_lock_is_two_sided_and_requires_small_std() -> None:
    spec = TesSacSpec(band=0.15, std_max=0.05)
    assert tes_sac_inventory_locked(mean=3.74, var=0.0, target=3.74, spec=spec)
    assert not tes_sac_inventory_locked(mean=3.45, var=0.0, target=3.74, spec=spec)
    assert not tes_sac_inventory_locked(mean=3.91, var=0.0, target=3.74, spec=spec)
    assert not tes_sac_inventory_locked(mean=3.74, var=0.04, target=3.74, spec=spec)
    assert not tes_sac_inventory_locked(mean=float('nan'), var=0.0, target=3.74, spec=spec)


def test_snap_undershoot_drops_target_to_mean() -> None:
    spec = TesSacSpec(band=0.15)
    assert tes_sac_snap_undershoot(mean=3.45, target=3.74, spec=spec) == pytest.approx(3.45)
    assert tes_sac_snap_undershoot(mean=3.70, target=3.74, spec=spec) == pytest.approx(3.74)
    assert tes_sac_snap_undershoot(mean=3.90, target=3.74, spec=spec) == pytest.approx(3.74)


def test_ratchet_multiplies_without_typed_floor() -> None:
    spec = TesSacSpec(ratchet=0.5)
    assert tes_sac_ratchet(target=4.0, spec=spec) == pytest.approx(2.0)
    assert tes_sac_ratchet(target=1.0, spec=TesSacSpec(ratchet=0.9)) == pytest.approx(0.9)
    assert 'floor' not in tes_sac_ratchet.__code__.co_varnames
