"""TES-SAC inventory lock, undershoot snap, and target ratchet."""

from __future__ import annotations

import math

from ip_claim.ssv.config import TesSacSpec


def finch_update(*, mean: float, var: float, live: float, decay: float) -> tuple[float, float]:
    """One Finch increment of mean and variance for a finite observation."""
    alpha = 1.0 - decay
    delta = live - mean
    incr = alpha * delta
    return mean + incr, max(decay * (var + delta * incr), 0.0)


def tes_sac_inventory_locked(
    *,
    mean: float,
    var: float,
    target: float,
    spec: TesSacSpec,
) -> bool:
    """True when Finch mean is two-sided in-band of the target and std is small."""
    finite = math.isfinite(mean) and math.isfinite(var) and math.isfinite(target)
    return finite and abs(mean - target) <= spec.band and math.sqrt(max(var, 0.0)) <= spec.std_max


def tes_sac_snap_undershoot(*, mean: float, target: float, spec: TesSacSpec) -> float:
    """Drop an inventory upper-bound target to the Finch mean when it undershoots."""
    undershot = math.isfinite(mean) and math.isfinite(target) and mean < target - spec.band
    return mean if undershot else target


def tes_sac_ratchet(*, target: float, spec: TesSacSpec) -> float:
    """Multiply a locked inventory target by the ratchet."""
    if not math.isfinite(target):
        return target
    return float(target * spec.ratchet)
