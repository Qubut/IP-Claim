"""Covering measurement reports, oracles, and named culprit loci.

Frozen Pydantic envelopes record schedule, inventory shape, and letter
geometry. Tests and ledger fixtures read these types; they do not encode
or load corpus rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

import pytest
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from ip_claim.collision.cover import Covering

PairScore = Callable[..., Tensor]
PairGap = Callable[..., float | None]
CulpritClass = Literal[
    '',
    'DATA',
    'SCORE',
    'FORWARD_SEAM',
    'REPRESENTATION',
    'BACKWARD_SEAM',
    'OPTIMIZER',
    'INTERACTION',
    'PROXY',
]


class FrozenModel(BaseModel):
    """Immutable Pydantic envelope shared by covering measurement reports."""

    model_config = ConfigDict(frozen=True)


class CoveringSchedule(FrozenModel):
    """Ledger fields recorded after the collator Factory override takes effect."""

    class Mix(FrozenModel):
        """Rho and MLM mix a scheduled covering ledger must carry."""

        mask_schedule: str = ''
        rho: float = 0.0
        mlm_probability: float = 0.0

    kind: Literal['floor', 'scheduled']
    rho: float = Field(ge=0.0, le=1.0)
    entropy_scale: float = Field(ge=0.0, le=1.0)
    mlm_probability: float = Field(ge=0.0, le=1.0)
    inject_scale: float = Field(default=0.0, ge=0.0, le=1.0)
    global_step: int = Field(ge=0)
    rho_warmup_steps: int = Field(ge=0)

    def reading(self) -> dict[str, object]:
        """Fields every covering gate JSON must record."""
        return {
            'mask_schedule': self.kind,
            'rho': self.rho,
            'entropy_scale': self.entropy_scale,
            'mlm_probability': self.mlm_probability,
            'inject_scale': self.inject_scale,
            'global_step': self.global_step,
            'rho_warmup_steps': self.rho_warmup_steps,
        }

    @staticmethod
    def refuse_inert_scheduled(payload: Mapping[str, object]) -> None:
        """Fail closed when a scheduled ledger would encode without graph-mask mix."""
        mix = CoveringSchedule.Mix.model_validate(payload)
        if mix.mask_schedule == 'scheduled' and (mix.rho <= 0.0 or mix.mlm_probability <= 0.0):
            pytest.fail(
                'scheduled mask encode claims rho_max but collate mix is inert '
                f'(rho={mix.rho}, mlm_probability={mix.mlm_probability})'
            )

    def require_honest(self) -> None:
        """Refuse a scheduled ledger whose collate mix cannot apply graph-mask rho."""
        self.refuse_inert_scheduled(self.reading())


class CulpritLocus(FrozenModel):
    """One covering measurement class plus a short named locus."""

    culprit: CulpritClass = ''
    locus: str = 'none'

    def reading(self) -> dict[str, str]:
        """Ledger fields. Empty culprit is a pass."""
        return {'culprit': self.culprit or 'none', 'culprit_name': self.locus}

    @classmethod
    def first(cls, *candidates: CulpritLocus | None) -> CulpritLocus:
        """Keep the first named locus. Empty means the measurement passed."""
        return next((item for item in candidates if item is not None), cls())


class InventoryShapeReport(FrozenModel):
    """Occupied-graph measurements at the covering operating point."""

    mask_schedule: str
    rho: float
    occupancy: float
    utilization_min: float
    usage_entropy: float
    ln_k: float
    termhood_mass: float
    overlay_norm: float
    compose_norm: float
    mixed_norm: float
    prefix_norm: float
    parent_moved: bool
    child_moved: bool
    entropy_pause_ratio: float = 0.95

    def culprit(self) -> CulpritLocus:
        """First inventory-shape failure, or an empty pass."""
        return CulpritLocus.first(
            self._rho_floor(),
            self._termhood(),
            self._dead(),
            self._uniform(),
            self._readout(),
            self._compose(),
        )

    def _rho_floor(self) -> CulpritLocus | None:
        if self.mask_schedule == 'scheduled' and self.rho <= 0.0:
            return CulpritLocus(culprit='PROXY', locus='RHO_FLOOR')
        return None

    def _termhood(self) -> CulpritLocus | None:
        if self.termhood_mass <= 0.0:
            return CulpritLocus(culprit='DATA', locus='TERMHOOD_STORE')
        return None

    def _dead(self) -> CulpritLocus | None:
        if self.occupancy < self.utilization_min:
            return CulpritLocus(culprit='REPRESENTATION', locus='DEAD_OCCUPANCY')
        return None

    def _uniform(self) -> CulpritLocus | None:
        if self.ln_k > 0.0 and self.usage_entropy >= self.entropy_pause_ratio * self.ln_k:
            return CulpritLocus(culprit='REPRESENTATION', locus='UNIFORM_ASSIGNMENT')
        return None

    def _readout(self) -> CulpritLocus | None:
        if self.overlay_norm <= 0.0 or self.prefix_norm <= 0.0:
            return CulpritLocus(culprit='FORWARD_SEAM', locus='GRAPH_READOUT')
        return None

    def _compose(self) -> CulpritLocus | None:
        parent_only = self.parent_moved and not self.child_moved
        mix_stuck = self.overlay_norm > 0.0 and abs(self.mixed_norm - self.compose_norm) <= 1e-12
        if parent_only or mix_stuck:
            return CulpritLocus(culprit='FORWARD_SEAM', locus='CPC_COMPOSE')
        return None


class LetterSeparabilityReport(FrozenModel):
    """ST.14 unpaid XA under matching, foreign, and text-only graph."""

    n_pairs: int
    graph_on_xa: float | None
    graph_off_xa: float | None
    text_only_xa: float | None
    noise: float
    parent_moved: bool
    unpaid_moved: bool
    saturated: bool
    intensities_ok: bool = True
    letter_a_unpaid: float | None = None
    xa_filter: Literal['', 'empty_scores', 'empty_xa', 'null_frac'] = ''

    def culprit(self) -> CulpritLocus:
        """First letter-separability failure, or an empty pass."""
        return CulpritLocus.first(
            self._pairs(),
            self._saturation(),
            self._query_macro(),
            self._intensities(),
            self._disclosure(),
            self._readout(),
            self._text_identity(),
        )

    def _pairs(self) -> CulpritLocus | None:
        if self.n_pairs <= 0:
            return CulpritLocus(culprit='DATA', locus='ST14_PAIRS')
        return None

    def _saturation(self) -> CulpritLocus | None:
        paid_a = self.letter_a_unpaid is not None and 0.0 <= self.letter_a_unpaid <= self.noise
        if self.saturated or (paid_a and not self.unpaid_moved):
            return CulpritLocus(culprit='SCORE', locus='SATURATION')
        return None

    def _query_macro(self) -> CulpritLocus | None:
        if (
            self.graph_on_xa is not None
            and self.graph_off_xa is not None
            and self.text_only_xa is not None
        ):
            return None
        locus = 'QUERY_MACRO_FRAC' if self.xa_filter == 'null_frac' else 'QUERY_MACRO_XA'
        return CulpritLocus(culprit='DATA', locus=locus)

    def _intensities(self) -> CulpritLocus | None:
        if not self.intensities_ok:
            return CulpritLocus(culprit='DATA', locus='TERMHOOD_STORE')
        return None

    def _disclosure(self) -> CulpritLocus | None:
        if self.parent_moved and not self.unpaid_moved:
            return CulpritLocus(culprit='REPRESENTATION', locus='DISCLOSURE_INTENSITY')
        return None

    def _readout(self) -> CulpritLocus | None:
        if not self.parent_moved:
            return CulpritLocus(culprit='FORWARD_SEAM', locus='GRAPH_READOUT')
        return None

    def _text_identity(self) -> CulpritLocus | None:
        on_xa = self.graph_on_xa
        off_xa = self.graph_off_xa
        text_xa = self.text_only_xa
        if on_xa is None or off_xa is None or text_xa is None:
            return None
        owned = (on_xa - off_xa) <= self.noise or (on_xa - text_xa) <= self.noise
        if owned:
            return CulpritLocus(culprit='PROXY', locus='TEXT_IDENTITY')
        return None


def require_named_culprit(verdict: CulpritLocus, *, significant: bool) -> CulpritLocus:
    """Refuse a null architecture reading that has no class and locus."""
    if significant:
        return CulpritLocus(culprit='', locus='none')
    if not verdict.culprit or verdict.locus in {'', 'none'}:
        pytest.fail('null covering result needs a named culprit locus')
    return verdict


def letter_separability_culprit(**reading: object) -> CulpritLocus:
    """Name why ST.14 unpaid geometry is missing or not graph-mediated."""
    return LetterSeparabilityReport.model_validate(reading).culprit()


class LetterOracleEnvelope(FrozenModel):
    """Host-free letter-culprit fixture. Live ST.14 XA is scored, not copied from here."""

    n_pairs: int = Field(ge=0)
    graph_on_xa: float
    graph_off_xa: float
    text_only_xa: float
    noise: float = Field(gt=0.0)
    parent_moved: bool
    unpaid_moved: bool
    saturated: bool
    intensities_ok: bool


class InventoryOracleEnvelope(FrozenModel):
    """Host-free occupied-graph fixture. Occupancy, entropy, and compose norms."""

    occupancy: float = Field(ge=0.0, le=1.0)
    usage_entropy: float = Field(ge=0.0)
    termhood_mass: float = Field(ge=0.0)
    overlay_norm: float = Field(ge=0.0)
    compose_norm: float = Field(ge=0.0)
    mixed_norm: float = Field(ge=0.0)
    prefix_norm: float = Field(ge=0.0)
    parent_moved: bool
    child_moved: bool


def unpaid_gap(covering: Covering, demand: Tensor, supply: Tensor) -> float | None:
    """Finite normalized unpaid gap, or None when the reducer is NaN."""
    value = covering.normalized_unpaid_gap(covering.pair_table(demand, supply))
    return None if torch.isnan(value) else float(value.item())
