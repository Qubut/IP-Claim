"""Covering draw sizes, step budgets, and encode overlay.

``CoveringPilotSpec`` loads the experiment YAML, then overlays
``COVERING_*`` environment values and ``--covering-*`` CLI flags. Tests
read the validated spec; they do not hardcode hygiene N or step counts.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Literal, Self, get_type_hints

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, computed_field, model_validator

from experiments.covering_objective.verdicts import (
    FrozenModel,
    InventoryOracleEnvelope,
    InventoryShapeReport,
    LetterOracleEnvelope,
)
from ip_claim.collision.data.citation_pairs import CitationPair
from ip_claim.ssv.config import SsvTrainConfig

_PILOT_YAML = Path(__file__).resolve().parent / 'pilot.yaml'


def covering_cli_flag(path: str) -> str:
    """Pytest flag for one covering-spec path."""
    return '--covering-' + path.replace('.', '-').replace('_', '-')


def covering_env_name(path: str) -> str:
    """Environment name for one covering-spec path."""
    return 'COVERING_' + path.replace('.', '_').upper()


class CoveringPilotSpec(FrozenModel):
    """Draw sizes, step budgets, and encode overlay for covering preflight."""

    seed: int = Field(ge=0)
    hygiene_n: int = Field(ge=2)
    hygiene_fraction: float = Field(gt=0.0, le=1.0)
    isolated_steps: int = Field(ge=1)
    isolated_readouts: tuple[int, ...] = Field(min_length=1)
    held_out_steps: int = Field(ge=1)
    held_out_ceiling_s: int = Field(ge=1)
    held_out_n: int = Field(ge=1)
    patch_n: int = Field(ge=2)
    functional_n: int = Field(ge=2)
    functional_step: float = Field(gt=0.0)
    functional_steps: tuple[float, ...] = Field(default=())
    encode_max_length: int = Field(ge=8)
    batch_size: int = Field(ge=1)
    algebraic_examples: int = Field(ge=1)
    disclosure_max_chunks: int = Field(default=8, ge=1)
    mask_schedule: Literal['floor', 'scheduled'] = 'scheduled'
    entropy_pause_ratio: float = Field(gt=0.0, le=1.0)
    letter: LetterOracleEnvelope
    inventory: InventoryOracleEnvelope

    def readout_steps(self) -> tuple[int, ...]:
        """Readout indices including the isolated-step budget."""
        return tuple(sorted({*self.isolated_readouts, self.isolated_steps}))

    def hygiene_floor(self) -> float:
        """Minimum valid-pair count for the paired-identity gate."""
        return self.hygiene_fraction * self.hygiene_n

    def letter_scored_pairs(self, pairs: Sequence[CitationPair]) -> tuple[CitationPair, ...]:
        """Development pairs whose unique stems fit inside ``hygiene_n``."""
        budget = self.hygiene_n

        def take(stems: frozenset[str], pair: CitationPair) -> frozenset[str]:
            extras = (
                frozenset((pair.query_application_number, pair.partner_application_number)) - stems
            )
            if extras and len(stems) + len(extras) > budget:
                return stems
            return stems | extras

        stems = reduce(take, pairs, frozenset())
        return tuple(
            pair
            for pair in pairs
            if pair.query_application_number in stems and pair.partner_application_number in stems
        )

    def inventory_envelope(self, job: SsvTrainConfig | None = None) -> InventoryShapeReport:
        """Occupied-graph oracle from this envelope plus the train contract."""
        contract = job if job is not None else SsvTrainConfig()
        bank = max(int(contract.arch.entity_bank_size), 1)
        return InventoryShapeReport(
            mask_schedule=self.mask_schedule,
            rho=float(contract.mlm.rho_max),
            occupancy=self.inventory.occupancy,
            utilization_min=float(contract.bank.utilization_min),
            usage_entropy=self.inventory.usage_entropy,
            ln_k=math.log(bank),
            termhood_mass=self.inventory.termhood_mass,
            overlay_norm=self.inventory.overlay_norm,
            compose_norm=self.inventory.compose_norm,
            mixed_norm=self.inventory.mixed_norm,
            prefix_norm=self.inventory.prefix_norm,
            parent_moved=self.inventory.parent_moved,
            child_moved=self.inventory.child_moved,
            entropy_pause_ratio=self.entropy_pause_ratio,
        )

    @computed_field
    @property
    def step_ladder(self) -> tuple[float, ...]:
        """Exact-step sizes, smallest first. The single step is the fallback."""
        return self.functional_steps or (self.functional_step,)

    @model_validator(mode='after')
    def _readouts_inside_budget(self) -> Self:
        if any(step < 0 or step > self.isolated_steps for step in self.isolated_readouts):
            msg = 'isolated_readouts must lie in [0, isolated_steps]'
            raise ValueError(msg)
        if any(step <= 0.0 for step in self.functional_steps):
            msg = 'functional_steps must be positive'
            raise ValueError(msg)
        return self

    @classmethod
    def option_paths(cls) -> tuple[str, ...]:
        """Dotted spec paths exposed as ``--covering-*`` and ``COVERING_*``."""

        def paths_of(model: type[BaseModel], prefix: str) -> tuple[str, ...]:
            hints = get_type_hints(model)

            def for_field(name: str) -> tuple[str, ...]:
                annotation = hints[name]
                nested = (
                    annotation
                    if isinstance(annotation, type) and issubclass(annotation, BaseModel)
                    else None
                )
                path = f'{prefix}{name}'
                return (path,) if nested is None else paths_of(nested, f'{path}.')

            return tuple(chain.from_iterable(for_field(name) for name in model.model_fields))

        return paths_of(cls, '')

    @staticmethod
    def _overlay_mapping(loaded: object, *, message: str) -> dict[str, object]:
        if not isinstance(loaded, dict):
            raise TypeError(message)
        return {str(key): value for key, value in loaded.items()}

    @staticmethod
    def _overlay_from_dotlist(pairs: tuple[tuple[str, str], ...]) -> dict[str, object]:
        if not pairs:
            return {}
        return CoveringPilotSpec._overlay_mapping(
            OmegaConf.to_container(
                OmegaConf.from_dotlist([f'{path}={value}' for path, value in pairs]),
                resolve=True,
            ),
            message='covering knob overlay must be a mapping',
        )

    @classmethod
    def _overlay_from_env(cls) -> dict[str, object]:
        pairs = tuple(
            (path, os.environ[env])
            for path in cls.option_paths()
            for env in (covering_env_name(path),)
            if os.environ.get(env, '').strip()
        )
        return cls._overlay_from_dotlist(pairs)

    @classmethod
    def _overlay_from_cli(cls, config: pytest.Config | None) -> dict[str, object]:
        if config is None:
            return {}
        pairs = tuple(
            (path, str(value))
            for path in cls.option_paths()
            for value in (config.getoption(covering_cli_flag(path), default=None),)
            if value is not None
        )
        return cls._overlay_from_dotlist(pairs)

    @classmethod
    def from_yaml(
        cls,
        path: Path | None = None,
        *,
        config: pytest.Config | None = None,
    ) -> CoveringPilotSpec:
        """Load the experiment envelope next to this collection, then env and CLI."""
        resolved = path if path is not None else _PILOT_YAML
        raw = cls._overlay_mapping(
            OmegaConf.to_container(OmegaConf.load(str(resolved)), resolve=True),
            message=f'covering pilot YAML must be a mapping: {resolved}',
        )
        merged = OmegaConf.merge(
            OmegaConf.create(raw),
            OmegaConf.create(cls._overlay_from_env()),
            OmegaConf.create(cls._overlay_from_cli(config)),
        )
        payload = cls._overlay_mapping(
            OmegaConf.to_container(merged, resolve=True),
            message=f'covering pilot overlay must stay a mapping: {resolved}',
        )
        return cls.model_validate(payload)
