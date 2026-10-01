"""Collision eval configuration schemas."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from ip_claim.collision.cover import CoveringKnobs

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COLLISION_EVAL_PATH = _PACKAGE_ROOT / 'configs' / 'collision_eval.yaml'
SHIPPED_COLLISION_EVAL_PATHS = (
    DEFAULT_COLLISION_EVAL_PATH,
    _PACKAGE_ROOT / 'configs' / 'collision_eval.smoke.yaml',
    _PACKAGE_ROOT / 'configs' / 'collision_eval.smoke.host_only.yaml',
)


class CollisionSplitSpec(BaseModel):
    """Query-application fractions for train, eval, and test pair sets."""

    model_config = ConfigDict(frozen=True)

    train: float = Field(default=0.6, ge=0.0, le=1.0)
    eval: float = Field(default=0.2, ge=0.0, le=1.0)
    test: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int = 42

    @model_validator(mode='after')
    def _fractions_sum_to_one(self) -> CollisionSplitSpec:
        total = self.train + self.eval + self.test
        if abs(total - 1.0) > 1e-6:
            msg = f'split fractions must sum to 1.0, got {total}'
            raise ValueError(msg)
        return self


class CollisionPrefixMode(StrEnum):
    """Whether encode prepends the graph-bank prefix or runs the host alone."""

    trunk = 'trunk'
    host_only = 'host_only'


class KeepGridCell(BaseModel):
    """One score-time slot-keep setting for a rank-from-shards probe."""

    model_config = ConfigDict(frozen=True)

    slot_mass_keep: float | None = Field(default=None, gt=0.0, le=1.0)
    slot_top_k: int | None = Field(default=None, ge=1)

    @model_validator(mode='after')
    def _has_slot_keep(self) -> KeepGridCell:
        if self.slot_mass_keep is None and self.slot_top_k is None:
            msg = 'keep grid cell needs slot_mass_keep or slot_top_k'
            raise ValueError(msg)
        return self

    def as_covering(self, base: CoveringKnobs) -> CoveringKnobs:
        """Copy locked scales from base and apply this cell's slot keep."""
        return base.model_copy(
            update={
                'slot_mass_keep': self.slot_mass_keep,
                'slot_top_k': self.slot_top_k,
            }
        )


class CollisionEvalConfig(BaseModel):
    """Offline covering eval: encode inventories, rank citation pairs, write artefacts."""

    model_config = ConfigDict(frozen=True)

    recall_k: tuple[int, ...] = Field(
        default=(5, 10, 20, 50),
        description='Cutoffs for Recall and MRR on examiner grades.',
    )
    dataset: str | None = Field(
        default=None,
        description='Citation-pair table (application ids and ST.14 letters).',
    )
    hupd_dir: str | None = Field(
        default=None,
        description='HUPD JSON tree used to resolve application stems at encode.',
    )
    encode_shards: str | None = Field(
        default=None,
        description='Existing intensity-shard directory when reuse_encode is set.',
    )
    output_dir: str = Field(
        default='artifacts/collision',
        description='Root for shards, covering.json, and explain dumps.',
    )
    checkpoint: str | None = Field(
        default=None,
        description='SSV Lightning checkpoint for trunk weights. Null skips load.',
    )
    ssv_config: str | None = Field(
        default=None,
        description='SSV train envelope that builds the frozen trunk architecture.',
    )
    split: CollisionSplitSpec = Field(
        default_factory=CollisionSplitSpec,
        description='Query-application fractions and seed for train, eval, and test.',
    )
    query_limit: int | None = Field(
        default=None,
        ge=1,
        description='Cap on unique query applications per split. Null uses the full split.',
    )
    disclosure_max_chunks: int | None = Field(
        default=None,
        ge=1,
        description=(
            'Extra description windows added onto document supply. '
            'Null leaves supply as the train string (claims, abstract, summary).'
        ),
    )
    encode_batch_size: int = Field(
        default=4,
        ge=1,
        description='Filings per encode step.',
    )
    use_gpu: bool = Field(
        default=True,
        description='Run encode and rank on CUDA when a device is visible.',
    )
    num_devices: int | None = Field(
        default=None,
        ge=1,
        description='Encode ranks. Null lets Lightning pick visible GPUs.',
    )
    dataloader_num_workers: int = Field(
        default=0,
        ge=0,
        description='Host processes that load HUPD JSON during encode.',
    )
    shard_flush_every: int = Field(
        default=50,
        ge=1,
        description='Encode batches between intensity-shard writes.',
    )
    prefix_mode: CollisionPrefixMode = Field(
        default=CollisionPrefixMode.trunk,
        description='trunk prepends soft-graph tokens; host_only is the LM without that prefix.',
    )
    covering: CoveringKnobs = Field(
        default_factory=CoveringKnobs,
        description='Saturation scales, relation mix, and optional score-time keep.',
    )
    keep_grid: tuple[KeepGridCell, ...] = Field(
        default=(),
        description='Optional slot-keep cells to re-rank already encoded shards.',
    )
    contour_tau: float = Field(
        default=0.3,
        gt=0.0,
        lt=1.0,
        description='Paid-mass threshold for the explain community filtration.',
    )
    explain_top_n: int = Field(
        default=5,
        ge=1,
        description='Top covering hits per query written as explain artefacts.',
    )
    reuse_encode: bool = Field(
        default=True,
        description='Reuse encode_shards when that directory already has intensities.',
    )
    rank_query_tile: int = Field(
        default=256,
        ge=1,
        description='Query rows scored together against the document banks.',
    )
    publish_to_hub: bool = Field(
        default=False,
        description='Upload the eval artefact directory after a finished run.',
    )
    hf_hub_repo_id: str | None = Field(
        default=None,
        description='Hub repo for that upload. Required when publish_to_hub is true.',
    )
    hf_hub_private: bool = Field(
        default=True,
        description='Create or keep the Hub repo private.',
    )
    hf_hub_run_name: str | None = Field(
        default=None,
        description='Optional run label stored with the Hub upload.',
    )
    hf_token: SecretStr | None = Field(
        default=None,
        description='Hub token. Empty string is treated as unset.',
    )

    @field_validator('hf_token', mode='before')
    @classmethod
    def _empty_token_as_none(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> CollisionEvalConfig:
        """Load eval fields from YAML with OmegaConf env interpolation."""
        resolved = path if path is not None else DEFAULT_COLLISION_EVAL_PATH
        raw: Any = OmegaConf.to_container(OmegaConf.load(str(resolved)), resolve=True)
        if not isinstance(raw, dict):
            msg = f'Collision eval YAML must be a mapping: {resolved}'
            raise TypeError(msg)
        # Host knobs belong on SsvTrainConfig; strip before validate.
        payload = {key: value for key, value in raw.items() if key != 'host'}
        return cls.model_validate(payload)


__all__ = [
    'DEFAULT_COLLISION_EVAL_PATH',
    'SHIPPED_COLLISION_EVAL_PATHS',
    'CollisionEvalConfig',
    'CollisionPrefixMode',
    'CollisionSplitSpec',
    'CoveringKnobs',
    'KeepGridCell',
]
