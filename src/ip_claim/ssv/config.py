"""SSV train / host configuration schemas."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from patent_ate.spec import AteSpec
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from ip_claim.shared.hf_hub import PublishResult, publish_local_run_to_hf

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAY_SCALING_PATH = _PACKAGE_ROOT / 'configs' / 'ssv_ray.yaml'
DEFAULT_SSV_TRAIN_PATH = _PACKAGE_ROOT / 'configs' / 'ssv_train.yaml'
_DISTRIBUTED_STRATEGIES = frozenset({'auto', 'ddp', 'fsdp_grad_op', 'fsdp_full'})


class HostSpec(BaseModel):
    """Identity of a host language model for LoRA attachment."""

    model_config = ConfigDict(frozen=True)

    name: str = 'hf-internal-testing/tiny-random-bert'
    d_model: int = Field(default=32, ge=1)
    tokenizer_id: str | None = None
    lora_target_modules: tuple[str, ...] = ('query', 'value')


class TesSacSpec(BaseModel):
    """TES-SAC inventory lock: two-sided band, small Finch std, then ratchet."""

    model_config = ConfigDict(frozen=True)

    band: float = Field(default=0.01, ge=0.0)
    std_max: float = Field(default=0.05, ge=0.0)
    ema_decay: float = Field(default=0.999, gt=0.0, lt=1.0)
    ratchet: float = Field(default=0.9, gt=0.0, lt=1.0)
    lock_steps: int = Field(default=20, ge=1)


class GraphProbeSpec(BaseModel):
    """Cadence for the periodic host-only-vs-trunk reliance probe."""

    model_config = ConfigDict(frozen=True)

    interval_steps: int = Field(
        default=50,
        ge=1,
        description='Training steps between eval-mode host-only forward probes.',
    )


class DestComparisonSpec(BaseModel):
    """Matching-versus-dest-shuffle unpaid addend on the train step."""

    model_config = ConfigDict(frozen=True)

    weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            'Scale on the dest-comparison addend. Zero skips the shuffled '
            'forward. A positive value is constant from step zero.'
        ),
    )
    margin: float = Field(
        default=0.0,
        description='Offset inside softplus of matching unpaid minus shuffled unpaid.',
    )
    dest_shift: int = Field(
        default=1,
        description='Destination-index roll of covering inventory n on the slot axis.',
    )


class OverlayShapeSpec(BaseModel):
    """Community-plus-subgraph addend of the existing overlay pair table."""

    model_config = ConfigDict(frozen=True)

    shape_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            'Scale on the overlay-shape addend. Zero skips the family. '
            'Destination leftover unpaid stays a separate head.'
        ),
    )
    shape_collapse_weight: float = Field(
        default=1.0,
        ge=0.0,
        description='Weight on the column-balance penalty inside the community addend.',
    )
    community_columns: int = Field(
        default=2,
        ge=2,
        description=(
            'Host width of the soft community assignment. It is not occupy '
            'support size and not leftover-unpaid-sufficient width.'
        ),
    )


class RayScalingSpec(BaseModel):
    """Ray Train ScalingConfig fields for SSV (worker count and GPU flag)."""

    model_config = ConfigDict(frozen=True)

    num_workers: int = Field(default=1, ge=1)
    use_gpu: bool = False

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> RayScalingSpec:
        """Load scaling fields from YAML; defaults to package ``ssv_ray.yaml``."""
        resolved = path if path is not None else DEFAULT_RAY_SCALING_PATH
        raw: Any = OmegaConf.to_container(OmegaConf.load(str(resolved)), resolve=True)
        if not isinstance(raw, dict):
            msg = f'Ray scaling YAML must be a mapping: {resolved}'
            raise TypeError(msg)
        return cls.model_validate(raw)


class ArchSpec(BaseModel):
    """Trunk geometry: banks, GNN, temperatures, overlay occupancy floor, injection.

    There is no integer occupy cap. The overlay mass is the per-filing
    leftover-unpaid-sufficient mass, with ``soft_occupied_floor`` as its
    lower bound.
    """

    model_config = ConfigDict(frozen=True)

    entity_bank_size: int = Field(
        default=64,
        ge=2,
        description='Entity codebook size K for soft assignment.',
    )
    relation_bank_size: int = Field(
        default=32,
        ge=2,
        description='Relation codebook size for pairwise soft edges.',
    )
    soft_dim: int = Field(
        default=128,
        ge=8,
        description='Width of entity and relation bank vectors before host projection.',
    )
    n_soft_tokens: int = Field(
        default=8,
        ge=1,
        description='Soft-token prefix length prepended to host embeddings.',
    )
    gnn_hidden: int = Field(
        default=128,
        ge=8,
        description='Hidden width of HGT and of each CompGCN-sub compose layer.',
    )
    gnn_heads: int = Field(
        default=4,
        ge=1,
        description='Attention heads per HGT convolution.',
    )
    gnn_layers: int = Field(
        default=2,
        ge=1,
        description='Depth of the HGT stack and of the CompGCN-sub compose stack.',
    )
    assign_temperature: float = Field(
        default=0.07,
        gt=0.0,
        description='Softmax temperature for entity bank assignment.',
    )
    relation_temperature: float = Field(
        default=0.07,
        gt=0.0,
        description='Softmax temperature for relation bank assignment.',
    )
    align_temperature: float = Field(
        default=0.07,
        gt=0.0,
        description='Temperature for the text-versus-graph alignment softmax.',
    )
    dea_temperature: float = Field(
        default=0.07,
        gt=0.0,
        description='Softmax temperature for entity-denoise bank scoring.',
    )
    max_length: int = Field(default=256, ge=8)
    soft_occupied_floor: float = Field(
        default=0.0,
        ge=0.0,
        description='Minimum mask-weighted occupancy for an entity-bank row to enter the overlay.',
    )
    mask_query_window: int = Field(
        default=8,
        ge=1,
        description=(
            'Token radius of unmasked context pooled around each masked position '
            'before FiLM-conditioning the soft-token queries.'
        ),
    )
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    memory_queue_size: int = Field(default=1024, ge=0)
    inject_ramp_steps: int = Field(
        default=100,
        ge=0,
        description=('Steps to ramp the hidden-slot residual scale from 0 to inject_ramp_end.'),
    )
    inject_ramp_end: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            'Ceiling on the RMS-matched hidden-slot residual, as a fraction of '
            'the host embedding RMS at that position. Default 0.5 keeps the '
            'mask embedding the larger summand when the residual is orthogonal.'
        ),
    )
    cpc_identity_dim: int = Field(
        default=16,
        ge=1,
        description=(
            'Width of the learnable CPC-prefix identity table and of filing '
            'node features (CPC and claim) that consume it.'
        ),
    )


class KendallSpec(BaseModel):
    """Kendall homoscedastic seeds and log-sigma clips for the reconstruction heads."""

    model_config = ConfigDict(frozen=True)

    beta_div: float = Field(
        default=0.1,
        ge=0.0,
        description='Initial Kendall weight on diversity (seeds log-sigma).',
    )
    beta_rel_div: float = Field(
        default=1.0,
        ge=0.0,
        description='Weight on relation-bank diversity inside the combined diversity term.',
    )
    kendall_log_sigma_init: float = Field(
        default=0.0,
        description='Initial log-sigma on the MLM and dEA heads (weight exp(-s)).',
    )
    kendall_s_min: float = Field(
        default=-8.0,
        description='Lower clip on Kendall log-sigma so a head weight cannot explode.',
    )
    kendall_s_max: float = Field(
        default=8.0,
        description='Upper clip on Kendall log-sigma so a head weight cannot vanish.',
    )
    include_ke: bool = Field(
        default=False,
        description=(
            'When true, the already-computed KE scalar enters the task mix. '
            'When false, KE stays logged and detached.'
        ),
    )

    @model_validator(mode='after')
    def _clip_box_ordered(self) -> KendallSpec:
        if self.kendall_s_min >= self.kendall_s_max:
            msg = 'kendall_s_min must be below kendall_s_max'
            raise ValueError(msg)
        return self


class DualSpec(BaseModel):
    """SAC dual leftovers: entropy targets, log-lambda box, and pause."""

    model_config = ConfigDict(frozen=True)

    usage_entropy_ratio: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description='Usage-entropy lower bound as a fraction of ln(K).',
    )
    row_entropy_target: float = Field(
        default=math.log(3.0),
        gt=0.0,
        description='Upper bound on mean token assignment entropy (nats).',
    )
    dual_step_size: float = Field(
        default=0.01,
        gt=0.0,
        description='Ascent step on log-lambda duals from detached primal residuals.',
    )
    log_lambda_min: float = Field(
        default=-8.0,
        description='Lower clip on log-lambda when the primal cannot hit a target.',
    )
    log_lambda_max: float = Field(
        default=8.0,
        description='Upper clip on log-lambda when the primal cannot hit a target.',
    )
    log_lambda_lr: float = Field(
        default=1e-4,
        gt=0.0,
        description='Adam ascent rate on log-lambda duals.',
    )
    lambda_usage: float = Field(
        default=1.0,
        ge=0.0,
        description='Initial usage-KL dual weight; live scale is exp(log-lambda_use).',
    )
    lambda_rel_usage: float = Field(
        default=1.0,
        ge=0.0,
        description='Weight on relation usage entropy gap inside diversity.',
    )
    lambda_entropy: float = Field(
        default=1.0,
        ge=0.0,
        description='Initial MAGVIT-gap dual weight; live scale is exp(log-lambda_H).',
    )
    lambda_commit: float = Field(
        default=1.0,
        ge=0.0,
        description='Weight on commitment (query toward assigned codes).',
    )
    lambda_codebook: float = Field(
        default=1.0,
        ge=0.0,
        description='Weight on codebook (codes toward assigned queries).',
    )
    normalize_assignment_entropy: bool = Field(
        default=True,
        description='Divide assignment entropy by ln(K) before the entropy weight.',
    )
    entropy_warmup_steps: int | None = Field(
        default=None,
        ge=0,
        description='Steps to ramp the entropy coefficient; null uses rho warmup length.',
    )
    entropy_pause_ratio: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description='Hold rho when mean row entropy stays above this fraction of ln(K).',
    )
    entropy_pause_steps: int = Field(
        default=20,
        ge=1,
        description='Consecutive high-entropy steps required before holding rho.',
    )
    log_eps: float = Field(
        default=1e-8,
        gt=0.0,
        description='Floor added inside log for entropy and usage terms.',
    )

    @model_validator(mode='after')
    def _clip_box_ordered(self) -> DualSpec:
        if self.log_lambda_min >= self.log_lambda_max:
            msg = 'log_lambda_min must be below log_lambda_max'
            raise ValueError(msg)
        return self


class MlmSpec(BaseModel):
    """MLM mask mix: span geometry, graph-guided rho, CPC literals, and assignment top-k."""

    model_config = ConfigDict(frozen=True)

    mlm_probability: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description='Target fraction of maskable tokens selected for MLM.',
    )
    mlm_span_mask: bool = Field(
        default=True,
        description='Hide contiguous Geo-length spans instead of independent tokens.',
    )
    mlm_span_geo_p: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description='Geometric success probability for span length (clipped at max).',
    )
    mlm_span_max_length: int = Field(
        default=10,
        ge=1,
        description='Upper clip on sampled span length in tokens.',
    )
    cpc_boost: float = Field(
        default=4.0,
        gt=0.0,
        description='Rate multiplier for CPC-literal hits when the aux mix is on.',
    )
    cpc_aux_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description='Mix weight of CPC-literal rates into graph mask rates; 0 disables.',
    )
    assignment_top_k: int = Field(
        default=1,
        ge=1,
        description='Top assignment masses summed per token for graph mask rates.',
    )
    rho_max: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description='Ceiling on the graph-guided mask mixture after warmup.',
    )
    rho_warmup_steps: int = Field(
        default=100,
        ge=0,
        description='Steps to ramp rho from 0 to rho_max.',
    )


class BankHealthSpec(BaseModel):
    """Codebook health: usage EMA, utilization floor, and k-means bank seed."""

    model_config = ConfigDict(frozen=True)

    usage_ema_momentum: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description='EMA momentum for bank-slot usage estimates.',
    )
    utilization_eps: float = Field(
        default=1e-3,
        ge=0.0,
        description='Usage floor that counts a codebook slot as used.',
    )
    utilization_min: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description='Minimum used-slot fraction before rho is treated as utilization-stuck.',
    )
    bank_init_from_batch: bool = Field(
        default=True,
        description='Seed banks from the first batch with k-means centroids.',
    )
    bank_init_kmeans_iters: int = Field(
        default=10,
        ge=1,
        description='K-means iterations when seeding banks from a batch.',
    )


class FitSpec(BaseModel):
    """Lightning fit: AdamW, batching, devices, strategy, checkpoints, and track patience."""

    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        description='AdamW peak learning rate for the Lightning module.',
    )
    lr_warmup_steps: int | None = Field(
        default=None,
        ge=0,
        description='Linear warmup length; null means no warmup. Peak LR stays after warmup.',
    )
    early_stop_patience: int = Field(
        default=20,
        ge=1,
        description=(
            'Consecutive train-batch checks for InventoryTrackStop (TES-SAC stall) '
            'and HostRotStop (NLL above the seeded host ceiling while λ_host is clipped).'
        ),
    )
    max_steps: int = Field(
        default=100_000,
        ge=1,
        description='Lightning safety step ceiling; fit ends earlier on inventory track stop.',
    )
    batch_size: int = Field(
        default=2,
        ge=1,
        description='Per-device train batch size.',
    )
    accumulate_grad_batches: int = Field(default=1, ge=1)
    num_devices: int | None = Field(
        default=None,
        ge=1,
        description='GPU count for local Lightning DDP; null uses all visible GPUs.',
    )
    dataloader_num_workers: int = Field(default=0, ge=0)
    dataloader_prefetch_factor: int = Field(
        default=2,
        ge=1,
        description='Queued batches per DataLoader worker when workers are on.',
    )
    dataloader_pin_memory: bool = Field(
        default=False,
        description='Pin collated host tensors for overlapped device copies.',
    )
    prime_jate_spans: bool = Field(
        default=False,
        description='Draw noun-phrase spans in DataLoader workers before the GPU step.',
    )
    log_every_n_steps: int = Field(default=10, ge=1)
    enable_progress_bar: bool = True
    enable_csv_logger: bool = True
    distributed_strategy: str = Field(
        default='ddp',
        description='Multi-GPU strategy: ddp, fsdp_grad_op, fsdp_full, or auto.',
    )
    ddp_static_graph: bool = True
    ddp_find_unused_parameters: bool = Field(
        default=True,
        description='DDP scans for parameters omitted from a step loss (dynamic hetero graphs).',
    )
    fsdp_activation_checkpointing: bool = False
    checkpoint_dir: str = 'artifacts/ssv'
    checkpoint_every_n_steps: int = Field(default=1000, ge=1)
    checkpoint_save_top_k: int = Field(default=-1, ge=-1)

    @field_validator('distributed_strategy')
    @classmethod
    def _validate_distributed_strategy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _DISTRIBUTED_STRATEGIES:
            msg = f'distributed_strategy must be one of {sorted(_DISTRIBUTED_STRATEGIES)}'
            raise ValueError(msg)
        return normalized


class LogSpec(BaseModel):
    """Per-step metric flags logged from the Lightning module."""

    model_config = ConfigDict(frozen=True)

    log_mlm_accuracy: bool = Field(
        default=True,
        description='Log token-level MLM accuracy on labeled positions.',
    )
    log_compose_ranking: bool = Field(
        default=True,
        description=(
            'Log filtered compose-stack pair ranking: TransE residual, MRR, '
            'mean rank, and Hits at 1, 3, and 10.'
        ),
    )
    log_learning_rate: bool = Field(
        default=True,
        description='Log the optimizer learning rate each step.',
    )
    log_trainable_params: bool = Field(
        default=True,
        description='Log trainable versus total parameter counts at fit start.',
    )
    log_grad_norm: bool = Field(
        default=True,
        description='Log the global gradient norm after backward.',
    )
    grad_norm_type: float = Field(
        default=2.0,
        gt=0.0,
        description='p-norm used when logging the global gradient norm.',
    )


class RuntimeSpec(BaseModel):
    """Hub publish, Ray worker flag, GPU pin, HUPD corpus, and optional warm-start path."""

    model_config = ConfigDict(frozen=True)

    hupd_dir: str | None = None
    hupd_limit: int | None = Field(default=None, ge=1)
    termhood_store_path: str | None = Field(
        default=None,
        description='Pre-built termhood store root mounted into the container.',
    )
    init_weights: str | None = Field(
        default=None,
        description=(
            'Lightning checkpoint used as a model-prefix warm start. '
            'Optimizer, Kendall log-sigma, and dual buffers stay at their fresh init.'
        ),
    )
    hf_token: SecretStr | None = None
    publish_to_hub: bool = False
    hf_hub_repo_id: str | None = None
    hf_hub_private: bool = True
    hf_hub_run_name: str | None = None
    ray: bool = False
    use_gpu: bool = False

    @field_validator('hf_token', 'init_weights', mode='before')
    @classmethod
    def _empty_str_as_none(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value


_NEST_SPECS: dict[str, type[BaseModel]] = {
    'arch': ArchSpec,
    'kendall': KendallSpec,
    'dual': DualSpec,
    'mlm': MlmSpec,
    'bank': BankHealthSpec,
    'fit': FitSpec,
    'log': LogSpec,
    'runtime': RuntimeSpec,
    'graph_probe': GraphProbeSpec,
    'ate': AteSpec,
    'dest_comparison': DestComparisonSpec,
    'overlay_shape': OverlayShapeSpec,
}

_FLAT_TO_NEST: dict[str, str] = {
    field_name: nest_name
    for nest_name, spec in _NEST_SPECS.items()
    for field_name in spec.model_fields
}


class SsvTrainConfig(BaseModel):
    """Single load envelope for SSV training hyperparameters."""

    model_config = ConfigDict(frozen=True)

    host: HostSpec = Field(
        default_factory=HostSpec,
        description='Host masked LM identity, hidden size, and LoRA target modules.',
    )
    tes_sac: TesSacSpec = Field(
        default_factory=TesSacSpec,
        description='TES-SAC lock band, Finch decay, ratchet, and consecutive lock count.',
    )
    graph_probe: GraphProbeSpec = Field(
        default_factory=GraphProbeSpec,
        description='Cadence for the periodic graph-reliance probe.',
    )
    dest_comparison: DestComparisonSpec = Field(
        default_factory=DestComparisonSpec,
        description='Matching-versus-dest-shuffle unpaid addend; zero weight disables it.',
    )
    overlay_shape: OverlayShapeSpec = Field(
        default_factory=OverlayShapeSpec,
        description='Overlay-shape addend of kept pair mass; zero weight disables it.',
    )
    ate: AteSpec = Field(
        default_factory=AteSpec,
        description='Occupancy term recognition from the ATE package.',
    )
    arch: ArchSpec = Field(
        default_factory=ArchSpec,
        description='Banks, GNN, temperatures, occupied-code cap, and injection ramp.',
    )
    kendall: KendallSpec = Field(
        default_factory=KendallSpec,
        description='Kendall head seeds and log-sigma clips.',
    )
    dual: DualSpec = Field(
        default_factory=DualSpec,
        description='Inventory and entropy dual leftovers still on the train contract.',
    )
    mlm: MlmSpec = Field(
        default_factory=MlmSpec,
        description='MLM span mix, rho schedule, CPC literals, and assignment top-k.',
    )
    bank: BankHealthSpec = Field(
        default_factory=BankHealthSpec,
        description='Usage EMA, utilization floor, and k-means bank seed.',
    )
    fit: FitSpec = Field(
        default_factory=FitSpec,
        description='Lightning optimizer, batching, devices, strategy, and checkpoints.',
    )
    log: LogSpec = Field(
        default_factory=LogSpec,
        description='Which Lightning scalars the module emits.',
    )
    runtime: RuntimeSpec = Field(
        default_factory=RuntimeSpec,
        description='Hub publish, Ray/GPU pin, HUPD corpus, and optional warm-start path.',
    )

    @model_validator(mode='before')
    @classmethod
    def _lift_flat_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        def nest_payload(raw: object) -> dict[str, Any] | None:
            if isinstance(raw, BaseModel):
                return raw.model_dump()
            if isinstance(raw, dict):
                return dict(raw)
            return None

        lifted = dict(data)
        nests = {
            name: payload
            for name in _NEST_SPECS
            if (payload := nest_payload(lifted.get(name))) is not None
        }
        remaining = {key: value for key, value in lifted.items() if key not in nests}
        for key in tuple(remaining):
            nest_name = _FLAT_TO_NEST.get(key)
            if nest_name is None:
                continue
            nests.setdefault(nest_name, {})[key] = remaining.pop(key)
        remaining.update(nests)
        return remaining

    def overlay(self, updates: Mapping[str, Any]) -> SsvTrainConfig:
        """Re-validate after applying flat or nested overrides onto a dump."""
        payload = self.model_dump()
        payload.update(dict(updates))
        return type(self).model_validate(payload)

    def resolved_entropy_warmup_steps(self) -> int:
        """Entropy-coefficient warmup length; falls back to rho warmup when unset."""
        if self.dual.entropy_warmup_steps is None:
            return int(self.mlm.rho_warmup_steps)
        return int(self.dual.entropy_warmup_steps)

    def resolved_lr_warmup_steps(self) -> int:
        """Linear warmup length; null is no warmup. Clipped to the safety ceiling."""
        declared = 0 if self.fit.lr_warmup_steps is None else int(self.fit.lr_warmup_steps)
        return min(int(self.fit.max_steps), declared)

    def resolved_usage_entropy_target(self) -> float:
        """Lower bound on batch usage entropy: ratio times ln(K)."""
        return float(self.dual.usage_entropy_ratio) * math.log(int(self.arch.entity_bank_size))

    def upload_to_hub(self, run_dir: Path) -> PublishResult | None:
        """Upload run metadata when ``publish_to_hub`` is enabled."""
        if not self.runtime.publish_to_hub:
            return None
        repo_id = (self.runtime.hf_hub_repo_id or '').strip()
        if not repo_id:
            msg = 'publish_to_hub requires hf_hub_repo_id'
            raise ValueError(msg)
        token = self.runtime.hf_token if self.runtime.hf_token is not None else SecretStr('')
        return publish_local_run_to_hf(
            run_dir,
            repo_id=repo_id,
            run_name=self.runtime.hf_hub_run_name,
            hf_token=token,
            private=bool(self.runtime.hf_hub_private),
        )

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> SsvTrainConfig:
        """Load train fields from YAML with OmegaConf env interpolation."""
        resolved = path if path is not None else DEFAULT_SSV_TRAIN_PATH
        raw: Any = OmegaConf.to_container(OmegaConf.load(str(resolved)), resolve=True)
        if not isinstance(raw, dict):
            msg = f'SSV train YAML must be a mapping: {resolved}'
            raise TypeError(msg)
        return cls.model_validate(raw)

    @classmethod
    def from_yaml_or_job(
        cls,
        job: SsvTrainConfig | None = None,
        config_path: Path | None = None,
        *,
        hupd_dir: Path | None = None,
    ) -> SsvTrainConfig:
        """Keep an injected job or load YAML, then overlay an optional HUPD root."""
        base = job or cls.from_yaml(
            config_path if config_path is not None else DEFAULT_SSV_TRAIN_PATH
        )
        return base if hupd_dir is None else base.overlay({'hupd_dir': str(hupd_dir)})

    def hupd_root(self) -> Path:
        """HUPD JSON directory from runtime, else package test fixtures."""
        listed = self.runtime.hupd_dir
        if listed:
            return Path(listed)
        repo = Path(str(DEFAULT_SSV_TRAIN_PATH.resolve().parents[1]))
        return repo / 'tests' / 'fixtures' / 'hupd'


__all__ = [
    'DEFAULT_RAY_SCALING_PATH',
    'DEFAULT_SSV_TRAIN_PATH',
    'ArchSpec',
    'AteSpec',
    'BankHealthSpec',
    'DestComparisonSpec',
    'DualSpec',
    'FitSpec',
    'GraphProbeSpec',
    'HostSpec',
    'KendallSpec',
    'LogSpec',
    'MlmSpec',
    'OverlayShapeSpec',
    'RayScalingSpec',
    'RuntimeSpec',
    'SsvTrainConfig',
    'TesSacSpec',
]
