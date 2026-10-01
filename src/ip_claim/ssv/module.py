"""Lightning training step for graph-guided MLM plus joint trunk losses.

Owns Kendall-weighted MLM, diversity, and entity-denoise heads. Destination
comparison is leftover unpaid of numbered-claim demand against covering n.
Overlay shape is an optional addend of the existing kept pair table. Duals on
``log lambda`` scale inventory (upper), usage (lower), row-entropy hygiene,
and host-NLL slack after Kendall. The covering-ready train reading logs
directional slacks so a falling document entropy cannot be read as rot.
The θ step divides the mix by one plus the dual sum. Ramps mask-mix ``rho`` with
occupancy and assignment-entropy pause, and enlarges InfoNCE negatives via
``all_gather`` plus a detached memory queue.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any, TypedDict, cast

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.grads import grad_norm
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from patent_ate.termhood import TermhoodTable
from peft import PeftModel
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from transformers.optimization import get_scheduler

from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmCollator
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.covering_gate import CoveringTrainReading
from ip_claim.ssv.encode import ComposeScoreReport
from ip_claim.ssv.graph_prefix import measure_graph_prefix_delta
from ip_claim.ssv.inventory import CoveringInventory, Inventory
from ip_claim.ssv.model import SoftTrunkModel, SoftTrunkOutput
from ip_claim.ssv.shape import shape_family
from ip_claim.ssv.steer.cheng import cheng_rescale

if TYPE_CHECKING:
    from ip_claim.collision.cover import Covering
from ip_claim.ssv.steer.duals import LogLambdaAdam
from ip_claim.ssv.steer.tes_sac import (
    finch_update,
    tes_sac_inventory_locked,
    tes_sac_ratchet,
    tes_sac_snap_undershoot,
)

_STAGE_MEAN_KEYS = (
    'mlm_nll',
    'div_loss',
    'ke_loss',
    'soft_ke_loss',
    'align_loss',
    'occupancy',
    'relation_occupancy',
    'n_dead_before_restart',
    'usage_entropy',
    'inverse_simpson',
    'assignment_perplexity',
    'row_entropy',
    'compose_transe',
    'compose_mrr',
    'compose_mr',
    'compose_hits_at_1',
    'compose_hits_at_3',
    'compose_hits_at_10',
    'dea_gap',
    'dea_loss',
)


class SsvLrSchedulerConfig(TypedDict):
    """Step interval and frequency for the AdamW learning-rate schedule."""

    scheduler: LRSchedulerTypeUnion
    interval: str
    frequency: int


class SsvOptimizerLRConfig(TypedDict):
    """AdamW plus the learning-rate schedule Lightning consumes after each step."""

    optimizer: Optimizer
    lr_scheduler: SsvLrSchedulerConfig


class SsvLightningModule(pl.LightningModule):
    """Train SoftTrunkModel with GM, diversity, KE, and patent-graph align."""

    inventory_entropy_target: Tensor
    inventory_track_count: Tensor
    inventory_target_seeded: Tensor
    usage_entropy_target: Tensor
    usage_target_seeded: Tensor
    row_entropy_target: Tensor
    row_target_seeded: Tensor
    inventory_entropy_ema: Tensor
    inventory_entropy_ema_var: Tensor
    inventory_ema_seeded: Tensor
    last_inventory_entropy: Tensor
    last_row_entropy: Tensor
    last_usage_entropy: Tensor
    last_host_nll: Tensor
    host_nll_star: Tensor
    host_nll_slack: Tensor
    host_nll_star_seeded: Tensor
    host_nll_seed_count: Tensor
    host_nll_ema: Tensor
    host_nll_ema_var: Tensor
    host_nll_ema_seeded: Tensor
    last_graph_reliance_delta: Tensor
    host_nll_star_metric: MeanMetric
    kendall_s_mlm: nn.Parameter
    kendall_s_div: nn.Parameter
    kendall_s_dea: nn.Parameter
    _zg_queue: Tensor
    _zg_queue_len: Tensor
    _zg_queue_ptr: Tensor
    duals: LogLambdaAdam
    model: SoftTrunkModel
    covering: Covering
    inventory: Inventory
    train_metrics: MetricCollection
    val_metrics: MetricCollection
    train_accuracy: MetricCollection
    val_accuracy: MetricCollection

    def __init__(
        self,
        config: SsvTrainConfig,
        model: SoftTrunkModel,
        covering: Covering | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        if covering is None:
            cover_mod = importlib.import_module('ip_claim.collision.cover')
            covering = cast(Covering, cover_mod.Covering(cover_mod.CoveringKnobs()))
        self.covering = covering
        self.inventory = (
            inventory
            if inventory is not None
            else Inventory(
                occupied_floor=float(config.arch.soft_occupied_floor),
            )
        )
        config_snapshot = config.model_dump(mode='json')
        config_snapshot.pop('hf_token', None)
        self.save_hyperparameters(config_snapshot)
        self._mask_collator: SoftMlmCollator | None = None
        self._rho = 0.0
        self._rho_ceiling = 0.0
        self._last_occupancy = 1.0
        self._entropy_uniform_streak = 0

        self.train_metrics = MetricCollection(
            {name: MeanMetric() for name in _STAGE_MEAN_KEYS},
            compute_groups=False,
        )
        self.val_metrics = self.train_metrics.clone(prefix='val_')

        vocab_size = int(model.host.config.vocab_size)
        self.train_accuracy = MetricCollection(
            {
                'mlm_accuracy': MulticlassAccuracy(
                    num_classes=vocab_size,
                    ignore_index=-100,
                    average='micro',
                )
            },
        )
        self.val_accuracy = self.train_accuracy.clone(prefix='val_')

        d_model = int(config.host.d_model)
        queue_size = int(config.arch.memory_queue_size)
        self._zg_queue = nn.Buffer(torch.zeros(queue_size, d_model), persistent=False)
        self._zg_queue_len = nn.Buffer(torch.zeros((), dtype=torch.long), persistent=False)
        self._zg_queue_ptr = nn.Buffer(torch.zeros((), dtype=torch.long), persistent=False)
        self.duals = LogLambdaAdam(
            ('inv', 'use', 'h', 'host'),
            lr=float(self.config.dual.log_lambda_lr),
            lo=float(self.config.dual.log_lambda_min),
            hi=float(self.config.dual.log_lambda_max),
        )

        def kendall_log(weight: float, *, mlm: bool) -> nn.Parameter:
            lo = float(self.config.kendall.kendall_s_min)
            hi = float(self.config.kendall.kendall_s_max)
            seed = float(self.config.kendall.kendall_log_sigma_init)
            if not mlm and weight > 0.0:
                seed = -math.log(weight)
            return nn.Parameter(torch.tensor(min(max(seed, lo), hi)))

        def init_steer_state() -> None:
            """Register TES-SAC entropy-target buffers."""
            self.register_buffer(
                'inventory_entropy_target',
                torch.tensor(math.log(int(self.config.arch.entity_bank_size))),
            )
            self.register_buffer('inventory_track_count', torch.zeros((), dtype=torch.long))
            self.register_buffer('inventory_target_seeded', torch.zeros((), dtype=torch.bool))
            self.register_buffer('usage_target_seeded', torch.zeros((), dtype=torch.bool))
            self.register_buffer('row_target_seeded', torch.zeros((), dtype=torch.bool))
            self.register_buffer('inventory_ema_seeded', torch.zeros((), dtype=torch.bool))
            nan = torch.tensor(float('nan'))
            self.register_buffer('inventory_entropy_ema', nan.clone())
            self.register_buffer('inventory_entropy_ema_var', torch.zeros(()))
            self.register_buffer('last_inventory_entropy', nan.clone())
            self.register_buffer('last_row_entropy', nan.clone())
            self.register_buffer('last_usage_entropy', nan.clone())
            self.register_buffer('usage_entropy_target', nan.clone())
            self.register_buffer('row_entropy_target', nan.clone())
            self.register_buffer('last_host_nll', nan.clone())
            self.register_buffer('host_nll_star', nan.clone())
            self.register_buffer('host_nll_slack', torch.zeros(()))
            self.register_buffer('host_nll_star_seeded', torch.zeros((), dtype=torch.bool))
            self.register_buffer('host_nll_seed_count', torch.zeros((), dtype=torch.long))
            self.register_buffer('host_nll_ema', nan.clone())
            self.register_buffer('host_nll_ema_var', torch.zeros(()))
            self.register_buffer('host_nll_ema_seeded', torch.zeros((), dtype=torch.bool))
            self.register_buffer('last_graph_reliance_delta', nan.clone())

        init_steer_state()
        self.kendall_s_mlm = kendall_log(1.0, mlm=True)
        self.kendall_s_div = kendall_log(float(self.config.kendall.beta_div), mlm=False)
        self.kendall_s_dea = kendall_log(1.0, mlm=True)
        self.host_nll_star_metric = MeanMetric()
        self._dual_clip_inv = False
        self._dual_clip_use = False
        self._dual_clip_h = False
        self.host_lambda_clipped = False

    @property
    def log_lambda_inv(self) -> Tensor:
        """Log-lambda for the inventory entropy upper bound."""
        return cast(Tensor, self.duals.log_lambda['inv'])

    @property
    def log_lambda_use(self) -> Tensor:
        """Log-lambda for the usage-entropy lower bound."""
        return cast(Tensor, self.duals.log_lambda['use'])

    @property
    def log_lambda_h(self) -> Tensor:
        """Log-lambda for the row-entropy upper bound."""
        return cast(Tensor, self.duals.log_lambda['h'])

    @property
    def log_lambda_host(self) -> Tensor:
        """Log-lambda for the host-NLL ceiling."""
        return cast(Tensor, self.duals.log_lambda['host'])

    def bind_mask_collator(self, collator: SoftMlmCollator) -> None:
        """Attach the shared MLM collator so rho ramps update mask mixing."""
        self._mask_collator = collator
        collator.set_rho(self._rho)
        embed = self.model.host.get_input_embeddings()
        collator.bind_assignment_source(self.model.soft_vocab, embed)
        collator.assignment_top_k = int(self.config.mlm.assignment_top_k)
        collator.cpc_aux_weight = float(self.config.mlm.cpc_aux_weight)
        collator.cpc_boost = float(self.config.mlm.cpc_boost)

    def forward(self, batch: SoftMlmBatch, living: Tensor | None = None) -> SoftTrunkOutput:
        """Delegate to the soft trunk with numbered-claim demand and living pairs."""
        live = self.model.overlay_living(living)
        return self.model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
            batch.texts,
            claim_texts=batch.claim_texts,
            living=live,
        )

    def on_train_start(self) -> None:
        """Log one-shot LoRA / trunk trainable parameter counts."""
        if not bool(self.config.log.log_trainable_params):
            return
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        self.log('trainable_params', float(trainable), on_step=False, on_epoch=True)
        self.log('total_params', float(total), on_step=False, on_epoch=True)
        host = self.model.host
        if isinstance(host, PeftModel):
            peft_trainable, peft_all = host.get_nb_trainable_parameters()
            self.log('peft_trainable_params', float(peft_trainable), on_step=False, on_epoch=True)
            self.log('peft_all_params', float(peft_all), on_step=False, on_epoch=True)
            pct = 100.0 * float(peft_trainable) / float(max(peft_all, 1))
            self.log('peft_trainable_pct', pct, on_step=False, on_epoch=True)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:  # noqa: C901
        """Steer rho, inject scale, entropy scale, duals, inventory target, and Kendall clips."""
        del batch, batch_idx

        def safety_rho_and_entropy_scale() -> tuple[float, bool]:
            """Ramp rho and entropy scale; freeze rho when occupancy or entropy is stuck."""
            entropy_warmup = max(1, self.config.resolved_entropy_warmup_steps())
            scale = min(1.0, float(self.global_step) / float(entropy_warmup))
            warmup = max(1, int(self.config.mlm.rho_warmup_steps))
            frac = min(1.0, float(self.global_step) / float(warmup))
            scheduled = float(self.config.mlm.rho_max) * frac
            stuck = self._entropy_uniform_streak >= int(self.config.dual.entropy_pause_steps)
            scale = 0.0 if stuck else scale
            if self._last_occupancy < float(self.config.bank.utilization_min) or stuck:
                self._rho = min(self._rho_ceiling, scheduled)
                return scale, True
            self._rho = scheduled
            self._rho_ceiling = scheduled
            return scale, False

        def steer_entropy_duals() -> None:
            """Adam ascent on each log-lambda from the last detached primal residual."""
            if not torch.isfinite(self.last_inventory_entropy):
                return
            usage_bound = float(self.usage_entropy_target)
            row_bound = float(self.row_entropy_target)
            residuals = {
                'inv': float(self.last_inventory_entropy) - float(self.inventory_entropy_target),
                'use': (
                    usage_bound - float(self.last_usage_entropy)
                    if bool(self.usage_target_seeded.item()) and math.isfinite(usage_bound)
                    else 0.0
                ),
                'h': (
                    float(self.last_row_entropy) - row_bound
                    if bool(self.row_target_seeded.item()) and math.isfinite(row_bound)
                    else 0.0
                ),
                'host': (
                    float(self.last_host_nll)
                    - float(self.host_nll_star)
                    - float(self.host_nll_slack)
                    if bool(self.host_nll_star_seeded.item()) and torch.isfinite(self.last_host_nll)
                    else 0.0
                ),
            }
            with torch.enable_grad():
                clipped = self.duals.step(residuals)
            self._dual_clip_inv = clipped['inv']
            self._dual_clip_use = clipped['use']
            self._dual_clip_h = clipped['h']
            self.host_lambda_clipped = clipped['host']

        def update_inventory_ema(live: float) -> None:
            """Seed or Finch-update the TES-SAC exponential window of inventory entropy."""
            self._finch_fill(
                self.inventory_entropy_ema,
                self.inventory_entropy_ema_var,
                self.inventory_ema_seeded,
                live,
            )

        def ratchet_inventory_target() -> None:
            """Drop the live inventory target after a TES-SAC lock window."""
            if not torch.isfinite(self.last_inventory_entropy):
                return
            live = float(self.last_inventory_entropy)
            update_inventory_ema(live)
            spec = self.config.tes_sac
            snapped = tes_sac_snap_undershoot(
                mean=float(self.inventory_entropy_ema),
                target=float(self.inventory_entropy_target),
                spec=spec,
            )
            _ = self.inventory_entropy_target.fill_(snapped)
            locked = tes_sac_inventory_locked(
                mean=float(self.inventory_entropy_ema),
                var=float(self.inventory_entropy_ema_var),
                target=snapped,
                spec=spec,
            )
            count = int(self.inventory_track_count.item()) + 1 if locked else 0
            _ = self.inventory_track_count.fill_(count)
            if count < int(spec.lock_steps):
                return
            nxt = tes_sac_ratchet(target=snapped, spec=spec)
            _ = self.inventory_entropy_target.fill_(nxt)
            _ = self.inventory_track_count.zero_()

        def clamp_kendall_logs() -> None:
            """Keep Kendall log-sigma inside the declared box so a head cannot vanish."""
            lo = float(self.config.kendall.kendall_s_min)
            hi = float(self.config.kendall.kendall_s_max)
            for parameter in (
                self.kendall_s_mlm,
                self.kendall_s_div,
                self.kendall_s_dea,
            ):
                _ = parameter.data.clamp_(lo, hi)

        def log_steer_scalars(entropy_scale: float, *, paused: bool) -> None:
            """Log duals, clips, Kendall weights, and rho safety from the train hook."""
            if self._trainer is None:
                return
            self.log('rho', self._rho, on_step=True, on_epoch=False, prog_bar=False)
            self.log(
                'inject_scale',
                self.model.inject_scale,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log('rho_paused', float(paused), on_step=True, on_epoch=False, prog_bar=False)
            self.log(
                'lambda_entropy_scale',
                entropy_scale,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log('log_lambda_inv', self.log_lambda_inv, on_step=True, on_epoch=False)
            self.log('log_lambda_use', self.log_lambda_use, on_step=True, on_epoch=False)
            self.log('log_lambda_h', self.log_lambda_h, on_step=True, on_epoch=False)
            self.log('log_lambda_host', self.log_lambda_host, on_step=True, on_epoch=False)
            self.log('lambda_inv', self.log_lambda_inv.exp(), on_step=True, on_epoch=False)
            self.log('lambda_use', self.log_lambda_use.exp(), on_step=True, on_epoch=False)
            self.log('lambda_h', self.log_lambda_h.exp(), on_step=True, on_epoch=False)
            self.log('lambda_host', self.log_lambda_host.exp(), on_step=True, on_epoch=False)
            self.log(
                'inventory_entropy_target',
                self.inventory_entropy_target,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'usage_entropy_target',
                self.usage_entropy_target,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'row_entropy_target',
                self.row_entropy_target,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'inventory_entropy_ema',
                self.inventory_entropy_ema,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'inventory_entropy_ema_std',
                math.sqrt(max(float(self.inventory_entropy_ema_var), 0.0)),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'log_lambda_inv_clipped',
                float(self._dual_clip_inv),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'log_lambda_use_clipped',
                float(self._dual_clip_use),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'log_lambda_h_clipped',
                float(self._dual_clip_h),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'log_lambda_host_clipped',
                float(self.host_lambda_clipped),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'graph_reliance_delta',
                self.last_graph_reliance_delta,
                on_step=True,
                on_epoch=False,
            )
            self.log('host_nll_star', self.host_nll_star, on_step=True, on_epoch=False)
            self.log('host_nll_slack', self.host_nll_slack, on_step=True, on_epoch=False)
            self.log('host_nll_ema', self.host_nll_ema, on_step=True, on_epoch=False)
            self.log(
                'kendall_w_mlm',
                self.kendall_s_mlm.detach().neg().exp(),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'kendall_w_div',
                self.kendall_s_div.detach().neg().exp(),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                'kendall_w_dea',
                self.kendall_s_dea.detach().neg().exp(),
                on_step=True,
                on_epoch=False,
            )

        def scheduled_inject_scale() -> float:
            """Linear ramp of the hidden-slot residual scale from 0 to the ceiling."""
            warmup = max(1, int(self.config.arch.inject_ramp_steps))
            frac = min(1.0, float(self.global_step) / float(warmup))
            return float(self.config.arch.inject_ramp_end) * frac

        entropy_scale, paused = safety_rho_and_entropy_scale()
        self.model.set_inject_scale(scheduled_inject_scale())
        self.model.soft_vocab.set_entropy_scale(entropy_scale)
        if self._mask_collator is not None:
            self._mask_collator.set_rho(self._rho)
        steer_entropy_duals()
        ratchet_inventory_target()
        clamp_kendall_logs()
        log_steer_scalars(entropy_scale, paused=paused)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Log total gradient p-norm via Lightning's grad_norm helper."""
        del optimizer
        if not bool(self.config.log.log_grad_norm) or self._trainer is None:
            return
        norm_type = float(self.config.log.grad_norm_type)
        norms = grad_norm(self, norm_type=norm_type)
        total_key = f'grad_{norm_type}_norm_total'
        total = norms.get(total_key)
        if total is None:
            return
        self.log('grad_norm', total, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch: SoftMlmBatch, batch_idx: int) -> Tensor:  # noqa: C901
        """Compute joint trunk loss and update train metrics."""
        del batch_idx

        def snapshot_host_nll_ema() -> None:
            """After seed, copy the live MeanMetric NLL onto the HostRotStop buffer."""
            if not bool(self.host_nll_star_seeded.item()):
                return
            live_mean = self.train_metrics['mlm_nll'].compute()
            if torch.isfinite(live_mean):
                _ = self.host_nll_ema.fill_(float(live_mean))

        def record_primal_entropies(out: SoftTrunkOutput) -> None:
            """Snapshot detached entropies; seed inventory, usage, and row bounds once."""
            inventory = float(out.inventory_entropy.detach())
            row = float(out.mean_row_entropy.detach())
            usage = out.soft_vocab.batch_usage.detach().clamp_min(0)
            mass = usage.sum().clamp_min(1e-12)
            probs = usage / mass
            usage_h = float(-(probs * probs.clamp_min(1e-12).log()).sum())
            _ = self.last_inventory_entropy.fill_(inventory)
            _ = self.last_row_entropy.fill_(row)
            _ = self.last_usage_entropy.fill_(usage_h)
            if not bool(self.inventory_target_seeded.item()):
                _ = self.inventory_entropy_target.fill_(inventory)
                _ = self.inventory_target_seeded.fill_(True)
            if not bool(self.usage_target_seeded.item()):
                _ = self.usage_entropy_target.fill_(usage_h)
                _ = self.usage_target_seeded.fill_(True)
            if not bool(self.row_target_seeded.item()):
                _ = self.row_entropy_target.fill_(row)
                _ = self.row_target_seeded.fill_(True)

        living = self.model.living_snapshot()
        out = self.forward(batch, living=living)
        align = self._align_loss(out.z_d, out.z_g)
        task_loss, cons_loss = self._task_and_constraint_losses(out)
        loss: Tensor
        scale: Tensor
        loss, scale = cheng_rescale(task_loss, cons_loss, self._constraint_lambdas())

        def leftover_unpaid(demand: Tensor, supply: Tensor) -> Tensor:
            return self.covering.unpaid_fraction(self.covering.pair_table(demand, supply))

        def late_inventory(hidden: Tensor) -> CoveringInventory:
            claim = self.inventory.claim_mask(
                self.model.host_tokenizer(),
                batch.texts,
                batch.claim_texts,
                batch.attention_mask,
                max_length=int(self.config.arch.max_length),
            )
            return self.inventory(
                hidden,
                batch.attention_mask,
                claim,
                model=self.model,
                texts=batch.texts,
                input_ids=batch.unmasked_input_ids,
                living=living,
                claim_texts=batch.claim_texts,
            )

        dest_weight = float(self.config.dest_comparison.weight)
        shape_weight = float(self.config.overlay_shape.shape_weight)
        if dest_weight > 0.0 or shape_weight > 0.0:
            matching = late_inventory(out.text_hidden)
            if dest_weight > 0.0:
                demand = matching.n_entity_claim
                supply = matching.n_entity_full
                matching_unpaid = leftover_unpaid(demand, supply)
                dest_shift = int(self.config.dest_comparison.dest_shift)
                shuffled_unpaid = leftover_unpaid(demand, supply.roll(dest_shift, dims=-1))
                dest_loss = torch.nanmean(
                    F.softplus(
                        matching_unpaid
                        - shuffled_unpaid
                        + float(self.config.dest_comparison.margin)
                    )
                )
                loss += dest_weight * dest_loss
                self.log('dest_loss', dest_loss, on_step=True, on_epoch=False)
                self.log(
                    'dest_unpaid_gap',
                    torch.nanmean(shuffled_unpaid - matching_unpaid).detach(),
                    on_step=True,
                    on_epoch=False,
                )
            if shape_weight > 0.0 and matching.full_labeled is not None:
                pair_mass = (
                    matching.full_labeled
                    if matching.full_labeled.ndim == 3
                    else matching.full_labeled.sum(dim=-1)
                )
                row = pair_mass.sum(dim=-1)
                col = pair_mass.sum(dim=-2)
                occupy = (matching.n_entity_full - row - col).clamp_min(0)
                occupied = occupy > float(self.inventory.occupied_floor.item())
                columns = int(self.config.overlay_shape.community_columns)
                slots = pair_mass.size(-1)
                labels = (torch.arange(slots, device=pair_mass.device) * columns) // slots
                assignment = F.softmax(
                    pair_mass @ F.one_hot(labels, columns).to(dtype=pair_mass.dtype),
                    dim=-1,
                )
                shape_loss = torch.nanmean(
                    shape_family(
                        pair_mass,
                        assignment,
                        occupied,
                        collapse_weight=float(self.config.overlay_shape.shape_collapse_weight),
                    ).total
                )
                loss += shape_weight * shape_loss
                self.log('shape_loss', shape_loss, on_step=True, on_epoch=False)
            if matching.full_labeled is not None:
                labeled = matching.full_labeled
                self.model.absorb_living(labeled if labeled.ndim == 3 else labeled.sum(dim=-1))

        record_primal_entropies(out)
        self._observe_host_nll(float(out.mlm_loss.detach()))
        if dest_weight <= 0.0:
            self._probe_graph_reliance(batch)
        self._last_occupancy = float(out.occupancy.detach().item())
        max_entropy = math.log(int(self.config.arch.entity_bank_size))
        entropy_ceiling = float(self.config.dual.entropy_pause_ratio) * max_entropy
        if float(out.mean_row_entropy.detach()) > entropy_ceiling:
            self._entropy_uniform_streak += 1
        else:
            self._entropy_uniform_streak = 0
        self._enqueue_graph_embeddings(out.z_g.detach())
        self._update_metrics(out, align, self.train_metrics, self.train_accuracy)
        snapshot_host_nll_ema()
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('cheng_z', scale.detach(), on_step=True, on_epoch=False)
        self.log('mlm_nll', out.mlm_loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
        task_div = out.diversity_terms.vq + out.diversity_terms.rel_vq
        self.log('div_loss', task_div.detach(), on_step=True, on_epoch=False)
        self.log('cons_loss', cons_loss.detach(), on_step=True, on_epoch=False)
        self.log('ke_loss', out.ke_loss.detach(), on_step=True, on_epoch=False)
        self.log('soft_ke_loss', out.soft_ke_loss.detach(), on_step=True, on_epoch=False)
        self.log('align_loss', align.detach(), on_step=True, on_epoch=False)
        self.log(
            'occupancy',
            out.occupancy.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            'relation_occupancy',
            out.relation_occupancy.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            'n_dead_before_restart',
            out.n_dead_before_restart.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log('row_entropy', out.mean_row_entropy.detach(), on_step=True, on_epoch=False)
        self.log(
            'inventory_entropy',
            out.inventory_entropy.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'n_restarts',
            out.soft_vocab.n_restarts.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'n_entity_restarts',
            out.soft_vocab.n_entity_restarts.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'n_relation_restarts',
            out.soft_vocab.n_relation_restarts.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'usage_entropy',
            out.soft_vocab.usage_entropy.detach(),
            on_step=True,
            on_epoch=False,
        )
        usage_target = (
            float(self.usage_entropy_target)
            if bool(self.usage_target_seeded.item())
            and math.isfinite(float(self.usage_entropy_target))
            else float(self.config.resolved_usage_entropy_target())
        )
        host_ceiling = (
            float(self.host_nll_star)
            if bool(self.host_nll_star_seeded.item()) and math.isfinite(float(self.host_nll_star))
            else None
        )
        host_nll = (
            float(self.host_nll_ema)
            if bool(self.host_nll_ema_seeded.item()) and math.isfinite(float(self.host_nll_ema))
            else float(out.mlm_loss.detach())
        )
        covering = CoveringTrainReading(
            inventory_entropy=float(out.inventory_entropy.detach()),
            ln_k=math.log(int(self.config.arch.entity_bank_size)),
            usage_entropy=float(out.soft_vocab.usage_entropy.detach()),
            usage_target=usage_target,
            row_entropy=float(out.mean_row_entropy.detach()),
            host_nll=host_nll,
            host_nll_ceiling=host_ceiling,
            dea_gap=float(out.dea_gap.detach()),
        )
        for name, value in covering.log_scalars().items():
            self.log(
                name,
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=name == 'inventory_below_ln_k',
            )
        self.log(
            'assignment_perplexity',
            torch.exp(out.mean_row_entropy.detach()),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            'inverse_simpson',
            out.soft_vocab.inverse_simpson.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'perplexity',
            torch.exp(out.mlm_loss.detach()),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log('dea_gap', out.dea_gap.detach(), on_step=True, on_epoch=False)
        self.log('dea_loss', out.dea_loss.detach(), on_step=True, on_epoch=False)
        if bool(self.config.log.log_mlm_accuracy):
            self.log(
                'mlm_accuracy',
                self.train_accuracy['mlm_accuracy'].compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        if bool(self.config.log.log_learning_rate):
            self._log_learning_rate()
        self._log_compose_scores(out.compose_scores)
        return loss

    def on_train_epoch_end(self) -> None:
        """Epoch perplexity is exp of mean NLL, not mean of batch perplexities."""
        self.log('perplexity', torch.exp(self.train_metrics['mlm_nll'].compute()), prog_bar=True)

    def validation_step(self, batch: SoftMlmBatch, batch_idx: int) -> Tensor:
        """Same joint objective for held-out NLL and representation health."""
        del batch_idx
        out = self.forward(batch)
        align = self._align_loss(out.z_d, out.z_g)
        task_loss, cons_loss = self._task_and_constraint_losses(out)
        loss: Tensor
        loss, _scale = cheng_rescale(task_loss, cons_loss, self._constraint_lambdas())
        self._update_metrics(out, align, self.val_metrics, self.val_accuracy)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        log_compose = bool(self.config.log.log_compose_ranking)
        self.log_dict(
            {
                name: metric
                for name, metric in self.val_metrics.items(keep_base=False, copy_state=False)
                if log_compose or not str(name).startswith('val_compose_')
            },
            on_step=False,
            on_epoch=True,
        )
        if bool(self.config.log.log_mlm_accuracy):
            self.log_dict(
                dict(self.val_accuracy.items(keep_base=False, copy_state=False)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Epoch val perplexity is exp of mean val NLL."""
        self.log('val_perplexity', torch.exp(self.val_metrics['mlm_nll'].compute()))

    def configure_optimizers(self) -> SsvOptimizerLRConfig:
        """AdamW plus linear warmup, then constant peak LR on trainer steps."""
        trunk = [param for name, param in self.named_parameters() if not name.startswith('duals.')]
        optimizer = AdamW(trunk, lr=float(self.config.fit.learning_rate))
        scheduler = get_scheduler(
            name='constant_with_warmup',
            optimizer=optimizer,
            num_warmup_steps=self.config.resolved_lr_warmup_steps(),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def _task_and_constraint_losses(
        self,
        out: SoftTrunkOutput,
    ) -> tuple[Tensor, Tensor]:
        """Kendall on mlm/diversity/dEA; dual-scale entropy and host pieces separately."""
        terms = out.diversity_terms
        task_div = terms.vq + terms.rel_vq
        task_loss = self._kendall_objective(out.mlm_loss, task_div, out.dea_loss)
        lams = self.duals.lambdas()
        cons_loss = (
            lams['inv'] * terms.inventory + lams['use'] * terms.usage_kl + lams['h'] * terms.gap
        )
        if bool(self.host_nll_star_seeded.item()):
            cons_loss += lams['host'] * (out.mlm_loss - self.host_nll_star - self.host_nll_slack)
        if bool(self.config.kendall.include_ke):
            task_loss += out.ke_loss
        return task_loss, cons_loss

    def _constraint_lambdas(self) -> Tensor:
        """Positive dual weights stacked in Cheng sum order."""
        lams = self.duals.lambdas()
        return torch.stack((lams['inv'], lams['use'], lams['h'], lams['host']))

    def _finch_fill(
        self,
        mean_buf: Tensor,
        var_buf: Tensor,
        seeded_buf: Tensor,
        live: float,
    ) -> None:
        """Seed or increment a Finch window stored on Lightning buffers."""
        if not math.isfinite(live):
            return
        if not bool(seeded_buf.item()):
            _ = mean_buf.fill_(live)
            _ = var_buf.zero_()
            _ = seeded_buf.fill_(True)
            return
        mean, var = finch_update(
            mean=float(mean_buf),
            var=float(var_buf),
            live=live,
            decay=float(self.config.tes_sac.ema_decay),
        )
        _ = mean_buf.fill_(mean)
        _ = var_buf.fill_(var)

    def _observe_host_nll(self, nll: float) -> None:
        """Record live NLL; seed L* from MeanMetric and slack from Finch std."""
        if not math.isfinite(nll):
            return
        _ = self.last_host_nll.fill_(nll)
        if bool(self.host_nll_star_seeded.item()):
            return
        self.host_nll_star_metric.update(torch.tensor(nll))
        self._finch_fill(
            self.host_nll_ema,
            self.host_nll_ema_var,
            self.host_nll_ema_seeded,
            nll,
        )
        count = int(self.host_nll_seed_count.item()) + 1
        _ = self.host_nll_seed_count.fill_(count)
        if count < int(self.config.tes_sac.lock_steps):
            return
        star = float(self.host_nll_star_metric.compute())
        slack = math.sqrt(max(float(self.host_nll_ema_var), 0.0))
        if not math.isfinite(star):
            return
        _ = self.host_nll_star.fill_(star)
        _ = self.host_nll_slack.fill_(slack)
        _ = self.host_nll_star_seeded.fill_(True)

    def _probe_graph_reliance(self, batch: SoftMlmBatch) -> None:
        """Measure host-only-vs-trunk NLL delta in eval mode, once every probe interval.

        Both forwards run in eval mode (real prefix vs zeroed prefix). Positive
        delta means the graph prefix lowers MLM NLL. The value is a wiring check;
        it does not enter dual ascent.
        """
        interval = int(self.config.graph_probe.interval_steps)
        if self.global_step % interval != 0:
            return

        delta = measure_graph_prefix_delta(self.model, batch).delta
        if not math.isfinite(delta):
            return
        _ = self.last_graph_reliance_delta.fill_(delta)

    def _kendall_objective(self, mlm: Tensor, diversity: Tensor, dea: Tensor) -> Tensor:
        """Homoscedastic mix of the mlm, diversity, and entity-denoise heads."""
        heads = (
            (self.kendall_s_mlm, mlm),
            (self.kendall_s_div, diversity),
            (self.kendall_s_dea, dea),
        )
        return cast(Tensor, sum(s.neg().exp() * loss + s for s, loss in heads))

    def transfer_batch_to_device(
        self,
        batch: SoftMlmBatch,
        device: torch.device,
        dataloader_idx: int = 0,
    ) -> SoftMlmBatch:
        """Move MLM tensors and HeteroData graphs onto the trainer device."""
        del dataloader_idx
        if batch.term_spans:
            self.model.graph_ingress.remember_inference_spans(
                dict(zip(batch.texts, batch.term_spans, strict=True))
            )
        delta = batch.termhood_delta
        if delta is not None and isinstance(self.model.graph_ingress.termhood, TermhoodTable):
            self.model.graph_ingress.termhood = self.model.graph_ingress.termhood.merged(delta)
        return SoftMlmBatch(
            input_ids=batch.input_ids.to(device, non_blocking=True),
            unmasked_input_ids=batch.unmasked_input_ids.to(device, non_blocking=True),
            attention_mask=batch.attention_mask.to(device, non_blocking=True),
            labels=batch.labels.to(device, non_blocking=True),
            graphs=tuple(graph.to(device) for graph in batch.graphs),
            texts=batch.texts,
            claim_texts=batch.claim_texts,
            disclosure_texts=batch.disclosure_texts,
            term_spans=batch.term_spans,
            termhood_delta=batch.termhood_delta,
            rho=batch.rho,
            require_supervised=batch.require_supervised,
        )

    def _log_compose_scores(self, scores: ComposeScoreReport) -> None:
        """On-step compose CSV scalars for the train split."""
        if not bool(self.config.log.log_compose_ranking) or self._trainer is None:
            return
        self.log(
            'compose_transe',
            scores.compose_transe.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'compose_mrr',
            scores.compose_mrr.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            'compose_mr',
            scores.compose_mr.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'compose_hits_at_1',
            scores.compose_hits_at_1.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'compose_hits_at_3',
            scores.compose_hits_at_3.detach(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            'compose_hits_at_10',
            scores.compose_hits_at_10.detach(),
            on_step=True,
            on_epoch=False,
        )

    def _log_learning_rate(self) -> None:
        if self._trainer is None:
            return
        opt = self.optimizers()
        first = opt[0] if isinstance(opt, (list, tuple)) else opt
        lr = float(first.param_groups[0]['lr'])
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=False)

    def _align_loss(self, z_d: Tensor, z_g: Tensor) -> Tensor:
        """InfoNCE between patent and graph embeddings with gathered + queue negatives."""
        temperature = float(self.config.arch.align_temperature)
        batch_size = int(z_d.size(0))
        anchor = F.normalize(z_d, dim=-1)
        if self._trainer is None:
            keys_live = z_g
            targets = torch.arange(batch_size, device=z_d.device)
        else:
            gathered = cast(Tensor, self.all_gather(z_g, sync_grads=False))
            if gathered.dim() == z_g.dim():
                keys_live = gathered
                targets = torch.arange(batch_size, device=z_d.device)
            else:
                world = int(gathered.size(0))
                keys_live = gathered.reshape(world * batch_size, z_g.size(-1))
                rank_offset = int(self.global_rank) * batch_size
                targets = torch.arange(batch_size, device=z_d.device) + rank_offset
        keys_live = F.normalize(keys_live, dim=-1)
        queue = self._queue_tensor()
        if queue.numel() == 0:
            keys = keys_live
        else:
            keys = torch.cat([keys_live, F.normalize(queue, dim=-1)], dim=0)
        logits = anchor @ keys.transpose(0, 1) / temperature
        return F.cross_entropy(logits, targets)

    def _queue_tensor(self) -> Tensor:
        queue = self._zg_queue
        queue_len = self._zg_queue_len
        length = int(queue_len.item())
        if length <= 0 or int(queue.size(0)) <= 0:
            return queue[:0]
        return queue[:length]

    def _enqueue_graph_embeddings(self, z_g: Tensor) -> None:
        queue = self._zg_queue
        queue_len = self._zg_queue_len
        queue_ptr = self._zg_queue_ptr
        capacity = int(queue.size(0))
        if capacity <= 0:
            return
        for row in z_g.detach().reshape(-1, z_g.size(-1)):
            ptr = int(queue_ptr.item())
            _ = queue[ptr].copy_(row)
            _ = queue_ptr.fill_((ptr + 1) % capacity)
            _ = queue_len.fill_(min(capacity, int(queue_len.item()) + 1))

    def _update_metrics(
        self,
        out: SoftTrunkOutput,
        align: Tensor,
        means: MetricCollection,
        accuracy: MetricCollection,
    ) -> None:
        """Fold detached trunk scalars into one MeanMetric collection."""
        scores = out.compose_scores
        values = {
            'mlm_nll': out.mlm_loss.detach(),
            'div_loss': (out.diversity_terms.vq + out.diversity_terms.rel_vq).detach(),
            'ke_loss': out.ke_loss.detach(),
            'soft_ke_loss': out.soft_ke_loss.detach(),
            'align_loss': align.detach(),
            'occupancy': out.occupancy.detach(),
            'relation_occupancy': out.relation_occupancy.detach(),
            'n_dead_before_restart': out.n_dead_before_restart.detach(),
            'usage_entropy': out.soft_vocab.usage_entropy.detach(),
            'inverse_simpson': out.soft_vocab.inverse_simpson.detach(),
            'assignment_perplexity': torch.exp(out.mean_row_entropy.detach()),
            'row_entropy': out.mean_row_entropy.detach(),
            'compose_transe': scores.compose_transe.detach(),
            'compose_mrr': scores.compose_mrr.detach(),
            'compose_mr': scores.compose_mr.detach(),
            'compose_hits_at_1': scores.compose_hits_at_1.detach(),
            'compose_hits_at_3': scores.compose_hits_at_3.detach(),
            'compose_hits_at_10': scores.compose_hits_at_10.detach(),
            'dea_gap': out.dea_gap.detach(),
            'dea_loss': out.dea_loss.detach(),
        }
        for name, metric in means.items(keep_base=True, copy_state=False):
            metric.update(values[str(name)])
        if bool(self.config.log.log_mlm_accuracy):
            accuracy['mlm_accuracy'].update(
                out.mlm_preds.detach().reshape(-1),
                out.mlm_label_ids.detach().reshape(-1),
            )


__all__ = [
    'SsvLightningModule',
]
