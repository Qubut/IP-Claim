"""Lightning ``train_func`` for SSV (local multi-GPU or Ray Train worker)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import structlog
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from returns.result import Failure, Success
from torch.utils.data import DataLoader

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ssv.config import SsvTrainConfig, TesSacSpec
from ip_claim.ssv.dataset import LazyHupdMlmDataset
from ip_claim.ssv.load_weights import resolve_fit_ckpt
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.steer.tes_sac import tes_sac_inventory_locked
from ip_claim.ssv.trainer_strategy import resolve_lightning_strategy

_TARGET_DROP_ATOL = 1e-6
_log = structlog.get_logger(__name__)


class InventoryTrackStop(Callback):
    """End Lightning fit when the inventory target stalls and EMA locks to that floor."""

    def __init__(
        self,
        *,
        tes_sac: TesSacSpec,
        patience: int,
    ) -> None:
        super().__init__()
        self.tes_sac = tes_sac
        self.patience = int(patience)
        self.in_band_streak = 0
        self._last_target: float | None = None
        self._empirical_floor: float | None = None

    @classmethod
    def from_config(cls, job: SsvTrainConfig) -> InventoryTrackStop:
        """Build from the train job's TES-SAC spec and early-stop patience."""
        return cls(
            tes_sac=job.tes_sac,
            patience=int(job.fit.early_stop_patience),
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Count consecutive stalls at the last locked target; stop when patience is met."""
        del outputs, batch, batch_idx
        module = cast(SsvLightningModule, pl_module)
        target = float(module.inventory_entropy_target)
        ema = float(module.inventory_entropy_ema)
        var = float(module.inventory_entropy_ema_var)
        locked_now = tes_sac_inventory_locked(
            mean=ema,
            var=var,
            target=target,
            spec=self.tes_sac,
        )
        if locked_now:
            self._empirical_floor = target
        dropped = (
            self._last_target is not None
            and math.isfinite(target)
            and math.isfinite(self._last_target)
            and target < self._last_target - _TARGET_DROP_ATOL
        )
        self._last_target = target
        stalled_at = self._empirical_floor
        locked_to_stall = stalled_at is not None and tes_sac_inventory_locked(
            mean=ema,
            var=var,
            target=stalled_at,
            spec=self.tes_sac,
        )
        tracking = (not dropped) and locked_to_stall
        self.in_band_streak = self.in_band_streak + 1 if tracking else 0
        if self.in_band_streak >= self.patience:
            trainer.should_stop = True


class HostRotStop(Callback):
    """End Lightning fit when host NLL stays above the seeded ceiling and λ_host is clipped."""

    def __init__(self, *, patience: int) -> None:
        super().__init__()
        self.patience = int(patience)
        self.high_streak = 0

    @classmethod
    def from_config(cls, job: SsvTrainConfig) -> HostRotStop:
        """Build from the train job's early-stop patience."""
        return cls(patience=int(job.fit.early_stop_patience))

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Count consecutive clipped host-NLL breaches; set should_stop when patience is met."""
        del outputs, batch, batch_idx
        module = cast(SsvLightningModule, pl_module)
        seeded = bool(module.host_nll_star_seeded.item())
        ema = float(module.host_nll_ema)
        ceiling = float(module.host_nll_star) + float(module.host_nll_slack)
        rotting = (
            seeded
            and math.isfinite(ema)
            and math.isfinite(ceiling)
            and ema > ceiling
            and bool(module.host_lambda_clipped)
        )
        self.high_streak = self.high_streak + 1 if rotting else 0
        if self.high_streak >= self.patience:
            trainer.should_stop = True


def _build_trainer(job: SsvTrainConfig, checkpoint_dir: Path) -> pl.Trainer:
    """Declare Lightning Trainer wiring (CSV logger, progress, Ray or local strategy)."""
    use_csv_logger = bool(job.fit.enable_csv_logger or job.runtime.publish_to_hub)
    logger: bool | CSVLogger = (
        CSVLogger(save_dir=str(checkpoint_dir), name='lightning') if use_csv_logger else False
    )
    step_ckpt = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='ssv-step{step:06d}',
        auto_insert_metric_name=False,
        every_n_train_steps=int(job.fit.checkpoint_every_n_steps),
        save_top_k=int(job.fit.checkpoint_save_top_k),
        save_last=True,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )
    track_stop = InventoryTrackStop.from_config(job)
    host_stop = HostRotStop.from_config(job)
    trainer_kwargs: dict[str, Any] = {
        'max_steps': int(job.fit.max_steps),
        'accumulate_grad_batches': int(job.fit.accumulate_grad_batches),
        'enable_checkpointing': True,
        'logger': logger,
        'enable_progress_bar': bool(job.fit.enable_progress_bar),
        'log_every_n_steps': int(job.fit.log_every_n_steps),
        'gradient_clip_val': 1.0,
    }
    if job.runtime.ray:
        ray_trainer = pl.Trainer(
            accelerator='gpu' if job.runtime.use_gpu else 'cpu',
            devices='auto',
            strategy=RayDDPStrategy(),
            plugins=[RayLightningEnvironment()],
            callbacks=[RayTrainReportCallback(), step_ckpt, track_stop, host_stop],
            **trainer_kwargs,
        )
        return cast(pl.Trainer, prepare_trainer(ray_trainer))
    devices: int | str = 1
    if job.runtime.use_gpu:
        devices = int(job.fit.num_devices) if job.fit.num_devices is not None else 'auto'
    return pl.Trainer(
        accelerator='gpu' if job.runtime.use_gpu else 'cpu',
        devices=devices,
        strategy=resolve_lightning_strategy(job),
        callbacks=[step_ckpt, track_stop, host_stop],
        **trainer_kwargs,
    )


def train_func(config: SsvTrainConfig | Mapping[str, Any]) -> Path:
    """Fit ``SsvLightningModule`` and return the written checkpoint path."""
    job = (
        config
        if isinstance(config, SsvTrainConfig)
        else SsvTrainConfig.model_validate(dict(config))
    )
    checkpoint_dir = Path(job.fit.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    container = SsvContainer(config=job)
    module: SsvLightningModule = container.lightning_module()
    collator = container.train_collator()
    module.bind_mask_collator(collator)

    dataset = LazyHupdMlmDataset(
        Path(job.runtime.hupd_dir) if job.runtime.hupd_dir else None,
        limit=job.runtime.hupd_limit,
        index_cache=checkpoint_dir / 'hupd_path_index.txt',
    )
    num_workers = int(job.fit.dataloader_num_workers)
    worker_opts = (
        {
            'persistent_workers': True,
            'prefetch_factor': int(job.fit.dataloader_prefetch_factor),
        }
        if num_workers > 0
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(job.fit.batch_size), max(len(dataset), 1)),
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=bool(job.fit.dataloader_pin_memory),
        **worker_opts,
    )

    trainer = _build_trainer(job, checkpoint_dir)
    init_path = Path(job.runtime.init_weights) if job.runtime.init_weights else None
    resume = resolve_fit_ckpt(
        last_ckpt=checkpoint_dir / 'last.ckpt',
        init_weights=init_path,
    )
    if resume is None and init_path is not None:
        match container.init_weights(path=init_path):
            case Failure(message):
                raise RuntimeError(message)
            case Success():
                _log.info('ssv.train.init_weights', path=str(init_path))
    elif resume is not None:
        _log.info('ssv.train.fit_resume', path=str(resume))
    trainer.fit(module, train_dataloaders=loader, ckpt_path=resume)

    checkpoint_path = checkpoint_dir / 'ssv.ckpt'
    trainer.save_checkpoint(str(checkpoint_path))
    if trainer.is_global_zero:
        job.upload_to_hub(checkpoint_dir)
    return checkpoint_path


__all__ = [
    'HostRotStop',
    'InventoryTrackStop',
    'train_func',
]
