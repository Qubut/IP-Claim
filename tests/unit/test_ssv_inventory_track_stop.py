"""Inventory TES-SAC stall stop: fires when the target stops dropping."""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ip_claim.ssv.config import DualSpec, FitSpec, HostSpec, RuntimeSpec, SsvTrainConfig, TesSacSpec
from ip_claim.ssv.steer.tes_sac import tes_sac_inventory_locked, tes_sac_ratchet
from ip_claim.ssv.train import InventoryTrackStop, _build_trainer
from tests._ssv_fixtures import lightning_trainer_callbacks, lightning_trainer_stub


class _TrackStub(pl.LightningModule):
    """Minimal Lightning module that exposes TES-SAC inventory buffers."""

    weight: nn.Parameter
    last_inventory_entropy: Tensor
    inventory_entropy_target: Tensor
    inventory_entropy_ema: Tensor
    inventory_entropy_ema_var: Tensor
    inventory_ema_seeded: Tensor

    def __init__(self, *, live: float, target: float, var: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.last_inventory_entropy = nn.Buffer(torch.tensor(live))
        self.inventory_entropy_target = nn.Buffer(torch.tensor(target))
        self.inventory_entropy_ema = nn.Buffer(torch.tensor(live))
        self.inventory_entropy_ema_var = nn.Buffer(torch.tensor(var))
        self.inventory_ema_seeded = nn.Buffer(torch.tensor(True))

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        del batch, batch_idx
        return self.weight.pow(2).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def _check(
    stop: InventoryTrackStop,
    trainer: pl.Trainer,
    *,
    live: float,
    target: float,
    var: float = 0.0,
) -> None:
    module = _TrackStub(live=live, target=target, var=var)
    stop.on_train_batch_end(trainer, module, None, None, 0)


def test_tes_sac_lock_requires_mean_band_and_low_std() -> None:
    spec = TesSacSpec(band=0.15, std_max=0.05)
    assert tes_sac_inventory_locked(mean=3.74, var=0.0, target=3.74, spec=spec)
    assert not tes_sac_inventory_locked(mean=3.45, var=0.0, target=3.74, spec=spec)
    assert not tes_sac_inventory_locked(mean=3.74, var=0.04, target=3.74, spec=spec)


def test_inventory_track_stop_has_no_typed_slot_floor() -> None:
    assert 'inventory_slots' not in DualSpec.model_fields
    assert 'floor' not in tes_sac_ratchet.__code__.co_varnames
    stop = InventoryTrackStop(tes_sac=TesSacSpec(band=0.15), patience=3)
    assert not hasattr(stop, 'floor')


def test_inventory_track_stop_fires_when_target_stalled_and_ema_locked() -> None:
    stop = InventoryTrackStop(tes_sac=TesSacSpec(band=0.15), patience=3)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, live=2.0, target=2.0)
    _check(stop, trainer, live=2.0, target=2.0)
    assert trainer.should_stop is False
    _check(stop, trainer, live=2.0, target=2.0)
    assert trainer.should_stop is True


def test_inventory_track_stop_patience_resets_on_noise() -> None:
    stop = InventoryTrackStop(tes_sac=TesSacSpec(band=0.15), patience=3)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, live=2.0, target=2.0)
    _check(stop, trainer, live=2.0, target=2.0)
    _check(stop, trainer, live=3.0, target=2.0)
    assert trainer.should_stop is False
    _check(stop, trainer, live=2.0, target=2.0)
    _check(stop, trainer, live=2.0, target=2.0)
    assert trainer.should_stop is False
    _check(stop, trainer, live=2.0, target=2.0)
    assert trainer.should_stop is True


def test_inventory_track_stop_ignores_high_ema_std() -> None:
    stop = InventoryTrackStop(tes_sac=TesSacSpec(band=0.15), patience=2)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, live=2.0, target=2.0, var=0.04)
    _check(stop, trainer, live=2.0, target=2.0, var=0.04)
    assert trainer.should_stop is False


def test_inventory_track_stop_ignores_live_outside_band() -> None:
    stop = InventoryTrackStop(tes_sac=TesSacSpec(band=0.15), patience=2)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, live=1.6, target=2.0)
    _check(stop, trainer, live=1.6, target=2.0)
    assert trainer.should_stop is False


def test_inventory_track_stop_resets_when_target_still_drops() -> None:
    stop = InventoryTrackStop(tes_sac=TesSacSpec(band=0.15), patience=2)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, live=5.0, target=5.0)
    _check(stop, trainer, live=4.5, target=4.5)
    assert trainer.should_stop is False
    _check(stop, trainer, live=4.0, target=4.0)
    assert trainer.should_stop is False


def test_inventory_track_stop_fires_after_infeasible_ratchet() -> None:
    stop = InventoryTrackStop(tes_sac=TesSacSpec(band=0.15), patience=3)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, live=3.0, target=3.0)
    _check(stop, trainer, live=3.0, target=2.0)
    _check(stop, trainer, live=3.0, target=2.0)
    _check(stop, trainer, live=3.0, target=2.0)
    assert trainer.should_stop is False
    _check(stop, trainer, live=3.0, target=2.0)
    assert trainer.should_stop is True


def test_inventory_track_stop_ends_lightning_fit() -> None:
    module = _TrackStub(live=2.0, target=2.0)
    stop = InventoryTrackStop(tes_sac=TesSacSpec(band=0.15), patience=3)
    trainer = pl.Trainer(
        max_steps=40,
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[stop],
    )
    loader = DataLoader(TensorDataset(torch.zeros(40, 1)), batch_size=1)
    trainer.fit(module, train_dataloaders=loader)
    assert trainer.should_stop
    assert trainer.global_step == 3


def test_build_trainer_attaches_inventory_track_stop(tmp_path: Path) -> None:
    job = SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        fit=FitSpec(
            max_steps=50,
            early_stop_patience=7,
            enable_csv_logger=False,
            checkpoint_dir=str(tmp_path),
        ),
        tes_sac=TesSacSpec(band=0.15),
        runtime=RuntimeSpec(ray=False, use_gpu=False),
    )
    trainer = _build_trainer(job, tmp_path)
    stops = [
        cb for cb in lightning_trainer_callbacks(trainer) if isinstance(cb, InventoryTrackStop)
    ]
    assert len(stops) == 1
    assert stops[0].patience == 7
    assert stops[0].tes_sac.band == pytest.approx(0.15)
    assert stops[0].tes_sac.std_max == pytest.approx(0.05)
    assert not hasattr(stops[0], 'floor')
    assert 'enable_early_stop' not in SsvTrainConfig.model_fields
    assert not hasattr(job, 'resolved_inventory_entropy_floor')
