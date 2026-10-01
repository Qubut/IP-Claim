"""HostRotStop: fires only when NLL EMA is above L*+r and λ_host is clipped."""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torch
from tests._ssv_fixtures import lightning_trainer_callbacks, lightning_trainer_stub
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ip_claim.ssv.config import FitSpec, HostSpec, RuntimeSpec, SsvTrainConfig
from ip_claim.ssv.train import HostRotStop, InventoryTrackStop, _build_trainer


class _HostStub(pl.LightningModule):
    """Minimal Lightning module that exposes host-NLL stop buffers."""

    weight: nn.Parameter
    host_nll_ema: Tensor
    host_nll_star: Tensor
    host_nll_slack: Tensor
    host_nll_star_seeded: Tensor

    def __init__(
        self,
        *,
        ema: float,
        star: float,
        slack: float,
        clipped: bool,
        seeded: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.host_nll_ema = nn.Buffer(torch.tensor(ema))
        self.host_nll_star = nn.Buffer(torch.tensor(star))
        self.host_nll_slack = nn.Buffer(torch.tensor(slack))
        self.host_nll_star_seeded = nn.Buffer(torch.tensor(seeded))
        self.host_lambda_clipped = clipped

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        del batch, batch_idx
        return self.weight.pow(2).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def _check(
    stop: HostRotStop,
    trainer: pl.Trainer,
    *,
    ema: float,
    star: float,
    slack: float,
    clipped: bool,
    seeded: bool = True,
) -> None:
    module = _HostStub(ema=ema, star=star, slack=slack, clipped=clipped, seeded=seeded)
    stop.on_train_batch_end(trainer, module, None, None, 0)


def test_host_rot_stop_fires_when_ema_high_and_lambda_clipped() -> None:
    stop = HostRotStop(patience=3)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=True)
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=True)
    assert trainer.should_stop is False
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=True)
    assert trainer.should_stop is True


def test_host_rot_stop_does_not_fire_when_lambda_not_clipped() -> None:
    stop = HostRotStop(patience=2)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=False)
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=False)
    assert trainer.should_stop is False


def test_host_rot_stop_does_not_fire_when_ema_at_ceiling() -> None:
    stop = HostRotStop(patience=2)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, ema=2.1, star=2.0, slack=0.1, clipped=True)
    _check(stop, trainer, ema=2.1, star=2.0, slack=0.1, clipped=True)
    assert trainer.should_stop is False


def test_host_rot_stop_resets_when_clip_drops() -> None:
    stop = HostRotStop(patience=3)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=True)
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=True)
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=False)
    assert trainer.should_stop is False
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=True)
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=True)
    assert trainer.should_stop is False
    _check(stop, trainer, ema=2.5, star=2.0, slack=0.1, clipped=True)
    assert trainer.should_stop is True


def test_host_rot_stop_ignores_unseeded_ceiling() -> None:
    stop = HostRotStop(patience=1)
    trainer = lightning_trainer_stub(should_stop=False)
    _check(stop, trainer, ema=9.0, star=1.0, slack=0.0, clipped=True, seeded=False)
    assert trainer.should_stop is False


def test_host_rot_stop_ends_lightning_fit() -> None:
    module = _HostStub(ema=3.0, star=1.0, slack=0.1, clipped=True)
    stop = HostRotStop(patience=3)
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


def test_build_trainer_attaches_host_rot_stop(tmp_path: Path) -> None:
    job = SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        fit=FitSpec(
            max_steps=50,
            early_stop_patience=7,
            enable_csv_logger=False,
            checkpoint_dir=str(tmp_path),
        ),
        runtime=RuntimeSpec(ray=False, use_gpu=False),
    )
    trainer = _build_trainer(job, tmp_path)
    attached = lightning_trainer_callbacks(trainer)
    host_stops = [cb for cb in attached if isinstance(cb, HostRotStop)]
    inventory_stops = [cb for cb in attached if isinstance(cb, InventoryTrackStop)]
    assert len(host_stops) == 1
    assert host_stops[0].patience == 7
    assert len(inventory_stops) == 1
    assert inventory_stops[0].patience == 7
