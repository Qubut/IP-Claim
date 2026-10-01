"""Train resume: init_weights warm start and last.ckpt contract gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import lightning.pytorch as pl
import torch
from dependency_injector import providers
from returns.result import Success
from torch.utils.data import DataLoader as TorchDataLoader

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ssv.config import ArchSpec, FitSpec, HostSpec, RuntimeSpec, SsvTrainConfig
from ip_claim.ssv.train import train_func


def _smoke_config(tmp_path: Path) -> SsvTrainConfig:
    return SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        arch=ArchSpec(
            gnn_hidden=32,
            gnn_heads=4,
            gnn_layers=1,
            n_soft_tokens=4,
            entity_bank_size=16,
            relation_bank_size=8,
            soft_dim=32,
            max_length=48,
        ),
        fit=FitSpec(
            max_steps=1,
            batch_size=2,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        ),
        runtime=RuntimeSpec(ray=False),
    )


def test_train_func_skips_foreign_last_ckpt_and_loads_init_weights(tmp_path: Path) -> None:
    init = tmp_path / 'ssv-step009000.ckpt'
    torch.save({'state_dict': {'model.soft_vocab.entity_bank': torch.ones(1, 1)}}, init)
    config = _smoke_config(tmp_path).overlay({'init_weights': str(init)})
    ckpt_dir = Path(config.fit.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {'kendall_s_mlm': torch.tensor(9.0)}}, ckpt_dir / 'last.ckpt')
    loaded: dict[str, Path] = {}
    captured: dict[str, object] = {}

    def fake_load(*, path: Path, **kwargs: object) -> Success[None]:
        del kwargs
        loaded['path'] = path
        return Success(None)

    def spy_fit(self: object, *args: object, **kwargs: object) -> None:
        del args
        captured['ckpt_path'] = kwargs.get('ckpt_path')

    def fake_save(self: object, path: str) -> None:
        Path(path).touch()

    with (
        SsvContainer.init_weights.override(providers.Callable(fake_load)),
        patch.object(pl.Trainer, 'fit', spy_fit),
        patch.object(pl.Trainer, 'save_checkpoint', fake_save),
    ):
        train_func(config)
    assert loaded['path'] == init
    assert captured['ckpt_path'] is None


def test_train_func_dataloader_uses_feed_knobs(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path).overlay({
        'dataloader_num_workers': 2,
        'dataloader_prefetch_factor': 4,
        'dataloader_pin_memory': True,
        'prime_jate_spans': True,
    })
    captured: dict[str, object] = {}

    def fake_save(self: object, path: str) -> None:
        Path(path).touch()

    def spy_fit(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs

    def spy_loader(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return TorchDataLoader(*args, **kwargs)

    with (
        patch('ip_claim.ssv.train.DataLoader', spy_loader),
        patch.object(pl.Trainer, 'fit', spy_fit),
        patch.object(pl.Trainer, 'save_checkpoint', fake_save),
    ):
        train_func(config)
    assert captured['num_workers'] == 2
    assert captured['prefetch_factor'] == 4
    assert captured['pin_memory'] is True
    assert captured['persistent_workers'] is True
    assert config.fit.prime_jate_spans is True
