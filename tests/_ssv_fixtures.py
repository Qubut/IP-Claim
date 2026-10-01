"""Shared SSV test fixtures: a tiny trained module and one collated batch."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ingestion.adapters.hupd_json.paths import patent_from_hupd_path
from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmCollator
from ip_claim.ssv.config import (
    ArchSpec,
    FitSpec,
    HostSpec,
    MlmSpec,
    RuntimeSpec,
    SsvTrainConfig,
)
from ip_claim.ssv.dataset import examples_from_patents
from ip_claim.ssv.host_tokenizer import load_host_tokenizer
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.soft_vocab import SoftVocabModule

SSV_HUPD_FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'hupd'


@contextmanager
def type_aligned_pairs(vocab: SoftVocabModule) -> Iterator[None]:
    """Score every candidate as leftover of the first relation-bank row."""
    typed = F.normalize(vocab.relation_bank.detach()[0], dim=-1)

    def pair_features(heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        _ = tails
        return typed.to(device=heads.device, dtype=heads.dtype).expand(heads.size(0), -1).clone()

    with patch.object(vocab, 'pair_features', side_effect=pair_features):
        yield


def ssv_tiny_vocab_config(
    *,
    d_model: int = 32,
    entity_bank_size: int = 8,
    relation_bank_size: int = 4,
    soft_dim: int = 16,
    assign_temperature: float = 1.0,
    relation_temperature: float = 1.0,
) -> SsvTrainConfig:
    """A bank-sized config for SoftVocabModule tests without a host checkpoint."""
    return SsvTrainConfig(
        host=HostSpec(d_model=d_model),
        arch=ArchSpec(
            entity_bank_size=entity_bank_size,
            relation_bank_size=relation_bank_size,
            soft_dim=soft_dim,
            assign_temperature=assign_temperature,
            relation_temperature=relation_temperature,
        ),
    )


def ssv_smoke_config(tmp_path: Path) -> SsvTrainConfig:
    """A tiny host/arch config sized for CPU-only tests."""
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
            soft_occupied_floor=0.0,
            max_length=48,
        ),
        fit=FitSpec(
            max_steps=1,
            batch_size=2,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        ),
        mlm=MlmSpec(mlm_probability=0.15),
        runtime=RuntimeSpec(ray=False),
    )


def ssv_smoke_module(tmp_path: Path, **updates: object) -> SsvLightningModule:
    """Build the Lightning module for :func:`ssv_smoke_config`, optionally overlaid."""
    config = ssv_smoke_config(tmp_path)
    if updates:
        config = config.overlay(updates)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SsvContainer(config=config).lightning_module()


def lightning_trainer_stub(**attrs: object) -> pl.Trainer:
    """Trainer-typed handle for callback unit checks that do not run a fit loop."""
    return cast(pl.Trainer, cast(object, SimpleNamespace(**attrs)))


def lightning_trainer_callbacks(trainer: pl.Trainer) -> list[object]:
    """Callbacks on a live Trainer instance (the attribute is not on the type)."""
    return list(vars(trainer)['callbacks'])


def attach_cpu_trainer(module: SsvLightningModule) -> None:
    """Attach a CPU trainer so train-hook logging is not skipped."""
    module._trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


def ssv_fixture_batch(module: SsvLightningModule) -> SoftMlmBatch:
    """Collate the two fixture HUPD patents through the module's own tokenizer."""
    tokenizer = load_host_tokenizer(module.config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    patents = tuple(
        patent_from_hupd_path(SSV_HUPD_FIXTURES / name)
        for name in ('13817165.json', '14111139.json')
    )
    return collator(examples_from_patents(patents))


__all__ = [
    'attach_cpu_trainer',
    'lightning_trainer_callbacks',
    'lightning_trainer_stub',
    'ssv_fixture_batch',
    'ssv_smoke_config',
    'ssv_smoke_module',
    'ssv_tiny_vocab_config',
    'type_aligned_pairs',
]
