"""Unit tests for SSV Lightning strategy resolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from lightning.pytorch.strategies import FSDPStrategy

from ip_claim.ssv.config import FitSpec, HostSpec, RuntimeSpec, SsvTrainConfig
from ip_claim.ssv.trainer_strategy import resolve_lightning_strategy


def _job(**updates: object) -> SsvTrainConfig:
    base = SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        fit=FitSpec(max_steps=1, batch_size=2, checkpoint_dir='artifacts/ssv'),
        runtime=RuntimeSpec(ray=False, use_gpu=False),
    )
    return base.overlay(updates)


def test_single_process_uses_auto_strategy() -> None:
    assert resolve_lightning_strategy(_job(use_gpu=False)) == 'auto'


def test_multi_gpu_ddp_is_explicit_strategy() -> None:
    job = _job(use_gpu=True, num_devices=4, distributed_strategy='ddp')
    with patch('ip_claim.ssv.trainer_strategy.DDPStrategy') as mock_ddp:
        sentinel = object()
        mock_ddp.return_value = sentinel
        strategy = resolve_lightning_strategy(job)
    assert strategy is sentinel
    mock_ddp.assert_called_once_with(
        static_graph=True,
        find_unused_parameters=True,
    )


def test_ddp_prod_flags_from_config() -> None:
    job = _job(
        use_gpu=True,
        num_devices=4,
        distributed_strategy='ddp',
        ddp_static_graph=False,
    )
    with patch('ip_claim.ssv.trainer_strategy.DDPStrategy') as mock_ddp:
        mock_ddp.return_value = object()
        resolve_lightning_strategy(job)
    mock_ddp.assert_called_once_with(
        static_graph=False,
        find_unused_parameters=True,
    )


def test_ddp_static_graph_can_disable_find_unused() -> None:
    job = _job(
        use_gpu=True,
        num_devices=4,
        distributed_strategy='ddp',
        ddp_static_graph=True,
        ddp_find_unused_parameters=False,
    )
    with patch('ip_claim.ssv.trainer_strategy.DDPStrategy') as mock_ddp:
        mock_ddp.return_value = object()
        resolve_lightning_strategy(job)
    mock_ddp.assert_called_once_with(
        static_graph=True,
        find_unused_parameters=False,
    )


def test_fsdp_grad_op_strategy() -> None:
    job = _job(
        use_gpu=True,
        num_devices=4,
        distributed_strategy='fsdp_grad_op',
    )
    strategy = resolve_lightning_strategy(job)
    assert isinstance(strategy, FSDPStrategy)


def test_invalid_strategy_rejected_at_config_load() -> None:
    with pytest.raises(ValueError, match='distributed_strategy'):
        SsvTrainConfig(
            host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
            fit=FitSpec(distributed_strategy='deepspeed'),
        )
