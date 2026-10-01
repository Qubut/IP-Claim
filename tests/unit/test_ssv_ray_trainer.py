"""Unit seams for Soft Structural Vocabulary Ray TorchTrainer entry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from ip_claim.ssv import __main__ as ssv_main
from ip_claim.ssv.config import DEFAULT_RAY_SCALING_PATH, HostSpec, RayScalingSpec, SsvTrainConfig
from ip_claim.ssv.train import train_func

_SMOKE_TRAIN = Path(__file__).resolve().parents[2] / 'configs' / 'ssv_train.smoke.yaml'


def test_ray_scaling_spec_loads_package_yaml() -> None:
    spec = RayScalingSpec.from_yaml(DEFAULT_RAY_SCALING_PATH)
    assert spec.num_workers == 4
    assert spec.use_gpu is True
    assert DEFAULT_RAY_SCALING_PATH.is_file()


def test_ray_scaling_spec_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / 'bad.yaml'
    path.write_text('- not\n- a\n- mapping\n', encoding='utf-8')
    with pytest.raises(TypeError, match='must be a mapping'):
        RayScalingSpec.from_yaml(path)


def test_main_local_preserves_yaml_use_gpu(tmp_path: Path) -> None:
    """``--local`` must not force CPU; honor package YAML ``use_gpu``."""
    captured: dict[str, Any] = {}

    def fake_train_func(config: Any) -> Path:
        captured['config'] = (
            config if isinstance(config, SsvTrainConfig) else SsvTrainConfig.model_validate(config)
        )
        return tmp_path / 'ssv.ckpt'

    with patch.object(ssv_main, 'train_func', fake_train_func):
        code = ssv_main.main([
            'train',
            '--local',
            '--config',
            str(_SMOKE_TRAIN),
            '--checkpoint-dir',
            str(tmp_path / 'ckpt'),
        ])

    assert code == 0
    cfg = captured['config']
    assert cfg.runtime.ray is False
    assert cfg.runtime.use_gpu is True


def test_main_ray_launches_torch_trainer(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeResult:
        metrics = {'loss': 0.0}

    class FakeTorchTrainer:
        def __init__(
            self,
            train_loop_per_worker: Any,
            *,
            train_loop_config: dict[str, Any],
            scaling_config: Any,
        ) -> None:
            captured['train_loop_per_worker'] = train_loop_per_worker
            captured['train_loop_config'] = train_loop_config
            captured['scaling_config'] = scaling_config

        def fit(self) -> FakeResult:
            return FakeResult()

    def fake_scaling_config(*, num_workers: int, use_gpu: bool) -> SimpleNamespace:
        return SimpleNamespace(num_workers=num_workers, use_gpu=use_gpu)

    with (
        patch.object(ssv_main, 'ScalingConfig', fake_scaling_config),
        patch.object(ssv_main, 'TorchTrainer', FakeTorchTrainer),
    ):
        code = ssv_main.main([
            'train',
            '--ray',
            '--config',
            str(_SMOKE_TRAIN),
            '--max-steps',
            '1',
            '--checkpoint-dir',
            str(tmp_path / 'ckpt'),
            '--batch-size',
            '2',
            '--d-model',
            '32',
        ])

    assert code == 0
    assert captured['train_loop_per_worker'] is train_func
    loop = captured['train_loop_config']
    assert loop['runtime']['ray'] is True
    assert loop['runtime']['use_gpu'] is True
    assert loop['fit']['max_steps'] == 1
    assert loop['fit']['checkpoint_dir'] == str(tmp_path / 'ckpt')
    validated = SsvTrainConfig.model_validate(loop)
    assert validated.host == HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32)
    scaling = captured['scaling_config']
    assert scaling.num_workers == 4
    assert scaling.use_gpu is True


def test_main_no_args_exits_nonzero_without_traceback() -> None:
    assert ssv_main.main([]) != 0


def test_main_train_neither_mode_exits_nonzero_without_traceback() -> None:
    assert ssv_main.main(['train']) != 0


def test_main_train_both_modes_exits_nonzero_without_traceback() -> None:
    assert ssv_main.main(['train', '--local', '--ray']) != 0
