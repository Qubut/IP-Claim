"""Live constraint seeds and Adam duals; prod YAML omits babysit keys."""

from __future__ import annotations

import inspect
import math
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from tests._ssv_fixtures import ssv_fixture_batch, ssv_smoke_module

from ip_claim.ssv.config import DualSpec, SsvTrainConfig, TesSacSpec
from ip_claim.ssv.steer.tes_sac import tes_sac_ratchet
from ip_claim.ssv.train import InventoryTrackStop

_PROD_YAML = Path(__file__).resolve().parents[3] / 'configs' / 'ssv_train.prod.yaml'
_SMOKE_YAML = Path(__file__).resolve().parents[3] / 'configs' / 'ssv_train.smoke.yaml'

_BABYSIT_KEYS = (
    'inventory_slots',
    'usage_entropy_ratio',
    'row_entropy_target',
    'dual_step_size',
    'log_lambda_min',
    'log_lambda_max',
    'lambda_codebook',
    'lambda_commit',
    'inventory_track_band',
    'inventory_track_std',
    'inventory_track_steps',
    'inventory_ema_decay',
    'inventory_ratchet_factor',
)


def test_tes_sac_spec_defaults_remain_xu_table_1() -> None:
    spec = TesSacSpec()
    assert spec.band == pytest.approx(0.01)
    assert spec.std_max == pytest.approx(0.05)
    assert spec.ema_decay == pytest.approx(0.999)
    assert spec.ratchet == pytest.approx(0.9)
    assert spec.lock_steps == 20


def test_prod_yaml_loads_without_inventory_slots() -> None:
    text = _PROD_YAML.read_text(encoding='utf-8')
    for key in _BABYSIT_KEYS:
        assert key not in text, key
    job = SsvTrainConfig.from_yaml(_PROD_YAML)
    assert 'inventory_slots' not in DualSpec.model_fields
    assert not hasattr(job, 'resolved_inventory_entropy_floor')
    assert 'floor' not in inspect.signature(tes_sac_ratchet).parameters
    stop = InventoryTrackStop.from_config(job)
    assert not hasattr(stop, 'floor')
    assert job.dual.log_lambda_lr == pytest.approx(1e-4)
    assert job.tes_sac == TesSacSpec()


def test_smoke_yaml_omits_inventory_track_leftovers() -> None:
    text = _SMOKE_YAML.read_text(encoding='utf-8')
    for key in _BABYSIT_KEYS:
        assert key not in text, key
    job = SsvTrainConfig.from_yaml(_SMOKE_YAML)
    assert job.tes_sac == TesSacSpec()


def test_seed_bounds_come_from_live_batch_not_yaml_leftovers(tmp_path: Path) -> None:
    leftover_row = 5.0
    leftover_ratio = 0.50
    module = ssv_smoke_module(
        tmp_path,
        usage_entropy_ratio=leftover_ratio,
        row_entropy_target=leftover_row,
    )
    batch = ssv_fixture_batch(module)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module.train()
        out = module.forward(batch)
    usage = out.soft_vocab.batch_usage.detach().clamp_min(0)
    mass = usage.sum().clamp_min(1e-12)
    probs = usage / mass
    live_usage = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    live_row = float(out.mean_row_entropy.detach())
    with (
        patch.object(module, 'forward', return_value=out),
        patch.object(module, 'log'),
    ):
        _ = module.training_step(batch, 0)
    yaml_usage = leftover_ratio * math.log(int(module.config.arch.entity_bank_size))
    assert bool(module.usage_target_seeded.item())
    assert bool(module.row_target_seeded.item())
    assert float(module.usage_entropy_target) == pytest.approx(live_usage)
    assert float(module.row_entropy_target) == pytest.approx(live_row)
    assert float(module.usage_entropy_target) <= live_usage + 1e-6
    assert float(module.usage_entropy_target) != pytest.approx(yaml_usage)
    assert float(module.row_entropy_target) != pytest.approx(leftover_row)


def test_raising_lambda_via_adam_does_not_change_kendall_diversity_input(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, log_lambda_lr=0.1)
    batch = ssv_fixture_batch(module)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module.train()
        out = module.forward(batch)
    task_div = out.diversity_terms.vq + out.diversity_terms.rel_vq
    _ = module.last_inventory_entropy.fill_(5.0)
    _ = module.inventory_entropy_target.fill_(1.0)
    _ = module.inventory_target_seeded.fill_(True)
    _ = module.last_usage_entropy.fill_(1.0)
    _ = module.usage_entropy_target.fill_(1.0)
    _ = module.usage_target_seeded.fill_(True)
    _ = module.last_row_entropy.fill_(1.0)
    _ = module.row_entropy_target.fill_(1.0)
    _ = module.row_target_seeded.fill_(True)
    task_before, _cons_before = module._task_and_constraint_losses(out)
    before = float(module.log_lambda_inv.detach())
    module.on_train_batch_start(None, 0)
    task_after, _cons_after = module._task_and_constraint_losses(out)
    assert float(module.log_lambda_inv.detach()) > before
    assert float(task_div.detach()) == pytest.approx(
        float((out.diversity_terms.vq + out.diversity_terms.rel_vq).detach())
    )
    assert float(task_before.detach()) == pytest.approx(float(task_after.detach()))
