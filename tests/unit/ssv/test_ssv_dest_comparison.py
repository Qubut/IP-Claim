"""Dest-comparison train addend: weight-off identity and live shuffled unpaid."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from tests._ssv_fixtures import ssv_fixture_batch, ssv_smoke_module

from ip_claim.ssv.collate import SoftMlmBatch
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.model import SoftTrunkOutput
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.steer.cheng import cheng_rescale

_CONFIGS = Path(__file__).resolve().parents[3] / 'configs'


def _counting_forward(module: SsvLightningModule) -> list[SoftTrunkOutput]:
    captured: list[SoftTrunkOutput] = []
    real = SsvLightningModule.forward

    def spy(
        self: SsvLightningModule,
        batch: SoftMlmBatch,
        living: torch.Tensor | None = None,
    ) -> SoftTrunkOutput:
        out = real(self, batch, living=living)
        captured.append(out)
        return out

    module.forward = spy.__get__(module, SsvLightningModule)  # type: ignore[method-assign]
    return captured


def test_zero_weight_is_one_forward_and_cheng_only(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    batch = ssv_fixture_batch(module)
    captured = _counting_forward(module)
    module.train()
    with patch.object(module, 'log'), patch.object(module, 'log_dict'):
        loss = module.training_step(batch, 0)
    assert len(captured) == 1
    task, cons = module._task_and_constraint_losses(captured[0])
    expected, _scale = cheng_rescale(task, cons, module._constraint_lambdas())
    assert torch.allclose(loss, expected)
    assert torch.isfinite(loss)


def test_positive_weight_adds_finite_dest_term_and_bank_grad(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
    batch = ssv_fixture_batch(module)
    captured = _counting_forward(module)
    logged: dict[str, torch.Tensor] = {}

    def capture_log(name: str, value: object = None, **_kwargs: object) -> None:
        if torch.is_tensor(value):
            logged[name] = cast(torch.Tensor, value)

    module.train()
    with patch.object(module, 'log', side_effect=capture_log), patch.object(module, 'log_dict'):
        loss = module.training_step(batch, 0)
    assert len(captured) == 1
    assert 'dest_loss' in logged
    dest_loss = logged['dest_loss']
    assert torch.isfinite(dest_loss)
    task, cons = module._task_and_constraint_losses(captured[0])
    cheng, _scale = cheng_rescale(task, cons, module._constraint_lambdas())
    assert torch.isfinite(loss)
    assert not torch.allclose(loss, cheng)
    loss.backward()
    bank = module.model.soft_vocab.entity_bank
    assert bank.grad is not None
    assert float(bank.grad.abs().sum()) > 0.0
    compose = next(module.model.encoder.compose_convs.parameters())
    assert compose.grad is not None
    assert float(compose.grad.abs().sum()) > 0.0
    projector = next(module.model.projector.parameters())
    assert projector.grad is not None
    assert float(projector.grad.abs().sum()) > 0.0
    lora_params = tuple(
        param
        for name, param in module.model.host.named_parameters()
        if param.requires_grad and 'lora' in name.lower()
    )
    assert lora_params
    assert all(param.grad is not None for param in lora_params)
    assert any(
        float(param.grad.abs().sum()) > 0.0 for param in lora_params if param.grad is not None
    )


def test_transfer_batch_keeps_claim_and_disclosure_texts(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    batch = ssv_fixture_batch(module)
    moved = module.transfer_batch_to_device(batch, torch.device('cpu'))
    assert moved.claim_texts == batch.claim_texts
    assert moved.disclosure_texts == batch.disclosure_texts
    assert moved.claim_texts
    assert any(blob for blob in moved.claim_texts)


def test_arm_overlays_cross_dest_and_ke() -> None:
    prod = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.yaml')
    dest = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.dest_compare.yaml')
    ke = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.ke.yaml')
    both = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.dest_compare.ke.yaml')
    assert prod.dest_comparison.weight == pytest.approx(0.0)
    assert prod.kendall.include_ke is False
    assert dest.dest_comparison.weight > 0.0
    assert dest.kendall.include_ke is False
    assert dest.dest_comparison.margin == pytest.approx(0.0)
    assert ke.dest_comparison.weight == pytest.approx(0.0)
    assert ke.kendall.include_ke is True
    assert both.dest_comparison.weight > 0.0
    assert both.kendall.include_ke is True
    assert dest.fit.checkpoint_dir == '/outputs/ssv-fixd-dest'
    assert ke.fit.checkpoint_dir == '/outputs/ssv-fixd-ke'
    assert both.fit.checkpoint_dir == '/outputs/ssv-fixd-dest-ke'


def test_include_ke_enters_task_mix(tmp_path: Path) -> None:
    off = ssv_smoke_module(tmp_path)
    on = ssv_smoke_module(tmp_path, include_ke=True)
    batch = ssv_fixture_batch(off)
    off.train()
    on.train()
    out = off.forward(batch)
    task_off, _cons_off = off._task_and_constraint_losses(out)
    task_on, _cons_on = on._task_and_constraint_losses(out)
    assert torch.allclose(
        task_on,
        task_off + out.ke_loss,
    )
