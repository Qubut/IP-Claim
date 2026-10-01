"""Prefix-ablation probe: paired eval forwards, cadence, logged wiring check."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
import torch
from tests._ssv_fixtures import attach_cpu_trainer, ssv_fixture_batch, ssv_smoke_module

from ip_claim.ssv.collate import SoftMlmBatch
from ip_claim.ssv.module import SsvLightningModule


def _fire_probe(module: SsvLightningModule, batch: SoftMlmBatch, *, step: int) -> None:
    with patch.object(type(module), 'global_step', new_callable=PropertyMock, return_value=step):
        module._probe_graph_reliance(batch)


def test_probe_delta_uses_paired_eval_forwards_not_stale_host_nll(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, graph_probe={'interval_steps': 1})
    batch = ssv_fixture_batch(module)
    module.train()
    _ = module.last_host_nll.fill_(999.0)
    module.model.eval()
    with torch.no_grad():
        trunk_out = module.model(
            graphs=batch.graphs,
            input_ids=batch.input_ids,
            unmasked_input_ids=batch.unmasked_input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            zero_prefix=False,
        )
        host_out = module.model(
            graphs=batch.graphs,
            input_ids=batch.input_ids,
            unmasked_input_ids=batch.unmasked_input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            zero_prefix=True,
        )
    module.model.train()
    expected = float(host_out.mlm_loss) - float(trunk_out.mlm_loss)
    _fire_probe(module, batch, step=0)
    assert float(module.last_graph_reliance_delta) == pytest.approx(expected)
    assert float(module.last_host_nll) == pytest.approx(999.0)


def test_probe_restores_model_training_mode(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, graph_probe={'interval_steps': 1})
    batch = ssv_fixture_batch(module)
    module.train()
    assert module.model.training is True
    _ = module.last_host_nll.fill_(1.0)
    _fire_probe(module, batch, step=0)
    assert module.model.training is True


def test_off_interval_steps_are_a_noop(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, graph_probe={'interval_steps': 50})
    batch = ssv_fixture_batch(module)
    _ = module.last_graph_reliance_delta.fill_(0.5)
    _ = module.last_host_nll.fill_(0.0)
    _fire_probe(module, batch, step=1)
    assert float(module.last_graph_reliance_delta) == pytest.approx(0.5)


def test_prefix_ablation_delta_is_logged_and_not_a_dual(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, graph_probe={'interval_steps': 1})
    batch = ssv_fixture_batch(module)
    _ = module.last_inventory_entropy.fill_(1.0)
    _ = module.last_row_entropy.fill_(1.0)
    _ = module.last_usage_entropy.fill_(1.0)
    _ = module.inventory_target_seeded.fill_(True)
    _ = module.usage_target_seeded.fill_(True)
    _ = module.row_target_seeded.fill_(True)
    _ = module.inventory_entropy_target.fill_(1.0)
    _ = module.usage_entropy_target.fill_(1.0)
    _ = module.row_entropy_target.fill_(1.0)
    _fire_probe(module, batch, step=0)
    assert torch.isfinite(module.last_graph_reliance_delta)
    assert module.duals.names == ('inv', 'use', 'h', 'host')
    assert 'graph' not in module.duals.log_lambda
    logged: dict[str, object] = {}

    def capture(name: str, value: object, **kwargs: object) -> None:
        del kwargs
        logged[name] = value

    attach_cpu_trainer(module)
    with (
        patch.object(type(module), 'global_step', new_callable=PropertyMock, return_value=0),
        patch.object(module, 'log', side_effect=capture),
    ):
        module.on_train_batch_start(None, 0)
    assert 'graph_reliance_delta' in logged
    assert torch.isfinite(torch.as_tensor(logged['graph_reliance_delta']))
    assert 'lambda_graph' not in logged
    assert 'graph_reliance_floor' not in logged
    assert 'log_lambda_graph_clipped' not in logged
