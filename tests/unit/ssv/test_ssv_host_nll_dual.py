"""Host-NLL dual activates only above L*+r; Cheng mix stays finite; Kendall is VQ-only."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import torch
from tests._ssv_fixtures import ssv_smoke_config

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ingestion.adapters.hupd_json.paths import patent_from_hupd_path
from ip_claim.ssv.collate import SoftMlmCollator
from ip_claim.ssv.dataset import examples_from_patents
from ip_claim.ssv.host_tokenizer import load_host_tokenizer
from ip_claim.ssv.model import SoftTrunkOutput
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.steer.cheng import cheng_rescale

_FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'hupd'


def _module(tmp_path: Path, **updates: object) -> SsvLightningModule:
    config = ssv_smoke_config(tmp_path)
    if updates:
        config = config.overlay(updates)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SsvContainer(config=config).lightning_module()


def _seed_primal(module: SsvLightningModule) -> None:
    _ = module.last_inventory_entropy.fill_(1.0)
    _ = module.last_row_entropy.fill_(1.0)
    _ = module.last_usage_entropy.fill_(1.0)
    _ = module.inventory_target_seeded.fill_(True)
    _ = module.inventory_entropy_target.fill_(1.0)
    _ = module.usage_target_seeded.fill_(True)
    _ = module.row_target_seeded.fill_(True)
    _ = module.usage_entropy_target.fill_(1.0)
    _ = module.row_entropy_target.fill_(1.0)


def _seed_host(module: SsvLightningModule, *, star: float, slack: float) -> None:
    _ = module.host_nll_star.fill_(star)
    _ = module.host_nll_slack.fill_(slack)
    _ = module.host_nll_star_seeded.fill_(True)


def _forward_fixture(module: SsvLightningModule) -> tuple[SoftTrunkOutput, torch.Tensor]:
    tokenizer = load_host_tokenizer(module.config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    patents = tuple(
        patent_from_hupd_path(_FIXTURES / name) for name in ('13817165.json', '14111139.json')
    )
    batch = collator(examples_from_patents(patents))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module.train()
        out = module.forward(batch)
    return out, module._align_loss(out.z_d, out.z_g)


def test_host_dual_does_not_raise_when_nll_at_or_below_ceiling(tmp_path: Path) -> None:
    module = _module(tmp_path)
    _seed_primal(module)
    _seed_host(module, star=2.0, slack=0.1)
    _ = module.last_host_nll.fill_(2.0)
    before = float(module.log_lambda_host.detach())
    module.on_train_batch_start(None, 0)
    assert float(module.log_lambda_host.detach()) <= before


def test_host_dual_raises_only_when_nll_exceeds_ceiling(tmp_path: Path) -> None:
    module = _module(tmp_path)
    _seed_primal(module)
    _seed_host(module, star=2.0, slack=0.1)
    _ = module.last_host_nll.fill_(2.5)
    before = float(module.log_lambda_host.detach())
    module.on_train_batch_start(None, 0)
    assert float(module.log_lambda_host.detach()) > before


def test_host_dual_inactive_until_star_is_seeded(tmp_path: Path) -> None:
    module = _module(tmp_path)
    _seed_primal(module)
    _ = module.last_host_nll.fill_(9.0)
    before = float(module.log_lambda_host.detach())
    module.on_train_batch_start(None, 0)
    assert float(module.log_lambda_host.detach()) == pytest.approx(before)
    assert not bool(module.host_nll_star_seeded.item())


def test_observe_host_nll_seeds_star_from_mean_and_slack_from_finch(tmp_path: Path) -> None:
    module = _module(tmp_path, tes_sac={'lock_steps': 2})
    module._observe_host_nll(2.0)
    assert not bool(module.host_nll_star_seeded.item())
    module._observe_host_nll(4.0)
    assert bool(module.host_nll_star_seeded.item())
    assert float(module.host_nll_star) == pytest.approx(3.0)
    assert float(module.host_nll_slack) > 0.0
    assert float(module.last_host_nll) == pytest.approx(4.0)


def test_cheng_mix_stays_finite_when_all_duals_are_clipped(tmp_path: Path) -> None:
    module = _module(tmp_path)
    hi = float(module.config.dual.log_lambda_max)
    with torch.no_grad():
        _ = module.log_lambda_inv.fill_(hi)
        _ = module.log_lambda_use.fill_(hi)
        _ = module.log_lambda_h.fill_(hi)
        _ = module.log_lambda_host.fill_(hi)
    _seed_host(module, star=1.0, slack=0.0)
    out, _align = _forward_fixture(module)
    task, cons = module._task_and_constraint_losses(out)
    mix, scale = cheng_rescale(task, cons, module._constraint_lambdas())
    assert torch.isfinite(mix)
    assert torch.isfinite(scale)
    assert float(scale.detach()) == pytest.approx(1.0 + 4.0 * math.exp(hi))


def test_raising_lambda_host_does_not_change_kendall_task(tmp_path: Path) -> None:
    module = _module(tmp_path)
    out, _align = _forward_fixture(module)
    _seed_host(module, star=float(out.mlm_loss.detach()), slack=0.0)
    with torch.no_grad():
        _ = module.log_lambda_host.fill_(0.0)
    task_lo, _cons_lo = module._task_and_constraint_losses(out)
    with torch.no_grad():
        _ = module.log_lambda_host.fill_(math.log(10.0))
    task_hi, cons_hi = module._task_and_constraint_losses(out)
    task_div = out.diversity_terms.vq + out.diversity_terms.rel_vq
    assert float(task_lo.detach()) == pytest.approx(float(task_hi.detach()))
    assert float(task_div.detach()) == pytest.approx(
        float((out.diversity_terms.vq + out.diversity_terms.rel_vq).detach())
    )
    assert torch.isfinite(cons_hi)
