"""Unit and smoke tests for Soft Structural Vocabulary LoRA MLM training."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from unittest.mock import PropertyMock, patch

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.grads import grad_norm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from unit.ssv.test_ssv_inject import _occupy_visible

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ingestion.adapters.hupd_json.paths import patent_from_hupd_path
from ip_claim.ssv.collate import SoftMlmCollator, SoftMlmExample
from ip_claim.ssv.config import SsvTrainConfig, TesSacSpec
from ip_claim.ssv.dataset import examples_from_patents
from ip_claim.ssv.graph_batch import graph_batch_from_hupd_dict
from ip_claim.ssv.host_tokenizer import load_host_tokenizer
from ip_claim.ssv.model import SoftTrunkOutput, build_soft_trunk
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.soft_graph import (
    SOFT_RELATED,
    build_soft_relation_bundle,
    merge_soft_overlay,
)
from ip_claim.ssv.soft_vocab import SoftVocabModule
from ip_claim.ssv.steer.cheng import cheng_rescale
from ip_claim.ssv.train import train_func
from tests._ssv_fixtures import ssv_smoke_config

_FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'hupd'


def _smoke_config(tmp_path: Path, *, max_steps: int = 1) -> SsvTrainConfig:
    return ssv_smoke_config(tmp_path).overlay({'max_steps': max_steps})


def _fixture_examples(*names: str) -> tuple[SoftMlmExample, ...]:
    return examples_from_patents(tuple(patent_from_hupd_path(_FIXTURES / name) for name in names))


def test_collator_stock_mlm_rho_zero(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.5, max_length=48, rho=0.0)
    examples = _fixture_examples('13817165.json', '14111139.json')
    batch = collator(examples)
    assert batch.rho == pytest.approx(0.0)
    assert batch.input_ids.shape[0] == 2
    assert batch.labels.shape == batch.input_ids.shape
    assert len(batch.graphs) == 2
    assert (batch.labels != -100).any()


def test_collator_accepts_nonzero_rho(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.5, max_length=48, rho=0.4)

    soft_vocab = SoftVocabModule(config)
    embed = torch.nn.Embedding(len(tokenizer), int(config.host.d_model))
    collator.bind_assignment_source(soft_vocab, embed)
    examples = _fixture_examples('13817165.json', '14111139.json')
    batch = collator(examples)
    assert batch.rho == pytest.approx(0.4)
    assert (batch.labels != -100).any()
    collator.set_rho(0.0)
    assert collator.rho == pytest.approx(0.0)


def test_di_builds_lightning_module(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        container = SsvContainer(config=config)
        module = container.lightning_module()
    assert isinstance(module, SsvLightningModule)
    assert module.config.kendall.beta_div == config.kendall.beta_div
    assert module.model.n_soft_tokens == config.arch.n_soft_tokens


def test_relation_occupancy_mean_metric_updates(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        container = SsvContainer(config=config)
        module = container.lightning_module()
        module.train()
        out = module.model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
        )
    align = torch.zeros((), dtype=out.mlm_loss.dtype, device=out.mlm_loss.device)
    module._update_metrics(out, align, module.train_metrics, module.train_accuracy)
    rel_occ = module.train_metrics['relation_occupancy'].compute()
    assert torch.isfinite(rel_occ)
    assert 0.0 <= float(rel_occ) <= 1.0


def test_stage_mean_metrics_and_mlm_accuracy_update(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
        module.train()
        out = module.model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
        )
    align = torch.zeros((), dtype=out.mlm_loss.dtype, device=out.mlm_loss.device)
    module._update_metrics(out, align, module.train_metrics, module.train_accuracy)
    assert all(
        torch.isfinite(module.train_metrics[str(name)].compute())
        for name in module.train_metrics.keys(keep_base=True)
    )
    assert torch.isfinite(module.train_metrics['mlm_nll'].compute())
    acc = module.train_accuracy['mlm_accuracy'].compute()
    assert torch.isfinite(acc)
    assert 0.0 <= float(acc) <= 1.0

    module.eval()
    module._update_metrics(out, align, module.val_metrics, module.val_accuracy)
    assert torch.isfinite(module.val_metrics['mlm_nll'].compute())
    assert torch.isfinite(module.val_metrics['relation_occupancy'].compute())
    assert torch.isfinite(module.val_metrics['dea_gap'].compute())
    assert torch.isfinite(module.val_metrics['dea_loss'].compute())
    val_acc = module.val_accuracy['mlm_accuracy'].compute()
    assert torch.isfinite(val_acc)
    assert 0.0 <= float(val_acc) <= 1.0


def test_entropy_scale_starts_at_zero_on_first_batch(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    module.on_train_batch_start(None, 0)
    assert module.model.soft_vocab._entropy_scale == pytest.approx(0.0)


def test_entropy_scale_held_at_zero_when_uniform_stuck(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    module._entropy_uniform_streak = int(config.dual.entropy_pause_steps)
    with patch.object(type(module), 'global_step', new_callable=PropertyMock, return_value=50):
        module.on_train_batch_start(None, 0)
    assert module.model.soft_vocab._entropy_scale == pytest.approx(0.0)


def test_entropy_scale_follows_warmup_when_not_stuck(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    module._entropy_uniform_streak = 0
    warmup = max(1, config.resolved_entropy_warmup_steps())
    with patch.object(type(module), 'global_step', new_callable=PropertyMock, return_value=50):
        module.on_train_batch_start(None, 0)
    assert module.model.soft_vocab._entropy_scale == pytest.approx(min(1.0, 50 / warmup))


def test_epoch_perplexity_is_exp_of_mean_nll(tmp_path: Path) -> None:
    """Multi-batch epoch PPL must be exp(mean NLL), not mean of batch PPLs."""
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()

    nll_a = torch.tensor(0.0)
    nll_b = torch.tensor(2.0)
    module.train_metrics['mlm_nll'].update(nll_a)
    module.train_metrics['mlm_nll'].update(nll_b)
    module.val_metrics['mlm_nll'].update(nll_a)
    module.val_metrics['mlm_nll'].update(nll_b)

    mean_nll = 1.0
    expected_ppl = float(torch.exp(torch.tensor(mean_nll)))
    mean_of_batch_ppl = float((torch.exp(nll_a) + torch.exp(nll_b)) / 2.0)
    assert expected_ppl != pytest.approx(mean_of_batch_ppl, rel=1e-3)

    logged: dict[str, float] = {}

    def _capture(name: str, value: object, **_kwargs: object) -> None:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        logged[name] = float(tensor.detach())

    with patch.object(module, 'log', side_effect=_capture):
        module.on_train_epoch_end()
        module.on_validation_epoch_end()

    assert logged['perplexity'] == pytest.approx(expected_ppl, rel=1e-5)
    assert logged['val_perplexity'] == pytest.approx(expected_ppl, rel=1e-5)
    assert logged['perplexity'] != pytest.approx(mean_of_batch_ppl, rel=1e-3)


def test_grad_norm_hook_logs_total(tmp_path: Path) -> None:

    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
        module.train()
        out = module.model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
        )
    out.mlm_loss.backward()
    norms = grad_norm(module, norm_type=float(config.log.grad_norm_type))
    total_key = f'grad_{float(config.log.grad_norm_type)}_norm_total'
    assert total_key in norms
    assert math.isfinite(float(norms[total_key]))
    # Without a Trainer, the hook must no-op (no self.log) rather than warn-as-error.
    configured = module.configure_optimizers()
    module.on_before_optimizer_step(configured['optimizer'])


def test_soft_trunk_forward_mlm_plus_div(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.train()
        out = model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
            batch.texts,
            claim_texts=batch.claim_texts,
        )
    assert isinstance(out, SoftTrunkOutput)
    assert bool(model.soft_vocab._banks_seeded.item())
    assert torch.isfinite(out.mlm_loss)
    assert torch.isfinite(out.diversity_loss)
    assert torch.isfinite(out.inventory_entropy)
    assert torch.isfinite(out.ke_loss)
    assert torch.isfinite(out.dea_loss)
    assert torch.isfinite(out.dea_gap)
    assert torch.isfinite(out.occupancy)
    assert torch.isfinite(out.relation_occupancy)
    assert 0.0 <= float(out.relation_occupancy.detach()) <= 1.0
    assert out.soft_vocab.relation_assignment is not None
    assert out.soft_vocab.relation_assignment.shape[-1] == config.arch.relation_bank_size
    loss = out.mlm_loss + config.kendall.beta_div * out.diversity_loss + out.ke_loss
    loss.backward()
    assert model.soft_vocab.relation_bank.grad is not None
    assert torch.isfinite(model.soft_vocab.relation_bank.grad).all()
    assert not torch.allclose(
        model.soft_vocab.relation_bank.grad,
        torch.zeros_like(model.soft_vocab.relation_bank.grad),
    )
    assert model.dea_head.to_bank.weight.grad is None
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_soft_trunk_soft_edges_present_on_merged_graphs(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        token_embeds = model.host.get_input_embeddings()(batch.unmasked_input_ids)
        assign, _ = model.soft_vocab.soft_assign(token_embeds)
        bundle = build_soft_relation_bundle(
            model.soft_vocab,
            assign,
            batch.attention_mask,
            mass_floor=config.arch.soft_occupied_floor,
            demand=model.soft_vocab.masked_intensity(assign, batch.attention_mask),
        )
        merged = merge_soft_overlay(
            batch.graphs[0],
            bundle.overlays[0],
            bank_size=int(model.soft_vocab.entity_bank.size(0)),
        )
    assert 'soft_entity' in merged.node_types
    assert SOFT_RELATED in merged.edge_types
    assert int(merged['soft_entity'].num_nodes) > 0
    assert int(merged[SOFT_RELATED].edge_index.size(1)) > 0
    assert merged[SOFT_RELATED].edge_attr.shape == (
        merged[SOFT_RELATED].edge_index.size(1),
        config.arch.soft_dim,
    )


def test_train_func_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ckpt_path = train_func(config)
    assert ckpt_path.is_file()
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'state_dict' in payload
    assert any(key.startswith('model.') for key in payload['state_dict'])


def test_train_func_writes_step_and_last_checkpoints(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=2)
    config = config.overlay({'checkpoint_every_n_steps': 1, 'checkpoint_save_top_k': -1})
    ckpt_dir = Path(config.fit.checkpoint_dir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        final_path = train_func(config)
    assert final_path.is_file()
    assert (ckpt_dir / 'last.ckpt').is_file()
    step_ckpts = tuple(ckpt_dir.glob('ssv-step*.ckpt'))
    assert len(step_ckpts) >= 1


_CSV_STEP_METRICS = (
    'mlm_nll',
    'div_loss',
    'ke_loss',
    'soft_ke_loss',
    'align_loss',
    'occupancy',
    'relation_occupancy',
    'n_dead_before_restart',
    'usage_entropy',
    'inverse_simpson',
    'assignment_perplexity',
    'row_entropy',
    'inventory_entropy',
    'inventory_ln_k',
    'inventory_below_ln_k',
    'usage_above_target',
    'mlm_accuracy',
    'log_lambda_inv',
    'inventory_entropy_target',
    'kendall_w_mlm',
    'kendall_w_dea',
    'dea_gap',
    'dea_loss',
    'log_lambda_inv_clipped',
)


def test_csv_logger_records_step_scalars(tmp_path: Path) -> None:
    """MeanMetric objects passed to self.log do not emit on_step CSV rows; scalars do."""

    config = _smoke_config(tmp_path, max_steps=2)
    config_dict = config.model_dump(mode='json')
    config_dict['enable_csv_logger'] = True
    config_dict['log_every_n_steps'] = 1
    job = SsvTrainConfig.model_validate(config_dict)
    tokenizer = load_host_tokenizer(job)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    examples = _fixture_examples('13817165.json', '14111139.json')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=job).lightning_module()
    module.bind_mask_collator(collator)
    loader = DataLoader(
        TensorDataset(torch.zeros(len(examples))),
        batch_size=1,
        collate_fn=lambda _: collator(examples),
    )
    logger = CSVLogger(save_dir=str(tmp_path / 'csv'), name='lightning')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        trainer = pl.Trainer(
            max_steps=2,
            accelerator='cpu',
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1,
        )
        trainer.fit(module, train_dataloaders=loader)
    metrics_path = tmp_path / 'csv' / 'lightning' / 'version_0' / 'metrics.csv'
    header = metrics_path.read_text(encoding='utf-8').splitlines()[0]
    for name in _CSV_STEP_METRICS:
        assert name in header, header


_VAL_EPOCH_MEAN_KEYS = (
    'val_mlm_nll',
    'val_div_loss',
    'val_ke_loss',
    'val_soft_ke_loss',
    'val_align_loss',
    'val_occupancy',
    'val_relation_occupancy',
    'val_n_dead_before_restart',
    'val_usage_entropy',
    'val_inverse_simpson',
    'val_assignment_perplexity',
    'val_row_entropy',
    'val_dea_gap',
    'val_dea_loss',
)
_VAL_COMPOSE_KEYS = (
    'val_compose_transe',
    'val_compose_mrr',
    'val_compose_mr',
    'val_compose_hits_at_1',
    'val_compose_hits_at_3',
    'val_compose_hits_at_10',
)


def _capture_step_log_keys(module: SsvLightningModule) -> tuple[set[str], set[str]]:
    tokenizer = load_host_tokenizer(module.config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    train_keys: set[str] = set()
    val_keys: set[str] = set()

    def bucket() -> set[str]:
        return train_keys if module.training else val_keys

    def capture_log(name: str, *_args: object, **_kwargs: object) -> None:
        bucket().add(name)

    def capture_log_dict(mapping: object, **_kwargs: object) -> None:
        keys_fn = getattr(mapping, 'keys', None)
        if keys_fn is None:
            return
        bucket().update(str(key) for key in keys_fn())

    with (
        patch.object(module, 'log', side_effect=capture_log),
        patch.object(module, 'log_dict', side_effect=capture_log_dict),
    ):
        module.train()
        _ = module.training_step(batch, 0)
        module.eval()
        _ = module.validation_step(batch, 0)
    return train_keys, val_keys


def test_train_and_val_logged_metric_key_sets(tmp_path: Path) -> None:
    """Logged names stay equivalent across splits; dEA joins the same collection path."""
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    train_keys, val_keys = _capture_step_log_keys(module)
    for name in (
        'mlm_nll',
        'div_loss',
        'ke_loss',
        'soft_ke_loss',
        'align_loss',
        'occupancy',
        'relation_occupancy',
        'n_dead_before_restart',
        'usage_entropy',
        'inverse_simpson',
        'assignment_perplexity',
        'row_entropy',
        'inventory_below_ln_k',
        'usage_above_target',
        'mlm_accuracy',
        'dea_gap',
        'dea_loss',
    ):
        assert name in train_keys, train_keys
    for name in _VAL_EPOCH_MEAN_KEYS:
        assert name in val_keys, val_keys
    for name in _VAL_COMPOSE_KEYS:
        assert name in val_keys, val_keys
    assert 'val_mlm_accuracy' in val_keys
    assert 'val_loss' in val_keys


def test_logged_metric_key_gating(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path).overlay({
        'log_mlm_accuracy': False,
        'log_compose_ranking': False,
    })
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    train_keys, val_keys = _capture_step_log_keys(module)
    assert 'mlm_accuracy' not in train_keys
    assert 'val_mlm_accuracy' not in val_keys
    assert 'compose_mrr' not in train_keys
    for name in _VAL_COMPOSE_KEYS:
        assert name not in val_keys
    assert 'dea_gap' in train_keys
    assert 'val_dea_gap' in val_keys


def test_graph_batch_text_feeds_collator() -> None:
    raw = json.loads((_FIXTURES / '13817165.json').read_text(encoding='utf-8'))
    batch = graph_batch_from_hupd_dict(raw)
    assert batch.text
    assert '<SOH>' not in batch.text
    assert '<EOH>' not in batch.text


def _steer_module(tmp_path: Path, **updates: object) -> SsvLightningModule:
    config = _smoke_config(tmp_path)
    if updates:
        config = config.overlay(updates)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SsvContainer(config=config).lightning_module()


def _optimizer_and_scheduler(module: SsvLightningModule) -> tuple[Optimizer, LRScheduler]:
    configured = module.configure_optimizers()
    scheduler = configured['lr_scheduler']['scheduler']
    assert isinstance(scheduler, LRScheduler)
    return configured['optimizer'], scheduler


def _seed_primal(
    module: SsvLightningModule,
    *,
    inventory: float,
    row: float,
    usage: float,
    target: float | None = None,
    usage_target: float | None = None,
    row_target: float | None = None,
) -> None:
    _ = module.last_inventory_entropy.fill_(inventory)
    _ = module.last_row_entropy.fill_(row)
    _ = module.last_usage_entropy.fill_(usage)
    _ = module.inventory_target_seeded.fill_(True)
    _ = module.usage_target_seeded.fill_(True)
    _ = module.row_target_seeded.fill_(True)
    if target is not None:
        _ = module.inventory_entropy_target.fill_(target)
    _ = module.usage_entropy_target.fill_(usage if usage_target is None else usage_target)
    _ = module.row_entropy_target.fill_(row if row_target is None else row_target)


def test_first_finite_inventory_entropy_seeds_target(tmp_path: Path) -> None:
    module = _steer_module(tmp_path)
    assert not bool(module.inventory_target_seeded.item())
    tokenizer = load_host_tokenizer(module.config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module.train()
        out = module.forward(batch)
    live = float(out.inventory_entropy.detach())
    assert math.isfinite(live)
    with (
        patch.object(module, 'forward', return_value=out),
        patch.object(module, 'log'),
    ):
        _ = module.training_step(batch, 0)
    assert bool(module.inventory_target_seeded.item())
    assert float(module.inventory_entropy_target) == pytest.approx(live)
    assert float(module.last_inventory_entropy) == pytest.approx(live)


def test_dual_raises_lambda_inv_when_inventory_above_target(tmp_path: Path) -> None:
    module = _steer_module(tmp_path)
    _seed_primal(module, inventory=5.0, row=1.0, usage=1.0, target=2.0)
    before = float(module.log_lambda_inv.detach())
    module.on_train_batch_start(None, 0)
    assert float(module.log_lambda_inv.detach()) > before


def test_dual_lowers_lambda_use_when_usage_at_ln_k(tmp_path: Path) -> None:
    module = _steer_module(tmp_path)
    ln_k = math.log(int(module.config.arch.entity_bank_size))
    _seed_primal(
        module,
        inventory=1.0,
        row=1.0,
        usage=ln_k,
        target=1.0,
        usage_target=0.9 * ln_k,
    )
    before = float(module.log_lambda_use.detach())
    module.on_train_batch_start(None, 0)
    assert float(module.log_lambda_use.detach()) < before


def test_dual_raises_lambda_use_when_usage_below_target(tmp_path: Path) -> None:
    module = _steer_module(tmp_path)
    _seed_primal(module, inventory=1.0, row=1.0, usage=0.2, target=1.0, usage_target=1.0)
    before = float(module.log_lambda_use.detach())
    module.on_train_batch_start(None, 0)
    assert float(module.log_lambda_use.detach()) > before


def test_dual_raises_lambda_h_when_row_above_target(tmp_path: Path) -> None:
    module = _steer_module(tmp_path)
    _seed_primal(module, inventory=1.0, row=3.0, usage=1.0, target=1.0, row_target=1.0)
    before = float(module.log_lambda_h.detach())
    module.on_train_batch_start(None, 0)
    assert float(module.log_lambda_h.detach()) > before


def test_inventory_ratchet_multiplies_when_tracking(tmp_path: Path) -> None:
    module = _steer_module(
        tmp_path,
        tes_sac=TesSacSpec(lock_steps=2, band=0.2, ratchet=0.5),
    )
    start = 4.0
    _seed_primal(module, inventory=start, row=1.0, usage=1.0, target=start)
    module.on_train_batch_start(None, 0)
    assert float(module.inventory_entropy_target) == pytest.approx(start)
    module.on_train_batch_start(None, 0)
    after = float(module.inventory_entropy_target)
    assert after < start
    assert after == pytest.approx(start * 0.5)


def test_inventory_ratchet_holds_when_ema_std_is_high(tmp_path: Path) -> None:
    module = _steer_module(
        tmp_path,
        tes_sac=TesSacSpec(lock_steps=2, band=0.15, std_max=0.05, ema_decay=0.999, ratchet=0.5),
    )
    start = 3.74
    _seed_primal(module, inventory=start, row=1.0, usage=1.0, target=start)
    _ = module.inventory_entropy_ema.fill_(start)
    _ = module.inventory_entropy_ema_var.fill_(0.04)
    _ = module.inventory_ema_seeded.fill_(True)
    module.on_train_batch_start(None, 0)
    module.on_train_batch_start(None, 0)
    assert float(module.inventory_entropy_target) == pytest.approx(start)


def test_inventory_target_snaps_when_ema_undershoots_band(tmp_path: Path) -> None:
    module = _steer_module(
        tmp_path,
        tes_sac=TesSacSpec(lock_steps=2, band=0.15, ratchet=0.5),
    )
    start = 3.74
    live = 3.45
    _seed_primal(module, inventory=live, row=1.0, usage=1.0, target=start)
    module.on_train_batch_start(None, 0)
    assert float(module.inventory_entropy_target) == pytest.approx(live)


def test_inventory_ratchet_holds_when_not_tracking(tmp_path: Path) -> None:
    module = _steer_module(
        tmp_path,
        tes_sac=TesSacSpec(lock_steps=2, band=0.1),
    )
    _seed_primal(module, inventory=5.0, row=1.0, usage=1.0, target=2.0)
    module.on_train_batch_start(None, 0)
    module.on_train_batch_start(None, 0)
    assert float(module.inventory_entropy_target) == pytest.approx(2.0)


def test_log_lambda_clips_when_primal_cannot_hit_target(tmp_path: Path) -> None:
    module = _steer_module(tmp_path, log_lambda_max=0.2, log_lambda_lr=1.0)
    _seed_primal(module, inventory=8.0, row=1.0, usage=1.0, target=1.0)
    module.on_train_batch_start(None, 0)
    assert float(module.log_lambda_inv.detach()) == pytest.approx(0.2)
    assert module._dual_clip_inv is True


def test_raising_lambda_inv_does_not_change_kendall_diversity_input(tmp_path: Path) -> None:
    module = _steer_module(tmp_path)
    tokenizer = load_host_tokenizer(module.config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module.train()
        out = module.forward(batch)
    task_div = out.diversity_terms.vq + out.diversity_terms.rel_vq
    with torch.no_grad():
        _ = module.log_lambda_inv.fill_(0.0)
    task_lo, cons_lo = module._task_and_constraint_losses(out)
    with torch.no_grad():
        _ = module.log_lambda_inv.fill_(math.log(10.0))
    task_hi, cons_hi = module._task_and_constraint_losses(out)
    assert float(task_div.detach()) == pytest.approx(
        float((out.diversity_terms.vq + out.diversity_terms.rel_vq).detach())
    )
    assert float(task_lo.detach()) == pytest.approx(float(task_hi.detach()))
    assert float(cons_hi.detach()) > float(cons_lo.detach())


def test_kendall_mlm_weight_cannot_vanish(tmp_path: Path) -> None:
    module = _steer_module(tmp_path, kendall_s_max=4.0)
    module.kendall_s_mlm.data.fill_(100.0)
    module.kendall_s_dea.data.fill_(100.0)
    module.on_train_batch_start(None, 0)
    weight = float(module.kendall_s_mlm.detach().neg().exp())
    floor = math.exp(-float(module.config.kendall.kendall_s_max))
    assert weight == pytest.approx(floor)
    assert weight > 0.0
    assert float(module.kendall_s_dea.detach().neg().exp()) == pytest.approx(floor)
    zero = torch.zeros(())
    ten = torch.tensor(10.0)
    with_mlm = float(module._kendall_objective(ten, zero, zero).detach())
    without = float(module._kendall_objective(zero, zero, zero).detach())
    assert with_mlm > without


def test_train_mix_backward_populates_dea_head_grad(tmp_path: Path) -> None:
    module = _steer_module(tmp_path)
    assert float(module.kendall_s_dea.detach()) == pytest.approx(
        float(module.kendall_s_mlm.detach())
    )
    tokenizer = load_host_tokenizer(module.config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module.train()
        with _occupy_visible(module.model):
            out = module.forward(batch)
    task_loss, cons_loss = module._task_and_constraint_losses(out)
    loss, _scale = cheng_rescale(task_loss, cons_loss, module._constraint_lambdas())
    loss.backward()
    grad = module.model.dea_head.to_bank.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert not torch.allclose(grad, torch.zeros_like(grad))
    ten = torch.tensor(10.0)
    zero = torch.zeros(())
    with_dea = float(module._kendall_objective(zero, zero, ten).detach())
    without = float(module._kendall_objective(zero, zero, zero).detach())
    assert with_dea > without


def test_resolved_entropy_targets(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    assert config.resolved_usage_entropy_target() == pytest.approx(
        0.95 * math.log(config.arch.entity_bank_size)
    )
    assert not hasattr(config, 'resolved_inventory_entropy_floor')
    assert config.dual.row_entropy_target == pytest.approx(math.log(3.0))


def test_configure_optimizers_returns_step_constant_warmup_scheduler(tmp_path: Path) -> None:
    module = _steer_module(tmp_path, max_steps=8, lr_warmup_steps=2, learning_rate=3e-4)
    configured = module.configure_optimizers()
    assert set(configured) == {'optimizer', 'lr_scheduler'}
    sched_cfg = configured['lr_scheduler']
    assert sched_cfg['interval'] == 'step'
    assert sched_cfg['frequency'] == 1
    assert 'scheduler' in sched_cfg
    assert 'enable_scheduler' not in SsvTrainConfig.model_fields


def test_constant_scheduler_holds_peak_without_warmup(tmp_path: Path) -> None:
    peak = 3e-4
    module = _steer_module(tmp_path, max_steps=8, lr_warmup_steps=0, learning_rate=peak)
    opt, sch = _optimizer_and_scheduler(module)
    start = float(opt.param_groups[0]['lr'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sch.step()
    after = float(opt.param_groups[0]['lr'])
    assert start == pytest.approx(peak)
    assert after == pytest.approx(peak)


def test_constant_scheduler_warmup_then_hold(tmp_path: Path) -> None:
    peak = 3e-4
    module = _steer_module(tmp_path, max_steps=8, lr_warmup_steps=2, learning_rate=peak)
    opt, sch = _optimizer_and_scheduler(module)
    at_zero = float(opt.param_groups[0]['lr'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sch.step()
        mid_warm = float(opt.param_groups[0]['lr'])
        sch.step()
        at_peak = float(opt.param_groups[0]['lr'])
        sch.step()
        after_peak = float(opt.param_groups[0]['lr'])
    assert at_zero < mid_warm < at_peak
    assert at_peak == pytest.approx(peak)
    assert after_peak == pytest.approx(peak)


def test_resolved_lr_warmup_clips_to_safety_ceiling(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=8).overlay({'lr_warmup_steps': 100})
    assert config.resolved_lr_warmup_steps() == 8


def test_resolved_lr_warmup_default_is_zero(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=100)
    assert config.fit.lr_warmup_steps is None
    assert config.resolved_lr_warmup_steps() == 0


def test_prod_train_yaml_declares_track_stop_not_step_budget() -> None:
    path = Path(__file__).resolve().parents[2] / 'configs' / 'ssv_train.prod.yaml'
    job = SsvTrainConfig.from_yaml(path)
    assert job.fit.max_steps == 100000
    assert job.fit.lr_warmup_steps == 1000
    assert job.resolved_lr_warmup_steps() == 1000
    assert job.fit.early_stop_patience == 500
    assert job.dual.usage_entropy_ratio == pytest.approx(0.95)
    assert job.tes_sac.band == pytest.approx(0.01)
    assert job.tes_sac.std_max == pytest.approx(0.05)
    assert job.tes_sac.ema_decay == pytest.approx(0.999)
    assert job.tes_sac.ratchet == pytest.approx(0.9)
    assert job.tes_sac.lock_steps == 20
    assert 'enable_scheduler' not in SsvTrainConfig.model_fields
    assert 'enable_early_stop' not in SsvTrainConfig.model_fields
