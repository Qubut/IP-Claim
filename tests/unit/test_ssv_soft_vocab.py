"""Unit tests for SSV soft vocab banks, assignment, and diversity."""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ip_claim.ssv.config import (
    ArchSpec,
    BankHealthSpec,
    DualSpec,
    HostSpec,
    KendallSpec,
    SsvTrainConfig,
)
from ip_claim.ssv.soft_vocab import DiversityTerms, SoftVocabModule, SoftVocabOutput


def _tiny_config(
    *,
    entity_bank_size: int = 8,
    relation_bank_size: int = 4,
    soft_dim: int = 16,
    d_model: int = 32,
    utilization_eps: float = 1e-3,
) -> SsvTrainConfig:
    return SsvTrainConfig(
        host=HostSpec(d_model=d_model),
        arch=ArchSpec(
            entity_bank_size=entity_bank_size,
            relation_bank_size=relation_bank_size,
            soft_dim=soft_dim,
            assign_temperature=1.0,
            relation_temperature=1.0,
        ),
        dual=DualSpec(
            lambda_usage=1.0,
            lambda_rel_usage=1.0,
            lambda_commit=0.25,
        ),
        kendall=KendallSpec(beta_rel_div=1.0),
        bank=BankHealthSpec(utilization_eps=utilization_eps),
    )


def test_bank_sizes_match_config() -> None:
    config = _tiny_config(entity_bank_size=12, relation_bank_size=6, soft_dim=24, d_model=40)
    module = SoftVocabModule(config)
    assert module.entity_bank.shape == (12, 24)
    assert module.relation_bank.shape == (6, 24)
    assert module.refuse_probe.shape == (24,)
    assert module.entity_bank_size == 12
    assert module.relation_bank_size == 6
    assert module.entity_usage_ema.shape == (12,)
    assert module.relation_usage_ema.shape == (6,)
    assert module.project.in_features == 40
    assert module.project.out_features == 24
    assert module.pair_project.in_features == 48
    assert module.pair_project.out_features == 24


def test_assignment_shapes_batched_and_unbatched() -> None:
    config = _tiny_config()
    module = SoftVocabModule(config)
    module.eval()

    single = torch.randn(7, config.host.d_model)
    out_single = module(single)
    assert isinstance(out_single, SoftVocabOutput)
    assert out_single.assignment.shape == (7, config.arch.entity_bank_size)
    assert out_single.projected.shape == (7, config.arch.soft_dim)
    assert out_single.soft_entities.shape == (7, config.arch.soft_dim)
    assert torch.allclose(out_single.assignment.sum(dim=-1), torch.ones(7), atol=1e-5)

    batch = torch.randn(3, 5, config.host.d_model)
    out_batch = module(batch)
    assert out_batch.assignment.shape == (3, 5, config.arch.entity_bank_size)
    assert out_batch.projected.shape == (3, 5, config.arch.soft_dim)
    assert out_batch.soft_entities.shape == (3, 5, config.arch.soft_dim)
    assert torch.allclose(
        out_batch.assignment.sum(dim=-1),
        torch.ones(3, 5),
        atol=1e-5,
    )


def test_mean_row_entropy_finite_and_bounded() -> None:
    config = _tiny_config(entity_bank_size=8)
    module = SoftVocabModule(config)
    module.eval()
    states = torch.randn(4, 6, config.host.d_model)
    out = module(states)

    assert torch.isfinite(out.mean_row_entropy)
    assert torch.isfinite(out.inventory_entropy)
    assert torch.isfinite(out.diversity_loss)
    assert float(out.inventory_entropy.detach()) >= 0.0
    max_entropy = math.log(config.arch.entity_bank_size)
    entropy = float(out.mean_row_entropy.detach())
    assert 0.0 <= entropy <= max_entropy + 1e-5


def test_utilization_buffers_update_in_train_mode() -> None:
    config = _tiny_config(entity_bank_size=8, utilization_eps=1e-3)
    module = SoftVocabModule(config)
    before = module.entity_usage_ema.clone()

    module.eval()
    _ = module(torch.randn(2, 4, config.host.d_model))
    assert torch.equal(module.entity_usage_ema, before)

    module.train()
    out = module(torch.randn(2, 4, config.host.d_model))
    assert not torch.equal(module.entity_usage_ema, before)
    assert torch.isfinite(out.occupancy)
    assert 0.0 <= float(out.occupancy.detach()) <= 1.0
    assert out.batch_usage.shape == (config.arch.entity_bank_size,)
    assert torch.allclose(out.batch_usage.sum(), torch.tensor(1.0), atol=1e-5)


def test_diversity_loss_backprop_to_banks() -> None:
    config = _tiny_config()
    module = SoftVocabModule(config)
    module.train()
    states = torch.randn(2, 5, config.host.d_model, requires_grad=True)
    out = module(states)
    out.diversity_loss.backward()
    assert module.entity_bank.grad is not None
    assert module.project.weight.grad is not None
    assert torch.isfinite(module.entity_bank.grad).all()


def test_relation_assignment_and_diversity_backprop() -> None:
    config = _tiny_config()
    module = SoftVocabModule(config)
    module.train()
    heads = torch.randn(5, config.arch.soft_dim)
    tails = torch.randn(5, config.arch.soft_dim)
    pair_feat = module.pair_features(heads, tails)
    scored = module.soft_assign_relations(pair_feat)
    rel_assign = scored.typed
    soft_rels = module.soft_relations(rel_assign)
    rel_div, mean_ent, usage = module.relation_diversity_loss(rel_assign)
    soft_ke = module.soft_transe_loss(heads, tails, soft_rels)
    zero = torch.zeros((), device=rel_div.device)
    combined = module.combine_diversity(
        DiversityTerms(vq=zero, inventory=zero, usage_kl=zero, gap=zero, rel_vq=rel_div)
    )
    loss = combined + soft_ke
    loss.backward()
    assert rel_assign.shape == (5, config.arch.relation_bank_size)
    assert soft_rels.shape == (5, config.arch.soft_dim)
    assert torch.allclose(rel_assign.sum(dim=-1) + scored.consumed, torch.ones(5), atol=1e-5)
    assert torch.isfinite(mean_ent)
    assert usage.shape == (config.arch.relation_bank_size,)
    assert module.relation_bank.grad is not None
    assert module.pair_project.weight.grad is not None
    assert torch.isfinite(module.relation_bank.grad).all()
    zero_rel_grad = torch.zeros_like(module.relation_bank.grad)
    assert not torch.allclose(module.relation_bank.grad, zero_rel_grad)


def test_relation_usage_ema_updates_in_train_mode() -> None:
    config = _tiny_config()
    module = SoftVocabModule(config)
    before = module.relation_usage_ema.clone()
    usage = torch.full((config.arch.relation_bank_size,), 1.0 / config.arch.relation_bank_size)
    pair_feat = torch.randn(5, config.arch.soft_dim)
    module.eval()
    n_eval, occ_eval, dead_eval = module.update_relation_usage(usage, pair_feat)
    assert torch.equal(module.relation_usage_ema, before)
    assert int(n_eval.detach().item()) == 0
    assert 0.0 <= float(occ_eval.detach()) <= 1.0
    assert float(dead_eval.detach()) >= 0.0
    module.train()
    skewed = torch.zeros(config.arch.relation_bank_size)
    skewed[0] = 1.0
    n_train, occ_train, dead_train = module.update_relation_usage(skewed, pair_feat)
    assert not torch.equal(module.relation_usage_ema, before)
    assert float(n_train.detach()) >= 0.0
    assert 0.0 <= float(occ_train.detach()) <= 1.0
    assert float(dead_train.detach()) >= 0.0


def test_first_train_forward_seeds_banks() -> None:
    torch.manual_seed(0)
    config = _tiny_config(entity_bank_size=8, relation_bank_size=4, soft_dim=64, d_model=256)
    module = SoftVocabModule(config)
    before_bank = module.entity_bank.detach().clone()
    states = torch.randn(8, 32, config.host.d_model)
    module.eval()
    before = module(states)
    module.train()
    after = module(states)
    assert bool(module._banks_seeded.item())
    assert not torch.allclose(module.entity_bank, before_bank)
    after_h = float(after.mean_row_entropy.detach())
    before_h = float(before.mean_row_entropy.detach())
    assert after_h < before_h


def test_seed_banks_from_projected_breaks_uniform_assignment() -> None:
    torch.manual_seed(0)
    config = _tiny_config(entity_bank_size=8, relation_bank_size=4, soft_dim=64, d_model=256)
    module = SoftVocabModule(config)
    module.eval()
    states = torch.randn(8, 32, config.host.d_model)
    before = module(states)
    before_bank = module.entity_bank.detach().clone()
    module.seed_banks_from_projected(before.projected)
    after = module(states)
    after_h = float(after.mean_row_entropy.detach())
    before_h = float(before.mean_row_entropy.detach())
    assert after_h < before_h
    assert not torch.allclose(module.entity_bank, before_bank)


def test_soft_assign_is_scale_invariant() -> None:
    config = _tiny_config()
    module = SoftVocabModule(config)
    module.eval()
    x = torch.randn(4, config.host.d_model)
    a1, _ = module.soft_assign(x)
    a2, _ = module.soft_assign(x * 7.0)
    assert torch.allclose(a1, a2, atol=1e-5)


def test_seeded_cosine_assign_beats_half_ln_k() -> None:
    torch.manual_seed(0)
    config = SsvTrainConfig(
        host=HostSpec(d_model=256),
        arch=ArchSpec(
            entity_bank_size=8,
            relation_bank_size=4,
            soft_dim=64,
            assign_temperature=0.07,
            relation_temperature=0.07,
        ),
        bank=BankHealthSpec(bank_init_from_batch=True),
    )
    module = SoftVocabModule(config)
    module.eval()
    states = torch.randn(8, 32, config.host.d_model)
    before = module(states)
    module.seed_banks_from_projected(before.projected)
    after = module(states)
    assert float(after.mean_row_entropy.detach()) < 0.5 * math.log(config.arch.entity_bank_size)


def test_seeded_flag_survives_state_dict() -> None:
    torch.manual_seed(0)
    config = _tiny_config(entity_bank_size=8, relation_bank_size=4, soft_dim=64, d_model=256)
    module = SoftVocabModule(config)
    module.train()
    _ = module(torch.randn(4, 16, config.host.d_model))
    loaded = SoftVocabModule(config)
    _ = loaded.load_state_dict(module.state_dict())
    loaded.train()
    bank = loaded.entity_bank.detach().clone()
    _ = loaded(torch.randn(4, 16, config.host.d_model))
    assert bool(loaded._banks_seeded.item())
    assert torch.allclose(loaded.entity_bank, bank)


def test_normalized_entropy_term_is_unit_scale() -> None:
    config = _tiny_config()
    module = SoftVocabModule(config)
    module.eval()
    module.set_entropy_scale(1.0)
    assert module.entropy_scale == pytest.approx(1.0)
    states = torch.randn(3, 5, config.host.d_model)
    out = module(states)
    assert float(out.diversity_loss.detach()) > -2.0


def test_token_usage_entropy_prefers_peaked_balanced_rows() -> None:
    config = SsvTrainConfig(
        host=HostSpec(d_model=32),
        arch=ArchSpec(entity_bank_size=8, relation_bank_size=4, soft_dim=16),
        dual=DualSpec(
            lambda_usage=0.0,
            lambda_codebook=0.0,
            lambda_commit=0.0,
            normalize_assignment_entropy=True,
        ),
    )
    module = SoftVocabModule(config)
    module.eval()
    module.set_entropy_scale(1.0)
    bank = config.arch.entity_bank_size
    peaked = torch.eye(bank)
    uniform = torch.full((bank, bank), 1.0 / bank)
    projected = torch.zeros(bank, config.arch.soft_dim)
    peaked_terms, _, _ = module.diversity_loss(peaked, projected)
    uniform_terms, _, _ = module.diversity_loss(uniform, projected)
    assert float(peaked_terms.inventory.detach()) == pytest.approx(math.log(bank), abs=1e-5)
    assert float(uniform_terms.inventory.detach()) == pytest.approx(math.log(bank), abs=1e-5)
    assert float(peaked_terms.gap.detach()) < float(uniform_terms.gap.detach())


def test_entropy_scale_zero_leaves_inventory_live() -> None:
    config = SsvTrainConfig(
        host=HostSpec(d_model=32),
        arch=ArchSpec(entity_bank_size=8, relation_bank_size=4, soft_dim=16),
        dual=DualSpec(
            lambda_usage=0.0,
            lambda_codebook=0.0,
            lambda_commit=0.0,
            normalize_assignment_entropy=True,
        ),
    )
    module = SoftVocabModule(config)
    module.eval()
    states = torch.randn(2, 6, config.host.d_model)
    module.set_entropy_scale(0.0)
    out = module(states)
    assert float(out.diversity_terms.gap.detach()) == pytest.approx(0.0, abs=1e-5)
    assert float(out.diversity_terms.inventory.detach()) == pytest.approx(
        float(out.inventory_entropy.detach()),
        abs=1e-5,
    )
    assert float(out.inventory_entropy.detach()) > 0.0
    assert float(out.diversity_loss.detach()) == pytest.approx(
        float((out.diversity_terms.vq + out.diversity_terms.rel_vq).detach()),
        abs=1e-5,
    )


def test_inventory_entropy_sparse_doc_used_batch() -> None:
    config = SsvTrainConfig(
        host=HostSpec(d_model=32),
        arch=ArchSpec(entity_bank_size=8, relation_bank_size=4, soft_dim=16),
        dual=DualSpec(
            lambda_usage=0.0,
            lambda_codebook=0.0,
            lambda_commit=0.0,
            normalize_assignment_entropy=True,
        ),
    )
    module = SoftVocabModule(config)
    module.eval()
    module.set_entropy_scale(0.0)
    bank = config.arch.entity_bank_size
    tokens = 4
    sparse = torch.zeros(2, tokens, bank)
    sparse[0, torch.arange(tokens), torch.arange(tokens)] = 1.0
    sparse[1, torch.arange(tokens), torch.arange(tokens, bank)] = 1.0
    smear = torch.zeros(2, bank, bank)
    smear[:, torch.arange(bank), torch.arange(bank)] = 1.0
    projected_sparse = torch.zeros(2, tokens, config.arch.soft_dim)
    projected_smear = torch.zeros(2, bank, config.arch.soft_dim)
    sparse_terms, _, usage_sparse = module.diversity_loss(sparse, projected_sparse)
    smear_terms, _, usage_smear = module.diversity_loss(smear, projected_smear)
    inv_sparse = sparse_terms.inventory
    inv_smear = smear_terms.inventory
    usage_entropy_sparse = float(
        torch.special.entr(usage_sparse.clamp_min(config.dual.log_eps)).sum().detach()
    )
    usage_entropy_smear = float(
        torch.special.entr(usage_smear.clamp_min(config.dual.log_eps)).sum().detach()
    )
    assert float(inv_sparse.detach()) == pytest.approx(math.log(tokens), abs=1e-5)
    assert float(inv_smear.detach()) == pytest.approx(math.log(bank), abs=1e-5)
    assert float(inv_sparse.detach()) < float(inv_smear.detach())
    assert usage_entropy_sparse == pytest.approx(math.log(bank), abs=1e-5)
    assert usage_entropy_smear == pytest.approx(math.log(bank), abs=1e-5)


def test_inventory_entropy_ignores_padded_tokens() -> None:
    config = SsvTrainConfig(
        host=HostSpec(d_model=32),
        arch=ArchSpec(entity_bank_size=8, relation_bank_size=4, soft_dim=16),
        dual=DualSpec(
            lambda_usage=0.0,
            lambda_codebook=0.0,
            lambda_commit=0.0,
        ),
    )
    module = SoftVocabModule(config)
    module.eval()
    module.set_entropy_scale(0.0)
    bank = config.arch.entity_bank_size
    assignment = torch.zeros(1, 4, bank)
    assignment[0, 0, 0] = 1.0
    assignment[0, 1, 1] = 1.0
    assignment[0, 2, :] = 1.0 / bank
    assignment[0, 3, :] = 1.0 / bank
    projected = torch.zeros(1, 4, config.arch.soft_dim)
    token_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    terms, _row, _usage = module.diversity_loss(
        assignment,
        projected,
        token_mask=token_mask,
    )
    assert float(terms.inventory.detach()) == pytest.approx(math.log(2.0), abs=1e-5)


def test_dead_entity_row_is_replaced_live_row_is_not() -> None:
    config = _tiny_config(entity_bank_size=8, utilization_eps=0.05)
    module = SoftVocabModule(config)
    module.train()
    _ = module._banks_seeded.fill_(True)
    dead_row = torch.full((config.arch.soft_dim,), -4.0)
    live_row = module.entity_bank[1].detach().clone()
    with torch.no_grad():
        module.entity_bank[0].copy_(dead_row)
    module.entity_usage_ema.zero_()
    module.entity_usage_ema[0] = 0.0
    module.entity_usage_ema[1:] = 1.0
    pool = torch.ones(3, config.arch.soft_dim)
    n = module.restart_dead_rows(module.entity_bank, module.entity_usage_ema, pool)
    assert int(n.item()) == 1
    expected = F.normalize(pool[0], dim=-1)
    assert torch.allclose(module.entity_bank[0], expected, atol=1e-5)
    assert torch.allclose(module.entity_bank[1], live_row)
    assert float(module.entity_usage_ema[0].item()) == pytest.approx(max(1.0 / 8, 0.05))
    assert float(module.entity_usage_ema[1].item()) == pytest.approx(1.0)


def test_train_forward_restarts_planted_dead_entity_row() -> None:
    config = SsvTrainConfig(
        host=HostSpec(d_model=32),
        arch=ArchSpec(
            entity_bank_size=8,
            relation_bank_size=4,
            soft_dim=16,
            assign_temperature=1.0,
        ),
        bank=BankHealthSpec(
            utilization_eps=0.05,
            usage_ema_momentum=1.0,
            bank_init_from_batch=False,
        ),
    )
    module = SoftVocabModule(config)
    module.train()
    dead_row = torch.full((config.arch.soft_dim,), -5.0)
    live_row = module.entity_bank[3].detach().clone()
    with torch.no_grad():
        module.entity_bank[0].copy_(dead_row)
    module.entity_usage_ema.fill_(1.0)
    module.entity_usage_ema[0] = 0.0
    out = module(torch.randn(2, 4, config.host.d_model))
    assert int(out.n_entity_restarts.item()) == 1
    assert int(out.n_restarts.item()) == 1
    assert int(out.n_dead_before_restart.item()) == 1
    assert float(out.occupancy.detach()) == pytest.approx(7 / 8)
    assert not torch.allclose(module.entity_bank[0], dead_row)
    assert torch.allclose(module.entity_bank[3], live_row)
    post, post_dead = module.occupancy_and_dead(module.entity_usage_ema)
    assert int(post_dead.item()) == 0
    assert float(post.detach()) == pytest.approx(1.0)


def test_occupancy_varies_when_codes_die() -> None:
    config = SsvTrainConfig(
        host=HostSpec(d_model=32),
        arch=ArchSpec(
            entity_bank_size=8,
            relation_bank_size=4,
            soft_dim=16,
            assign_temperature=1.0,
        ),
        bank=BankHealthSpec(
            utilization_eps=0.4,
            usage_ema_momentum=0.99,
            bank_init_from_batch=False,
        ),
    )
    module = SoftVocabModule(config)
    module.train()
    module.entity_usage_ema.zero_()
    module.entity_usage_ema[0] = 1.0
    out = module(torch.randn(2, 4, config.host.d_model))
    assert int(out.n_dead_before_restart.item()) > 0
    assert float(out.occupancy.detach()) < 1.0
    post, _ = module.occupancy_and_dead(module.entity_usage_ema)
    assert float(post.detach()) == pytest.approx(1.0)


def test_eval_forward_does_not_restart_dead_entity_row() -> None:
    config = _tiny_config(entity_bank_size=8, utilization_eps=0.05)
    module = SoftVocabModule(config)
    module.eval()
    dead_row = module.entity_bank[0].detach().clone()
    module.entity_usage_ema.zero_()
    out = module(torch.randn(2, 4, config.host.d_model))
    assert int(out.n_entity_restarts.item()) == 0
    assert int(out.n_restarts.item()) == 0
    assert torch.allclose(module.entity_bank[0], dead_row)


def test_dead_relation_row_is_replaced_live_row_is_not() -> None:
    config = _tiny_config(relation_bank_size=4, utilization_eps=0.05)
    module = SoftVocabModule(config)
    module.train()
    dead_row = torch.full((config.arch.soft_dim,), -6.0)
    live_row = module.relation_bank[1].detach().clone()
    with torch.no_grad():
        module.relation_bank[0].copy_(dead_row)
    module.relation_usage_ema.zero_()
    module.relation_usage_ema[0] = 0.0
    module.relation_usage_ema[1:] = 1.0
    usage = torch.zeros(config.arch.relation_bank_size)
    usage[1] = 1.0
    pool = torch.ones(2, config.arch.soft_dim)
    n, occupancy, n_dead = module.update_relation_usage(usage, pool)
    assert int(n.item()) == 1
    assert int(n_dead.item()) == 1
    assert float(occupancy.detach()) == pytest.approx(3 / 4)
    expected = F.normalize(pool[0], dim=-1)
    assert torch.allclose(module.relation_bank[0], expected, atol=1e-5)
    assert torch.allclose(module.relation_bank[1], live_row)
    assert float(module.relation_usage_ema[0].item()) == pytest.approx(max(1.0 / 4, 0.05))


def test_restart_non_rank0_applies_broadcast_slab(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_config(entity_bank_size=4, utilization_eps=0.05)
    module = SoftVocabModule(config)
    module.train()
    dead_row = torch.full((config.arch.soft_dim,), -7.0)
    live_row = module.entity_bank[1].detach().clone()
    with torch.no_grad():
        module.entity_bank[0].copy_(dead_row)
    module.entity_usage_ema.zero_()
    module.entity_usage_ema[0] = 0.0
    module.entity_usage_ema[1:] = 1.0
    payload = F.normalize(torch.full((1, config.arch.soft_dim), 2.0), dim=-1)

    def fake_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        del src
        if tensor.ndim == 0:
            _ = tensor.fill_(1)
            return tensor
        _ = tensor.copy_(payload)
        return tensor

    monkeypatch.setattr(dist, 'is_available', lambda: True)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(dist, 'broadcast', fake_broadcast)
    n = module.restart_dead_rows(
        module.entity_bank,
        module.entity_usage_ema,
        torch.randn(3, config.arch.soft_dim),
    )
    assert int(n.item()) == 1
    assert torch.allclose(module.entity_bank[0], payload.squeeze(0), atol=1e-5)
    assert torch.allclose(module.entity_bank[1], live_row)


def test_ssv_train_config_has_no_restart_off_flag() -> None:
    assert 'restart_dead_codes' not in SsvTrainConfig.model_fields
    dumped = SsvTrainConfig().model_dump()
    assert 'restart_dead_codes' not in dumped


def test_ssv_train_config_has_no_dual_or_kendall_off_flag() -> None:
    forbidden = (
        'enable_duals',
        'enable_kendall',
        'freeze_kendall',
        'lambda_inventory',
    )
    fields = SsvTrainConfig.model_fields
    dumped = SsvTrainConfig().model_dump()
    for name in forbidden:
        assert name not in fields
        assert name not in dumped
    assert 'inventory_slots' not in DualSpec.model_fields
    assert 'inventory_slots' not in dumped['dual']
    assert dumped['dual']['usage_entropy_ratio'] == pytest.approx(0.95)
    assert dumped['dual']['row_entropy_target'] == pytest.approx(math.log(3.0))


def test_default_ssv_train_config_bank_sizes() -> None:
    config = SsvTrainConfig()
    module = SoftVocabModule(config)
    assert module.entity_bank_size == config.arch.entity_bank_size == 64
    assert module.relation_bank_size == config.arch.relation_bank_size == 32
    assert module.entity_bank.shape == (64, config.arch.soft_dim)
    assert module.relation_bank.shape == (32, config.arch.soft_dim)
