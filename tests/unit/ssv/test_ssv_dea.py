"""Entity denoise head, soft CE at hidden nodes, and modal-code gap."""

from __future__ import annotations

import inspect
import warnings
from itertools import starmap
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from unit.ssv.test_ssv_inject import _fixture_batch, _smoke_config

from ip_claim.ssv.config import ArchSpec, HostSpec, SsvTrainConfig
from ip_claim.ssv.dea import EntityDenoiseHead, entity_denoise_terms
from ip_claim.ssv.model import build_soft_trunk
from ip_claim.ssv.soft_graph import build_soft_relation_bundle


def _tiny_config(*, gnn_hidden: int = 8, soft_dim: int = 8, k_e: int = 5) -> SsvTrainConfig:
    return SsvTrainConfig(
        host=HostSpec(d_model=16),
        arch=ArchSpec(
            gnn_hidden=gnn_hidden,
            gnn_heads=2,
            gnn_layers=1,
            soft_dim=soft_dim,
            entity_bank_size=k_e,
            dea_temperature=0.5,
        ),
    )


def test_dea_temperature_is_on_arch_spec() -> None:
    assert SsvTrainConfig().arch.dea_temperature == pytest.approx(0.07)
    assert _tiny_config().arch.dea_temperature == pytest.approx(0.5)


def test_head_log_probs_shape_is_bank_size() -> None:
    config = _tiny_config(gnn_hidden=8, soft_dim=8, k_e=5)
    head = EntityDenoiseHead(config)
    node_states = torch.randn(2, 4, config.arch.gnn_hidden)
    bank = torch.randn(config.arch.entity_bank_size, config.arch.soft_dim)
    log_probs = head(node_states, bank)
    assert log_probs.shape == (2, 4, config.arch.entity_bank_size)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_denoise_query_does_not_accept_input_ids() -> None:
    assert 'input_ids' not in inspect.signature(EntityDenoiseHead.forward).parameters
    assert 'input_ids' not in inspect.signature(entity_denoise_terms).parameters


def test_loss_is_soft_ce_on_hidden_nodes_only() -> None:
    config = _tiny_config()
    head = EntityDenoiseHead(config)
    node_states = torch.randn(1, 3, config.arch.gnn_hidden, requires_grad=True)
    bank = torch.randn(config.arch.entity_bank_size, config.arch.soft_dim)
    teacher = F.softmax(torch.randn(1, 3, config.arch.entity_bank_size), dim=-1)
    teacher.requires_grad_(True)
    labels = torch.tensor([[-100, 4, -100]])
    attention = torch.ones(1, 3, dtype=torch.long)
    node_mask = torch.ones(1, 3, dtype=torch.bool)

    terms = entity_denoise_terms(head, node_states, node_mask, bank, teacher, labels, attention)
    log_probs = head(node_states, bank)
    expected = -(teacher[0, 1].detach() * log_probs[0, 1]).sum()
    assert torch.allclose(terms.dea_loss, expected)

    terms.dea_loss.backward()
    assert teacher.grad is None
    assert node_states.grad is not None

    shifted = teacher.clone().detach()
    codes = torch.arange(config.arch.entity_bank_size, dtype=torch.float)
    shifted[0, 0] = F.softmax(codes, dim=-1)
    shifted[0, 2] = F.softmax(codes.flip(0), dim=-1)
    other = entity_denoise_terms(
        head, node_states.detach(), node_mask, bank, shifted, labels, attention
    )
    assert torch.allclose(other.dea_loss, expected)


def _head_with_log_probs(config: SsvTrainConfig, log_probs: torch.Tensor) -> EntityDenoiseHead:
    class _FixedHead(EntityDenoiseHead):
        def forward(self, node_states: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
            del node_states, bank
            return log_probs

    return _FixedHead(config)


def test_gap_is_top1_minus_modal_baseline() -> None:
    config = _tiny_config(k_e=3)
    k_e = config.arch.entity_bank_size
    log_probs = torch.log(
        torch.tensor([
            [
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.9],
                [0.2, 0.7, 0.1],
            ]
        ])
    )
    teacher = torch.zeros(1, 4, k_e)
    teacher[0, 0, 0] = 1.0
    teacher[0, 1, 1] = 1.0
    teacher[0, 2, 2] = 1.0
    teacher[0, 3, 0] = 1.0
    labels = torch.ones(1, 4, dtype=torch.long)
    attention = torch.ones(1, 4, dtype=torch.long)
    node_mask = torch.ones(1, 4, dtype=torch.bool)
    node_states = torch.zeros(1, 4, config.arch.gnn_hidden)
    bank = torch.zeros(k_e, config.arch.soft_dim)
    terms = entity_denoise_terms(
        _head_with_log_probs(config, log_probs),
        node_states,
        node_mask,
        bank,
        teacher,
        labels,
        attention,
    )
    # Predictions 0,1,2,1 versus gold 0,1,2,0: top-1 = 0.75. Modal gold is 0 (2/4).
    assert float(terms.dea_gap) == pytest.approx(0.75 - 0.5)


def test_padded_tokens_do_not_enter_denoise() -> None:
    config = _tiny_config(k_e=3)
    log_probs = torch.log(
        torch.tensor([
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
            ]
        ])
    )
    teacher = torch.zeros(1, 4, 3)
    teacher[0, 0, 0] = 1.0
    teacher[0, 1, 1] = 1.0
    teacher[0, 2, 2] = 1.0
    teacher[0, 3, 2] = 1.0
    labels = torch.tensor([[3, 3, 3, 3]])
    attention = torch.tensor([[1, 1, 0, 0]])
    node_mask = torch.tensor([[True, True]])
    node_states = torch.zeros(1, 2, config.arch.gnn_hidden)
    bank = torch.zeros(3, config.arch.soft_dim)
    terms = entity_denoise_terms(
        _head_with_log_probs(config, log_probs),
        node_states,
        node_mask,
        bank,
        teacher,
        labels,
        attention,
    )
    expected = -(teacher[0, :2] * log_probs[0]).sum(dim=-1).mean()
    assert torch.allclose(terms.dea_loss, expected)
    # Both packed gold codes are unique and correctly predicted: top-1 1.0, modal 0.5.
    assert float(terms.dea_gap) == pytest.approx(0.5)


def test_token_node_count_mismatch_raises() -> None:
    config = _tiny_config()
    head = EntityDenoiseHead(config)
    node_states = torch.zeros(1, 2, config.arch.gnn_hidden)
    node_mask = torch.ones(1, 2, dtype=torch.bool)
    bank = torch.zeros(config.arch.entity_bank_size, config.arch.soft_dim)
    teacher = torch.zeros(1, 4, config.arch.entity_bank_size)
    labels = torch.full((1, 4), -100)
    attention = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match='do not match'):
        entity_denoise_terms(head, node_states, node_mask, bank, teacher, labels, attention)


def test_empty_node_grid_returns_zero_terms() -> None:
    config = _tiny_config()
    head = EntityDenoiseHead(config)
    node_states = torch.zeros(2, 0, config.arch.gnn_hidden)
    node_mask = torch.zeros(2, 0, dtype=torch.bool)
    bank = torch.randn(config.arch.entity_bank_size, config.arch.soft_dim)
    teacher = F.softmax(torch.randn(2, 5, config.arch.entity_bank_size), dim=-1)
    labels = torch.full((2, 5), 3)
    attention = torch.zeros(2, 5, dtype=torch.long)
    terms = entity_denoise_terms(head, node_states, node_mask, bank, teacher, labels, attention)
    assert float(terms.dea_loss) == pytest.approx(0.0)
    assert float(terms.dea_gap) == pytest.approx(0.0)


def test_empty_hidden_raises_when_compose_nodes_exist() -> None:
    config = _tiny_config()
    head = EntityDenoiseHead(config)
    node_states = torch.randn(2, 3, config.arch.gnn_hidden)
    node_mask = torch.ones(2, 3, dtype=torch.bool)
    bank = torch.randn(config.arch.entity_bank_size, config.arch.soft_dim)
    teacher = F.softmax(torch.randn(2, 3, config.arch.entity_bank_size), dim=-1)
    labels = torch.full((2, 3), -100)
    attention = torch.ones(2, 3, dtype=torch.long)
    with pytest.raises(RuntimeError, match='no hidden positions'):
        entity_denoise_terms(head, node_states, node_mask, bank, teacher, labels, attention)


def test_permuting_hidden_tokens_does_not_change_occupied_codes(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    hidden = batch.labels.ge(0)
    assert hidden.any()
    hidden_ids = batch.unmasked_input_ids[hidden]
    rolled = hidden_ids.roll(1)
    if torch.equal(rolled, hidden_ids):
        rolled = hidden_ids + 1
    permuted = batch.unmasked_input_ids.clone()
    permuted[hidden] = rolled
    assert not torch.equal(permuted[hidden], hidden_ids)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
    model.eval()
    seed = model.host.get_input_embeddings()(batch.unmasked_input_ids)
    model.soft_vocab.ensure_banks_seeded(seed)
    visible = batch.attention_mask * (~hidden).to(dtype=batch.attention_mask.dtype)

    def occupied_code_ids(unmasked_input_ids: Tensor) -> tuple[Tensor, ...]:
        embeds = model.host.get_input_embeddings()(unmasked_input_ids)
        assignment, _ = model.soft_vocab.soft_assign(embeds)
        bundle = build_soft_relation_bundle(
            model.soft_vocab,
            assignment,
            visible,
            mass_floor=model.soft_occupied_floor,
            demand=model.soft_vocab.masked_intensity(assignment, visible),
        )
        return tuple(overlay.code_ids.detach().clone() for overlay in bundle.overlays)

    with torch.no_grad():
        base = occupied_code_ids(batch.unmasked_input_ids)
        shuffled = occupied_code_ids(permuted)
    assert base
    assert all(starmap(torch.equal, zip(base, shuffled, strict=True)))


def test_random_init_dea_gap_does_not_beat_modal_baseline(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    assert batch.labels.ge(0).any()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
    model.eval()
    with torch.no_grad():
        out = model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
        )
    gap = float(out.dea_gap)
    assert gap <= 0, f'untrained dea_gap {gap} beats the modal baseline'
