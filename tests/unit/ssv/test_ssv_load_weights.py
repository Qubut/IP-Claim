"""SSV Lightning payload load, model prefix, and HGT key align."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import torch
from returns.result import Failure, Success
from torch import Tensor, nn

from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.load_weights import (
    align_hgt_key,
    load_lightning_payload,
    load_model_weights,
    resolve_fit_ckpt,
    take_model_prefix,
)
from ip_claim.ssv.soft_vocab import SoftVocabModule


def test_align_hgt_key_inserts_conv_after_convs_index() -> None:
    stale = 'model.encoder.convs.0.lin.weight'
    aligned = 'model.encoder.convs.0.conv.lin.weight'
    assert align_hgt_key(stale) == aligned
    assert align_hgt_key(aligned) == aligned
    assert align_hgt_key('kendall_s_mlm') == 'kendall_s_mlm'


def test_take_model_prefix_keeps_trunk_tensors() -> None:
    state = {
        'model.soft_vocab.entity_bank': torch.ones(2, 2),
        'kendall_s_mlm': torch.zeros(()),
        'log_lambda_inv': torch.ones(()),
    }
    taken = take_model_prefix(state)
    assert set(taken) == {'soft_vocab.entity_bank'}
    assert torch.equal(taken['soft_vocab.entity_bank'], torch.ones(2, 2))


def test_load_lightning_payload_missing_file_is_failure(tmp_path: Path) -> None:
    missing = tmp_path / 'absent.ckpt'
    loaded = load_lightning_payload(missing)
    assert isinstance(loaded, Failure)


def test_load_lightning_payload_reads_state_dict(tmp_path: Path) -> None:
    path = tmp_path / 'ssv.ckpt'
    state = {'model.encoder.convs.0.lin.weight': torch.ones(1)}
    torch.save({'state_dict': state, 'epoch': 1}, path)
    loaded = load_lightning_payload(path)
    assert isinstance(loaded, Success)
    assert set(loaded.unwrap()) == set(state)


def test_load_lightning_payload_without_state_dict_is_failure(tmp_path: Path) -> None:
    path = tmp_path / 'raw.ckpt'
    torch.save({'epoch': 1}, path)
    loaded = load_lightning_payload(path)
    assert isinstance(loaded, Failure)


class _Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.soft_vocab = SoftVocabModule(SsvTrainConfig())


class _Steer(nn.Module):
    model: _Trunk
    kendall_s_mlm: nn.Parameter
    log_lambda_inv: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.model = _Trunk()
        self.kendall_s_mlm = nn.Parameter(torch.tensor(0.25))
        self.log_lambda_inv = nn.Buffer(torch.tensor(-1.5))


def test_load_model_weights_keeps_steer_fresh_and_seeds_banks(tmp_path: Path) -> None:
    job = SsvTrainConfig()
    bank = torch.full((job.arch.entity_bank_size, job.arch.soft_dim), 7.0)
    path = tmp_path / 'v16.ckpt'
    torch.save(
        {
            'state_dict': {
                'model.soft_vocab.entity_bank': bank,
                'model.soft_vocab.relation_bank': torch.full(
                    (job.arch.relation_bank_size, job.arch.soft_dim),
                    3.0,
                ),
                'kendall_s_mlm': torch.tensor(9.0),
                'log_lambda_inv': torch.tensor(8.0),
            },
            'optimizer_states': [{'param_groups': [{'lr': 1e-8}]}],
        },
        path,
    )
    steer = _Steer()
    loaded = load_model_weights(steer.model, path)
    assert isinstance(loaded, Success)
    assert torch.allclose(steer.model.soft_vocab.entity_bank, bank)
    assert bool(steer.model.soft_vocab._banks_seeded.item())
    assert torch.equal(steer.kendall_s_mlm, torch.tensor(0.25))
    assert torch.equal(steer.log_lambda_inv, torch.tensor(-1.5))


def test_load_model_weights_drops_shape_mismatch_and_copies_lora(tmp_path: Path) -> None:
    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin_dict = nn.ModuleDict({
                'cpc': nn.Linear(16, 128, bias=False),
                'claim': nn.Linear(16, 128, bias=False),
            })

    class Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_B = nn.Parameter(torch.zeros(4, 4))

    class Trunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.host = Host()

    trunk = Trunk()
    lin_cpc = cast(nn.Linear, trunk.encoder.lin_dict['cpc'])
    lin_claim = cast(nn.Linear, trunk.encoder.lin_dict['claim'])
    init_cpc = lin_cpc.weight.detach().clone()
    init_claim = lin_claim.weight.detach().clone()
    path = tmp_path / 'clash.ckpt'
    trained = torch.full((4, 4), 0.7)
    torch.save(
        {
            'state_dict': {
                'model.encoder.lin_dict.cpc.weight': torch.ones(128, 1),
                'model.encoder.lin_dict.claim.weight': torch.ones(128, 1),
                'model.host.lora_B': trained,
            }
        },
        path,
    )
    loaded = load_model_weights(trunk, path)
    assert isinstance(loaded, Success)
    loaded_b = trunk.host.lora_B.detach()
    assert float(loaded_b.abs().max()) > 0.0
    assert torch.equal(loaded_b, trained)
    assert torch.equal(lin_cpc.weight.detach(), init_cpc)
    assert torch.equal(lin_claim.weight.detach(), init_claim)
    assert tuple(lin_cpc.weight.shape) == (128, 16)
    assert tuple(lin_claim.weight.shape) == (128, 16)


def test_load_model_weights_skips_module_extra_state_dict(tmp_path: Path) -> None:
    class Ingress(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2, 2))
            self._root = ''

        def get_extra_state(self) -> dict[str, object]:
            return {'termhood_root': self._root}

        def set_extra_state(self, state: object) -> None:
            if isinstance(state, dict):
                self._root = str(state.get('termhood_root', ''))

    class Trunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ingress = Ingress()

    path = tmp_path / 'extra.ckpt'
    trained = torch.full((2, 2), 0.4)
    torch.save(
        {
            'state_dict': {
                'model.ingress.w': trained,
                'model.ingress._extra_state': {'termhood_root': '/outputs/ate'},
            }
        },
        path,
    )
    trunk = Trunk()
    loaded = load_model_weights(trunk, path)
    assert isinstance(loaded, Success)
    assert torch.equal(trunk.ingress.w.detach(), trained)
    assert not trunk.ingress._root


def test_load_model_weights_without_banks_leaves_seed_flag(tmp_path: Path) -> None:
    path = tmp_path / 'host_only.ckpt'
    torch.save({'state_dict': {'model.missing_head.weight': torch.ones(2, 2)}}, path)
    trunk = _Trunk()
    assert not bool(trunk.soft_vocab._banks_seeded.item())
    loaded = load_model_weights(trunk, path)
    assert isinstance(loaded, Success)
    assert not bool(trunk.soft_vocab._banks_seeded.item())


def test_resolve_fit_ckpt_ignores_last_without_contract_ema(tmp_path: Path) -> None:
    last = tmp_path / 'last.ckpt'
    torch.save({'state_dict': {'kendall_s_mlm': torch.zeros(())}}, last)
    assert resolve_fit_ckpt(last_ckpt=last, init_weights=None) is None


def test_resolve_fit_ckpt_uses_this_contract_last_when_init_unset(tmp_path: Path) -> None:
    last = tmp_path / 'last.ckpt'
    torch.save({'state_dict': {'inventory_entropy_ema': torch.zeros(())}}, last)
    assert resolve_fit_ckpt(last_ckpt=last, init_weights=None) == last


def test_resolve_fit_ckpt_prefers_newer_last_over_init_weights(tmp_path: Path) -> None:
    init = tmp_path / 'v16.ckpt'
    last = tmp_path / 'last.ckpt'
    torch.save({'state_dict': {'model.x': torch.zeros(1)}}, init)
    torch.save({'state_dict': {'inventory_entropy_ema': torch.zeros(())}}, last)
    os.utime(init, (1, 1))
    os.utime(last, (2, 2))
    assert resolve_fit_ckpt(last_ckpt=last, init_weights=init) == last


def test_resolve_fit_ckpt_ignores_older_last_when_init_weights_set(tmp_path: Path) -> None:
    init = tmp_path / 'v16.ckpt'
    last = tmp_path / 'last.ckpt'
    torch.save({'state_dict': {'model.x': torch.zeros(1)}}, init)
    torch.save({'state_dict': {'inventory_entropy_ema': torch.zeros(())}}, last)
    os.utime(last, (1, 1))
    os.utime(init, (2, 2))
    assert resolve_fit_ckpt(last_ckpt=last, init_weights=init) is None
