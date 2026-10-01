"""Covering fixture keeps the prod Lightning warm-start and loads LoRA."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from returns.result import Success
from torch import nn

from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.load_weights import load_model_weights
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = pytest.mark.experiment

_PROD_YAML = Path(__file__).resolve().parents[2] / 'configs' / 'ssv_train.prod.yaml'
_PROD_INIT = SsvTrainConfig.from_yaml(_PROD_YAML).runtime.init_weights
_PROD_CKPT = Path(_PROD_INIT) if _PROD_INIT else Path()


class TestCoveringWarmStart:
    """Prod checkpoint path survives the covering overlay and fills LoRA."""

    def test_ssv_job_keeps_prod_init_weights(self, ssv_job: SsvTrainConfig) -> None:
        """Covering overlay leaves the shipped warm-start path intact."""
        assert ssv_job.runtime.init_weights == _PROD_INIT
        assert _PROD_INIT == '/outputs/ssv/ssv-step009000.ckpt'

    def test_load_model_weights_fills_lora_b(self, tmp_path: Path) -> None:
        """Lightning model-prefix load populates a zero-init LoRA B slot."""

        class Host(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lora_B = nn.Parameter(torch.zeros(4, 4))

        class Trunk(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.host = Host()

        trunk = Trunk()
        path = tmp_path / 'warm.ckpt'
        trained = torch.ones(4, 4)
        torch.save({'state_dict': {'model.host.lora_B': trained}}, path)
        loaded = load_model_weights(trunk, path)
        assert isinstance(loaded, Success)
        loaded_b = trunk.host.lora_B.detach()
        assert float(loaded_b.abs().max()) > 0.0
        assert torch.equal(loaded_b, trained)

    def test_load_model_weights_drops_shape_mismatch(self, tmp_path: Path) -> None:
        """Shape-clashing lin_dict rows stay at init; matching LoRA B copies."""

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

    @pytest.mark.skipif(
        not _PROD_CKPT.is_file(),
        reason='prod Lightning warm-start is not on this host',
    )
    def test_ssv_model_lora_b_is_trained(self, ssv_model: SoftTrunkModel) -> None:
        """Session trunk LoRA B is nonzero after the job warm-start load."""
        weights = tuple(
            parameter.detach()
            for name, parameter in ssv_model.named_parameters()
            if 'lora_B' in name
        )
        assert weights
        assert any(float(weight.float().abs().max()) > 0.0 for weight in weights)
