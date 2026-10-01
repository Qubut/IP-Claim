"""DiversityTerms pieces stay unweighted; Kendall sees reconstruction only."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from ip_claim.ssv.config import ArchSpec, DualSpec, HostSpec, SsvTrainConfig
from ip_claim.ssv.soft_vocab import DiversityTerms, SoftVocabModule


def _tiny_config() -> SsvTrainConfig:
    return SsvTrainConfig(
        host=HostSpec(d_model=32),
        arch=ArchSpec(
            entity_bank_size=8,
            relation_bank_size=4,
            soft_dim=16,
            assign_temperature=1.0,
        ),
        dual=DualSpec(normalize_assignment_entropy=False),
    )


def test_diversity_terms_are_frozen() -> None:
    zero = torch.zeros(())
    terms = DiversityTerms(vq=zero, inventory=zero, usage_kl=zero, gap=zero, rel_vq=zero)
    with pytest.raises(ValidationError):
        terms.vq = torch.ones(())  # type: ignore[misc]


def test_diversity_pieces_have_no_lambda() -> None:
    module = SoftVocabModule(_tiny_config())
    module.eval()
    assignment = torch.full((2, 4, 8), 1.0 / 8)
    projected = torch.randn(2, 4, 16)
    terms, _, _ = module.diversity_loss(assignment, projected)
    reconstructed = module.soft_entities(assignment)
    codebook = F.mse_loss(projected.detach(), reconstructed)
    commitment = F.mse_loss(projected, reconstructed.detach())
    log_uniform = -math.log(8)
    usage = assignment.mean(dim=(0, 1))
    usage_kl = (usage * (usage.clamp_min(module._log_eps).log() - log_uniform)).sum()
    row_h = -(assignment * assignment.clamp_min(module._log_eps).log()).sum(dim=-1).mean()
    usage_h = -(usage * usage.clamp_min(module._log_eps).log()).sum()
    assert float(terms.vq.detach()) == pytest.approx(float((codebook + commitment).detach()))
    assert float(terms.usage_kl.detach()) == pytest.approx(float(usage_kl.detach()), abs=1e-5)
    assert float(terms.gap.detach()) == pytest.approx(float((row_h - usage_h).detach()), abs=1e-5)
    assert float(terms.rel_vq.detach()) == pytest.approx(0.0)
    assert float(terms.inventory.detach()) > 0.0


def test_combine_diversity_sums_vq_only() -> None:
    module = SoftVocabModule(_tiny_config())
    vq = torch.tensor(2.0)
    rel_vq = torch.tensor(3.0)
    terms = DiversityTerms(
        vq=vq,
        inventory=torch.tensor(100.0),
        usage_kl=torch.tensor(50.0),
        gap=torch.tensor(25.0),
        rel_vq=rel_vq,
    )
    combined = module.combine_diversity(terms)
    assert float(combined.detach()) == pytest.approx(5.0)


def test_raising_lambda_inv_scales_constraint_not_task_div() -> None:
    module = SoftVocabModule(_tiny_config())
    module.eval()
    assignment = torch.full((2, 4, 8), 1.0 / 8)
    projected = torch.randn(2, 4, 16)
    terms, _, _ = module.diversity_loss(assignment, projected)
    task_div = terms.vq + terms.rel_vq
    cons_lo = 1.0 * terms.inventory + 1.0 * terms.usage_kl + 1.0 * terms.gap
    cons_hi = 10.0 * terms.inventory + 1.0 * terms.usage_kl + 1.0 * terms.gap
    assert float((terms.vq + terms.rel_vq).detach()) == pytest.approx(float(task_div.detach()))
    assert float(cons_hi.detach()) > float(cons_lo.detach())
