"""Cheng mix rescale: divide the Lagrange sum by one plus the dual total."""

from __future__ import annotations

import math

import pytest
import torch

from ip_claim.ssv.steer.cheng import cheng_rescale


def test_cheng_rescale_divides_by_one_plus_lambda_sum() -> None:
    task = torch.tensor(4.0)
    constraint = torch.tensor(2.0)
    lambdas = torch.tensor([1.0, 3.0])
    mix, scale = cheng_rescale(task, constraint, lambdas)
    assert float(scale) == pytest.approx(5.0)
    assert float(mix) == pytest.approx(6.0 / 5.0)


def test_cheng_rescale_is_identity_when_duals_are_zero() -> None:
    task = torch.tensor(1.5)
    constraint = torch.tensor(0.5)
    mix, scale = cheng_rescale(task, constraint, torch.zeros(3))
    assert float(scale) == pytest.approx(1.0)
    assert float(mix) == pytest.approx(2.0)


def test_cheng_rescale_stays_finite_at_large_lambda() -> None:
    task = torch.tensor(10.0)
    constraint = torch.tensor(1.0e6)
    lambdas = torch.full((4,), math.exp(8.0))
    mix, scale = cheng_rescale(task, constraint, lambdas)
    assert torch.isfinite(mix)
    assert torch.isfinite(scale)
    assert float(scale) == pytest.approx(1.0 + 4.0 * math.exp(8.0))
    assert float(mix) == pytest.approx(float((task + constraint) / scale))
