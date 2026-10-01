"""Adam ascent on log-lambda."""

from __future__ import annotations

import math

import pytest
import torch

from ip_claim.ssv.steer.duals import LogLambdaAdam


def test_log_lambda_adam_raises_when_residual_positive() -> None:
    duals = LogLambdaAdam(('inv',), lr=0.1, lo=-8.0, hi=8.0)
    before = float(duals.log_lambda['inv'].detach())
    clipped = duals.step({'inv': 1.0})
    after = float(duals.log_lambda['inv'].detach())
    assert after > before
    assert clipped['inv'] is False
    assert float(duals.lambdas()['inv'].detach()) == pytest.approx(math.exp(after))


def test_log_lambda_adam_clips_at_hi() -> None:
    names = ('inv', 'use', 'h', 'host')
    duals = LogLambdaAdam(names, lr=1.0, lo=-0.5, hi=0.2)
    with torch.no_grad():
        for name in names:
            duals.log_lambda[name].fill_(0.19)
    clipped = duals.step(dict.fromkeys(names, 10.0))
    for name in names:
        assert float(duals.log_lambda[name].detach()) == pytest.approx(0.2)
        assert clipped[name] is True
