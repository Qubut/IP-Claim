"""Adam ascent on log-lambda duals, clipped to a numerical box."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.optim import Adam


class LogLambdaAdam(nn.Module):
    """Adam on log-lambda; ``lambdas`` are ``exp(log λ)`` after the clip box."""

    log_lambda: nn.ParameterDict
    names: tuple[str, ...]
    lo: float
    hi: float
    _opt: Adam

    def __init__(
        self,
        names: tuple[str, ...],
        lr: float,
        lo: float = -8.0,
        hi: float = 8.0,
    ) -> None:
        super().__init__()
        if not names:
            msg = 'LogLambdaAdam needs at least one dual name'
            raise ValueError(msg)
        if lo >= hi:
            msg = 'log-lambda lo must be below hi'
            raise ValueError(msg)
        self.names = names
        self.lo = float(lo)
        self.hi = float(hi)
        self.log_lambda = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(0.0)) for name in names},
        )
        self._opt = Adam(list(self.log_lambda.parameters()), lr=lr)

    def lambdas(self) -> dict[str, Tensor]:
        """Positive dual weights ``exp(log λ)`` for each named constraint."""
        return {name: self.log_lambda[name].exp() for name in self.names}

    def step(self, residuals: Mapping[str, float]) -> dict[str, bool]:
        """Ascend each log-lambda by its residual; return per-name clip flags."""
        self._opt.zero_grad()
        loss = sum(
            (-self.log_lambda[name] * float(residuals[name]) for name in self.names),
            start=torch.tensor(0.0),
        )
        loss.backward()
        _ = self._opt.step()
        clipped: dict[str, bool] = {}
        with torch.no_grad():
            for name in self.names:
                parameter = self.log_lambda[name]
                _ = parameter.clamp_(self.lo, self.hi)
                value = float(parameter)
                clipped[name] = value <= self.lo or value >= self.hi
        return clipped
