"""Leftover unpaid identities on synthetic demand and partner supply.

The product Covering Factory already owns the Hadamard residual. These
tests lock coordinate monotonicity, the gap formula, the full-support
discrimination bound, and empty-demand NaN. They do not train a trunk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from torch import Tensor

from ip_claim.collision.cover import Covering

pytestmark = pytest.mark.experiment

_SLOTS = 3
_DEMAND_FLOOR = 1e-3
_MASS = st.floats(
    min_value=0.0,
    max_value=8.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_DEMAND_MASS = st.one_of(
    st.just(0.0),
    st.floats(
        min_value=_DEMAND_FLOOR,
        max_value=8.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
_UNIT = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_ALPHA = st.floats(
    min_value=0.0,
    max_value=2.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_SLACK = st.floats(
    min_value=0.05,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_INTENSITY = arrays(dtype=np.float64, shape=(_SLOTS,), elements=_MASS)
_DEMAND = arrays(dtype=np.float64, shape=(_SLOTS,), elements=_DEMAND_MASS)
_HAND_DEMAND = np.array((2.0, 1.0, 0.0), dtype=np.float64)
_HAND_P_X = np.array((1.0, 1.0, 7.0), dtype=np.float64)
_HAND_P_A = np.array((0.0, 0.0, 100.0), dtype=np.float64)
_SUBSET_DEMAND = np.array((1.0, 1.0), dtype=np.float64)
_SUBSET_P_X = np.array((1.0, 0.0), dtype=np.float64)
_SUBSET_P_A = np.array((0.0, 100.0), dtype=np.float64)


class TestLeftoverUnpaidIdentities:
    """Host-free leftover unpaid on ``CollisionContainer.covering``."""

    @staticmethod
    def leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
        """Demand-weighted unpaid fraction of one query against one supply."""
        return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()

    @staticmethod
    def bound_delta(sigma: Tensor, alpha: float, beta: float) -> Tensor:
        """Full-support leftover drop when every demanded slot meets the bounds."""
        scale = sigma.to(dtype=torch.float64)
        return scale * (beta - alpha) / ((beta + scale) * (alpha + scale))

    def test_unpaid_strictly_decreases_on_demanded_slots(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Raising supply on a demanded slot strictly lowers leftover unpaid."""
        demand = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
        supply = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        unpaid = self.leftover(covering, demand, supply)
        unpaid.backward()
        scale = covering.sigma.to(dtype=torch.float64)
        expected = -(demand / demand.sum()) * scale / (supply.detach() + scale).square()
        grad = supply.grad
        assert grad is not None
        request.node.user_properties.extend((
            ('partial_demanded', grad[0].item()),
            ('partial_silent', grad[2].item()),
        ))
        assert torch.allclose(grad, expected)
        assert grad[0].item() < 0.0
        assert grad[1].item() < 0.0
        assert grad[2].item() == pytest.approx(0.0, abs=1e-12)
        bumped = self.leftover(
            covering,
            demand,
            supply.detach() + torch.tensor([0.25, 0.0, 0.0]),
        )
        assert bumped.item() < unpaid.item()

    def test_unpaid_constant_on_silent_slots(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Supply on a zero-demand slot leaves leftover unpaid unchanged."""
        demand = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
        paid = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
        diverted = torch.tensor([1.0, 2.0, 8.0], dtype=torch.float64)
        base = self.leftover(covering, demand, paid)
        extra = self.leftover(covering, demand, diverted)
        scored = covering(demand, paid)
        diverted_score = covering(demand, diverted)
        request.node.user_properties.extend((
            ('unpaid_base', base.item()),
            ('unpaid_diverted', extra.item()),
        ))
        assert torch.allclose(base, extra)
        assert torch.allclose(scored.residual, diverted_score.residual)
        assert torch.allclose(scored.covering, diverted_score.covering)

    @given(demand=_DEMAND, partner_x=_INTENSITY, partner_a=_INTENSITY)
    @example(demand=_HAND_DEMAND, partner_x=_HAND_P_X, partner_a=_HAND_P_A)
    def test_gap_identity_matches_weighted_presence_increment(
        self,
        covering: Covering,
        demand: NDArray[np.float64],
        partner_x: NDArray[np.float64],
        partner_a: NDArray[np.float64],
        request: pytest.FixtureRequest,
    ) -> None:
        """Difference of leftover unpaid is the demand-weighted presence increment."""
        query = torch.as_tensor(demand, dtype=torch.float64)
        assume(bool(query.sum() >= _DEMAND_FLOOR))
        p_x = torch.as_tensor(partner_x, dtype=torch.float64)
        p_a = torch.as_tensor(partner_a, dtype=torch.float64)
        unpaid_x = self.leftover(covering, query, p_x)
        unpaid_a = self.leftover(covering, query, p_a)
        scale = covering.sigma.to(dtype=query.dtype)
        weights = query / query.sum()
        increment = weights * scale * (p_x - p_a) / ((p_x + scale) * (p_a + scale))
        scored_x = covering(query, p_x)
        scored_a = covering(query, p_a)
        residual_gap = (scored_a.residual - scored_x.residual).sum() / scored_x.demand_l1
        request.node.user_properties.extend((
            ('unpaid_x', unpaid_x.item()),
            ('unpaid_a', unpaid_a.item()),
            ('increment_sum', increment.sum().item()),
        ))
        assert torch.allclose(unpaid_a - unpaid_x, increment.sum())
        assert torch.allclose(unpaid_a - unpaid_x, residual_gap)
        assert torch.allclose(
            unpaid_x,
            scored_x.unpaid_mass / scored_x.demand_l1,
        )

    @given(
        demand=_DEMAND,
        alpha=_ALPHA,
        slack=_SLACK,
        unit=arrays(dtype=np.float64, shape=(_SLOTS,), elements=_UNIT),
        extra=arrays(dtype=np.float64, shape=(_SLOTS,), elements=_MASS),
        extra_x=arrays(dtype=np.float64, shape=(_SLOTS,), elements=_MASS),
    )
    @example(
        demand=_HAND_DEMAND,
        alpha=0.0,
        slack=1.0,
        unit=np.zeros(_SLOTS, dtype=np.float64),
        extra=_HAND_P_A,
        extra_x=np.zeros(_SLOTS, dtype=np.float64),
    )
    def test_full_support_bounds_guarantee_named_delta(
        self,
        covering: Covering,
        demand: NDArray[np.float64],
        alpha: float,
        slack: float,
        unit: NDArray[np.float64],
        extra: NDArray[np.float64],
        extra_x: NDArray[np.float64],
        request: pytest.FixtureRequest,
    ) -> None:
        """When every demanded slot meets the payment bounds, leftover drops by delta."""
        query = torch.as_tensor(demand, dtype=torch.float64)
        assume(bool(query.sum() >= _DEMAND_FLOOR))
        beta = alpha + slack
        demanded = query >= _DEMAND_FLOOR
        p_x = torch.where(
            demanded,
            torch.as_tensor(beta + extra_x, dtype=torch.float64),
            torch.as_tensor(extra_x, dtype=torch.float64),
        )
        p_a = torch.where(
            demanded,
            torch.as_tensor(alpha * unit, dtype=torch.float64),
            torch.as_tensor(extra, dtype=torch.float64),
        )
        unpaid_x = self.leftover(covering, query, p_x)
        unpaid_a = self.leftover(covering, query, p_a)
        delta = self.bound_delta(covering.sigma, alpha, beta)
        request.node.user_properties.extend((
            ('unpaid_x', unpaid_x.item()),
            ('unpaid_a', unpaid_a.item()),
            ('delta', float(delta.item())),
            ('alpha', alpha),
            ('beta', beta),
        ))
        assert unpaid_x.item() <= unpaid_a.item() - float(delta.item()) + 1e-12
        scored_x = covering(query, p_x)
        assert torch.allclose(
            unpaid_x,
            scored_x.residual.sum() / scored_x.demand_l1,
        )

    def test_hand_full_support_meets_exact_delta(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """L=(2,1,0), alpha=0, beta=1 meets the bound with equality on demanded slots."""
        demand = torch.as_tensor(_HAND_DEMAND, dtype=torch.float64)
        p_x = torch.as_tensor(_HAND_P_X, dtype=torch.float64)
        p_a = torch.as_tensor(_HAND_P_A, dtype=torch.float64)
        unpaid_x = self.leftover(covering, demand, p_x)
        unpaid_a = self.leftover(covering, demand, p_a)
        delta = self.bound_delta(covering.sigma, 0.0, 1.0)
        request.node.user_properties.extend((
            ('unpaid_x', unpaid_x.item()),
            ('unpaid_a', unpaid_a.item()),
            ('delta', float(delta.item())),
            ('sigma', float(covering.sigma.item())),
        ))
        assert float(covering.sigma.item()) == pytest.approx(1.0)
        assert unpaid_x.item() == pytest.approx(0.5, abs=1e-12)
        assert unpaid_a.item() == pytest.approx(1.0, abs=1e-12)
        assert (unpaid_a - unpaid_x).item() == pytest.approx(float(delta.item()), abs=1e-12)

    def test_strict_payment_makes_leftover_strict(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """A demanded slot above the lower payment bound makes leftover strictly smaller."""
        demand = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
        p_x = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
        p_a = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        unpaid_x = self.leftover(covering, demand, p_x)
        unpaid_a = self.leftover(covering, demand, p_a)
        delta = self.bound_delta(covering.sigma, 0.0, 1.0)
        request.node.user_properties.extend((
            ('unpaid_x', unpaid_x.item()),
            ('unpaid_a', unpaid_a.item()),
            ('delta', float(delta.item())),
        ))
        assert unpaid_x.item() < unpaid_a.item() - float(delta.item())

    def test_proper_subset_support_is_outside_full_demand_bounds(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """L=(1,1), P_X=(1,0), P_A=(0,100) inverts leftover; it is not a delta win."""
        demand = torch.as_tensor(_SUBSET_DEMAND, dtype=torch.float64)
        p_x = torch.as_tensor(_SUBSET_P_X, dtype=torch.float64)
        p_a = torch.as_tensor(_SUBSET_P_A, dtype=torch.float64)
        unpaid_x = self.leftover(covering, demand, p_x)
        unpaid_a = self.leftover(covering, demand, p_a)
        false_delta = self.bound_delta(covering.sigma, 0.0, 1.0)
        x_floor = torch.min(p_x)
        a_ceil = torch.max(p_a)
        request.node.user_properties.extend((
            ('unpaid_x', unpaid_x.item()),
            ('unpaid_a', unpaid_a.item()),
            ('false_delta', float(false_delta.item())),
            ('x_floor', float(x_floor.item())),
            ('a_ceil', float(a_ceil.item())),
        ))
        assert float(covering.sigma.item()) == pytest.approx(1.0)
        assert unpaid_x.item() == pytest.approx(0.75, abs=1e-12)
        assert unpaid_a.item() == pytest.approx(0.5 + 1.0 / 202.0, abs=1e-12)
        assert unpaid_x.item() > unpaid_a.item()
        assert unpaid_x.item() > unpaid_a.item() - float(false_delta.item())
        assert float(x_floor.item()) <= float(a_ceil.item())

    def test_empty_demand_unpaid_fraction_is_nan(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Empty demand is NaN on unpaid_fraction. The residual score is not a paid zero."""
        empty = torch.zeros(3, dtype=torch.float64)
        supply = torch.ones(3, dtype=torch.float64)
        table = covering.pair_table(empty, supply)
        frac = covering.unpaid_fraction(table)
        scored = covering(empty, supply)
        request.node.user_properties.extend((
            ('empty_covering', scored.covering.item()),
            ('empty_unpaid_mass', scored.unpaid_mass.item()),
            ('frac_isnan', bool(torch.isnan(frac).all())),
        ))
        assert torch.isnan(frac).all()
        assert torch.allclose(scored.unpaid_mass, torch.zeros_like(scored.unpaid_mass))
        assert torch.allclose(scored.demand_l1, torch.zeros_like(scored.demand_l1))
        assert torch.allclose(scored.covering, torch.ones_like(scored.covering))
        assert not torch.isfinite(frac).any()
