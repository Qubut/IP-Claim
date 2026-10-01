"""Combination residual versus window compose and occupancy sum.

Two partner intensities join by the product of residual gaps. That is
leftover unpaid of owner-indexed window compose for one owner. Occupancy
sum and the cheaper pairwise residual are named fails on the complementary
and shared-field fixtures. Empty demand is NaN. These tests do not train
a trunk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from torch import Tensor

from ip_claim.collision.cover import Covering

pytestmark = pytest.mark.experiment

_SLOTS = 3
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
        min_value=1e-3,
        max_value=8.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
_INTENSITY = arrays(dtype=np.float64, shape=(_SLOTS,), elements=_MASS)
_DEMAND = arrays(dtype=np.float64, shape=(_SLOTS,), elements=_DEMAND_MASS)
_HAND_DEMAND = np.array((1.0, 1.0, 0.0), dtype=np.float64)
_HAND_FIRST = np.array((3.0, 0.0, 0.0), dtype=np.float64)
_HAND_SECOND = np.array((0.0, 3.0, 0.0), dtype=np.float64)
_HAND_THIRD = np.array((0.0, 0.0, 3.0), dtype=np.float64)
_SHARED = np.array((0.0, 2.0, 0.0), dtype=np.float64)


class TestCombinationResidual:
    """Host-free combination residual on ``CollisionContainer.covering``."""

    @staticmethod
    def leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
        """Demand-weighted unpaid fraction of one query against one supply."""
        return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()

    def test_combination_residual_matches_compose_windows_identity(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Two partners as one owner compose to the same leftover unpaid."""
        demand = torch.as_tensor(_HAND_DEMAND, dtype=torch.float64)
        first = torch.as_tensor(_HAND_FIRST, dtype=torch.float64)
        second = torch.as_tensor(_HAND_SECOND, dtype=torch.float64)
        partners = torch.stack((first, second))
        owners = torch.zeros(partners.size(0), dtype=torch.long)
        composed = covering.compose_windows(partners, owners, 1)
        from_windows = covering.unpaid_fraction(covering.pair_table(demand, composed))
        from_partners = covering.combination_residual(demand, partners)
        request.node.user_properties.extend((
            ('from_partners', from_partners.squeeze().item()),
            ('from_windows', from_windows.squeeze().item()),
        ))
        assert torch.allclose(from_partners, from_windows)
        assert torch.allclose(
            covering.combination_residual(torch.eye(_SLOTS, dtype=torch.float64), partners),
            covering.unpaid_fraction(
                covering.pair_table(torch.eye(_SLOTS, dtype=torch.float64), composed)
            ),
        )

    @given(demand=_DEMAND, first=_INTENSITY, second=_INTENSITY)
    @example(demand=_HAND_DEMAND, first=_HAND_FIRST, second=_HAND_SECOND)
    def test_random_partners_match_single_owner_compose(
        self,
        covering: Covering,
        demand: NDArray[np.float64],
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        request: pytest.FixtureRequest,
    ) -> None:
        """Hypothesis pairs stay identical to window compose, including empty demand."""
        query = torch.as_tensor(demand, dtype=torch.float64)
        partners = torch.stack((
            torch.as_tensor(first, dtype=torch.float64),
            torch.as_tensor(second, dtype=torch.float64),
        ))
        owners = torch.zeros(partners.size(0), dtype=torch.long)
        composed = covering.compose_windows(partners, owners, 1)
        from_windows = covering.unpaid_fraction(covering.pair_table(query, composed))
        from_partners = covering.combination_residual(query, partners)
        request.node.user_properties.extend((
            ('demand_l1', float(query.sum().item())),
            ('finite', bool(torch.isfinite(from_partners).all())),
        ))
        assert torch.allclose(from_partners, from_windows, equal_nan=True)

    def test_three_way_fold_is_the_same_product(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """A third partner folds as one more gap, not a new join."""
        demand = torch.as_tensor(_HAND_DEMAND, dtype=torch.float64)
        first = torch.as_tensor(_HAND_FIRST, dtype=torch.float64)
        second = torch.as_tensor(_HAND_SECOND, dtype=torch.float64)
        third = torch.as_tensor(_HAND_THIRD, dtype=torch.float64)
        stacked = torch.stack((first, second, third))
        owners = torch.zeros(stacked.size(0), dtype=torch.long)
        u_stack = covering.combination_residual(demand, stacked)
        composed_pair = covering.compose_windows(
            torch.stack((first, second)),
            torch.zeros(2, dtype=torch.long),
            1,
        )
        u_fold = covering.combination_residual(
            demand,
            torch.cat((composed_pair, third.unsqueeze(0)), dim=0),
        )
        u_windows = covering.unpaid_fraction(
            covering.pair_table(demand, covering.compose_windows(stacked, owners, 1))
        )
        request.node.user_properties.extend((
            ('u_stack', u_stack.squeeze().item()),
            ('u_fold', u_fold.squeeze().item()),
            ('u_windows', u_windows.squeeze().item()),
        ))
        assert torch.allclose(u_stack, u_fold)
        assert torch.allclose(u_stack, u_windows)

    def test_min_pairwise_fails_on_complementary_slots(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Each partner pays a different demanded slot. The cheaper residual stays high."""
        demand = torch.as_tensor(_HAND_DEMAND, dtype=torch.float64)
        first = torch.as_tensor(_HAND_FIRST, dtype=torch.float64)
        second = torch.as_tensor(_HAND_SECOND, dtype=torch.float64)
        unpaid_first = self.leftover(covering, demand, first)
        unpaid_second = self.leftover(covering, demand, second)
        cheaper = torch.minimum(unpaid_first, unpaid_second)
        combined = covering.combination_residual(demand, torch.stack((first, second))).squeeze()
        request.node.user_properties.extend((
            ('unpaid_first', unpaid_first.item()),
            ('unpaid_second', unpaid_second.item()),
            ('min_pairwise', cheaper.item()),
            ('combination', combined.item()),
        ))
        assert cheaper.item() == pytest.approx(unpaid_first.item(), abs=1e-12)
        assert cheaper.item() > combined.item()
        assert combined.item() < unpaid_first.item()
        assert combined.item() < unpaid_second.item()

    def test_union_covering_fails_on_shared_field_mass(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Additive mass is not the gap product when both partners occupy one slot."""
        demand = torch.as_tensor(_HAND_DEMAND, dtype=torch.float64)
        shared = torch.as_tensor(_SHARED, dtype=torch.float64)
        combined = covering.combination_residual(demand, torch.stack((shared, shared))).squeeze()
        summed = covering.union_covering(demand, shared, shared)
        u_sum = self.leftover(covering, demand, shared + shared)
        cheaper = torch.minimum(
            self.leftover(covering, demand, shared),
            self.leftover(covering, demand, shared),
        )
        request.node.user_properties.extend((
            ('combination', combined.item()),
            ('union_unpaid', u_sum.item()),
            ('min_pairwise', cheaper.item()),
            ('union_score_unpaid', (summed.unpaid_mass / summed.demand_l1).item()),
        ))
        assert u_sum.item() == pytest.approx(
            (summed.unpaid_mass / summed.demand_l1).item(),
            abs=1e-12,
        )
        assert combined.item() != pytest.approx(u_sum.item(), abs=1e-8)
        assert combined.item() != pytest.approx(cheaper.item(), abs=1e-8)
        assert cheaper.item() > combined.item()

    def test_empty_demand_combination_residual_is_nan(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Empty demand stays NaN on the combination residual."""
        empty = torch.zeros(_SLOTS, dtype=torch.float64)
        first = torch.as_tensor(_HAND_FIRST, dtype=torch.float64)
        second = torch.as_tensor(_HAND_SECOND, dtype=torch.float64)
        frac = covering.combination_residual(empty, torch.stack((first, second)))
        request.node.user_properties.extend((('frac_isnan', bool(torch.isnan(frac).all())),))
        assert torch.isnan(frac).all()
        assert not torch.isfinite(frac).any()
