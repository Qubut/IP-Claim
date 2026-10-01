"""Algebraic covering oracle: unpaid-gap sign, anti-shortcuts, fail-closed, saturation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from ip_claim.collision.cover import Covering

pytestmark = pytest.mark.experiment

_SLOTS = 4
_MASS = st.floats(min_value=0.0, max_value=8.0, allow_nan=False, allow_infinity=False, width=64)
_INTENSITY = arrays(dtype=np.float64, shape=(_SLOTS, _SLOTS), elements=_MASS)
_SCALE = st.floats(min_value=0.25, max_value=8.0, allow_nan=False, allow_infinity=False, width=64)
_HAND_QUERY = np.array(
    ((2.0, 0.0, 0.0, 0.0), (0.0, 2.0, 0.0, 0.0), (0.25, 0.0, 0.0, 0.0), (0.0, 0.0, 0.25, 0.0)),
    dtype=np.float64,
)
_HAND_MATCH = np.array(
    ((4.0, 0.0, 0.0, 0.0), (0.0, 4.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    dtype=np.float64,
)
_GRAD_UNUSABLE = 1e-8


class TestAlgebraicOracle:
    """Host-free unpaid-gap identities on the product Covering Factory."""

    def test_matching_supply_lowers_normalized_unpaid(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Adding supply on a demanded slot strictly lowers normalized unpaid."""
        query = torch.tensor([2.0, 0.0], dtype=torch.float64)
        before = covering.unpaid_fraction(
            covering.pair_table(query, torch.tensor([0.5, 0.0], dtype=torch.float64)),
        )
        after = covering.unpaid_fraction(
            covering.pair_table(query, torch.tensor([1.5, 0.0], dtype=torch.float64)),
        )
        request.node.user_properties.extend((
            ('unpaid_frac_before', before.item()),
            ('unpaid_frac_after', after.item()),
        ))
        assert after.item() < before.item()

    def test_unused_slot_supply_does_not_pay(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Supply on a silent slot leaves unpaid mass and covering unchanged."""
        query = torch.tensor([2.0, 0.0], dtype=torch.float64)
        demanded = covering.pair_table(query, torch.tensor([1.0, 0.0], dtype=torch.float64))
        diverted = covering.pair_table(query, torch.tensor([1.0, 8.0], dtype=torch.float64))
        request.node.user_properties.extend((
            ('unpaid_demanded', demanded.unpaid_mass.item()),
            ('unpaid_diverted', diverted.unpaid_mass.item()),
        ))
        assert torch.allclose(demanded.unpaid_mass, diverted.unpaid_mass)
        assert torch.allclose(demanded.covering, diverted.covering)

    @given(query=_INTENSITY, document=_INTENSITY, order=st.permutations(range(_SLOTS)))
    @example(query=_HAND_QUERY, document=_HAND_MATCH, order=[1, 0, 2, 3])
    def test_joint_slot_permutation_invariant(
        self,
        covering: Covering,
        query: NDArray[np.float64],
        document: NDArray[np.float64],
        order: Sequence[int],
        request: pytest.FixtureRequest,
    ) -> None:
        """A shared slot permutation leaves unpaid, covering, and the gap unchanged."""
        demand = torch.as_tensor(query, dtype=torch.float64)
        supply = torch.as_tensor(document, dtype=torch.float64)
        assume(bool(demand.sum() > 0))
        perm = torch.as_tensor(order, dtype=torch.long)
        table = covering.pair_table(demand, supply)
        shuffled = covering.pair_table(demand[:, perm], supply[:, perm])
        gap = covering.normalized_unpaid_gap(table)
        gap_perm = covering.normalized_unpaid_gap(shuffled)
        request.node.user_properties.extend((
            ('gap', 'nan' if torch.isnan(gap) else gap.item()),
            ('gap_permuted', 'nan' if torch.isnan(gap_perm) else gap_perm.item()),
        ))
        assert torch.allclose(table.unpaid_mass, shuffled.unpaid_mass)
        assert torch.allclose(table.covering, shuffled.covering, equal_nan=True)
        assert torch.allclose(table.demand_l1, shuffled.demand_l1)
        assert torch.allclose(gap, gap_perm, equal_nan=True)

    def test_disclosure_only_permutation_raises_diagonal_unpaid(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Permuting only disclosure slots breaks matching payment."""
        query = torch.tensor(((2.0, 0.0), (0.0, 2.0)), dtype=torch.float64)
        matched = torch.tensor(((4.0, 0.0), (0.0, 4.0)), dtype=torch.float64)
        broken = matched[:, torch.tensor((1, 0))]
        match_diag = covering.unpaid_fraction(covering.pair_table(query, matched)).diagonal().mean()
        broken_diag = covering.unpaid_fraction(covering.pair_table(query, broken)).diagonal().mean()
        request.node.user_properties.extend((
            ('diag_unpaid_matched', match_diag.item()),
            ('diag_unpaid_broken', broken_diag.item()),
        ))
        assert broken_diag.item() > match_diag.item()

    @given(query=_INTENSITY, document=_INTENSITY, scale=_SCALE)
    @example(query=_HAND_QUERY, document=_HAND_MATCH, scale=3.0)
    def test_query_mass_scale_invariant_unpaid_fraction(
        self,
        covering: Covering,
        query: NDArray[np.float64],
        document: NDArray[np.float64],
        scale: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """Positive query scale changes raw unpaid mass, not the unpaid fraction."""
        demand = torch.as_tensor(query, dtype=torch.float64)
        supply = torch.as_tensor(document, dtype=torch.float64)
        table = covering.pair_table(demand, supply)
        scaled = covering.pair_table(demand * scale, supply)
        assume(bool((table.demand_l1 > 0).all() and (scaled.demand_l1 > 0).all()))
        frac = covering.unpaid_fraction(table)
        frac_scaled = covering.unpaid_fraction(scaled)
        request.node.user_properties.extend((
            ('unpaid_mean', table.unpaid_mass.mean().item()),
            ('unpaid_scaled_mean', scaled.unpaid_mass.mean().item()),
            ('scale', scale),
        ))
        assert torch.allclose(scaled.unpaid_mass, table.unpaid_mass * scale)
        assert torch.allclose(scaled.covering, table.covering, equal_nan=True)
        assert torch.allclose(frac, frac_scaled, equal_nan=True)

    def test_empty_demand_reducer_is_nan(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Empty demand is NaN on unpaid fraction and on the square gap."""
        empty = covering.pair_table(
            torch.zeros(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
        )
        frac = covering.unpaid_fraction(empty)
        gap = covering.normalized_unpaid_gap(
            covering.pair_table(
                torch.zeros(2, 2, dtype=torch.float64),
                torch.ones(2, 2, dtype=torch.float64),
            ),
        )
        request.node.user_properties.extend((
            ('empty_query_covering', empty.covering.item()),
            ('empty_gap_isnan', bool(torch.isnan(gap))),
        ))
        assert torch.isnan(frac).all()
        assert torch.isnan(gap)
        assert torch.allclose(empty.covering, torch.ones(1, 1, dtype=torch.float64))

    def test_empty_disclosure_leaves_demand_unpaid(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Empty disclosure pays nothing: covering 0, unpaid fraction 1."""
        table = covering.pair_table(
            torch.tensor([2.0, 0.0], dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        )
        frac = covering.unpaid_fraction(table)
        request.node.user_properties.extend((
            ('empty_doc_covering', table.covering.item()),
            ('empty_doc_unpaid_frac', frac.item()),
        ))
        assert torch.allclose(table.covering, torch.zeros(1, 1, dtype=torch.float64))
        assert torch.allclose(frac, torch.ones(1, 1, dtype=torch.float64))

    def test_saturated_disclosure_gradient_is_unusable(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Saturated matching disclosure has a vanishing document-intensity gradient."""
        query = torch.tensor([2.0, 0.0], dtype=torch.float64)

        def document_grad(mass: float) -> float:
            document = torch.tensor([mass, 0.0], dtype=torch.float64, requires_grad=True)
            covering.unpaid_fraction(covering.pair_table(query, document)).squeeze().backward()
            grad = document.grad
            return float('nan') if grad is None else float(grad[0].item())

        unsaturated = document_grad(1.0)
        saturated = document_grad(1.0e6)
        request.node.user_properties.extend((
            ('grad_unsaturated', unsaturated),
            ('grad_saturated', saturated),
        ))
        assert abs(saturated) < _GRAD_UNUSABLE < abs(unsaturated)

    def test_normalized_unpaid_gap_sign_on_hand_matrix(
        self,
        covering: Covering,
        request: pytest.FixtureRequest,
    ) -> None:
        """Hand 2x2: matching gap is 0.8; swapped documents flip the sign."""
        query = torch.tensor(((2.0, 0.0), (0.0, 2.0)), dtype=torch.float64)
        matched = torch.tensor(((4.0, 0.0), (0.0, 4.0)), dtype=torch.float64)
        match_gap = covering.normalized_unpaid_gap(covering.pair_table(query, matched))
        swap_gap = covering.normalized_unpaid_gap(covering.pair_table(query, matched.flip(0)))
        expected = 0.8
        request.node.user_properties.extend((
            ('gap_matched', match_gap.item()),
            ('gap_swapped', swap_gap.item()),
            ('expected_matched', expected),
        ))
        assert match_gap.item() == pytest.approx(expected, abs=1e-12)
        assert match_gap.item() > 0.0
        assert swap_gap.item() < 0.0
