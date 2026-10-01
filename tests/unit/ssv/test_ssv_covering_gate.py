"""Covering-ready train reading: bound direction and slack sign."""

from __future__ import annotations

import math

import pytest

from ip_claim.ssv.covering_gate import (
    DEA_BOUND,
    HOST_BOUND,
    INVENTORY_BOUND,
    ROW_HYGIENE,
    TERMINAL_ACCEPTANCE,
    USAGE_BOUND,
    CoveringTrainReading,
)


def test_inventory_slack_grows_as_document_entropy_falls() -> None:
    ln_k = math.log(256)
    smear = CoveringTrainReading(
        inventory_entropy=4.90,
        ln_k=ln_k,
        usage_entropy=5.20,
        usage_target=0.95 * ln_k,
        row_entropy=math.log(3.0),
    )
    sparse = smear.model_copy(update={'inventory_entropy': 3.04})
    assert smear.inventory_below_ln_k == pytest.approx(ln_k - 4.90)
    assert sparse.inventory_below_ln_k > smear.inventory_below_ln_k
    assert 'upper' in INVENTORY_BOUND
    assert 'fall well below ln K' in INVENTORY_BOUND


def test_usage_slack_is_a_lower_bound() -> None:
    ln_k = math.log(256)
    target = 0.95 * ln_k
    short = CoveringTrainReading(
        inventory_entropy=3.04,
        ln_k=ln_k,
        usage_entropy=4.59,
        usage_target=target,
        row_entropy=math.log(3.0),
    )
    held = short.model_copy(update={'usage_entropy': target})
    assert short.usage_above_target < 0.0
    assert held.usage_above_target == pytest.approx(0.0)
    assert 'lower' in USAGE_BOUND
    assert 'hold near target' in USAGE_BOUND


def test_report_rows_state_bound_and_role() -> None:
    reading = CoveringTrainReading(
        inventory_entropy=3.04,
        ln_k=math.log(256),
        usage_entropy=4.59,
        usage_target=5.20,
        row_entropy=2.85,
        host_nll=1.022,
        host_nll_ceiling=0.912,
        dea_gap=-0.0027,
    )
    rows = {row['arm']: row for row in reading.report_rows('earlier')}
    assert rows['inventory_entropy']['bound'] == INVENTORY_BOUND
    assert rows['inventory_entropy']['role'] == 'leading'
    assert rows['usage_entropy']['bound'] == USAGE_BOUND
    assert rows['host_nll']['bound'] == HOST_BOUND
    assert rows['host_nll']['role'] == 'host_health'
    assert rows['dea_gap']['bound'] == DEA_BOUND
    assert rows['dea_gap']['role'] == 'graph_soundness'
    assert rows['row_entropy']['bound'] == ROW_HYGIENE
    assert rows['row_entropy']['role'] == 'hygiene'
    assert rows['covering_eval']['bound'] == TERMINAL_ACCEPTANCE
    assert rows['covering_eval']['role'] == 'terminal'
    slacks = reading.log_scalars()
    assert slacks['inventory_below_ln_k'] == reading.inventory_below_ln_k
    assert slacks['usage_above_target'] == reading.usage_above_target
    assert slacks['host_nll_above_ceiling'] > 0.0
