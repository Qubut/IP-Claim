"""Isolated-block covering steps on the hygiene draw. No checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from pydantic import TypeAdapter, ValidationError
from torch.optim import AdamW

from experiments.covering_objective.corpus import HupdLoader
from experiments.covering_objective.encode_pool import (
    PARAM_GROUPS,
    clone_module_state,
    owned_parameters,
)
from experiments.covering_objective.ledgers import (
    GateRecorder,
    session_gate_extra,
    session_live_gate,
)
from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.verdicts import PairGap, PairScore
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = [pytest.mark.experiment]

_EFFECTIVE_GROUPS = TypeAdapter(tuple[str, ...])


@dataclass(frozen=True)
class ArmReading:
    """One frozen or isolated-group unpaid-gap trajectory."""

    name: str
    readings: dict[int, float | None]


@pytest.fixture
def covering_effective_groups(request: pytest.FixtureRequest) -> tuple[str, ...]:
    """Groups announced this session, else cached names when functional is absent."""
    raw = session_gate_extra('covering/effective_groups')
    if raw is None and not session_live_gate('functional_steps'):
        raw = request.config.cache.get('covering/effective_groups', ())
    try:
        return _EFFECTIVE_GROUPS.validate_python(() if raw is None else raw)
    except ValidationError:
        return ()


@pytest.mark.live
class TestIsolatedPilot:
    """Isolated-block covering steps on the hygiene draw."""

    def test_isolated_cover_only_arms(
        self,
        ssv_job: SsvTrainConfig,
        covering_pilot: CoveringPilotSpec,
        restored_trunk: SoftTrunkModel,
        load_hupd_draw: HupdLoader,
        attach_termhood: Callable[[tuple[str, ...]], int],
        pair_gap: PairGap,
        pair_score: PairScore,
        require_gate: Callable[..., None],
        announce_gate: GateRecorder,
        covering_effective_groups: tuple[str, ...],
    ) -> None:
        """Cheap isolated AdamW arms on groups that passed the functional-step gate."""
        require_gate('functional_steps', 'functional-step gate did not run', ran=True)
        groups = covering_effective_groups or tuple(PARAM_GROUPS)
        draw = load_hupd_draw()
        termhood_n = attach_termhood(draw.lemma_keys(restored_trunk))
        lr = float(ssv_job.fit.learning_rate)
        snapshot = clone_module_state(restored_trunk)
        readouts = covering_pilot.readout_steps()
        budget = covering_pilot.isolated_steps

        def arm(name: str) -> ArmReading:
            _ = restored_trunk.load_state_dict(snapshot, strict=False)
            owned = tuple(owned_parameters(restored_trunk, name).values())
            _ = tuple(parameter.requires_grad_(False) for parameter in restored_trunk.parameters())
            _ = tuple(parameter.requires_grad_(True) for parameter in owned)
            opt = AdamW(owned, lr=lr)
            readings: dict[int, float | None] = {0: pair_gap(draw.claim_rows, draw.disc_rows)}

            def step(index: int) -> int:
                opt.zero_grad(set_to_none=True)
                loss = -pair_score(draw.claim_rows, draw.disc_rows, retain_graph=True)
                if torch.isfinite(loss):
                    torch.autograd.backward(loss)
                    opt.step()
                if index in readouts:
                    readings[index] = pair_gap(draw.claim_rows, draw.disc_rows)
                print(f'isolated {name} step {index}/{budget}', flush=True)
                return index

            _ = tuple(step(index) for index in range(1, budget + 1))
            return ArmReading(name=name, readings=readings)

        frozen = ArmReading(
            name='frozen',
            readings={0: pair_gap(draw.claim_rows, draw.disc_rows)},
        )
        reports = (frozen, *tuple(arm(name) for name in groups))
        start = frozen.readings[0]

        def improved(row: ArmReading) -> bool:
            end = row.readings.get(budget)
            return end is not None and start is not None and end > start

        moved = any(improved(row) for row in reports if row.name != 'frozen')
        culprit = '' if moved else 'OPTIMIZER'
        announce_gate(
            'isolated_pilot',
            {
                'n_patents': len(draw.rows),
                'n_pool': draw.n_pool,
                'seed': covering_pilot.seed,
                'isolated_steps': budget,
                'termhood_n': termhood_n,
                'groups': groups,
                'arms': tuple({'name': row.name, 'readings': row.readings} for row in reports),
                'culprit': culprit or 'none',
            },
            culprit=culprit,
            n_patents=len(draw.rows),
        )
        assert moved
