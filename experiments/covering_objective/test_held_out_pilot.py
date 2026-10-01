"""Held-out unpaid-gap pilot. ST.14 letters are a report, not a pass bar."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
import torch
from torch.optim import AdamW

from experiments.covering_objective.corpus import HupdLoader
from experiments.covering_objective.ledgers import GateRecorder
from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.verdicts import PairGap, PairScore
from ip_claim.collision.data.citation_pairs import CitationPair
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = [pytest.mark.experiment]


@pytest.mark.live
class TestHeldOutPilot:
    """Held-out unpaid-gap pilot. Letters are a report."""

    def test_held_out_gap_and_letter_report(
        self,
        ssv_job: SsvTrainConfig,
        covering_pilot: CoveringPilotSpec,
        restored_trunk: SoftTrunkModel,
        load_hupd_draw: HupdLoader,
        attach_termhood: Callable[[tuple[str, ...]], int],
        pair_gap: PairGap,
        pair_score: PairScore,
        citation_pairs: tuple[CitationPair, ...],
        require_gate: Callable[..., None],
        announce_gate: GateRecorder,
    ) -> None:
        """Bounded train-slice steps. Letters are recorded, not required to order."""
        require_gate('isolated_pilot', 'isolated-block pilot did not run', ran=True)
        train_n = covering_pilot.hygiene_n
        held_n = covering_pilot.held_out_n
        draw = load_hupd_draw(train_n + held_n)
        termhood_n = attach_termhood(draw.lemma_keys(restored_trunk))
        train = draw.take(0, train_n)
        held = draw.take(train_n, train_n + held_n)
        if not held.rows:
            announce_gate(
                'held_out_pilot',
                {
                    'n_train': len(train.rows),
                    'n_held': 0,
                    'n_pool': draw.n_pool,
                    'termhood_n': termhood_n,
                },
                culprit='DATA',
            )
            pytest.fail(f'held-out draw is empty; pool={draw.n_pool}')
        start_held = pair_gap(held.claim_rows, held.disc_rows)
        opt = AdamW(
            (value for value in restored_trunk.parameters() if value.requires_grad),
            lr=float(ssv_job.fit.learning_rate),
        )
        t0 = time.monotonic()
        ceiling = covering_pilot.held_out_ceiling_s
        budget = covering_pilot.held_out_steps

        def step(index: int) -> int:
            if time.monotonic() - t0 >= ceiling:
                return index
            opt.zero_grad(set_to_none=True)
            loss = -pair_score(train.claim_rows, train.disc_rows, retain_graph=True)
            if torch.isfinite(loss):
                torch.autograd.backward(loss)
                opt.step()
            return index

        _ = tuple(step(index) for index in range(budget))
        end_held = pair_gap(held.claim_rows, held.disc_rows)
        delta = None if start_held is None or end_held is None else end_held - start_held
        culprit = (
            ''
            if delta is not None and delta > 0.0
            else ('PROXY' if delta is not None and delta <= 0.0 else 'DATA')
        )
        announce_gate(
            'held_out_pilot',
            {
                'n_train': len(train.rows),
                'n_held': len(held.rows),
                'n_pool': draw.n_pool,
                'termhood_n': termhood_n,
                'seed': covering_pilot.seed,
                'held_out_steps': budget,
                'start_held_gap': start_held,
                'end_held_gap': end_held,
                'delta': delta,
                'seconds': time.monotonic() - t0,
                'letters': {
                    'n_pairs': len(citation_pairs),
                    'note': (
                        f'ST.14 letters are a report. X < Y < A is not a {budget}-step pass bar.'
                    ),
                },
                'culprit': culprit or 'none',
            },
            culprit=culprit,
            n_held=len(held.rows),
            delta=delta,
        )
        assert delta is not None
        assert delta > 0.0
