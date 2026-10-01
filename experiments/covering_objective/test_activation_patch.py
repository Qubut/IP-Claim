"""On-distribution necessity and recovery patches at covering seams."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from torch import Tensor

from experiments.covering_objective.corpus import HupdLoader
from experiments.covering_objective.encode_pool import EncodeView
from experiments.covering_objective.ledgers import GateRecorder
from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.verdicts import unpaid_gap
from ip_claim.collision.cover import Covering
from ip_claim.ssv.covering_trace import keep_covering_seams, patch_covering
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = [pytest.mark.experiment]

_SEAMS = (
    'early_assignment',
    'termhood_weights',
    'termhood_assignment',
    'overlay_state',
    'compose_state',
    'mixed_compose',
    'graph_readout',
    'projected_prefix',
    'token_residual',
    'host_text_state',
    'late_assignment',
    'disclosure_intensity',
)


@pytest.mark.live
class TestActivationPatch:
    """On-distribution necessity and recovery patches."""

    def test_necessity_and_recovery_patches(
        self,
        covering_pilot: CoveringPilotSpec,
        ssv_model: SoftTrunkModel,
        load_hupd_draw: HupdLoader,
        attach_termhood: Callable[[Sequence[str]], int],
        encode_view: EncodeView,
        covering: Covering,
        require_gate: Callable[[str, str], None],
        announce_gate: GateRecorder,
    ) -> None:
        """Patch clean and corrupt activations from a matched foreign pair."""
        require_gate('paired_identity', 'paired-identity hygiene did not pass')
        needed = covering_pilot.patch_n
        draw = load_hupd_draw(needed)
        _ = attach_termhood(draw.lemma_keys(ssv_model))
        if len(draw.rows) < needed:
            announce_gate('activation_patch', {'n_patents': len(draw.rows)}, culprit='DATA')
            pytest.fail(f'activation patches need >= {needed} pairs; got {len(draw.rows)}')
        foreign = draw.rolled()
        claims, _trace = encode_view(draw.claim_rows, as_claim=True)
        patch_seams = frozenset(_SEAMS)
        with keep_covering_seams(patch_seams):
            clean_n, clean_trace = encode_view(draw.disc_rows, as_claim=False, full_trace=True)
        corrupt_rows = draw.views(
            tuple(draw.disc_texts()[index] for index in foreign),
            tuple(draw.graphs()[index] for index in foreign),
        )
        with keep_covering_seams(patch_seams):
            corrupt_n, corrupt_trace = encode_view(corrupt_rows, as_claim=False, full_trace=True)

        def gap(supply: Tensor) -> float | None:
            return unpaid_gap(covering, claims, supply)

        def patched(name: str, replacement: Tensor, rows: tuple[object, ...]) -> Tensor:
            with patch_covering({name: replacement}):
                supply, _trace = encode_view(rows, as_claim=False)
            return supply

        clean_gap = gap(clean_n)
        corrupt_gap = gap(corrupt_n)
        contrast = (
            abs((clean_gap or 0.0) - (corrupt_gap or 0.0))
            if clean_gap is not None and corrupt_gap is not None
            else 0.0
        )
        effects = {
            name: (
                {
                    'necessity': gap(patched(name, corrupt.rename(None), draw.disc_rows)),
                    'recovery': gap(patched(name, clean.rename(None), corrupt_rows)),
                    'present': True,
                }
                if (clean := clean_trace.tensors.get(name)) is not None
                and (corrupt := corrupt_trace.tensors.get(name)) is not None
                else {'necessity': None, 'recovery': None, 'present': False}
            )
            for name in _SEAMS
        }
        parent_moved = any(bool(effect['present']) for effect in effects.values())
        culprit = (
            ''
            if parent_moved and contrast > 0.0
            else ('REPRESENTATION' if parent_moved else 'FORWARD_SEAM')
        )
        announce_gate(
            'activation_patch',
            {
                'n_patents': len(draw.rows),
                'n_pool': draw.n_pool,
                'clean_gap': clean_gap,
                'corrupt_gap': corrupt_gap,
                'contrast': contrast,
                'effects': effects,
                'culprit': culprit or 'none',
            },
            culprit=culprit,
            n_patents=len(draw.rows),
            contrast=contrast,
        )
        assert parent_moved
        assert effects['disclosure_intensity']['present']
