"""Occupied inventory and graph-compose shape at the covering operating point."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from experiments.covering_objective.corpus import HupdLoader
from experiments.covering_objective.encode_pool import EncodeView
from experiments.covering_objective.ledgers import GateRecorder
from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.verdicts import InventoryShapeReport, require_named_culprit
from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ssv.collate import SoftMlmCollator
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.covering_trace import keep_covering_seams
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = pytest.mark.experiment


class TestInventoryShape:
    """Host-free occupancy and compose-shape culprits."""

    def test_null_shape_without_locus_fails_closed(self, covering_pilot: CoveringPilotSpec) -> None:
        """A dead KG reading cannot pass as 'not significant' with no locus."""
        with pytest.raises(pytest.fail.Exception, match='named culprit'):
            _ = require_named_culprit(
                covering_pilot.inventory_envelope().culprit(),
                significant=False,
            )

    def test_scheduled_zero_rho_is_rho_floor(self, covering_pilot: CoveringPilotSpec) -> None:
        """Claiming the train pin while encoding at rho=0 is a proxy, not a KG."""
        verdict = covering_pilot.inventory_envelope().model_copy(update={'rho': 0.0}).culprit()
        assert verdict.culprit == 'PROXY'
        assert verdict.locus == 'RHO_FLOOR'
        _ = require_named_culprit(verdict, significant=False)

    def test_empty_termhood_is_data_not_a_kg(self, covering_pilot: CoveringPilotSpec) -> None:
        """A termhood join is necessary and not proof the graph is occupied."""
        verdict = (
            covering_pilot.inventory_envelope().model_copy(update={'termhood_mass': 0.0}).culprit()
        )
        assert verdict.culprit == 'DATA'
        assert verdict.locus == 'TERMHOOD_STORE'

    def test_occupancy_below_utilization_min_is_dead(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """Stuck occupancy is the live-train rho freeze, not a useful codebook."""
        job = SsvTrainConfig()
        vocab = SsvContainer(config=job).soft_trunk().soft_vocab
        occupancy, n_dead = vocab.occupancy_and_dead(
            vocab.entity_usage_ema.new_zeros(vocab.entity_bank_size)
        )
        assert float(occupancy.item()) < float(job.bank.utilization_min)
        assert int(n_dead.item()) == vocab.entity_bank_size
        verdict = (
            covering_pilot
            .inventory_envelope(job)
            .model_copy(update={'occupancy': float(occupancy.item())})
            .culprit()
        )
        assert verdict.culprit == 'REPRESENTATION'
        assert verdict.locus == 'DEAD_OCCUPANCY'

    def test_usage_entropy_at_ln_k_is_uniform_assignment(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """Leave the uniform fixed point. Pretty H in 0.3-0.9 is not the bar."""
        envelope = covering_pilot.inventory_envelope()
        verdict = envelope.model_copy(
            update={'usage_entropy': covering_pilot.entropy_pause_ratio * envelope.ln_k}
        ).culprit()
        assert verdict.culprit == 'REPRESENTATION'
        assert verdict.locus == 'UNIFORM_ASSIGNMENT'

    def test_cpc_mix_identical_to_compose_is_forward_seam(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """Overlay mass that never changes mixed compose is the CPC disconnect."""
        compose = covering_pilot.inventory.compose_norm
        verdict = (
            covering_pilot
            .inventory_envelope()
            .model_copy(update={'mixed_norm': compose, 'compose_norm': compose})
            .culprit()
        )
        assert verdict.culprit == 'FORWARD_SEAM'
        assert verdict.locus == 'CPC_COMPOSE'

    def test_zero_overlay_or_prefix_is_graph_readout(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """Covering without overlay or projected prefix never used the graph path."""
        envelope = covering_pilot.inventory_envelope()
        overlay = envelope.model_copy(update={'overlay_norm': 0.0}).culprit()
        prefix = envelope.model_copy(update={'prefix_norm': 0.0}).culprit()
        assert overlay.culprit == 'FORWARD_SEAM'
        assert overlay.locus == 'GRAPH_READOUT'
        assert prefix.locus == 'GRAPH_READOUT'

    def test_overlay_parent_without_compose_child_is_cpc_compose(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """A moved overlay that leaves mixed compose unchanged is the CPC seam."""
        verdict = (
            covering_pilot
            .inventory_envelope()
            .model_copy(update={'parent_moved': True, 'child_moved': False})
            .culprit()
        )
        assert verdict.culprit == 'FORWARD_SEAM'
        assert verdict.locus == 'CPC_COMPOSE'

    def test_healthy_occupied_mix_has_no_culprit(self, covering_pilot: CoveringPilotSpec) -> None:
        """Non-uniform occupancy, termhood mass, and a CPC mix can pass this gate."""
        verdict = require_named_culprit(
            covering_pilot.inventory_envelope().culprit(),
            significant=True,
        )
        assert not verdict.culprit
        assert verdict.locus == 'none'


@pytest.mark.live
class TestInventoryShapeLive:
    """Occupied inventory on the initialized covering trunk."""

    def test_inventory_shape_on_pinned_trunk(
        self,
        ssv_job: SsvTrainConfig,
        covering_pilot: CoveringPilotSpec,
        covering_collator: SoftMlmCollator,
        ssv_model: SoftTrunkModel,
        load_hupd_draw: HupdLoader,
        encode_view: EncodeView,
        attach_termhood: Callable[[tuple[str, ...]], int],
        announce_gate: GateRecorder,
    ) -> None:
        """Occupied rows, entropy vs ln K, and prefix/compose mass on the real trunk."""
        draw = load_hupd_draw()
        termhood_n = attach_termhood(draw.lemma_keys(ssv_model))
        _claims, _trace = encode_view(draw.claim_rows, as_claim=True)
        shape_seams = frozenset({
            'late_assignment',
            'early_assignment',
            'termhood_weights',
            'overlay_state',
            'compose_state',
            'mixed_compose',
            'projected_prefix',
        })
        with keep_covering_seams(shape_seams):
            _supply, trace = encode_view(draw.disc_rows, as_claim=False, full_trace=True)
        occupancy, n_dead = ssv_model.soft_vocab.occupancy_and_dead(
            ssv_model.soft_vocab.entity_usage_ema
        )
        bank_k = max(int(ssv_job.arch.entity_bank_size), 1)
        ln_k = math.log(bank_k)

        def mass(name: str) -> float:
            tensor = trace.tensors.get(name)
            return 0.0 if tensor is None else float(tensor.rename(None).abs().sum().item())

        def usage_entropy() -> float:
            assignment = trace.tensors.get('late_assignment')
            if assignment is None:
                assignment = trace.tensors.get('early_assignment')
            if assignment is None:
                return ln_k
            usage = assignment.rename(None).mean(dim=tuple(range(assignment.ndim - 1)))
            usage = usage.clamp_min(1e-12)
            usage /= usage.sum().clamp_min(1e-12)
            return float((-(usage * usage.log())).sum().item())

        occupy = trace.tensors.get('termhood_weights')
        overlay_norm = mass('overlay_state')
        compose_norm = mass('compose_state')
        mixed_norm = mass('mixed_compose')
        prefix_norm = mass('projected_prefix')
        entropy = usage_entropy()
        occupy_mass = 0.0 if occupy is None else float(occupy.rename(None).sum().item())
        termhood_mass = occupy_mass if termhood_n else 0.0
        verdict = InventoryShapeReport(
            mask_schedule=covering_pilot.mask_schedule,
            rho=float(covering_collator.rho),
            occupancy=float(occupancy.item()),
            utilization_min=float(ssv_job.bank.utilization_min),
            usage_entropy=entropy,
            ln_k=ln_k,
            termhood_mass=termhood_mass,
            overlay_norm=overlay_norm,
            compose_norm=compose_norm,
            mixed_norm=mixed_norm,
            prefix_norm=prefix_norm,
            parent_moved=overlay_norm > 0.0,
            child_moved=abs(mixed_norm - compose_norm) > 1e-12,
        ).culprit()
        significant = not verdict.culprit
        require_named_culprit(verdict, significant=significant)
        announce_gate(
            'inventory_shape',
            {
                'n_patents': len(draw.rows),
                'occupancy': float(occupancy.item()),
                'n_dead': float(n_dead.item()),
                'utilization_min': float(ssv_job.bank.utilization_min),
                'usage_entropy': entropy,
                'ln_k': ln_k,
                'termhood_mass': termhood_mass,
                'inject_scale': float(ssv_model.inject_scale),
                'overlay_norm': overlay_norm,
                'compose_norm': compose_norm,
                'mixed_norm': mixed_norm,
                'prefix_norm': prefix_norm,
                'seams': tuple(trace.tensors),
                **verdict.reading(),
            },
            culprit=verdict.culprit,
            culprit_name=verdict.locus,
            occupancy=float(occupancy.item()),
            inject_scale=float(ssv_model.inject_scale),
        )
        assert significant
