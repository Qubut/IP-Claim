"""Overlay-shape train mix imports shape_family. Dest stays leftover unpaid.

The addend is community plus subgraph energy of the existing pair table
Phi already consumes. Zero shape weight leaves dest identity put. Dest is
not a shape scalar and not edge unpaid of pair masses. Community column
count is a host width, not occupy support. Citation identifiers are not
edges. These tests do not launch leftover-unpaid train.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import NamedTuple, cast
from unittest.mock import patch

import pytest
import torch
from tests._ssv_fixtures import ssv_fixture_batch, ssv_smoke_module
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.config import ArchSpec, OverlayShapeSpec, SsvTrainConfig
from ip_claim.ssv.inventory import CoveringInventory
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.shape import shape_family
from ip_claim.ssv.steer.cheng import cheng_rescale

_SLOTS = 16
_COMMUNITY_COLUMNS = 2
_INTRA_MASS = 1.0
_OCCUPY_MASS = 1.0
_CLAIM_MASS = 2.0
_BATCH = 2
_CONFIGS = Path(__file__).resolve().parents[3] / 'configs'


class MixOverlay(NamedTuple):
    """Occupy, kept pair table, numbered-claim demand, and covering n."""

    occupy: Tensor
    pair_mass: Tensor
    demand: Tensor
    supply: Tensor


def _labels() -> Tensor:
    return (torch.arange(_SLOTS) * _COMMUNITY_COLUMNS) // _SLOTS


def _clique_table() -> Tensor:
    same = _labels().unsqueeze(-1) == _labels().unsqueeze(-2)
    identity = torch.eye(_SLOTS, dtype=torch.bool)
    return (same & ~identity).to(dtype=torch.float32) * _INTRA_MASS


def _strength_matched_cut() -> Tensor:
    neighbors = _INTRA_MASS * (_SLOTS // _COMMUNITY_COLUMNS - 1)
    other = float(_SLOTS // _COMMUNITY_COLUMNS)
    cut = _labels().unsqueeze(-1) != _labels().unsqueeze(-2)
    return cut.to(dtype=torch.float32) * (neighbors / other)


def _scene(pair_mass: Tensor) -> MixOverlay:
    occupy = torch.full((_BATCH, _SLOTS), _OCCUPY_MASS)
    batched = pair_mass.expand(_BATCH, -1, -1)
    return MixOverlay(
        occupy=occupy,
        pair_mass=batched,
        demand=torch.full((_BATCH, _SLOTS), _CLAIM_MASS),
        supply=occupy + batched.sum(dim=-1) + batched.sum(dim=-2),
    )


def _planted_inventory(scene: MixOverlay) -> CoveringInventory:
    labeled = scene.pair_mass.unsqueeze(-1)
    return CoveringInventory(
        n_entity_claim=scene.demand,
        n_entity_full=scene.supply,
        n_relation_claim=torch.zeros_like(scene.demand),
        n_relation_full=torch.zeros_like(scene.supply),
        mean_row_entropy=torch.zeros(()),
        batch_usage=torch.zeros(_SLOTS),
        relation_row_entropy=0.0,
        claim_labeled=labeled,
        full_labeled=labeled,
    )


def _dest(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply))


def _run_step(
    module: SsvLightningModule,
    payload: CoveringInventory,
) -> dict[str, Tensor]:
    batch = ssv_fixture_batch(module)
    logged: dict[str, Tensor] = {}

    def capture_log(name: str, value: object = None, **_kwargs: object) -> None:
        if torch.is_tensor(value):
            logged[name] = cast(Tensor, value)

    module.train()
    with (
        patch.object(module.inventory, 'forward', return_value=payload),
        patch.object(module, 'log', side_effect=capture_log),
        patch.object(module, 'log_dict'),
    ):
        logged['loss'] = module.training_step(batch, 0)
    return logged


class TestOverlayShapeTrainMix:
    """The train mix multiplies shape_family by the shape weight. Dest stays U."""

    def test_train_step_imports_shape_family_and_keeps_dest_leftover_unpaid(self) -> None:
        """The step calls shape_family. Dest still scores unpaid_fraction of pair_table."""
        step = inspect.getsource(SsvLightningModule.training_step)
        assert 'shape_family' in step
        assert 'pair_table' in step
        assert 'unpaid_fraction' in step
        assert 'n_entity_claim' in step
        assert 'n_entity_full' in step
        assert 'full_labeled' in step
        assert 'edge_unpaid_fraction' not in step
        assert 'modularity' not in step
        assert 'shapor' not in step
        assert 'cited' not in step

    def test_zero_shape_weight_leaves_cheng_identity(self, tmp_path: Path) -> None:
        """Ablating the shape mix weight does not add a shape scalar to the mix."""
        module = ssv_smoke_module(tmp_path)
        batch = ssv_fixture_batch(module)
        logged: dict[str, Tensor] = {}
        captured: list[object] = []
        real = SsvLightningModule.forward

        def spy(
            self: SsvLightningModule,
            seen: object,
            living: Tensor | None = None,
        ) -> object:
            out = real(self, seen, living=living)
            captured.append(out)
            return out

        def capture_log(name: str, value: object = None, **_kwargs: object) -> None:
            if torch.is_tensor(value):
                logged[name] = cast(Tensor, value)

        module.forward = spy.__get__(module, SsvLightningModule)  # type: ignore[method-assign]
        module.train()
        with patch.object(module, 'log', side_effect=capture_log), patch.object(module, 'log_dict'):
            loss = module.training_step(batch, 0)
        out = captured[0]
        task, cons = module._task_and_constraint_losses(out)
        expected, _scale = cheng_rescale(task, cons, module._constraint_lambdas())
        assert module.config.overlay_shape.shape_weight == pytest.approx(0.0)
        assert 'shape_loss' not in logged
        assert 'dest_loss' not in logged
        assert torch.isfinite(loss)
        assert torch.allclose(loss, expected)

    def test_phi_preserving_rewiring_moves_shape_and_holds_dest(
        self,
        tmp_path: Path,
    ) -> None:
        """Strength-matched cut fill moves the wired shape scalar; leftover unpaid stays put."""
        covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))
        cliques = _scene(_clique_table())
        filled = _scene(_strength_matched_cut())
        module = ssv_smoke_module(tmp_path, overlay_shape={'shape_weight': 1.0})
        clique_logs = _run_step(module, _planted_inventory(cliques))
        filled_logs = _run_step(module, _planted_inventory(filled))
        dest_clique = _dest(covering, cliques.demand, cliques.supply)
        dest_filled = _dest(covering, filled.demand, filled.supply)
        labels = _labels()
        assignment = torch.nn.functional.one_hot(labels, _COMMUNITY_COLUMNS).to(
            dtype=cliques.pair_mass.dtype,
        )
        occupied = torch.ones(_SLOTS, dtype=torch.bool)
        family_clique = shape_family(cliques.pair_mass[0], assignment, occupied)
        family_filled = shape_family(filled.pair_mass[0], assignment, occupied)
        assert torch.allclose(cliques.supply, filled.supply)
        assert torch.allclose(dest_clique, dest_filled)
        assert torch.isfinite(dest_clique).all()
        assert 'shape_loss' in clique_logs
        assert 'dest_loss' not in clique_logs
        assert torch.isfinite(clique_logs['shape_loss'])
        assert not torch.allclose(clique_logs['shape_loss'], filled_logs['shape_loss'])
        assert not torch.allclose(
            clique_logs['shape_loss'],
            dest_clique.to(dtype=clique_logs['shape_loss'].dtype).mean(),
        )
        assert family_clique.community.item() != family_filled.community.item()
        assert family_clique.subgraph.item() != family_filled.subgraph.item()
        assert torch.allclose(
            family_clique.total,
            family_clique.community + family_clique.subgraph,
        )

    def test_dest_weight_keeps_leftover_unpaid_when_shape_is_on(
        self,
        tmp_path: Path,
    ) -> None:
        """A positive shape weight does not rewrite dest leftover unpaid."""
        covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))
        scene = _scene(_clique_table())
        payload = _planted_inventory(scene)
        dest_only = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
        both = ssv_smoke_module(
            tmp_path,
            dest_comparison={'weight': 1.0, 'margin': 0.0},
            overlay_shape={'shape_weight': 1.0},
        )
        dest_logs = _run_step(dest_only, payload)
        mix_logs = _run_step(both, payload)
        leftover = torch.nanmean(_dest(covering, scene.demand, scene.supply))
        edge = covering.edge_unpaid_fraction(scene.pair_mass, scene.pair_mass)
        assert 'dest_loss' in dest_logs
        assert 'shape_loss' not in dest_logs
        assert 'dest_loss' in mix_logs
        assert 'shape_loss' in mix_logs
        assert torch.allclose(mix_logs['dest_loss'], dest_logs['dest_loss'])
        assert torch.isfinite(leftover)
        assert not torch.allclose(mix_logs['dest_loss'], mix_logs['shape_loss'])
        assert not torch.allclose(
            leftover.to(dtype=edge.dtype),
            edge,
        )

    def test_community_columns_is_host_width_not_occupy_cap(self, tmp_path: Path) -> None:
        """Community column count is a host knob. It is not occupy support size."""
        module = ssv_smoke_module(tmp_path)
        spec = OverlayShapeSpec()
        assert spec.community_columns == 2
        assert spec.shape_weight == pytest.approx(0.0)
        assert module.config.overlay_shape.community_columns == 2
        assert module.config.arch.entity_bank_size == _SLOTS
        assert 'soft_occupied_max' not in ArchSpec.model_fields
        assert spec.community_columns != module.config.arch.entity_bank_size

    def test_prod_yaml_keeps_shape_weight_off(self) -> None:
        """Shipped leftover-unpaid dest YAML does not turn the shape addend on."""
        prod = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.yaml')
        dest = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.dest_compare.yaml')
        assert prod.overlay_shape.shape_weight == pytest.approx(0.0)
        assert dest.overlay_shape.shape_weight == pytest.approx(0.0)
        assert dest.dest_comparison.weight > 0.0

    def test_cite_fields_unused(self) -> None:
        """Citation identifiers do not index the mix overlay or the shape import."""
        step = inspect.getsource(SsvLightningModule.training_step)
        assert 'cited' not in MixOverlay._fields
        assert 'category' not in MixOverlay._fields
        assert 'hupd' not in MixOverlay._fields
        assert 'cited' not in inspect.signature(shape_family).parameters
        assert 'category' not in inspect.signature(shape_family).parameters
        assert 'cited_hupd' not in step
