"""Dest train leftover unpaid of claim-mask demand against covering n."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch
from tests._ssv_fixtures import ssv_fixture_batch, ssv_smoke_module

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.inventory import CoveringInventory
from ip_claim.ssv.module import SsvLightningModule


def _planted_inventory(
    demand: torch.Tensor,
    supply: torch.Tensor,
    *,
    claim_labeled: torch.Tensor | None = None,
    full_labeled: torch.Tensor | None = None,
) -> CoveringInventory:
    slots = int(demand.size(-1))
    return CoveringInventory(
        n_entity_claim=demand,
        n_entity_full=supply,
        n_relation_claim=torch.zeros_like(demand),
        n_relation_full=torch.zeros_like(supply),
        mean_row_entropy=torch.zeros(()),
        batch_usage=torch.zeros(slots),
        relation_row_entropy=0.0,
        claim_labeled=claim_labeled,
        full_labeled=full_labeled,
    )


def _dest_loss(
    module: SsvLightningModule,
    payload: CoveringInventory,
) -> torch.Tensor:
    batch = ssv_fixture_batch(module)
    logged: dict[str, torch.Tensor] = {}

    def capture_log(name: str, value: object = None, **_kwargs: object) -> None:
        if torch.is_tensor(value):
            logged[name] = cast(torch.Tensor, value)

    module.train()
    with (
        patch.object(module.inventory, 'forward', return_value=payload),
        patch.object(module, 'log', side_effect=capture_log),
        patch.object(module, 'log_dict'),
    ):
        _ = module.training_step(batch, 0)
    return logged['dest_loss']


def test_dest_loss_moves_when_inventory_n_moves_on_demand(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
    demand = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
    off_support = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    on_support = torch.tensor([[3.0, 2.0, 0.0, 0.0]])
    off_loss = _dest_loss(module, _planted_inventory(demand, off_support))
    on_loss = _dest_loss(module, _planted_inventory(demand, on_support))
    assert torch.isfinite(off_loss)
    assert torch.isfinite(on_loss)
    assert not torch.allclose(off_loss, on_loss)


def test_dest_loss_ignores_pair_table_edge_unpaid_when_n_is_fixed(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
    covering = Covering(CoveringKnobs(sigma_edge=1.0))
    demand = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
    supply = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    matching = torch.zeros(1, 4, 4, 1)
    matching[0, 0, 1, 0] = 1.0
    matching[0, 1, 0, 0] = 1.0
    dest_shifted = matching.roll(1, dims=-2)
    match_edge = covering.edge_unpaid_fraction(
        matching.reshape(1, -1, 1),
        matching.reshape(1, -1, 1),
    )
    dest_edge = covering.edge_unpaid_fraction(
        matching.reshape(1, -1, 1),
        dest_shifted.reshape(1, -1, 1),
    )
    assert not torch.allclose(match_edge, dest_edge)
    same_n_matching = _dest_loss(
        module,
        _planted_inventory(demand, supply, claim_labeled=matching, full_labeled=matching),
    )
    same_n_shifted = _dest_loss(
        module,
        _planted_inventory(demand, supply, claim_labeled=matching, full_labeled=dest_shifted),
    )
    assert torch.allclose(same_n_matching, same_n_shifted)


def test_empty_demand_dest_loss_is_nan(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path, dest_comparison={'weight': 1.0, 'margin': 0.0})
    demand = torch.zeros(1, 4)
    supply = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    dest_loss = _dest_loss(module, _planted_inventory(demand, supply))
    assert dest_loss.isnan().all()
