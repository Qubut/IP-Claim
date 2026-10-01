"""Retrieval head reads persistable Phi. Invert is layout. Cosine on z_g fails.

Covering leftover unpaid stays the current train head. The retrieval head
is a later map of the same Phi store. These tests do not add a second
encoder, do not rewire dest, and do not write cite edges.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.inventory import CoveringInventory
from ip_claim.ssv.model import TrunkExport
from ip_claim.ssv.phi import PhiIntensity

_SLOTS = 4
_PAID = 0
_BRIDGE = 2
_SILENT = 3
_CLAIM_MASS = 2.0
_OCCUPY_MASS = 1.0
_BRIDGE_OCCUPY = 0.2
_BRIDGE_EDGE = 2.0


class PlantedFiling(NamedTuple):
    """Ingress occupy, kept pair table, and persistable Phi."""

    occupy: Tensor
    pair_mass: Tensor
    ingress: Tensor
    store: CoveringInventory


class InvertLayout(NamedTuple):
    """Slot postings of filings whose persistable Phi is live on that slot."""

    postings: tuple[tuple[int, tuple[int, ...]], ...]


class OccupyOnlyKey(NamedTuple):
    """Named fail: a retrieval tower that reads occupy and drops kept addends."""

    occupy: Tensor


class GraphReadoutKey(NamedTuple):
    """Named fail: cosine on the GNN readout."""

    z_g: Tensor


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def overlay() -> PhiIntensity:
    return PhiIntensity(_SLOTS)


def _leftover(covering: Covering, demand: Tensor, supply: Tensor) -> Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


def _query_demand() -> Tensor:
    demand = torch.zeros(_SLOTS, dtype=torch.float64)
    demand[_PAID] = _CLAIM_MASS
    return demand


def _persist(overlay: PhiIntensity, occupy: Tensor, pair_mass: Tensor) -> CoveringInventory:
    live = occupy > 0
    intensity = overlay(occupy, pair_mass, live)
    zeros = intensity.new_zeros(intensity.shape)
    return CoveringInventory(
        n_entity_claim=intensity,
        n_entity_full=intensity,
        n_relation_claim=zeros,
        n_relation_full=zeros,
        mean_row_entropy=intensity.new_zeros(()),
        batch_usage=intensity.new_zeros(_SLOTS),
        relation_row_entropy=0.0,
    )


def _paid_document(overlay: PhiIntensity) -> PlantedFiling:
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_PAID] = _OCCUPY_MASS
    pair_mass = occupy.new_zeros(_SLOTS, _SLOTS)
    return PlantedFiling(
        occupy=occupy,
        pair_mass=pair_mass,
        ingress=occupy > 0,
        store=_persist(overlay, occupy, pair_mass),
    )


def _silent_document(overlay: PhiIntensity) -> PlantedFiling:
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_SILENT] = _OCCUPY_MASS
    pair_mass = occupy.new_zeros(_SLOTS, _SLOTS)
    return PlantedFiling(
        occupy=occupy,
        pair_mass=pair_mass,
        ingress=occupy > 0,
        store=_persist(overlay, occupy, pair_mass),
    )


def _bridge_document(overlay: PhiIntensity) -> PlantedFiling:
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_PAID] = _BRIDGE_OCCUPY
    occupy[_BRIDGE] = _OCCUPY_MASS
    pair_mass = overlay.kept_pair_addends(
        torch.tensor((_BRIDGE,)),
        torch.tensor((_PAID,)),
        torch.tensor((_BRIDGE_EDGE,), dtype=torch.float64),
        torch.tensor((True,)),
    )
    return PlantedFiling(
        occupy=occupy,
        pair_mass=pair_mass,
        ingress=occupy > 0,
        store=_persist(overlay, occupy, pair_mass),
    )


def _invert(stores: tuple[CoveringInventory, ...]) -> InvertLayout:
    slots = tuple(range(stores[0].n_entity_full.size(-1)))
    return InvertLayout(
        postings=tuple(
            (
                slot,
                tuple(
                    index
                    for index, store in enumerate(stores)
                    if bool(store.n_entity_full[slot] > 0)
                ),
            )
            for slot in slots
        )
    )


def _reversed_invert(layout: InvertLayout) -> InvertLayout:
    return InvertLayout(
        postings=tuple((slot, filing_ids[::-1]) for slot, filing_ids in layout.postings)
    )


def _posting(layout: InvertLayout, slot: int) -> tuple[int, ...]:
    return next(ids for key, ids in layout.postings if key == slot)


class TestRetrievalHeadOnPersistablePhi:
    """Later retrieval head reads the same Phi store. Covering dest is unchanged."""

    def test_retrieval_key_is_the_same_phi_store(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Dest leftover unpaid and retrieval rank both read n_entity_full."""
        paid = _paid_document(overlay)
        query = _query_demand()
        dest_supply = paid.store.n_entity_full
        retrieval_key = paid.store.n_entity_full
        dest = _leftover(covering, query, dest_supply)
        retrieval = _leftover(covering, query, retrieval_key)
        assert retrieval_key is dest_supply
        assert torch.allclose(dest, retrieval)
        assert 'n_entity_claim' in CoveringInventory.model_fields
        assert 'n_entity_full' in CoveringInventory.model_fields
        assert 'z_g' in TrunkExport.model_fields
        assert 'z_g' not in CoveringInventory.model_fields
        assert 'n_entity_full' not in TrunkExport.model_fields

    def test_invert_of_occupy_is_layout_not_dest(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Posting order is a layout of Phi. Dest leftover unpaid does not move with it."""
        paid = _paid_document(overlay)
        bridge = _bridge_document(overlay)
        query = _query_demand()
        stores = (paid.store, bridge.store)
        layout = _invert(stores)
        reversed_layout = _reversed_invert(layout)
        dest_paid = _leftover(covering, query, paid.store.n_entity_full)
        dest_bridge = _leftover(covering, query, bridge.store.n_entity_full)
        assert dest_paid.ndim == 0
        assert dest_bridge.ndim == 0
        assert not torch.allclose(dest_paid, dest_bridge)
        assert _posting(layout, _PAID) == (0, 1)
        assert _posting(reversed_layout, _PAID) == (1, 0)
        assert torch.allclose(
            _leftover(covering, query, paid.store.n_entity_full),
            dest_paid,
        )
        assert torch.allclose(
            _leftover(covering, query, bridge.store.n_entity_full),
            dest_bridge,
        )
        assert len(_posting(layout, _PAID)) == 2

    def test_cosine_on_graph_readout_is_named_fail(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Cosine on z_g ranks the silent filing first. Leftover unpaid ranks the paid one."""
        paid = _paid_document(overlay)
        silent = _silent_document(overlay)
        query = _query_demand()
        leftovers = torch.stack((
            _leftover(covering, query, paid.store.n_entity_full),
            _leftover(covering, query, silent.store.n_entity_full),
        ))
        query_z = GraphReadoutKey(z_g=torch.tensor((1.0, 0.0), dtype=torch.float64))
        paid_z = GraphReadoutKey(z_g=torch.tensor((0.0, 1.0), dtype=torch.float64))
        silent_z = GraphReadoutKey(z_g=torch.tensor((1.0, 0.0), dtype=torch.float64))
        cosine = torch.stack((
            cosine_similarity(query_z.z_g, paid_z.z_g, dim=0),
            cosine_similarity(query_z.z_g, silent_z.z_g, dim=0),
        ))
        leftover_order = leftovers.argsort()
        cosine_order = cosine.argsort(descending=True)
        assert leftovers[0] < leftovers[1]
        assert cosine[1] > cosine[0]
        assert int(leftover_order[0]) == 0
        assert int(cosine_order[0]) == 1
        assert leftover_order.tolist() != cosine_order.tolist()

    def test_second_encoder_that_drops_kept_addends_is_named_fail(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Occupy-only cosine ranks the paid filing first. Leftover unpaid ranks the bridge."""
        paid = _paid_document(overlay)
        bridge = _bridge_document(overlay)
        query = _query_demand()
        leftovers = torch.stack((
            _leftover(covering, query, paid.store.n_entity_full),
            _leftover(covering, query, bridge.store.n_entity_full),
        ))
        query_occupy = OccupyOnlyKey(occupy=_query_demand() / _CLAIM_MASS)
        paid_key = OccupyOnlyKey(occupy=paid.occupy)
        bridge_key = OccupyOnlyKey(occupy=bridge.occupy)
        occupy_cosine = torch.stack((
            cosine_similarity(query_occupy.occupy, paid_key.occupy, dim=0),
            cosine_similarity(query_occupy.occupy, bridge_key.occupy, dim=0),
        ))
        leftover_order = leftovers.argsort()
        occupy_order = occupy_cosine.argsort(descending=True)
        assert leftovers[1] < leftovers[0]
        assert occupy_cosine[0] > occupy_cosine[1]
        assert int(leftover_order[0]) == 1
        assert int(occupy_order[0]) == 0
        assert leftover_order.tolist() != occupy_order.tolist()
        assert not torch.allclose(bridge.store.n_entity_full, bridge.occupy)
        assert torch.allclose(paid.store.n_entity_full, paid.occupy)

    def test_covering_head_dest_stays_leftover_unpaid_of_phi(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """The covering head still scores unpaid_fraction of pair_table(L, Phi)."""
        paid = _paid_document(overlay)
        query = _query_demand()
        leftover = _leftover(covering, query, paid.store.n_entity_full)
        assert leftover.ndim == 0
        assert torch.isfinite(leftover)
        assert leftover > 0
        assert 'cited' not in PlantedFiling._fields
        assert 'category' not in PlantedFiling._fields
        assert 'cited' not in InvertLayout._fields
        assert 'cited' not in OccupyOnlyKey._fields
        assert 'cited' not in GraphReadoutKey._fields
        assert 'cited' not in CoveringInventory.model_fields
