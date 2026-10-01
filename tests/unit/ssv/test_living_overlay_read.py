"""Living typed links from another filing enter Phi and leftover unpaid.

A reader filing occupies shared slots. A typed kept pair written by a
different filing is added into the reader's pair table and scored by
PhiIntensity. Dest-axis roll of that living link moves leftover unpaid of
numbered-claim demand against Phi. Dest-axis roll of a pair that does not
pay demanded support, and occupy swap among leftover-unpaid-irrelevant
slots, leave leftover unpaid put. Smallest leftover-unpaid-sufficient
occupy gains the living-link endpoint. Citation fields are not edges.
These tests do not rewire production occupy select.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
from tests.unit.ssv.test_leftover_unpaid_sufficient_occupy import (
    PlantedFiling,
    _is_leftover_unpaid_sufficient,
    _leftover_on_claims,
    _mask,
    _smallest_leftover_unpaid_sufficient,
)
from torch import Tensor

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.phi import PhiIntensity

_SLOTS = 4
_DEMANDED = 0
_UNRELATED = 1
_LIVE_SRC = 2
_SILENT = 3
_OCCUPY_MASS = 1.0
_LIGHT_MASS = 0.4
_SILENT_MASS = 0.25
_EDGE_MASS = 0.5
_CLAIM_MASS = 2.0
_DEST_SHIFT = 1
_READER = 'this-filing'
_WRITER = 'other-filing'


class LivingRead(NamedTuple):
    """Reader occupy and demand plus a living pair table written elsewhere."""

    reader: PlantedFiling
    living: Tensor
    written_by: str


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def overlay() -> PhiIntensity:
    return PhiIntensity(_SLOTS)


def _pair(overlay: PhiIntensity, source: int, destination: int) -> Tensor:
    return overlay.kept_pair_addends(
        torch.tensor((source,)),
        torch.tensor((destination,)),
        torch.tensor((_EDGE_MASS,), dtype=torch.float64),
        torch.tensor((True,)),
    )


def _other_filing(overlay: PhiIntensity) -> PlantedFiling:
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_DEMANDED] = _OCCUPY_MASS
    occupy[_LIVE_SRC] = _OCCUPY_MASS
    demand = occupy.new_zeros(_SLOTS)
    demand[_DEMANDED] = _CLAIM_MASS
    return PlantedFiling(
        occupy=occupy,
        pair_mass=_pair(overlay, _LIVE_SRC, _DEMANDED),
        ingress=occupy > 0,
        demands=(demand,),
    )


def _reader_own() -> PlantedFiling:
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_DEMANDED] = _OCCUPY_MASS
    occupy[_UNRELATED] = _LIGHT_MASS
    occupy[_LIVE_SRC] = _OCCUPY_MASS
    occupy[_SILENT] = _SILENT_MASS
    demand = occupy.new_zeros(_SLOTS)
    demand[_DEMANDED] = _CLAIM_MASS
    return PlantedFiling(
        occupy=occupy,
        pair_mass=occupy.new_zeros(_SLOTS, _SLOTS),
        ingress=occupy > 0,
        demands=(demand,),
    )


def _unrelated_other_filing(overlay: PhiIntensity) -> PlantedFiling:
    occupy = torch.zeros(_SLOTS, dtype=torch.float64)
    occupy[_UNRELATED] = _OCCUPY_MASS
    occupy[_SILENT] = _OCCUPY_MASS
    demand = occupy.new_zeros(_SLOTS)
    demand[_UNRELATED] = _CLAIM_MASS
    return PlantedFiling(
        occupy=occupy,
        pair_mass=_pair(overlay, _SILENT, _UNRELATED),
        ingress=occupy > 0,
        demands=(demand,),
    )


def _read_living(reader: PlantedFiling, living: Tensor) -> PlantedFiling:
    return PlantedFiling(
        occupy=reader.occupy,
        pair_mass=reader.pair_mass + living,
        ingress=reader.ingress,
        demands=reader.demands,
    )


def _scene(overlay: PhiIntensity) -> LivingRead:
    writer = _other_filing(overlay)
    return LivingRead(
        reader=_reader_own(),
        living=writer.pair_mass,
        written_by=_WRITER,
    )


class TestLivingOverlayRead:
    """Other-filing typed kept pairs are read into Phi. Dest leftover unpaid moves."""

    def test_other_filing_living_link_moves_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """A typed pair written on another filing pays the reader's demanded slot."""
        scene = _scene(overlay)
        writer = _other_filing(overlay)
        own = _leftover_on_claims(
            covering,
            scene.reader.demands,
            overlay(scene.reader.occupy, scene.reader.pair_mass, scene.reader.ingress),
        )
        read = _read_living(scene.reader, scene.living)
        with_living = _leftover_on_claims(
            covering,
            read.demands,
            overlay(read.occupy, read.pair_mass, read.ingress),
        )
        assert scene.written_by == _WRITER
        assert scene.written_by != _READER
        assert torch.equal(scene.living, writer.pair_mass)
        assert not torch.equal(writer.ingress, scene.reader.ingress)
        assert not torch.allclose(own, with_living)
        assert torch.all(with_living < own)

    def test_dest_roll_of_living_link_moves_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Dest-axis roll of the living pair moves leftover unpaid on demanded support."""
        scene = _scene(overlay)
        read = _read_living(scene.reader, scene.living)
        paid = _leftover_on_claims(
            covering,
            read.demands,
            overlay(read.occupy, read.pair_mass, read.ingress),
        )
        rolled = _leftover_on_claims(
            covering,
            read.demands,
            overlay(
                read.occupy,
                scene.reader.pair_mass + scene.living.roll(_DEST_SHIFT, dims=-1),
                read.ingress,
            ),
        )
        assert not torch.allclose(paid, rolled)

    def test_dest_roll_of_unrelated_living_pair_leaves_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Dest-axis roll of a living pair that does not pay demand leaves U put."""
        scene = _scene(overlay)
        extra = _unrelated_other_filing(overlay)
        living = scene.living + extra.pair_mass
        read = _read_living(scene.reader, living)
        paid = _leftover_on_claims(
            covering,
            read.demands,
            overlay(read.occupy, read.pair_mass, read.ingress),
        )
        rolled = _leftover_on_claims(
            covering,
            read.demands,
            overlay(
                read.occupy,
                scene.reader.pair_mass + scene.living + extra.pair_mass.roll(_DEST_SHIFT, dims=-1),
                read.ingress,
            ),
        )
        assert extra.pair_mass[_SILENT, _UNRELATED].item() == _EDGE_MASS
        assert extra.pair_mass.roll(_DEST_SHIFT, dims=-1)[_SILENT, _LIVE_SRC].item() == _EDGE_MASS
        assert torch.allclose(paid, rolled)

    def test_unrelated_slot_occupy_roll_leaves_leftover_unpaid(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Swapping occupy on leftover-unpaid-irrelevant slots leaves leftover unpaid put."""
        scene = _scene(overlay)
        read = _read_living(scene.reader, scene.living)
        paid = _leftover_on_claims(
            covering,
            read.demands,
            overlay(read.occupy, read.pair_mass, read.ingress),
        )
        swapped = read.occupy.clone()
        unrelated = swapped[_UNRELATED].clone()
        swapped[_UNRELATED] = swapped[_SILENT].clone()
        swapped[_SILENT] = unrelated
        rolled = _leftover_on_claims(
            covering,
            read.demands,
            overlay(swapped, read.pair_mass, read.ingress),
        )
        assert not torch.equal(swapped, read.occupy)
        assert torch.allclose(paid, rolled)

    def test_living_link_endpoint_enters_smallest_support(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """S-star gains the other-filing endpoint that addend-pays demanded occupy."""
        scene = _scene(overlay)
        own = scene.reader
        read = _read_living(own, scene.living)
        star_own = _smallest_leftover_unpaid_sufficient(covering, overlay, own)
        star_read = _smallest_leftover_unpaid_sufficient(covering, overlay, read)
        dropped = read.ingress.clone()
        dropped[_LIVE_SRC] = False
        assert star_own == frozenset((_DEMANDED,))
        assert star_read == frozenset((_DEMANDED, _LIVE_SRC))
        assert len(star_own) != len(star_read)
        assert len(read.demands) != len(star_read)
        assert _LIVE_SRC not in star_own
        assert _LIVE_SRC in star_read
        assert _UNRELATED not in star_read
        assert _SILENT not in star_read
        assert not _is_leftover_unpaid_sufficient(covering, overlay, read, dropped)

    def test_smallest_support_reconstructs_ambient_after_living_read(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """Phi on S-star, with the living pair table, leaves leftover unpaid put."""
        scene = _scene(overlay)
        read = _read_living(scene.reader, scene.living)
        star = _smallest_leftover_unpaid_sufficient(covering, overlay, read)
        ambient = _leftover_on_claims(
            covering,
            read.demands,
            overlay(read.occupy, read.pair_mass, read.ingress),
        )
        reconstructed = _leftover_on_claims(
            covering,
            read.demands,
            overlay(read.occupy, read.pair_mass, _mask(star)),
        )
        assert torch.allclose(ambient, reconstructed)
        assert torch.all(torch.isfinite(ambient))
        assert not torch.allclose(ambient, ambient.new_zeros(ambient.shape))

    def test_destination_type_is_covering_leftover_unpaid_of_phi(
        self,
        covering: Covering,
        overlay: PhiIntensity,
    ) -> None:
        """The covering head reads unpaid_fraction of pair_table(L, Phi)."""
        scene = _scene(overlay)
        read = _read_living(scene.reader, scene.living)
        supply = overlay(read.occupy, read.pair_mass, read.ingress)
        leftover = covering.unpaid_fraction(covering.pair_table(read.demands[0], supply)).squeeze()
        assert leftover.ndim == 0
        assert torch.isfinite(leftover)
        assert scene.written_by == _WRITER
        assert scene.written_by != _READER
        assert 'cited' not in LivingRead._fields
        assert 'category' not in LivingRead._fields
        assert 'cited' not in PlantedFiling._fields
        assert 'category' not in PlantedFiling._fields
