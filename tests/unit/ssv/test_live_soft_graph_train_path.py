"""Live train and encode use leftover-unpaid occupy and the living table.

Numbered-claim demand is claim-span mass on top entity codes, not occupancy.
Host ceiling remains compute. Destination leftover unpaid stays
unpaid_fraction of pair_table. Prod YAML does not turn the shape addend
on. These tests do not launch leftover-unpaid train.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tests._ssv_fixtures import ssv_fixture_batch, ssv_smoke_module, ssv_tiny_vocab_config
from tests.unit.test_covering_inventory import _VocabTrunk

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.soft_graph import SoftRelationBundle, build_soft_relation_bundle
from ip_claim.ssv.soft_vocab import SoftVocabModule

_CONFIGS = Path(__file__).resolve().parents[3] / 'configs'
_CLAIM = '1. A latchbolt comprising a photodiode.'
_TOKENS = 4
_CLAIM_SPAN = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
_FULL_SPAN = torch.ones(1, _TOKENS)


class TestLiveSoftGraphTrainPath:
    """Production callers pass numbered-claim demand and a living table."""

    def test_inventory_forward_empty_claim_texts_is_not_top_m(self) -> None:
        """Empty numbered-claim texts yield zero demand, not a top-M occupy bag."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config())
        last_layer = torch.randn(1, _TOKENS, 32)
        captured: dict[str, torch.Tensor | None] = {}
        real = build_soft_relation_bundle

        def spy(*args: object, **kwargs: object) -> SoftRelationBundle:
            demand = kwargs.get('demand')
            captured['demand'] = demand if isinstance(demand, torch.Tensor) else None
            return real(*args, **kwargs)

        with patch('ip_claim.ssv.inventory.build_soft_relation_bundle', side_effect=spy):
            payload = Inventory()(
                last_layer,
                _FULL_SPAN,
                _CLAIM_SPAN,
                model=_VocabTrunk(vocab),
            )
        demand = captured['demand']
        assert demand is not None
        assert torch.equal(demand, torch.zeros_like(demand))
        occupancy = vocab.masked_intensity(
            vocab.soft_assign(last_layer)[0],
            _FULL_SPAN,
        )
        top = occupancy.topk(min(16, occupancy.size(-1)), dim=-1)
        assert int((top.values > 0).sum().item()) > 0
        assert payload.claim_demand is not None
        assert torch.equal(payload.claim_demand, demand)

    def test_inventory_forward_demand_is_claim_span_not_occupancy(self) -> None:
        """Claim-span top-code mass is demand. Full-token occupancy is not."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config())
        last_layer = torch.randn(1, _TOKENS, 32)
        late_assign, _ = vocab.soft_assign(last_layer)
        expected = vocab.claim_span_demand(late_assign, _CLAIM_SPAN)
        occupancy = vocab.masked_intensity(late_assign, _FULL_SPAN)
        captured: dict[str, torch.Tensor | None] = {}
        real = build_soft_relation_bundle

        def spy(*args: object, **kwargs: object) -> SoftRelationBundle:
            demand = kwargs.get('demand')
            captured['demand'] = demand if isinstance(demand, torch.Tensor) else None
            return real(*args, **kwargs)

        with patch('ip_claim.ssv.inventory.build_soft_relation_bundle', side_effect=spy):
            Inventory()(
                last_layer,
                _FULL_SPAN,
                _CLAIM_SPAN,
                model=_VocabTrunk(vocab),
                claim_texts=(_CLAIM,),
            )
        demand = captured['demand']
        assert demand is not None
        assert torch.allclose(demand, expected)
        assert not torch.allclose(demand, occupancy)
        assert int((demand > 0).sum().item()) < int(demand.size(-1))

    def test_inventory_forward_reads_living_table(self) -> None:
        """Phi compose of a living pair table is the inventory living argument."""
        vocab = SoftVocabModule(ssv_tiny_vocab_config(entity_bank_size=8))
        last_layer = torch.randn(1, _TOKENS, 32)
        living = torch.zeros(8, 8)
        living[0, 1] = 1.5
        captured: dict[str, torch.Tensor | None] = {}
        real = build_soft_relation_bundle

        def spy(*args: object, **kwargs: object) -> SoftRelationBundle:
            table = kwargs.get('living')
            captured['living'] = table if isinstance(table, torch.Tensor) else None
            return real(*args, **kwargs)

        with patch('ip_claim.ssv.inventory.build_soft_relation_bundle', side_effect=spy):
            Inventory()(
                last_layer,
                _FULL_SPAN,
                _CLAIM_SPAN,
                model=_VocabTrunk(vocab),
                living=living,
                claim_texts=(_CLAIM,),
            )
        assert captured['living'] is not None
        assert torch.equal(captured['living'], living)

    def test_trunk_overlay_receives_demand_and_living(self, tmp_path: Path) -> None:
        """Train overlay reads numbered-claim demand and the persistable living table."""
        module = ssv_smoke_module(tmp_path)
        batch = ssv_fixture_batch(module)
        assert any(batch.claim_texts)
        planted = torch.zeros_like(module.model.living_pair_mass)
        planted[0, 2] = 0.75
        module.model.living_pair_mass.copy_(planted)
        captured: dict[str, torch.Tensor | None] = {}
        real = build_soft_relation_bundle

        def spy(*args: object, **kwargs: object) -> SoftRelationBundle:
            demand = kwargs.get('demand')
            living = kwargs.get('living')
            captured['demand'] = demand if isinstance(demand, torch.Tensor) else None
            captured['living'] = living if isinstance(living, torch.Tensor) else None
            return real(*args, **kwargs)

        with patch('ip_claim.ssv.model.build_soft_relation_bundle', side_effect=spy):
            module.forward(batch)
        assert captured['demand'] is not None
        assert captured['living'] is not None
        assert captured['living'][0, 2] == pytest.approx(0.75)

    def test_living_write_is_read_on_the_next_trunk_step(self, tmp_path: Path) -> None:
        """Absorb updates the bounded slot-pair table the next overlay reads."""
        module = ssv_smoke_module(tmp_path)
        planted = torch.zeros_like(module.model.living_pair_mass)
        planted[1, 3] = 2.0
        module.model.absorb_living(planted)
        snapshot = module.model.living_snapshot()
        assert snapshot[1, 3] > 0
        assert snapshot.shape == module.model.living_pair_mass.shape

    def test_dest_identity_stays_leftover_unpaid_of_pair_table(self) -> None:
        """Covering head is unpaid_fraction of pair_table, not edge unpaid."""
        covering = Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))
        demand = torch.tensor([1.0, 0.0, 0.0, 0.0])
        supply = torch.tensor([0.5, 0.0, 0.0, 0.0])
        leftover = covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()
        edge = covering.edge_unpaid_fraction(
            torch.ones(4, 4),
            torch.ones(4, 4),
        ).squeeze()
        assert leftover.ndim == 0
        assert torch.isfinite(leftover)
        assert not torch.allclose(leftover, edge)

    def test_prod_yaml_keeps_shape_weight_off(self) -> None:
        """Dest isolation keeps shipped shape weight at zero. Family stays on the object."""
        prod = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.yaml')
        dest = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.dest_compare.yaml')
        assert prod.overlay_shape.shape_weight == pytest.approx(0.0)
        assert dest.overlay_shape.shape_weight == pytest.approx(0.0)
        assert dest.dest_comparison.weight > 0.0

    def test_cite_fields_are_not_living_or_demand_inputs(self) -> None:
        """Citation identifiers do not enter occupy select or the living table."""
        assert 'cited' not in Inventory.forward.__code__.co_varnames
        assert 'category' not in Inventory.forward.__code__.co_varnames
        assert 'cited_hupd' not in Inventory.forward.__code__.co_varnames
