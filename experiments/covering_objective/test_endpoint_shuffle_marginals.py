"""GPU-free marginals for endpoint and relation-label shuffles."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from experiments.covering_objective.relational_arms import (
    DECLARED_EDGE_SWEEP,
    RELATIONAL_CONDITIONS,
    compact_encode_seams,
    endpoint_marginals,
    endpoint_shuffle,
    relation_label_shuffle,
    relational_arms,
    retain_seams,
    write_frozen_campaign,
)
from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.covering_trace import CoveringTrace


def _toy_labeled() -> Tensor:
    table = torch.zeros(2, 4, 4, 3)
    table[0, 0, 1, 0] = 2.0
    table[0, 0, 2, 1] = 1.0
    table[0, 1, 3, 2] = 3.0
    table[1, 2, 0, 0] = 4.0
    table[1, 3, 1, 1] = 5.0
    return table


def test_endpoint_shuffle_preserves_source_relation_edges_and_mass() -> None:
    """Dest-axis roll keeps source mass, relation totals, support, and mass."""
    labeled = _toy_labeled()
    shuffled = endpoint_shuffle(labeled, shift=1)
    before = endpoint_marginals(labeled)
    after = endpoint_marginals(shuffled)
    torch.testing.assert_close(after.source_mass, before.source_mass)
    torch.testing.assert_close(after.relation_totals, before.relation_totals)
    torch.testing.assert_close(after.edge_count, before.edge_count)
    torch.testing.assert_close(after.total_mass, before.total_mass)
    torch.testing.assert_close(after.dest_mass, before.dest_mass.roll(1, dims=-1))
    assert shuffled.shape == labeled.shape
    assert not torch.equal(shuffled, labeled)


def test_relation_label_shuffle_preserves_endpoints() -> None:
    """Relation-code roll keeps the collapsed table and endpoint totals."""
    labeled = _toy_labeled()
    shuffled = relation_label_shuffle(labeled, shift=1)
    before = endpoint_marginals(labeled)
    after = endpoint_marginals(shuffled)
    torch.testing.assert_close(after.collapsed, before.collapsed)
    torch.testing.assert_close(after.source_mass, before.source_mass)
    torch.testing.assert_close(after.dest_mass, before.dest_mass)
    torch.testing.assert_close(after.edge_count, before.edge_count)
    torch.testing.assert_close(after.total_mass, before.total_mass)
    torch.testing.assert_close(after.relation_totals, before.relation_totals.roll(1, dims=-1))
    assert not torch.equal(shuffled, labeled)
    torch.testing.assert_close(shuffled[0, 0, 1, 1], torch.tensor(2.0))


def test_declared_condition_matrix_has_eight_rows() -> None:
    """The frozen campaign names eight conditions before any encode."""
    assert RELATIONAL_CONDITIONS == (
        'matching',
        'rolled_filing',
        'endpoint_shuffle',
        'relation_label_shuffle',
        'prefix_off',
        'residual_off',
        'both_off',
        'adapter_off',
    )
    assert len(DECLARED_EDGE_SWEEP) == 3


def test_write_frozen_campaign_marks_every_declared_condition(tmp_path: Path) -> None:
    """Each named condition writes arms, retained seams, and a complete marker."""
    covering = Covering(CoveringKnobs())
    query = torch.tensor([[2.0, 1.0]])
    document = torch.tensor([[1.0, 0.5]])
    labeled = torch.zeros(1, 2, 2, 2)
    labeled[0, 0, 1, 0] = 3.0
    arms = relational_arms(
        n_query=query,
        n_document=document,
        n_rel_query=labeled.sum(dim=(-3, -2)),
        n_rel_document=labeled.sum(dim=(-3, -2)),
        labeled_query=labeled,
        labeled_document=labeled,
        collapsed_query=labeled.sum(dim=-1),
        collapsed_document=labeled.sum(dim=-1),
        covering=covering,
    )
    trace = CoveringTrace(
        tensors={
            'token_residual': torch.ones(1, 4, 8),
            'projected_prefix': torch.ones(1, 2, 8),
        }
    )
    payload = {
        name: (
            {DECLARED_EDGE_SWEEP[0]: arms},
            retain_seams(trace, name, labeled_composed=labeled),
        )
        for name in RELATIONAL_CONDITIONS
    }
    root = write_frozen_campaign(tmp_path, conditions=payload)
    complete = tuple((root / name / 'complete.json').is_file() for name in RELATIONAL_CONDITIONS)
    assert all(complete)
    assert (root / 'campaign.json').is_file()
    assert 'token_residual' in torch.load(root / 'matching' / 'seams.pt', weights_only=False)


def test_compact_encode_seams_drops_unknown_and_token_means_host() -> None:
    """Actor returns keep declared scopes only; token-major host state is averaged."""
    fat = torch.ones(2, 4, 8, 3)
    kept = compact_encode_seams({'host_text_state': fat, 'unknown_cube': fat})
    assert 'unknown_cube' not in kept
    assert tuple(kept['host_text_state'].shape) == (2, 4, 3)


def test_compact_encode_seams_keeps_occupy_and_explicit_unknown() -> None:
    """Occupy seams stay on the actor return; an explicit keep list keeps extras."""
    weights = torch.ones(2, 5)
    extra = torch.ones(2, 3)
    kept = compact_encode_seams(
        {'termhood_weights': weights, 'unknown_cube': extra},
        keep=frozenset({'unknown_cube'}),
    )
    assert tuple(kept['termhood_weights'].shape) == (2, 5)
    assert tuple(kept['unknown_cube'].shape) == (2, 3)


def test_compact_then_retain_adapter_delta_keeps_module_axis() -> None:
    """First adapter_delta scope stays token-major so later norms keep modules."""
    cube = torch.ones(2, 64, 8)
    compacted = compact_encode_seams({'adapter_delta': cube})
    assert tuple(compacted['adapter_delta'].shape) == (2, 64, 8)
    trace = CoveringTrace(tensors=compacted)
    retained = retain_seams(trace, 'matching')
    assert tuple(retained['adapter_delta'].shape) == (2, 64)
