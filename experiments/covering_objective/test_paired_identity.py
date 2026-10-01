"""Paired identity on the initialized train-YAML SSV host."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import pytest
import torch
from patent_ate.termhood import TermhoodStore

from experiments.covering_objective.corpus import HupdLoader
from experiments.covering_objective.encode_pool import EncodeView
from experiments.covering_objective.ledgers import GateRecorder
from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.verdicts import unpaid_gap
from ip_claim.collision.cover import Covering
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.covering_trace import keep_covering_seams
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = [pytest.mark.experiment]


@pytest.mark.live
class TestPairedIdentity:
    """Paired identity on the initialized train-YAML SSV host."""

    def test_crossed_text_graph_identity(
        self,
        ssv_job: SsvTrainConfig,
        covering_pilot: CoveringPilotSpec,
        ssv_model: SoftTrunkModel,
        load_hupd_draw: HupdLoader,
        encode_view: EncodeView,
        covering: Covering,
        attach_termhood: Callable[[tuple[str, ...]], int],
        termhood_store: TermhoodStore | None,
        announce_gate: GateRecorder,
    ) -> None:
        """Four text/graph cells through ``Covering.pair_table`` on real patents."""
        draw = load_hupd_draw()
        required = covering_pilot.hygiene_n
        floor = covering_pilot.hygiene_floor()
        attach_termhood(draw.lemma_keys(ssv_model))
        available = tuple(
            bool(claim.claim_blob.strip()) and bool(disc.disclosure.strip())
            for claim, disc in zip(draw.claim_rows, draw.disc_rows, strict=True)
        )
        foreign = draw.rolled()
        text_views = {
            'match': draw.disc_texts(),
            'foreign': tuple(draw.disc_texts()[index] for index in foreign),
        }
        graph_views = {
            'match': draw.graphs(),
            'foreign': tuple(draw.graphs()[index] for index in foreign),
        }
        cells = {
            f'{text_key}_text_{graph_key}_graph': (text_views[text_key], graph_views[graph_key])
            for text_key, graph_key in product(('match', 'foreign'), repeat=2)
        }
        claims, _trace = encode_view(draw.claim_rows, as_claim=True)
        with keep_covering_seams(frozenset({'termhood_weights'})):
            encoded = {
                key: encode_view(draw.views(texts, graphs), as_claim=False, full_trace=True)
                for key, (texts, graphs) in cells.items()
            }
        tables = {
            key: covering.pair_table(claims, supply) for key, (supply, _trace) in encoded.items()
        }
        match = tables['match_text_match_graph']
        frac = covering.unpaid_fraction(match)
        diag = frac.diagonal()
        finite = torch.isfinite(diag) & (match.demand_l1 > 0)
        valid_frac = float(finite.float().mean().item()) if diag.numel() else 0.0
        demand = match.demand_l1.detach().cpu()
        occupy = encoded['match_text_match_graph'][1].tensors.get('termhood_weights')
        termhood_mass = float(occupy.rename(None).sum().item()) if occupy is not None else 0.0
        unpaid_all_nan = bool((~torch.isfinite(frac)).all().item())
        store_missing = termhood_store is None
        hygiene_n = valid_frac * len(draw.rows)
        culprit = (
            'DATA'
            if (
                draw.n_pool < required
                or hygiene_n < floor
                or store_missing
                or not all(available)
                or not bool((demand > 0).any().item())
            )
            else ('SCORE' if unpaid_all_nan else '')
        )
        announce_gate(
            'paired_identity',
            {
                'host': ssv_job.host.name,
                'n_patents': len(draw.rows),
                'n_pool': draw.n_pool,
                'hygiene_required': required,
                'hupd_dir': str(ssv_job.hupd_root()),
                'termhood_root': None if termhood_store is None else str(termhood_store.root),
                'available': list(available),
                'valid_pair_fraction': valid_frac,
                'demand_l1': demand.tolist(),
                'unpaid_all_nan': unpaid_all_nan,
                'gaps': {
                    key: unpaid_gap(covering, claims, supply)
                    for key, (supply, _trace) in encoded.items()
                },
                'termhood_mass': termhood_mass,
                'culprit': culprit or 'none',
                'culprit_name': (
                    'TERMHOOD_STORE'
                    if store_missing
                    else ('HUPD_SAMPLE' if draw.n_pool < required else culprit or 'none')
                ),
            },
            culprit=culprit,
            n_patents=len(draw.rows),
            n_pool=draw.n_pool,
            host=ssv_job.host.name,
            valid_pair_fraction=valid_frac,
            termhood_mass=termhood_mass,
            culprit_name=(
                'TERMHOOD_STORE'
                if store_missing
                else ('HUPD_SAMPLE' if draw.n_pool < required else culprit or 'none')
            ),
        )
        assert 'tiny-random' not in ssv_job.host.name
        assert all(available)
        assert draw.n_pool >= required
        assert hygiene_n >= floor
        assert not store_missing
        assert bool((demand > 0).any().item())
        assert not unpaid_all_nan
