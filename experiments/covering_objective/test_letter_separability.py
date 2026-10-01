"""Graph contribution to ST.14 X/Y/A unpaid geometry, with named culprits."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from experiments.covering_objective.corpus import HupdLoader
from experiments.covering_objective.encode_pool import EncodeView
from experiments.covering_objective.ledgers import GateRecorder
from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.verdicts import (
    LetterSeparabilityReport,
    letter_separability_culprit,
    require_named_culprit,
)
from ip_claim.collision.collide import PatentEmbeddingRecord
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.cover import Covering
from ip_claim.collision.data.citation_pairs import CitationPair, St14Mark, split_pairs_by_query
from ip_claim.collision.diagnose import (
    IntensityIndex,
    QueryPartnerScore,
    citation_letter,
    intensity_index,
    query_macro_unpaid_fraction_xa,
    query_macro_xa_filter,
    score_cited_pairs,
)
from ip_claim.ssv.collate import SoftMlmCollator
from ip_claim.ssv.covering_trace import CoveringTrace, keep_covering_seams, patch_covering
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = pytest.mark.experiment


def _pair(query: str, partner: str, *marks: St14Mark) -> CitationPair:
    return CitationPair.model_validate({
        'query_application_number': query,
        'partner_application_number': partner,
        'marks': marks,
    })


def _record(app: str, claim: Sequence[float], full: Sequence[float]) -> PatentEmbeddingRecord:
    return PatentEmbeddingRecord(
        application_number=app,
        z_d=torch.zeros(2),
        n_entity_claim=torch.tensor(claim),
        n_entity_full=torch.tensor(full),
        n_relation_claim=torch.zeros(2),
        n_relation_full=torch.zeros(2),
    )


def _letter_unpaid(scores: Sequence[QueryPartnerScore], mark: str) -> float | None:
    unpaid = tuple(row.unpaid for row in scores if row.mark == mark)
    return None if not unpaid else sum(unpaid) / len(unpaid)


def _frac(covering: Covering, query: Tensor, document: Tensor) -> float:
    return float(covering.unpaid_fraction(covering.pair_table(query, document)).item())


class TestLetterSeparability:
    """Host-free ST.14 letter geometry and named culprits."""

    def test_x_mark_wins_over_y_and_letters_stay_distinct(self) -> None:
        """ST.14 keeps X and Y. Do not collapse both into CLEF-IP grade 2."""
        both = citation_letter(_pair('1', '11', 'X', 'Y'))
        only_y = citation_letter(_pair('1', '12', 'Y'))
        only_x = citation_letter(_pair('1', '13', 'X'))
        only_a = citation_letter(_pair('1', '14', 'A'))
        assert both == 'X'
        assert only_y == 'Y'
        assert only_x == 'X'
        assert only_a == 'A'
        assert only_x != only_y

    def test_empty_cited_pairs_are_st14_data(self, covering_pilot: CoveringPilotSpec) -> None:
        """A development slice with no scored pairs cannot show letter geometry."""
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'n_pairs': 0,
        })
        assert verdict.culprit == 'DATA'
        assert verdict.locus == 'ST14_PAIRS'
        require_named_culprit(verdict, significant=False)

    def test_empty_graph_intensities_fail_closed(self, covering_pilot: CoveringPilotSpec) -> None:
        """Pairs without claim/full intensities are not a letter measurement."""
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'intensities_ok': False,
        })
        assert verdict.culprit == 'DATA'
        assert verdict.locus == 'TERMHOOD_STORE'
        require_named_culprit(verdict, significant=False)

    def test_matching_graph_xa_exceeds_shuffled_and_text_only(
        self,
        covering: Covering,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """Matching, foreign, and prefix-off supplies are three cells, not one alias."""
        query = torch.tensor([8.0, 0.0])
        matched_x = torch.tensor([8.0, 0.0])
        matched_a = torch.tensor([0.0, 8.0])
        shuffled_x = matched_a
        shuffled_a = matched_x
        text_x = torch.tensor([4.0, 4.0])
        text_a = torch.tensor([5.0, 3.0])
        graph_on = _frac(covering, query, matched_a) - _frac(covering, query, matched_x)
        graph_off = _frac(covering, query, shuffled_a) - _frac(covering, query, shuffled_x)
        text_only = _frac(covering, query, text_a) - _frac(covering, query, text_x)
        assert not torch.equal(matched_x, shuffled_x)
        assert not torch.equal(matched_x, text_x)
        assert not torch.equal(shuffled_x, text_x)
        assert graph_on != graph_off
        assert graph_off != text_only
        assert graph_on != text_only
        assert graph_on > text_only
        verdict = letter_separability_culprit(
            n_pairs=2,
            graph_on_xa=graph_on,
            graph_off_xa=graph_off,
            text_only_xa=text_only,
            noise=covering_pilot.letter.noise,
            parent_moved=True,
            unpaid_moved=True,
            saturated=False,
        )
        assert not verdict.culprit
        require_named_culprit(verdict, significant=True)

    def test_graph_on_below_text_only_is_text_identity(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """A graph that hurts XA relative to prefix-off is a text proxy, not a KG."""
        envelope = covering_pilot.letter
        verdict = letter_separability_culprit(**{
            **envelope.model_dump(),
            'graph_on_xa': envelope.graph_off_xa + 2.0 * envelope.noise,
            'graph_off_xa': envelope.graph_off_xa,
            'text_only_xa': envelope.graph_on_xa,
        })
        assert verdict.culprit == 'PROXY'
        assert verdict.locus == 'TEXT_IDENTITY'
        require_named_culprit(verdict, significant=False)

    def test_unjoined_cited_pairs_are_st14_not_termhood(
        self,
        covering: Covering,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """Non-empty parquet that scores nothing against encoded apps is ST.14 data."""
        records = (_record('1', [8.0, 0.0], [8.0, 0.0]),)
        pairs = (_pair('1', '11', 'X'), _pair('1', '13', 'A'))
        scores = score_cited_pairs(pairs, intensity_index(records), covering)
        assert len(scores) < len(pairs)
        assert scores == ()
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'n_pairs': len(scores),
        })
        assert verdict.culprit == 'DATA'
        assert verdict.locus == 'ST14_PAIRS'
        require_named_culprit(verdict, significant=False)

    def test_development_split_is_eval_not_train(self) -> None:
        """Letter geometry uses the frozen eval query split, not train or hygiene."""
        pairs = tuple(
            _pair(f'{10_000_000 + index}', f'{20_000_000 + index}', 'X') for index in range(10)
        )
        eval_config = CollisionEvalConfig.from_yaml()
        train, development, held = split_pairs_by_query(
            pairs,
            train=eval_config.split.train,
            eval_fraction=eval_config.split.eval,
            test=eval_config.split.test,
            seed=eval_config.split.seed,
            query_limit=eval_config.query_limit,
        )
        train_q = {pair.query_application_number for pair in train}
        eval_q = {pair.query_application_number for pair in development}
        test_q = {pair.query_application_number for pair in held}
        assert eval_q
        assert train_q.isdisjoint(eval_q)
        assert eval_q.isdisjoint(test_q)
        assert len(train_q) == 6
        assert len(eval_q) == 2
        assert len(test_q) == 2

    def test_equal_xa_with_graph_destroyed_is_text_identity(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """Letter ranking that survives graph destruction is host text, not the KG."""
        envelope = covering_pilot.letter
        tied = envelope.graph_off_xa
        verdict = letter_separability_culprit(**{
            **envelope.model_dump(),
            'graph_on_xa': tied,
            'graph_off_xa': tied,
            'text_only_xa': tied,
        })
        assert verdict.culprit == 'PROXY'
        assert verdict.locus == 'TEXT_IDENTITY'
        require_named_culprit(verdict, significant=False)

    def test_query_mass_does_not_flip_letter_fraction_ranking(
        self,
        covering: Covering,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """E0 scale invariance must hold on letter tables. A flip is saturation."""
        query = torch.tensor([2.0, 0.0])
        x_doc = torch.tensor([4.0, 0.0])
        a_doc = torch.tensor([0.0, 4.0])
        x_frac = _frac(covering, query, x_doc)
        a_frac = _frac(covering, query, a_doc)
        x_scaled = _frac(covering, query * 3.0, x_doc)
        a_scaled = _frac(covering, query * 3.0, a_doc)
        flipped = (x_frac < a_frac) != (x_scaled < a_scaled)
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'saturated': flipped,
            'graph_on_xa': a_frac - x_frac,
        })
        assert not flipped
        assert x_frac < a_frac
        assert not verdict.culprit

    def test_letter_scored_pairs_fit_hygiene_n(self, covering_pilot: CoveringPilotSpec) -> None:
        """Live ST.14 encode stays inside the hygiene patent budget."""
        extra = tuple(
            _pair(f'{10_000_000 + index}', f'{20_000_000 + index}', 'X')
            for index in range(covering_pilot.hygiene_n)
        )
        scored = covering_pilot.letter_scored_pairs(extra)
        stems = {
            stem
            for pair in scored
            for stem in (pair.query_application_number, pair.partner_application_number)
        }
        assert scored
        assert extra[0] in scored
        assert extra[-1] not in scored
        assert len(stems) <= covering_pilot.hygiene_n

    def test_saturated_letters_are_score(self, covering_pilot: CoveringPilotSpec) -> None:
        """When covering already pays every letter, X and A are unseparable by construction."""
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'saturated': True,
        })
        assert verdict.culprit == 'SCORE'
        assert verdict.locus == 'SATURATION'

    def test_parent_graph_without_unpaid_move_is_disclosure_intensity(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """A live overlay that leaves unpaid ranking still is the disclosure seam."""
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'unpaid_moved': False,
        })
        assert verdict.culprit == 'REPRESENTATION'
        assert verdict.locus == 'DISCLOSURE_INTENSITY'

    def test_unmoved_graph_parent_is_readout_seam(self, covering_pilot: CoveringPilotSpec) -> None:
        """Letter margins with no overlay or prefix motion never used the graph."""
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'parent_moved': False,
        })
        assert verdict.culprit == 'FORWARD_SEAM'
        assert verdict.locus == 'GRAPH_READOUT'

    def test_score_cited_pairs_preserves_x_y_a_marks(self, covering: Covering) -> None:
        """Product letter scoring keeps X, Y, and A on planted intensities."""
        records = (
            _record('1', [8.0, 0.0], [8.0, 0.0]),
            _record('11', [8.0, 0.0], [8.0, 0.0]),
            _record('12', [8.0, 0.0], [4.0, 4.0]),
            _record('13', [8.0, 0.0], [0.0, 8.0]),
        )
        pairs = (_pair('1', '11', 'X'), _pair('1', '12', 'Y'), _pair('1', '13', 'A'))
        scores = score_cited_pairs(pairs, intensity_index(records), covering)
        marks = tuple(row.mark for row in scores)
        unpaid = {row.mark: row.unpaid for row in scores}
        assert marks == ('X', 'Y', 'A')
        assert unpaid['X'] < unpaid['Y']
        assert unpaid['Y'] < unpaid['A']
        assert query_macro_unpaid_fraction_xa(scores) is not None
        assert (query_macro_unpaid_fraction_xa(scores) or 0.0) > 0.0

    def test_empty_demand_letter_xa_is_none(self) -> None:
        """Shared-query empty demand is a null fraction, not a missing X-and-A join."""
        scores = (
            QueryPartnerScore(
                query='1',
                partner='11',
                mark='X',
                unpaid=0.0,
                covering=0.0,
                demand_l1=0.0,
            ),
            QueryPartnerScore(
                query='1',
                partner='13',
                mark='A',
                unpaid=0.0,
                covering=0.0,
                demand_l1=0.0,
            ),
        )
        named = query_macro_xa_filter(scores)
        assert query_macro_unpaid_fraction_xa(scores) is None
        assert named.reason == 'null_frac'
        assert named.n_queries_xa == 1
        assert named.n_live == 0

    def test_disjoint_letter_queries_are_empty_xa_not_null_frac(self) -> None:
        """X and A on different queries leave XA None because the join is empty."""
        scores = (
            QueryPartnerScore(
                query='1',
                partner='11',
                mark='X',
                unpaid=1.59,
                covering=0.2,
                demand_l1=4.0,
            ),
            QueryPartnerScore(
                query='2',
                partner='21',
                mark='A',
                unpaid=0.0,
                covering=1.0,
                demand_l1=4.0,
            ),
            QueryPartnerScore(
                query='3',
                partner='31',
                mark='Y',
                unpaid=3.08,
                covering=0.1,
                demand_l1=4.0,
            ),
        )
        named = query_macro_xa_filter(scores)
        assert query_macro_unpaid_fraction_xa(scores) is None
        assert named.reason == 'empty_xa'
        assert named.n_scores == 3
        assert named.n_queries_xa == 0

    def test_scored_pairs_with_none_xa_are_not_empty_st14(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """A scored table with no X-and-A join is the query-macro filter."""
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'n_pairs': 302,
            'graph_on_xa': None,
            'graph_off_xa': None,
            'text_only_xa': None,
            'unpaid_moved': False,
            'saturated': False,
            'letter_a_unpaid': 1.59,
            'xa_filter': 'empty_xa',
        })
        assert verdict.culprit == 'DATA'
        assert verdict.locus == 'QUERY_MACRO_XA'
        require_named_culprit(verdict, significant=False)

    def test_null_frac_without_paid_a_is_query_macro_frac(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """X-and-A rows with null fractions are not an empty ST.14 join."""
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'n_pairs': 302,
            'graph_on_xa': None,
            'graph_off_xa': None,
            'text_only_xa': None,
            'unpaid_moved': False,
            'saturated': False,
            'letter_a_unpaid': 1.59,
            'xa_filter': 'null_frac',
        })
        assert verdict.culprit == 'DATA'
        assert verdict.locus == 'QUERY_MACRO_FRAC'
        require_named_culprit(verdict, significant=False)

    def test_paid_a_without_unpaid_move_is_saturation(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """A unpaid near zero with still-None XA is saturation, not empty ST.14."""
        verdict = letter_separability_culprit(**{
            **covering_pilot.letter.model_dump(),
            'n_pairs': 302,
            'graph_on_xa': None,
            'graph_off_xa': None,
            'text_only_xa': None,
            'parent_moved': True,
            'unpaid_moved': False,
            'saturated': False,
            'letter_a_unpaid': 0.0,
            'xa_filter': 'empty_xa',
        })
        assert verdict.culprit == 'SCORE'
        assert verdict.locus == 'SATURATION'
        require_named_culprit(verdict, significant=False)

    def test_null_letter_reading_without_locus_fails_closed(
        self,
        covering_pilot: CoveringPilotSpec,
    ) -> None:
        """'Not significant' with no locus is not an architecture result."""
        with pytest.raises(pytest.fail.Exception, match='named culprit'):
            require_named_culprit(
                letter_separability_culprit(**covering_pilot.letter.model_dump()),
                significant=False,
            )


@pytest.mark.live
class TestLetterSeparabilityLive:
    """ST.14 unpaid under matching, foreign, and prefix-off graph."""

    def test_letter_separability_on_development_pairs(
        self,
        covering_pilot: CoveringPilotSpec,
        ssv_model: SoftTrunkModel,
        load_hupd_draw: HupdLoader,
        attach_termhood: Callable[[tuple[str, ...]], int],
        encode_view: EncodeView,
        covering: Covering,
        covering_collator: SoftMlmCollator,
        citation_pairs: tuple[CitationPair, ...],
        require_gate: Callable[[str, str], None],
        announce_gate: GateRecorder,
    ) -> None:
        """ST.14 unpaid under matching, foreign, and prefix-off graph."""
        require_gate('paired_identity', 'paired-identity hygiene did not pass')
        eval_config = CollisionEvalConfig.from_yaml()
        _train, development, _held = split_pairs_by_query(
            citation_pairs,
            train=eval_config.split.train,
            eval_fraction=eval_config.split.eval,
            test=eval_config.split.test,
            seed=eval_config.split.seed,
            query_limit=eval_config.query_limit,
        )
        scored = covering_pilot.letter_scored_pairs(development)
        stems = tuple(
            dict.fromkeys(
                stem
                for pair in scored
                for stem in (pair.query_application_number, pair.partner_application_number)
            )
        )
        draw = load_hupd_draw(stems=stems)
        _ = attach_termhood(draw.lemma_keys(ssv_model))

        def index_for(claims: Tensor, supply: Tensor) -> IntensityIndex:
            records = tuple(
                PatentEmbeddingRecord(
                    application_number=row.application_number,
                    z_d=torch.zeros(1),
                    n_entity_claim=claim.rename(None),
                    n_entity_full=full.rename(None),
                    n_relation_claim=torch.zeros(1),
                    n_relation_full=torch.zeros(1),
                )
                for row, claim, full in zip(draw.rows, claims, supply, strict=True)
            )
            return intensity_index(records)

        def score_cells() -> tuple[LetterSeparabilityReport, Mapping[str, object]]:
            foreign = draw.rolled()
            graphs = draw.graphs()
            claims, _claim_trace = encode_view(draw.claim_rows, as_claim=True)

            def composed_supply(
                filing: Sequence[HeteroData],
            ) -> tuple[Tensor, CoveringTrace]:
                windows, owners = draw.window_rows(
                    covering_collator.tokenizer,
                    max_length=int(covering_collator.max_length),
                    max_chunks=covering_pilot.disclosure_max_chunks,
                    graphs=filing,
                )
                with keep_covering_seams(frozenset({'projected_prefix', 'overlay_state'})):
                    encoded, trace = encode_view(windows, as_claim=False, full_trace=True)
                return covering.compose_windows(
                    encoded,
                    torch.tensor(owners, device=encoded.device, dtype=torch.long),
                    len(draw.rows),
                ), trace

            match_n, match_trace = composed_supply(graphs)
            off_n, off_trace = composed_supply(tuple(graphs[index] for index in foreign))
            prefix = match_trace.tensors.get('projected_prefix')
            zeros = None if prefix is None else torch.zeros_like(prefix.rename(None))
            with patch_covering({} if zeros is None else {'projected_prefix': zeros}):
                text_n, text_trace = composed_supply(graphs)
            match_scores = score_cited_pairs(scored, index_for(claims, match_n), covering)
            match_filter = query_macro_xa_filter(match_scores)
            graph_on = match_filter.delta
            graph_off = query_macro_xa_filter(
                score_cited_pairs(scored, index_for(claims, off_n), covering),
            ).delta
            text_only = query_macro_xa_filter(
                score_cited_pairs(scored, index_for(claims, text_n), covering),
            ).delta
            overlay = match_trace.tensors.get('overlay_state')
            letter_a = _letter_unpaid(match_scores, 'A')
            unpaid_moved = (
                graph_on is not None
                and graph_off is not None
                and abs(graph_on - graph_off) > covering_pilot.letter.noise
            )
            paid_a = letter_a is not None and 0.0 <= letter_a <= covering_pilot.letter.noise
            report = LetterSeparabilityReport(
                n_pairs=len(match_scores),
                graph_on_xa=graph_on,
                graph_off_xa=graph_off,
                text_only_xa=text_only,
                noise=covering_pilot.letter.noise,
                parent_moved=(
                    overlay is not None and float(overlay.rename(None).abs().sum().item()) > 0.0
                ),
                unpaid_moved=unpaid_moved,
                saturated=paid_a and not unpaid_moved,
                letter_a_unpaid=letter_a,
                xa_filter='' if match_filter.reason == 'ok' else match_filter.reason,
                intensities_ok=match_n.numel() > 0,
            )
            return report, {
                'letters': {
                    'X': _letter_unpaid(match_scores, 'X'),
                    'Y': _letter_unpaid(match_scores, 'Y'),
                    'A': letter_a,
                },
                'xa_filter': match_filter.reason,
                'n_queries_xa': match_filter.n_queries_xa,
                'n_live_xa': match_filter.n_live,
                'seams': tuple(match_trace.tensors),
                'text_seams': tuple(text_trace.tensors),
                'off_seams': tuple(off_trace.tensors),
            }

        report, ledger = (
            score_cells()
            if scored and draw.rows
            else (
                LetterSeparabilityReport(
                    n_pairs=0,
                    graph_on_xa=None,
                    graph_off_xa=None,
                    text_only_xa=None,
                    noise=covering_pilot.letter.noise,
                    parent_moved=False,
                    unpaid_moved=False,
                    saturated=False,
                    intensities_ok=False,
                    xa_filter='empty_scores',
                ),
                {
                    'letters': {'X': None, 'Y': None, 'A': None},
                    'xa_filter': 'empty_scores',
                    'n_queries_xa': 0,
                    'n_live_xa': 0,
                    'seams': (),
                    'text_seams': (),
                    'off_seams': (),
                },
            )
        )
        verdict = report.culprit()
        significant = not verdict.culprit
        require_named_culprit(verdict, significant=significant)
        announce_gate(
            'letter_separability',
            {
                'n_pairs': report.n_pairs,
                'n_loaded_pairs': len(citation_pairs),
                'n_development_pairs': len(development),
                'n_scored_pairs': len(scored),
                'n_patents': len(draw.rows),
                'letters': ledger['letters'],
                'letter_a_unpaid': report.letter_a_unpaid,
                'saturated': report.saturated,
                'unpaid_moved': report.unpaid_moved,
                'xa_filter': ledger['xa_filter'],
                'n_queries_xa': ledger['n_queries_xa'],
                'n_live_xa': ledger['n_live_xa'],
                'graph_on_xa': report.graph_on_xa,
                'graph_off_xa': report.graph_off_xa,
                'text_only_xa': report.text_only_xa,
                'x_less_y_less_a': 'report only; not a cheap pass bar',
                'seams': ledger['seams'],
                'text_seams': ledger['text_seams'],
                'off_seams': ledger['off_seams'],
                **verdict.reading(),
            },
            culprit=verdict.culprit,
            culprit_name=verdict.locus,
        )
        assert significant
