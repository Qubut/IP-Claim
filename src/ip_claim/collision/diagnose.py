"""Offline covering judgment on existing encode shards.

Cited partners are scored with one batched ``Covering`` call. Per-query
letter means, X/A deltas, and two-Y selection are Polars group/join.
Does not encode, train, or change YAML.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl
import structlog
import torch
from pydantic import BaseModel, ConfigDict
from returns.maybe import Maybe
from returns.methods import partition
from torch import Tensor

from ip_claim.collision.collide import PatentEmbeddingRecord
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.cover import Covering
from ip_claim.collision.data.citation_pairs import (
    CitationPair,
    EpoProcessorCitationPairSource,
    split_pairs_by_query,
)
from ip_claim.collision.encode_job import CollisionEncodeShardStore

_log = structlog.get_logger(__name__)

Letter = Literal['X', 'Y', 'A']
XaFilterReason = Literal['ok', 'empty_scores', 'empty_xa', 'null_frac']
TOP_SLOT_K = 5
OVERLAP_PER_LETTER = 500
DEFAULT_SHARD_JUDGMENT = 'shard-judgment.json'


class QueryPartnerScore(BaseModel):
    """One cited partner scored against its query claim demand."""

    model_config = ConfigDict(frozen=True)

    query: str
    partner: str
    mark: Letter
    unpaid: float
    covering: float
    demand_l1: float
    residual: tuple[float, ...] = ()


class QueryMacroXaFilter(BaseModel):
    """Why query-macro XA is a number or None.

    None is a missing X-and-A query join or a null unpaid fraction on that
    join. It is not an empty pair table and not a compose NaN.
    """

    model_config = ConfigDict(frozen=True, ser_json_inf_nan='null')

    reason: XaFilterReason
    n_scores: int
    n_queries_xa: int
    n_live: int
    delta: float | None = None


class PairedDeltaSplit(BaseModel):
    """Per-query unpaid and covering deltas for one pair table."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan='null')

    queries_with_x_and_a: int
    queries_with_y_and_a: int
    median_u_x_minus_a: float | None = None
    median_u_y_minus_a: float | None = None
    median_c_x_minus_a: float | None = None
    median_c_y_minus_a: float | None = None
    fraction_x_below_a: float | None = None
    fraction_y_below_a: float | None = None
    fraction_x_below_a_by_demand_quartile: tuple[float | None, ...] = ()


class LetterOverlap(BaseModel):
    """Mean inventory entropy and top-slot overlap for one citation letter."""

    model_config = ConfigDict(frozen=True)

    n_apps: int
    mean_n_entropy: float
    mean_top_slot_overlap: float


class InventorySample(BaseModel):
    """Slot overlap and n entropy on a sampled partner set."""

    model_config = ConfigDict(frozen=True)

    ln_bank: float
    letters: dict[str, LetterOverlap]


class UnionYSplit(BaseModel):
    """Union covering of the two lowest-unpaid Y partners versus A and X."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan='null')

    queries_with_two_y: int
    median_union_u: float | None = None
    median_single_y_u: float | None = None
    median_a_u: float | None = None
    median_x_u: float | None = None
    fraction_union_below_a: float | None = None
    fraction_union_below_x: float | None = None


class ShardJudgment(BaseModel):
    """Covering judgment for the eval and test pair tables."""

    model_config = ConfigDict(frozen=True)

    encoded_apps: int
    scored_pairs: int
    eval_paired: PairedDeltaSplit
    test_paired: PairedDeltaSplit
    inventory: InventorySample
    eval_union_y: UnionYSplit
    test_union_y: UnionYSplit


class IntensityIndex(BaseModel):
    """Claim and full intensities keyed by application number."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    claim: dict[str, Tensor]
    full: dict[str, Tensor]


def citation_letter(pair: CitationPair) -> Letter | None:
    """ST.14 letter for one pair. X wins over Y. Grade fills empty letters."""
    letters = set(pair.marks)
    if 'X' in letters or (not letters and pair.grade >= 2):
        return 'X'
    if 'Y' in letters:
        return 'Y'
    if 'A' in letters or pair.grade == 1:
        return 'A'
    return None


def intensity_index(records: Sequence[PatentEmbeddingRecord]) -> IntensityIndex:
    """Keep applications that stored both claim demand and full-text supply."""
    kept, _ = partition(
        Maybe.do(
            (record.application_number, claim, full)
            for claim in Maybe.from_optional(record.n_entity_claim)
            for full in Maybe.from_optional(record.n_entity_full)
        )
        for record in records
    )
    return IntensityIndex(
        claim={app: claim for app, claim, _full in kept},
        full={app: full for app, _claim, full in kept},
    )


def score_cited_pairs(
    pairs: Sequence[CitationPair],
    index: IntensityIndex,
    covering: Covering,
) -> tuple[QueryPartnerScore, ...]:
    """One reciprocal covering score on stacked cited pairs.

    Leftover unpaid is of stored covering n (occupy plus kept-edge addends).
    """
    live, _ = partition(
        Maybe.do(
            (pair, mark, claim, full)
            for mark in Maybe.from_optional(citation_letter(pair))
            for claim in Maybe.from_optional(index.claim.get(pair.query_application_number))
            for full in Maybe.from_optional(index.full.get(pair.partner_application_number))
        )
        for pair in pairs
    )
    if not live:
        return ()
    scored = covering.reciprocal_score(
        torch.stack(tuple(n_query for _pair, _mark, n_query, _n_doc in live)),
        torch.stack(tuple(n_doc for _pair, _mark, _n_query, n_doc in live)),
    )
    unpaid = scored.unpaid_mass.reshape(-1)
    paid = scored.covering.reshape(-1)
    demand = scored.demand_l1.reshape(-1)
    leftover = scored.residual.reshape(len(live), -1)
    return tuple(
        QueryPartnerScore(
            query=pair.query_application_number,
            partner=pair.partner_application_number,
            mark=mark,
            unpaid=float(unpaid[i].item()),
            covering=float(paid[i].item()),
            demand_l1=float(demand[i].item()),
            residual=tuple(float(slot) for slot in leftover[i].tolist()),
        )
        for i, (pair, mark, _n_query, _n_doc) in enumerate(live)
    )


def partner_frame(scores: Sequence[QueryPartnerScore]) -> pl.DataFrame:
    """Cited-pair scores as a frame for group/join."""
    return pl.DataFrame(tuple(row.model_dump() for row in scores))


def query_macro_xa_filter(scores: Sequence[QueryPartnerScore]) -> QueryMacroXaFilter:
    """Name the query-macro filter that yields XA or None."""
    n_scores = len(scores)
    empty = QueryMacroXaFilter(
        reason='empty_scores',
        n_scores=n_scores,
        n_queries_xa=0,
        n_live=0,
    )
    if not scores:
        return empty
    means = (
        partner_frame(scores)
        .with_columns(
            pl
            .when(pl.col('demand_l1') > 0)
            .then(pl.col('unpaid') / pl.col('demand_l1'))
            .otherwise(None)
            .alias('frac')
        )
        .group_by(['query', 'mark'])
        .agg(pl.col('frac').mean())
    )
    xa = (
        means
        .filter(pl.col('mark') == 'X')
        .select('query', 'frac')
        .join(
            means.filter(pl.col('mark') == 'A').select(
                'query',
                pl.col('frac').alias('frac_a'),
            ),
            on='query',
        )
    )
    n_queries_xa = xa.height
    if n_queries_xa == 0:
        return QueryMacroXaFilter(
            reason='empty_xa',
            n_scores=n_scores,
            n_queries_xa=0,
            n_live=0,
        )
    live = xa.filter(pl.col('frac').is_not_null() & pl.col('frac_a').is_not_null())
    n_live = live.height
    if n_live == 0:
        return QueryMacroXaFilter(
            reason='null_frac',
            n_scores=n_scores,
            n_queries_xa=n_queries_xa,
            n_live=0,
        )
    delta = live.select((pl.col('frac_a') - pl.col('frac')).mean()).item()
    return QueryMacroXaFilter(
        reason='ok',
        n_scores=n_scores,
        n_queries_xa=n_queries_xa,
        n_live=n_live,
        delta=None if delta is None else float(delta),
    )


def query_macro_unpaid_fraction_xa(scores: Sequence[QueryPartnerScore]) -> float | None:
    """Mean unpaid fraction of A minus X, after a per-query letter mean.

    Raw pair unpaid mass scales with claim L1, so one long X query can dominate
    the letter margin while covering (the fraction) does not move. Empty demand
    does not count as paid. Missing X or A on a query drops that query.
    """
    return query_macro_xa_filter(scores).delta


def _letter_means(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by(['query', 'mark']).agg(
        pl.col('unpaid').mean(),
        pl.col('covering').mean(),
        pl.col('demand_l1').first(),
    )


def _median_col(frame: pl.DataFrame, name: str) -> float | None:
    if frame.is_empty():
        return None
    return float(frame.select(pl.col(name).median()).item())


def _win_rate(frame: pl.DataFrame, name: str) -> float | None:
    if frame.is_empty():
        return None
    return float(frame.select((pl.col(name) < 0).mean()).item())


def paired_delta_split(scores: Sequence[QueryPartnerScore]) -> PairedDeltaSplit:
    """Median and win-rate of per-query mean U and C, plus demand-quartile X/A."""

    def join_letters(means: pl.DataFrame, left: Letter, right: Letter) -> pl.DataFrame:
        lhs = means.filter(pl.col('mark') == left)
        rhs = means.filter(pl.col('mark') == right).select('query', 'unpaid', 'covering')
        return lhs.join(rhs, on='query', suffix='_r').with_columns(
            (pl.col('unpaid') - pl.col('unpaid_r')).alias('u_delta'),
            (pl.col('covering') - pl.col('covering_r')).alias('c_delta'),
        )

    def quartile_win_rate(xa: pl.DataFrame) -> tuple[float | None, ...]:
        if xa.is_empty():
            return (None, None, None, None)
        buckets = (
            xa
            .with_columns(
                pl
                .col('demand_l1')
                .qcut(4, labels=['0', '1', '2', '3'], allow_duplicates=True)
                .alias('bucket')
            )
            .group_by('bucket')
            .agg((pl.col('u_delta') < 0).mean().alias('win'))
            .sort('bucket')
        )
        by_label = {str(row['bucket']): float(row['win']) for row in buckets.iter_rows(named=True)}
        return tuple(by_label.get(label) for label in ('0', '1', '2', '3'))

    if not scores:
        return PairedDeltaSplit(queries_with_x_and_a=0, queries_with_y_and_a=0)
    means = _letter_means(partner_frame(scores))
    xa = join_letters(means, 'X', 'A')
    ya = join_letters(means, 'Y', 'A')
    return PairedDeltaSplit(
        queries_with_x_and_a=xa.height,
        queries_with_y_and_a=ya.height,
        median_u_x_minus_a=_median_col(xa, 'u_delta'),
        median_u_y_minus_a=_median_col(ya, 'u_delta'),
        median_c_x_minus_a=_median_col(xa, 'c_delta'),
        median_c_y_minus_a=_median_col(ya, 'c_delta'),
        fraction_x_below_a=_win_rate(xa, 'u_delta'),
        fraction_y_below_a=_win_rate(ya, 'u_delta'),
        fraction_x_below_a_by_demand_quartile=quartile_win_rate(xa),
    )


def n_entropy(intensity: Tensor) -> float:
    """Shannon entropy of one normalized intensity row."""
    return float(_row_entropy(intensity.unsqueeze(0)).item())


def top_slot_overlap(query: Tensor, document: Tensor, *, top_k: int = TOP_SLOT_K) -> float:
    """Fraction of the query's top slots that also lead the document."""
    return float(_row_overlap(query.unsqueeze(0), document.unsqueeze(0), top_k=top_k).item())


def _row_entropy(intensity: Tensor) -> Tensor:
    mass = intensity.to(dtype=torch.float64).clamp_min(0.0)
    total = mass.sum(dim=-1, keepdim=True)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    probs = torch.where(total > 0, mass / safe_total, torch.zeros_like(mass))
    live = probs > 0
    safe_p = torch.where(live, probs, torch.ones_like(probs))
    return torch.where(live, -(probs * safe_p.log()), torch.zeros_like(probs)).sum(dim=-1)


def _row_overlap(query: Tensor, document: Tensor, *, top_k: int) -> Tensor:
    k = min(int(top_k), int(query.size(-1)), int(document.size(-1)))
    if k < 1:
        return query.new_zeros(query.size(0))
    q_idx = query.topk(k, dim=-1).indices.unsqueeze(-1)
    d_idx = document.topk(k, dim=-1).indices.unsqueeze(-2)
    return (q_idx == d_idx).any(dim=-1).sum(dim=-1).to(dtype=torch.float64) / k


def inventory_sample(
    scores: Sequence[QueryPartnerScore],
    index: IntensityIndex,
    *,
    per_letter: int = OVERLAP_PER_LETTER,
    seed: int = 42,
) -> InventorySample:
    """Entropy and top-slot overlap on a declared sample of X / Y / A / random."""
    bank = next(iter(index.full.values())).numel() if index.full else 0
    frame = partner_frame(scores) if scores else pl.DataFrame()
    cited = set(frame['partner'].to_list()) if scores else set()
    generator = torch.Generator().manual_seed(int(seed))
    unused = tuple(app for app in index.full if app not in cited)

    def take_letter(mark: str) -> LetterOverlap:
        rows = frame.filter(pl.col('mark') == mark) if scores else frame
        n = min(per_letter, rows.height)
        if n == 0:
            return LetterOverlap(n_apps=0, mean_n_entropy=0.0, mean_top_slot_overlap=0.0)
        order = torch.randperm(rows.height, generator=generator).tolist()[:n]
        picked = rows[order]
        documents = torch.stack(tuple(index.full[app] for app in picked['partner']))
        queries = torch.stack(tuple(index.claim[app] for app in picked['query']))
        return LetterOverlap(
            n_apps=n,
            mean_n_entropy=float(_row_entropy(documents).mean().item()),
            mean_top_slot_overlap=float(
                _row_overlap(queries, documents, top_k=TOP_SLOT_K).mean().item()
            ),
        )

    letters = {mark: take_letter(mark) for mark in ('X', 'Y', 'A')}
    if unused and scores:
        pick = min(per_letter, len(unused))
        order = torch.randperm(len(unused), generator=generator).tolist()[:pick]
        apps = tuple(unused[i] for i in order)
        anchors = tuple(scores[i % len(scores)].query for i in range(pick))
        documents = torch.stack(tuple(index.full[app] for app in apps))
        queries = torch.stack(tuple(index.claim[app] for app in anchors if app in index.claim))
        letters['random'] = LetterOverlap(
            n_apps=pick,
            mean_n_entropy=float(_row_entropy(documents).mean().item()),
            mean_top_slot_overlap=float(
                _row_overlap(queries, documents[: queries.size(0)], top_k=TOP_SLOT_K).mean().item()
            )
            if queries.size(0)
            else 0.0,
        )
    ln_bank = float(torch.tensor(float(bank)).log().item()) if bank else 0.0
    return InventorySample(ln_bank=ln_bank, letters=letters)


def union_y_split(
    scores: Sequence[QueryPartnerScore],
    index: IntensityIndex,
    covering: Covering,
) -> UnionYSplit:
    """Union of the two lowest-unpaid Y partners versus that query's A and X."""
    empty = UnionYSplit(queries_with_two_y=0)
    if not scores:
        return empty
    frame = partner_frame(scores)
    ranked = (
        frame
        .filter(pl.col('mark') == 'Y')
        .sort(['query', 'unpaid'])
        .with_columns(pl.int_range(pl.len()).over('query').alias('ord'))
    )
    combo = ranked.filter(pl.col('ord') == 0).join(
        ranked.filter(pl.col('ord') == 1).select('query', 'partner'),
        on='query',
        suffix='_b',
    )
    queries = combo['query'].to_list()
    if not queries:
        return empty
    union = covering.union_covering(
        torch.stack(tuple(index.claim[query] for query in queries)),
        torch.stack(tuple(index.full[app] for app in combo['partner'])),
        torch.stack(tuple(index.full[app] for app in combo['partner_b'])),
    )
    letter_u = _letter_means(frame).select('query', 'mark', 'unpaid')
    union_rows = (
        pl
        .DataFrame({
            'query': queries,
            'union_u': union.unpaid_mass.reshape(-1).tolist(),
            'single_y': combo['unpaid'].to_list(),
        })
        .join(
            letter_u.filter(pl.col('mark') == 'A').select('query', pl.col('unpaid').alias('a_u')),
            on='query',
            how='left',
        )
        .join(
            letter_u.filter(pl.col('mark') == 'X').select('query', pl.col('unpaid').alias('x_u')),
            on='query',
            how='left',
        )
    )
    vs_a = union_rows.filter(pl.col('a_u').is_not_null())
    vs_x = union_rows.filter(pl.col('x_u').is_not_null())
    return UnionYSplit(
        queries_with_two_y=union_rows.height,
        median_union_u=_median_col(union_rows, 'union_u'),
        median_single_y_u=_median_col(union_rows, 'single_y'),
        median_a_u=_median_col(vs_a, 'a_u'),
        median_x_u=_median_col(vs_x, 'x_u'),
        fraction_union_below_a=_win_rate(
            vs_a.with_columns((pl.col('union_u') - pl.col('a_u')).alias('u_delta')),
            'u_delta',
        )
        if not vs_a.is_empty()
        else None,
        fraction_union_below_x=_win_rate(
            vs_x.with_columns((pl.col('union_u') - pl.col('x_u')).alias('u_delta')),
            'u_delta',
        )
        if not vs_x.is_empty()
        else None,
    )


def judge_shards(
    records: Sequence[PatentEmbeddingRecord],
    *,
    eval_pairs: Sequence[CitationPair],
    test_pairs: Sequence[CitationPair],
    covering: Covering,
    overlap_per_letter: int = OVERLAP_PER_LETTER,
    seed: int = 42,
) -> ShardJudgment:
    """Paired deltas, inventory sample, and union-Y on eval and test."""
    index = intensity_index(records)
    eval_scores = score_cited_pairs(eval_pairs, index, covering)
    test_scores = score_cited_pairs(test_pairs, index, covering)
    return ShardJudgment(
        encoded_apps=len(index.full),
        scored_pairs=len(eval_scores) + len(test_scores),
        eval_paired=paired_delta_split(eval_scores),
        test_paired=paired_delta_split(test_scores),
        inventory=inventory_sample(
            (*eval_scores, *test_scores),
            index,
            per_letter=overlap_per_letter,
            seed=seed,
        ),
        eval_union_y=union_y_split(eval_scores, index, covering),
        test_union_y=union_y_split(test_scores, index, covering),
    )


def write_shard_judgment(path: Path, judgment: ShardJudgment) -> Path:
    """Write the judgment JSON next to covering artefacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(judgment.model_dump_json(indent=2) + '\n', encoding='utf-8')
    return path


def judge_from_paths(
    *,
    shard_dir: Path,
    dataset: Path,
    output_dir: Path,
    config: CollisionEvalConfig,
    overlap_per_letter: int = OVERLAP_PER_LETTER,
) -> Path:
    """Load shards and the pair table, then write shard-judgment.json."""
    records = CollisionEncodeShardStore(root=shard_dir).try_read_complete()
    if records is None:
        msg = f'complete encode shards are required under {shard_dir}'
        raise FileNotFoundError(msg)
    pairs = EpoProcessorCitationPairSource().load_pairs(dataset)
    _train, eval_pairs, test_pairs = split_pairs_by_query(
        pairs,
        train=config.split.train,
        eval_fraction=config.split.eval,
        test=config.split.test,
        seed=config.split.seed,
        query_limit=config.query_limit,
    )
    judgment = judge_shards(
        records,
        eval_pairs=eval_pairs,
        test_pairs=test_pairs,
        covering=Covering(config.covering),
        overlap_per_letter=overlap_per_letter,
        seed=config.split.seed,
    )
    written = write_shard_judgment(output_dir / DEFAULT_SHARD_JUDGMENT, judgment)
    _log.info(
        'collision.diagnose.written',
        path=str(written),
        encoded_apps=judgment.encoded_apps,
        scored_pairs=judgment.scored_pairs,
        eval_fraction_x_below_a=judgment.eval_paired.fraction_x_below_a,
        eval_union_below_a=judgment.eval_union_y.fraction_union_below_a,
    )
    return written
