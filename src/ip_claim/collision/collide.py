"""Collision ranking metrics on covering intensities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import groupby
from operator import itemgetter
from typing import Any, TypeVar, cast

import ray
import torch
from pydantic import BaseModel, ConfigDict, Field
from returns.functions import raise_exception
from returns.io import IOResultE, impure_safe
from returns.pipeline import managed
from returns.result import Result
from returns.unsafe import unsafe_perform_io
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torchmetrics.functional.retrieval import (
    retrieval_normalized_dcg,
    retrieval_recall,
    retrieval_reciprocal_rank,
)

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.shared.ray_runtime import ensure_local_ray

DEFAULT_RECALL_K = (5, 10, 20, 50)
RANK_QUERY_TILE = 256
# RetrievalRecall / RetrievalMRR zero a target wherever preds <= 0.
COSINE_RETRIEVAL_SHIFT = 2.0
MARK_NONE = 0
MARK_A = 1
MARK_X = 2
MARK_Y = 3

RankingBatch = Data
CorpusIndex = Data
CorpusRankBanks = Data
_PoolUse = TypeVar('_PoolUse')


def _mark_code(letters: set[str], grade: float) -> int:
    """Encode one ST.14 letter set. X wins over Y. Empty letters use the grade."""
    if 'X' in letters or (not letters and grade >= 2):
        return MARK_X
    if 'Y' in letters:
        return MARK_Y
    if 'A' in letters or grade == 1:
        return MARK_A
    return MARK_NONE


def _query_macro_mean(values: Tensor, query_ids: Tensor, pair_mask: Tensor) -> float:
    """Mean of per-query means. No selected pairs is not a zero unpaid score."""
    if not bool(pair_mask.any()):
        return float('nan')
    ids = query_ids[pair_mask]
    chosen = values[pair_mask]
    n_query = int(query_ids.max().item()) + 1
    totals = scatter(chosen, ids, dim=0, dim_size=n_query, reduce='sum')
    counts = scatter(torch.ones_like(chosen), ids, dim=0, dim_size=n_query, reduce='sum')
    live = counts > 0
    if not bool(live.any()):
        return float('nan')
    return float((totals[live] / counts[live]).mean().item())


def _query_macro_random(
    table: Tensor,
    query_ids: Tensor,
    cited: Tensor,
    finite: Tensor,
) -> float:
    """Mean over queries of the mean uncited finite corpus cell."""
    n_query = int(query_ids.max().item()) + 1
    cited_mass = scatter(
        cited.to(dtype=table.dtype),
        query_ids,
        dim=0,
        dim_size=n_query,
        reduce='sum',
    )
    positions = torch.arange(query_ids.size(0), device=query_ids.device)
    first = scatter(positions, query_ids, dim=0, dim_size=n_query, reduce='min')
    rows = table[first]
    keep = finite[first] & (cited_mass <= 0)
    weight = keep.to(dtype=table.dtype)
    mass = (rows * weight).sum(dim=1)
    denom = weight.sum(dim=1)
    live = denom > 0
    if not bool(live.any()):
        return float('nan')
    return float((mass[live] / denom[live]).mean().item())


class PatentEmbeddingRecord(BaseModel):
    """One patent's trunk embedding, covering intensities, and optional filing claim number."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    application_number: str
    z_d: Tensor
    n_entity_claim: Tensor | None = None
    n_entity_full: Tensor | None = None
    n_relation_claim: Tensor | None = None
    n_relation_full: Tensor | None = None
    cpc_section: str | None = None
    claim_number: int | None = None


class CollisionEvalResult(BaseModel):
    """Offline covering tables for one query split against the n-corpus."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan='null')

    recall_at_k: Mapping[int, float]
    particular_recall_at_k: Mapping[int, float]
    cpc_hard_recall_at_k: Mapping[int, float]
    particular_cpc_hard_recall_at_k: Mapping[int, float]
    mrr: float
    particular_mrr: float
    ndcg_at_k: Mapping[int, float]
    queries: int
    queries_x: int = 0
    queries_y: int = 0
    queries_a: int = 0
    unpaid_x: float = float('nan')
    unpaid_y: float = float('nan')
    unpaid_a: float = float('nan')
    unpaid_random: float = float('nan')
    covering_x: float = float('nan')
    covering_y: float = float('nan')
    covering_a: float = float('nan')
    covering_random: float = float('nan')
    edge_unpaid_x: float = float('nan')
    edge_unpaid_y: float = float('nan')
    edge_unpaid_a: float = float('nan')
    edge_unpaid_random: float = float('nan')


class CollisionEvalRequest(BaseModel):
    """Inputs for one offline collision ranking evaluation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    query_records: tuple[PatentEmbeddingRecord, ...]
    corpus_records: tuple[PatentEmbeddingRecord, ...]
    partner_apps: tuple[str, ...]
    relevances: tuple[float, ...] = Field(default_factory=tuple)
    marks: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    k_values: tuple[int, ...] = DEFAULT_RECALL_K


def corpus_index(records: Sequence[PatentEmbeddingRecord]) -> Data:
    """Stack corpus embeddings and attach application and CPC lookup maps."""
    recs = tuple(records)
    section_lists: dict[str, list[int]] = {}
    for idx, row in enumerate(recs):
        if row.cpc_section is not None:
            section_lists.setdefault(row.cpc_section, []).append(idx)
    z_d = torch.stack([row.z_d for row in recs], dim=0) if recs else torch.empty((0,))
    index = Data(z_d=z_d)
    index.records = recs
    index.app_to_index = {row.application_number: idx for idx, row in enumerate(recs)}
    index.section_to_indices = {
        section: tuple(indices) for section, indices in section_lists.items()
    }
    return index


def section_id_tables(
    query_sections: Sequence[str | None],
    corpus_records: Sequence[PatentEmbeddingRecord],
    *,
    device: torch.device,
) -> tuple[dict[str, int], Tensor, Tensor]:
    """Map CPC section strings to ids and build query/corpus section index tensors."""
    section_labels = sorted({
        *(s for s in query_sections if s is not None),
        *(r.cpc_section for r in corpus_records if r.cpc_section is not None),
    })
    section_to_id = dict(zip(section_labels, range(len(section_labels)), strict=True))
    corpus_section_ids = torch.tensor(
        [
            section_to_id.get(row.cpc_section, -1) if row.cpc_section else -1
            for row in corpus_records
        ],
        dtype=torch.long,
        device=device,
    )
    query_section_ids = torch.tensor(
        [
            section_to_id.get(section, -1) if section is not None else -1
            for section in query_sections
        ],
        dtype=torch.long,
        device=device,
    )
    return section_to_id, query_section_ids, corpus_section_ids


def unique_query_records(
    records: Sequence[PatentEmbeddingRecord],
) -> tuple[tuple[PatentEmbeddingRecord, ...], tuple[int, ...]]:
    """First record per application and the pair-to-unique row map."""
    first: dict[str, PatentEmbeddingRecord] = {}
    for row in records:
        first.setdefault(row.application_number, row)
    id_of = {app: index for index, app in enumerate(first)}
    return tuple(first.values()), tuple(id_of[row.application_number] for row in records)


def empty_collision_eval_result(k_values: Sequence[int]) -> CollisionEvalResult:
    """Zeroed metrics when no queries or corpus rows are available."""
    empty = dict.fromkeys(k_values, 0.0)
    return CollisionEvalResult(
        recall_at_k=empty,
        particular_recall_at_k=empty,
        cpc_hard_recall_at_k=empty,
        particular_cpc_hard_recall_at_k=empty,
        mrr=0.0,
        particular_mrr=0.0,
        ndcg_at_k=empty,
        queries=0,
    )


def partner_rank_positions(scores: Tensor, partner_indices: Tensor) -> Tensor:
    """1-based rank of each query's partner column in the score matrix."""
    partner_scores = scores.gather(1, partner_indices.view(-1, 1)).squeeze(1)
    better = (scores > partner_scores.unsqueeze(1)).sum(dim=1)
    return better + 1


def retrieval_labels(
    batch: Data,
    *,
    min_grade: float,
    pool_mask: Tensor | None = None,
    graded: bool = False,
) -> Data:
    """Label partner columns on unique-query rows. Labels stay on the score device."""
    scores = batch.scores.detach()
    query_count, corpus_size = scores.shape
    pair_rows = batch.pair_rows.to(device=scores.device)
    partner = batch.partner_indices.to(device=scores.device)
    relevances = batch.relevances.to(device=scores.device)
    target = torch.zeros((query_count, corpus_size), dtype=torch.long, device=scores.device)
    live = relevances >= min_grade
    gains = relevances.to(dtype=torch.long) if graded else live.to(dtype=torch.long)
    if graded:
        gains = torch.where(live, gains, torch.zeros_like(gains))
    target[pair_rows, partner] = gains
    if pool_mask is not None:
        pool = pool_mask.to(device=scores.device)
        scores = scores.masked_fill(~pool, float('-inf'))
        in_pool = pool[pair_rows, partner]
        target[pair_rows[~in_pool], partner[~in_pool]] = 0
    rows = torch.arange(query_count, device=scores.device)
    return Data(
        preds=scores + COSINE_RETRIEVAL_SHIFT,
        target=target,
        query_index=rows.unsqueeze(1).expand(query_count, corpus_size),
    )


def compute_retrieval_slice(
    batch: Data,
    k_values: Sequence[int],
    *,
    min_grade: float,
    pool_mask: Tensor | None = None,
    include_ndcg: bool = False,
) -> dict[str, float]:
    """Score one grade threshold and optional CPC pool through torchmetrics retrieval."""
    scores = batch.scores.detach()
    if pool_mask is not None:
        scores = scores.masked_fill(~pool_mask.to(device=scores.device), float('-inf'))
    pair_rows = batch.pair_rows.to(device=scores.device)
    partners = batch.partner_indices.to(device=scores.device)
    relevances = batch.relevances.to(device=scores.device)
    live = relevances >= min_grade
    corpus = int(scores.size(1))

    def row_metrics(query: int) -> dict[str, float] | None:
        take = (pair_rows == query) & live
        if not bool(take.any()):
            return None
        hits = partners[take]
        binary = torch.zeros(corpus, dtype=torch.long, device=scores.device)
        binary[hits] = 1
        if not bool((binary > 0).any()):
            return None
        preds = scores[query] + COSINE_RETRIEVAL_SHIFT
        graded = relevances[take].to(dtype=torch.long) if include_ndcg else None
        ndcg_target = torch.zeros(corpus, dtype=torch.long, device=scores.device)
        if graded is not None:
            ndcg_target[hits] = graded
        return {
            **{
                f'recall@{k}': float(retrieval_recall(preds, binary, top_k=k).item())
                for k in k_values
            },
            'mrr': float(retrieval_reciprocal_rank(preds, binary).item()),
            'queries': 1.0,
            **{
                f'ndcg@{k}': (
                    float(retrieval_normalized_dcg(preds, ndcg_target, top_k=k).item())
                    if include_ndcg
                    else 0.0
                )
                for k in k_values
            },
        }

    return _merge_retrieval_slices(
        tuple(
            metrics
            for query in range(int(scores.size(0)))
            if (metrics := row_metrics(query)) is not None
        ),
        k_values,
    )


def _weighted_mean(
    parts: Sequence[float],
    weights: Sequence[int],
    *,
    empty: float,
) -> float:
    """Weighted mean of finite positive-weight cells; ``empty`` when none live."""
    values = torch.tensor(list(parts), dtype=torch.float32)
    mass = torch.tensor(list(weights), dtype=torch.float32)
    live = torch.isfinite(values) & (mass > 0)
    if not bool(live.any()):
        return empty
    return float((values[live] * mass[live]).sum().item() / mass[live].sum().item())


def _merge_retrieval_slices(
    slices: Sequence[Mapping[str, float]],
    k_values: Sequence[int],
) -> dict[str, float]:
    """Mean retrieval metrics across tiles, weighted by labeled pair-rows."""
    if not slices:
        return {
            **{f'recall@{k}': 0.0 for k in k_values},
            'mrr': 0.0,
            **{f'ndcg@{k}': 0.0 for k in k_values},
            'queries': 0.0,
        }
    weights = tuple(int(item.get('queries', 0)) for item in slices)
    return {
        **{
            f'recall@{k}': _weighted_mean(
                tuple(float(item[f'recall@{k}']) for item in slices),
                weights,
                empty=0.0,
            )
            for k in k_values
        },
        'mrr': _weighted_mean(tuple(float(item['mrr']) for item in slices), weights, empty=0.0),
        **{
            f'ndcg@{k}': _weighted_mean(
                tuple(float(item.get(f'ndcg@{k}', 0.0)) for item in slices),
                weights,
                empty=0.0,
            )
            for k in k_values
        },
        'queries': float(sum(weights)),
    }


def _stack_intensity(rows: Sequence[PatentEmbeddingRecord], name: str) -> Tensor | None:
    """Stack one intensity column. Missing values fail closed."""
    values = tuple(getattr(row, name) for row in rows)
    if not values or any(item is None for item in values):
        return None
    return torch.stack(values, dim=0)


def corpus_rank_banks(
    records: Sequence[PatentEmbeddingRecord],
    devices: Sequence[torch.device],
) -> Data | None:
    """Stack corpus intensities once and keep one full document replica.

    More than one CUDA home leaves that replica on CPU so Ray workers can
    claim the cards. A single home receives the replica on that device.
    """
    recs = tuple(records)
    n_entity = _stack_intensity(recs, 'n_entity_full')
    n_relation = _stack_intensity(recs, 'n_relation_full')
    if n_entity is None or n_relation is None:
        return None
    homes = tuple(devices) if devices else (n_entity.device,)
    cuda = tuple(home for home in homes if home.type == 'cuda')
    home = torch.device('cpu') if len(cuda) > 1 else homes[0]
    banks = Data()
    banks.entity_shards = (n_entity.to(device=home, non_blocking=home.type == 'cuda'),)
    banks.relation_shards = (n_relation.to(device=home, non_blocking=home.type == 'cuda'),)
    banks.lookup = corpus_index(recs)
    return banks


@ray.remote
class CoveringRankActor:
    """One rank worker that scores unique-query tiles against a full document replica."""

    def __init__(
        self,
        knobs: Mapping[str, object],
        n_entity: Tensor,
        n_relation: Tensor,
        lookup: Data,
        *,
        use_gpu: bool,
    ) -> None:
        device = torch.device('cuda' if use_gpu else 'cpu')
        self.covering = Covering(CoveringKnobs.model_validate(knobs)).to(device)
        self.banks = Data()
        self.banks.entity_shards = (n_entity.to(device=device, non_blocking=device.type == 'cuda'),)
        self.banks.relation_shards = (
            n_relation.to(device=device, non_blocking=device.type == 'cuda'),
        )
        self.banks.lookup = lookup

    def score_tile(
        self,
        request: CollisionEvalRequest,
        pair_ids: tuple[int, ...],
        id_of: Mapping[str, int],
    ) -> Data | None:
        """Score one unique-query tile and return host tensors for the driver."""
        tile = score_ranking_tile(request, self.covering, pair_ids, id_of, self.banks)
        if tile is None:
            return None

        def host_value(value: object) -> object:
            if isinstance(value, Tensor):
                return value.detach().cpu()
            return value

        hosted = Data()
        hosted.unpaid = host_value(tile.unpaid)
        hosted.covering = host_value(tile.covering)
        hosted.edge = host_value(tile.edge)
        hosted.marks = host_value(tile.marks)
        hosted.query_ids = host_value(tile.query_ids)
        hosted.cited = tile.cited
        hosted.particular = tile.particular
        hosted.cpc = tile.cpc
        hosted.particular_cpc = tile.particular_cpc
        hosted.random_u = tile.random_u
        hosted.random_c = tile.random_c
        hosted.random_e = tile.random_e
        hosted.weight = tile.weight
        return hosted


class CoveringRankPool:
    """Query-tile rank workers. Ray GPU actors when more than one CUDA device is live."""

    def __init__(
        self,
        covering: Covering,
        banks: Data,
        actors: tuple[Any, ...] | None,
        *,
        backend: str,
        worker_count: int,
        owns_ray: bool,
    ) -> None:
        self.covering = covering
        self.banks = banks
        self.actors = actors
        self.backend = backend
        self.worker_count = worker_count
        self.owns_ray = owns_ray

    @classmethod
    def local(cls, covering: Covering, banks: Data) -> CoveringRankPool:
        """Single-process tiles on the driver replica."""
        return cls(
            covering,
            banks,
            None,
            backend='local',
            worker_count=1,
            owns_ray=False,
        )

    @classmethod
    def ray_workers(
        cls,
        covering: Covering,
        banks: Data,
        *,
        count: int,
        use_gpu: bool,
    ) -> CoveringRankPool:
        """Start one Ray actor per worker. Each actor holds a full document replica."""

        def knobs_payload() -> dict[str, object]:
            def optional_int(buffer: Tensor) -> int | None:
                value = int(buffer.item())
                return None if value < 1 else value

            def optional_mass(buffer: Tensor) -> float | None:
                value = float(buffer.item())
                return None if value <= 0.0 else value

            return cast(
                dict[str, object],
                CoveringKnobs(
                    sigma=float(covering.sigma.item()),
                    sigma_edge=float(covering.sigma_edge.item()),
                    lambda_relation=float(covering.lambda_relation.item()),
                    row_top_k=optional_int(covering.row_top_k),
                    row_mass_keep=optional_mass(covering.row_mass_keep),
                    slot_top_k=optional_int(covering.slot_top_k),
                    slot_mass_keep=optional_mass(covering.slot_mass_keep),
                ).model_dump(),
            )

        def slim_lookup(lookup: Data) -> Data:
            slim = Data()
            slim.app_to_index = dict(lookup.app_to_index)
            slim.records = tuple(
                PatentEmbeddingRecord(
                    application_number=row.application_number,
                    z_d=torch.empty(0),
                    cpc_section=row.cpc_section,
                )
                for row in lookup.records
            )
            return slim

        owns_ray = ensure_local_ray(num_gpus=count if use_gpu else 0)
        knobs = knobs_payload()
        entity_ref = ray.put(banks.entity_shards[0].detach().cpu())
        relation_ref = ray.put(banks.relation_shards[0].detach().cpu())
        lookup_ref = ray.put(slim_lookup(banks.lookup))
        remote_actor = cast(Any, CoveringRankActor)
        actors = tuple(
            remote_actor.options(num_gpus=1 if use_gpu else 0).remote(
                knobs,
                entity_ref,
                relation_ref,
                lookup_ref,
                use_gpu=use_gpu,
            )
            for _ in range(count)
        )
        return cls(
            covering,
            banks,
            actors,
            backend='ray',
            worker_count=count,
            owns_ray=owns_ray,
        )

    @classmethod
    def open(
        cls,
        covering: Covering,
        banks: Data,
        devices: Sequence[torch.device],
    ) -> CoveringRankPool:
        """Ray GPU actors when more than one CUDA device is visible; else local tiles."""
        cuda = tuple(device for device in devices if device.type == 'cuda')
        if len(cuda) > 1:
            return cls.ray_workers(covering, banks, count=len(cuda), use_gpu=True)
        return cls.local(covering, banks)

    def score_tiles(
        self,
        request: CollisionEvalRequest,
        pair_tiles: Sequence[tuple[int, ...]],
        id_of: Mapping[str, int],
    ) -> tuple[Data | None, ...]:
        """Score unique-query tiles locally or across Ray workers."""
        tiles = tuple(pair_tiles)
        if self.actors is None:
            return tuple(
                score_ranking_tile(request, self.covering, pair_ids, id_of, self.banks)
                for pair_ids in tiles
            )

        def tile_request(pair_ids: tuple[int, ...]) -> CollisionEvalRequest:
            def take_pairs(name: str) -> tuple[object, ...]:
                values = getattr(request, name)
                if not values:
                    return ()
                return tuple(values[index] for index in pair_ids)

            return request.model_copy(
                update={
                    'query_records': tuple(request.query_records[index] for index in pair_ids),
                    'corpus_records': (),
                    'partner_apps': tuple(request.partner_apps[index] for index in pair_ids),
                    'relevances': take_pairs('relevances'),
                    'marks': take_pairs('marks'),
                }
            )

        id_ref = ray.put(dict(id_of))
        return tuple(
            ray.get([
                self.actors[index % len(self.actors)].score_tile.remote(
                    tile_request(pair_ids),
                    tuple(range(len(pair_ids))),
                    id_ref,
                )
                for index, pair_ids in enumerate(tiles)
            ])
        )

    def close(self) -> None:
        """Release actors and, when this pool started Ray, shut the local runtime down."""
        if self.actors is not None:
            for actor in self.actors:
                ray.kill(actor)
            self.actors = None
        if self.owns_ray and ray.is_initialized():
            ray.shutdown()
            self.owns_ray = False

    @staticmethod
    @impure_safe
    def acquire(
        covering: Covering,
        banks: Data,
        devices: Sequence[torch.device],
    ) -> CoveringRankPool:
        """Open workers for one rank pass."""
        return CoveringRankPool.open(covering, banks, devices)

    @staticmethod
    @impure_safe
    def release(pool: CoveringRankPool, _: Result[object, Exception]) -> None:
        """Tear down workers after use, including a failed score."""
        pool.close()

    @staticmethod
    def run(
        acquire: IOResultE[CoveringRankPool],
        use: Callable[[CoveringRankPool], IOResultE[_PoolUse]],
    ) -> _PoolUse:
        """Acquire, use, always release; re-raise the original exception at this edge."""
        pipeline = cast(
            Callable[[IOResultE[CoveringRankPool]], IOResultE[_PoolUse]],
            managed(use, CoveringRankPool.release),
        )
        return unsafe_perform_io(pipeline(acquire).alt(raise_exception).unwrap())

    @staticmethod
    def score_request(
        covering: Covering,
        banks: Data,
        devices: Sequence[torch.device],
        request: CollisionEvalRequest,
        pair_tiles: Sequence[tuple[int, ...]],
        id_of: Mapping[str, int],
        pool: CoveringRankPool | None,
    ) -> tuple[Data | None, ...]:
        """Score tiles on a borrowed pool, or acquire, score, and release."""
        if pool is not None:
            return pool.score_tiles(request, pair_tiles, id_of)

        @impure_safe
        def score_owned(workers: CoveringRankPool) -> tuple[Data | None, ...]:
            return workers.score_tiles(request, pair_tiles, id_of)

        return CoveringRankPool.run(
            CoveringRankPool.acquire(covering, banks, devices),
            score_owned,
        )


def cpc_hard_pool_mask(
    request: CollisionEvalRequest,
    index: Data | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Same-CPC-section corpus mask that excludes each query's own row."""
    lookup = index if index is not None else corpus_index(request.corpus_records)
    home = device if device is not None else lookup.z_d.device
    n_corpus = len(lookup.records)
    unique_rows, _ = unique_query_records(request.query_records)
    query_sections = tuple(row.cpc_section for row in unique_rows)
    _, query_section_ids, corpus_section_ids = section_id_tables(
        query_sections,
        lookup.records,
        device=home,
    )
    query_self_indices = torch.tensor(
        [lookup.app_to_index.get(row.application_number, -1) for row in unique_rows],
        dtype=torch.long,
        device=home,
    )
    corpus_idx = torch.arange(n_corpus, device=home)
    same_section = query_section_ids.unsqueeze(1) == corpus_section_ids.unsqueeze(0)
    not_self = corpus_idx.unsqueeze(0) != query_self_indices.unsqueeze(1)
    return same_section & not_self & (query_section_ids.unsqueeze(1) >= 0)


def cpc_hard_rank_positions(
    query_records: Sequence[PatentEmbeddingRecord],
    corpus_records: Sequence[PatentEmbeddingRecord],
    partner_apps: Sequence[str],
    *,
    covering: Covering | None = None,
) -> Tensor:
    """1-based partner rank within each query's same-CPC-section pool."""
    request = CollisionEvalRequest(
        query_records=tuple(query_records),
        corpus_records=tuple(corpus_records),
        partner_apps=tuple(partner_apps),
    )
    batch = prepare_ranking_batch(
        request, covering if covering is not None else Covering(CoveringKnobs())
    )
    if batch is None:
        return torch.tensor([], dtype=torch.long)
    pool_mask = cpc_hard_pool_mask(request).to(device=batch.scores.device)
    partner_in_pool = pool_mask[batch.pair_rows, batch.partner_indices]
    if not partner_in_pool.any():
        return torch.tensor([], dtype=torch.long, device=batch.scores.device)
    masked = batch.scores.masked_fill(~pool_mask, float('-inf'))
    return partner_rank_positions(masked[batch.pair_rows], batch.partner_indices)[partner_in_pool]


def prepare_ranking_batch(
    request: CollisionEvalRequest,
    covering: Covering,
    banks: Data | None = None,
) -> Data | None:
    """Score query claim intensity against corpus full-text intensity with covering."""
    if not request.query_records:
        return None
    if not request.corpus_records and banks is None:
        return None
    unique_rows, pair_row_ids = unique_query_records(request.query_records)
    n_query = _stack_intensity(unique_rows, 'n_entity_claim')
    n_rel_query = _stack_intensity(unique_rows, 'n_relation_claim')
    live = (
        banks
        if banks is not None
        else corpus_rank_banks(
            request.corpus_records,
            (n_query.device,) if n_query is not None else (torch.device('cpu'),),
        )
    )
    if n_query is None or n_rel_query is None or live is None:
        return None
    home = live.entity_shards[0].device
    n_query = n_query.to(device=home, non_blocking=home.type == 'cuda')
    n_rel_query = n_rel_query.to(device=home, non_blocking=home.type == 'cuda')
    entity = covering.pair_tables(n_query, live.entity_shards)
    relation = covering.relation_tables(n_rel_query, live.relation_shards)
    mix = covering.lambda_relation.to(device=entity.covering.device, dtype=entity.covering.dtype)
    scores = entity.covering + mix * relation.covering
    unpaid = entity.unpaid_mass
    edge_unpaid = 1.0 - relation.covering
    lookup = live.lookup
    partner_indices = torch.tensor(
        [lookup.app_to_index[app] for app in request.partner_apps],
        dtype=torch.long,
        device=scores.device,
    )
    pair_rows = torch.tensor(pair_row_ids, dtype=torch.long, device=scores.device)
    row_indices = torch.arange(len(unique_rows), device=scores.device)
    self_indices = torch.tensor(
        [lookup.app_to_index.get(row.application_number, -1) for row in unique_rows],
        dtype=torch.long,
        device=scores.device,
    )
    valid_self = self_indices >= 0
    scores = scores.clone()
    scores[row_indices[valid_self], self_indices[valid_self]] = float('-inf')
    relevances = torch.tensor(
        list(request.relevances) if request.relevances else [1.0] * len(request.partner_apps),
        dtype=torch.float32,
        device=scores.device,
    )
    grades = relevances.detach().cpu().tolist()
    marks = request.marks
    return Data(
        scores=scores,
        unpaid=unpaid,
        covering=entity.covering,
        demand=entity.demand_l1,
        edge_unpaid=edge_unpaid,
        pair_rows=pair_rows,
        partner_indices=partner_indices,
        relevances=relevances,
        mark_codes=torch.tensor(
            [
                _mark_code(
                    set(marks[index]) if index < len(marks) else set(),
                    grades[index] if index < len(grades) else 0.0,
                )
                for index in range(len(request.partner_apps))
            ],
            dtype=torch.long,
            device=scores.device,
        ),
    )


def score_ranking_tile(
    request: CollisionEvalRequest,
    covering: Covering,
    pair_ids: tuple[int, ...],
    id_of: Mapping[str, int],
    banks: Data | None = None,
) -> Data | None:
    """Score one unique-query tile. Retrieval state is not kept across tiles."""

    def take_pairs(name: str) -> tuple[object, ...]:
        values = getattr(request, name)
        if not values:
            return ()
        return tuple(values[index] for index in pair_ids)

    piece = request.model_copy(
        update={
            'query_records': tuple(request.query_records[index] for index in pair_ids),
            'partner_apps': tuple(request.partner_apps[index] for index in pair_ids),
            'relevances': take_pairs('relevances'),
            'marks': take_pairs('marks'),
        }
    )
    batch = prepare_ranking_batch(piece, covering, banks)
    if batch is None:
        return None
    pool = cpc_hard_pool_mask(
        piece,
        index=None if banks is None else banks.lookup,
        device=batch.scores.device,
    )
    k_values = request.k_values
    tile_apps = tuple(request.query_records[index].application_number for index in pair_ids)
    local_of = {app: index for index, app in enumerate(dict.fromkeys(tile_apps))}
    pair_rows = batch.pair_rows
    partners = batch.partner_indices
    cited_cells = torch.zeros_like(batch.unpaid, dtype=torch.bool)
    cited_cells[pair_rows, partners] = True
    finite = torch.isfinite(batch.scores)
    local_ids = torch.arange(batch.scores.size(0), device=batch.scores.device)
    tile = Data(
        unpaid=batch.unpaid[pair_rows, partners],
        covering=batch.covering[pair_rows, partners],
        edge=batch.edge_unpaid[pair_rows, partners],
        marks=batch.mark_codes,
        query_ids=torch.tensor(
            [id_of[app] for app in tile_apps],
            dtype=torch.long,
            device=batch.scores.device,
        ),
    )
    tile.cited = compute_retrieval_slice(batch, k_values, min_grade=1.0, include_ndcg=True)
    tile.particular = compute_retrieval_slice(batch, k_values, min_grade=2.0)
    tile.cpc = compute_retrieval_slice(batch, k_values, min_grade=1.0, pool_mask=pool)
    tile.particular_cpc = compute_retrieval_slice(batch, k_values, min_grade=2.0, pool_mask=pool)
    tile.random_u = _query_macro_random(batch.unpaid, local_ids, cited_cells, finite)
    tile.random_c = _query_macro_random(batch.covering, local_ids, cited_cells, finite)
    tile.random_e = _query_macro_random(batch.edge_unpaid, local_ids, cited_cells, finite)
    tile.weight = len(local_of)
    return tile


def evaluate_collision_ranking(
    query_records: Sequence[PatentEmbeddingRecord],
    corpus_records: Sequence[PatentEmbeddingRecord],
    partner_apps: Sequence[str],
    *,
    relevances: Sequence[float] | None = None,
    marks: Sequence[Sequence[str]] | None = None,
    k_values: Sequence[int] = DEFAULT_RECALL_K,
    covering: Covering | None = None,
    query_tile: int = RANK_QUERY_TILE,
    devices: Sequence[torch.device] | None = None,
    banks: Data | None = None,
    pool: CoveringRankPool | None = None,
) -> CollisionEvalResult:
    """Rank the n-corpus by saturation covering and tabulate U by ST.14 letter."""
    request = CollisionEvalRequest(
        query_records=tuple(query_records),
        corpus_records=tuple(corpus_records),
        partner_apps=tuple(partner_apps),
        relevances=tuple(relevances) if relevances is not None else (),
        marks=tuple(tuple(item) for item in marks) if marks is not None else (),
        k_values=tuple(k_values),
    )
    scorer = covering if covering is not None else Covering(CoveringKnobs())
    live_banks = (
        banks
        if banks is not None
        else corpus_rank_banks(
            request.corpus_records,
            devices or (torch.device('cpu'),),
        )
    )
    if live_banks is None:
        return empty_collision_eval_result(k_values)
    apps = tuple(row.application_number for row in request.query_records)
    if not apps:
        return empty_collision_eval_result(k_values)
    id_of = {app: index for index, app in enumerate(dict.fromkeys(apps))}
    pairs_by_query = {
        app: tuple(index for index, _name in group)
        for app, group in groupby(
            sorted(enumerate(apps), key=itemgetter(1)),
            key=itemgetter(1),
        )
    }
    unique_apps = tuple(id_of)
    limit = max(int(query_tile), 1)
    pair_tiles = tuple(
        tuple(index for app in unique_apps[start : start + limit] for index in pairs_by_query[app])
        for start in range(0, len(unique_apps), limit)
    )
    scored = CoveringRankPool.score_request(
        scorer,
        live_banks,
        devices or (torch.device('cpu'),),
        request,
        pair_tiles,
        id_of,
        pool,
    )
    if not scored or any(item is None for item in scored):
        return empty_collision_eval_result(k_values)
    tiles = tuple(item for item in scored if item is not None)
    unpaid = torch.cat(tuple(item.unpaid for item in tiles))
    covering_cited = torch.cat(tuple(item.covering for item in tiles))
    edge = torch.cat(tuple(item.edge for item in tiles))
    codes = torch.cat(tuple(item.marks for item in tiles))
    query_ids = torch.cat(tuple(item.query_ids for item in tiles))
    x_mask = codes == MARK_X
    y_mask = codes == MARK_Y
    a_mask = codes == MARK_A
    cited = _merge_retrieval_slices(tuple(item.cited for item in tiles), k_values)
    particular = _merge_retrieval_slices(tuple(item.particular for item in tiles), k_values)
    cpc_hard = _merge_retrieval_slices(tuple(item.cpc for item in tiles), k_values)
    particular_cpc = _merge_retrieval_slices(tuple(item.particular_cpc for item in tiles), k_values)

    def live_query_count(pair_mask: Tensor) -> int:
        if not bool(pair_mask.any()):
            return 0
        n_query = int(query_ids.max().item()) + 1
        counts = scatter(
            torch.ones(int(pair_mask.sum().item()), device=query_ids.device),
            query_ids[pair_mask],
            dim=0,
            dim_size=n_query,
            reduce='sum',
        )
        return int((counts > 0).sum().item())

    return CollisionEvalResult(
        recall_at_k={k: cited[f'recall@{k}'] for k in k_values},
        particular_recall_at_k={k: particular[f'recall@{k}'] for k in k_values},
        cpc_hard_recall_at_k={k: cpc_hard[f'recall@{k}'] for k in k_values},
        particular_cpc_hard_recall_at_k={k: particular_cpc[f'recall@{k}'] for k in k_values},
        mrr=cited['mrr'],
        particular_mrr=particular['mrr'],
        ndcg_at_k={k: cited[f'ndcg@{k}'] for k in k_values},
        queries=len(id_of),
        queries_x=live_query_count(x_mask),
        queries_y=live_query_count(y_mask),
        queries_a=live_query_count(a_mask),
        unpaid_x=_query_macro_mean(unpaid, query_ids, x_mask),
        unpaid_y=_query_macro_mean(unpaid, query_ids, y_mask),
        unpaid_a=_query_macro_mean(unpaid, query_ids, a_mask),
        unpaid_random=_weighted_mean(
            tuple(item.random_u for item in tiles),
            tuple(item.weight for item in tiles),
            empty=float('nan'),
        ),
        covering_x=_query_macro_mean(covering_cited, query_ids, x_mask),
        covering_y=_query_macro_mean(covering_cited, query_ids, y_mask),
        covering_a=_query_macro_mean(covering_cited, query_ids, a_mask),
        covering_random=_weighted_mean(
            tuple(item.random_c for item in tiles),
            tuple(item.weight for item in tiles),
            empty=float('nan'),
        ),
        edge_unpaid_x=_query_macro_mean(edge, query_ids, x_mask),
        edge_unpaid_y=_query_macro_mean(edge, query_ids, y_mask),
        edge_unpaid_a=_query_macro_mean(edge, query_ids, a_mask),
        edge_unpaid_random=_weighted_mean(
            tuple(item.random_e for item in tiles),
            tuple(item.weight for item in tiles),
            empty=float('nan'),
        ),
    )


__all__ = [
    'COSINE_RETRIEVAL_SHIFT',
    'DEFAULT_RECALL_K',
    'RANK_QUERY_TILE',
    'CollisionEvalRequest',
    'CollisionEvalResult',
    'CorpusIndex',
    'CorpusRankBanks',
    'CoveringRankActor',
    'CoveringRankPool',
    'PatentEmbeddingRecord',
    'RankingBatch',
    'compute_retrieval_slice',
    'corpus_index',
    'corpus_rank_banks',
    'cpc_hard_pool_mask',
    'cpc_hard_rank_positions',
    'empty_collision_eval_result',
    'evaluate_collision_ranking',
    'partner_rank_positions',
    'prepare_ranking_batch',
    'retrieval_labels',
    'score_ranking_tile',
    'section_id_tables',
]
