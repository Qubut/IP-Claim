"""Collision eval orchestration: load citation pairs, encode HUPD patents, rank."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import lightning.pytorch as pl
import structlog
import torch
from pydantic import BaseModel, ConfigDict, Field
from returns.io import impure_safe
from returns.result import Failure
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from ip_claim.collision.artefacts import (
    ExplainArtefact,
    ExplainState,
    artefacts_for_keys,
    report_from_splits,
    top_hit_keys,
    write_covering_report,
)
from ip_claim.collision.collide import (
    RANK_QUERY_TILE,
    CollisionEvalResult,
    CoveringRankPool,
    PatentEmbeddingRecord,
    corpus_rank_banks,
    empty_collision_eval_result,
    evaluate_collision_ranking,
)
from ip_claim.collision.config import CollisionEvalConfig, CollisionPrefixMode
from ip_claim.collision.cover import Covering
from ip_claim.collision.data.citation_pairs import (
    CitationPair,
    CitationPairSource,
    split_pairs_by_query,
)
from ip_claim.collision.encode_job import (
    CollisionEncodeBatch,
    CollisionEncodeCollate,
    CollisionEncodeDataset,
    CollisionEncodeModule,
    CollisionEncodeProbe,
    CollisionEncodeRow,
    CollisionEncodeShardStore,
    CollisionEncodeStep,
    CollisionEncodeTrainerSpec,
    CollisionEncodeWriter,
    CollisionExplainWriter,
    collate_encode_rows,
    collision_encode_trainer,
    encode_collision_records,
    encode_rows_from_hupd,
    encode_trainer_kwargs,
    explain_states_from_batch,
    export_encode_batch,
    resolve_encode_devices,
)
from ip_claim.collision.explain import Explain
from ip_claim.ingestion.adapters.hupd_json.paths import HupdPathIndex, hupd_json_files
from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmExample
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.load_weights import align_hgt_key, load_lightning_payload
from ip_claim.ssv.model import SoftTrunkModel, build_soft_trunk
from ip_claim.ssv.module import SsvLightningModule

_log = structlog.get_logger(__name__)


class CollisionJobResult(BaseModel):
    """Covering tables and explain hits for the train, eval, and test splits."""

    model_config = ConfigDict(frozen=True)

    train: CollisionEvalResult
    eval: CollisionEvalResult
    test: CollisionEvalResult
    checkpoint: str | None = None
    encoded_apps: int = 0
    explain: Mapping[str, tuple[ExplainArtefact, ...]] = Field(default_factory=dict)


class CollisionEvalQueue(BaseModel):
    """Split citation pairs plus the HUPD paths queued for trunk export."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    train_pairs: tuple[CitationPair, ...]
    eval_pairs: tuple[CitationPair, ...]
    test_pairs: tuple[CitationPair, ...]
    items: tuple[tuple[str, Path], ...]


def hupd_stem_index(hupd_dir: Path, index_cache: Path | None = None) -> Mapping[str, Path]:
    """Map HUPD application-number stems to JSON paths under the dump root."""
    paths = (
        HupdPathIndex(root=hupd_dir, cache_path=index_cache).load()
        if index_cache is not None
        else tuple(hupd_json_files(hupd_dir))
    )
    return {path.stem: path for path in paths}


def resolve_checkpoint(path: Path) -> Path:
    """Resolve a checkpoint file or the newest trained weights under a directory."""
    if path.is_file():
        return path
    last = path / 'last.ckpt'
    if last.is_file():
        return last
    steps = tuple(sorted(path.glob('ssv-step*.ckpt'), key=lambda item: item.stat().st_mtime))
    if steps:
        return steps[-1]
    msg = f'no collision trunk checkpoint under {path}'
    raise FileNotFoundError(msg)


def load_collision_trunk(eval_config: CollisionEvalConfig) -> tuple[SsvTrainConfig, SoftTrunkModel]:
    """Build the trunk from the SSV train YAML and optionally load Lightning weights."""
    ssv_path = Path(eval_config.ssv_config) if eval_config.ssv_config else None
    ssv_config = SsvTrainConfig.from_yaml(ssv_path) if ssv_path is not None else SsvTrainConfig()
    model = build_soft_trunk(ssv_config)
    if not eval_config.checkpoint:
        return ssv_config, model
    ckpt = resolve_checkpoint(Path(eval_config.checkpoint))
    loaded = load_lightning_payload(ckpt)
    if isinstance(loaded, Failure):
        raise FileNotFoundError(loaded.failure())
    aligned = {align_hgt_key(key): value for key, value in loaded.unwrap().items()}
    module = SsvLightningModule(ssv_config, model)
    module.load_state_dict(aligned, strict=True)
    _log.info('collision.eval.checkpoint', path=str(ckpt))
    return ssv_config, module.model


def resolve_rank_devices(
    num_devices: int | None,
    *,
    use_gpu: bool,
) -> tuple[torch.device, ...]:
    """List visible rank devices. More than one CUDA card starts Ray query workers."""
    if not use_gpu or not torch.cuda.is_available():
        return (torch.device('cpu'),)
    visible = torch.cuda.device_count()
    count = visible if num_devices is None else min(int(num_devices), visible)
    return tuple(torch.device(f'cuda:{index}') for index in range(max(count, 1)))


def resolve_explain_device(*, use_gpu: bool) -> torch.device:
    """Pin explain re-encode to CUDA when requested.

    Lightning predict teardown leaves the trunk on CPU.
    """
    if not use_gpu:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        msg = 'collision explain requires CUDA when use_gpu is true'
        raise RuntimeError(msg)
    return torch.device('cuda')


def explain_predict_strategy(
    num_devices: int | None,
    *,
    use_gpu: bool,
) -> CollisionEncodeTrainerSpec:
    """Trainer spec for explain. Multi-GPU uses spawn so the CLI is not re-entered."""
    spec = encode_trainer_kwargs(num_devices, use_gpu=use_gpu)
    if spec['accelerator'] != 'gpu':
        return spec
    count = torch.cuda.device_count() if spec['devices'] == 'auto' else int(spec['devices'])
    if count <= 1:
        return spec
    return CollisionEncodeTrainerSpec(
        accelerator=spec['accelerator'],
        devices=count,
        strategy='ddp_spawn',
    )


def collect_explain_states(
    items: Sequence[tuple[str, Path | Mapping[str, object]]],
    *,
    collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch],
    model: SoftTrunkModel,
    inventory: Inventory,
    explain: Explain,
    prefix_mode: CollisionPrefixMode,
    batch_size: int,
    use_gpu: bool,
    num_devices: int | None = None,
    num_workers: int = 0,
) -> dict[str, ExplainState]:
    """Re-encode only the explained applications; do not persist assignment on disk."""
    if not items:
        return {}
    if use_gpu:
        _ = resolve_explain_device(use_gpu=True)
    spec = explain_predict_strategy(num_devices, use_gpu=use_gpu)
    _log.info(
        'collision.eval.explain_start',
        apps=len(items),
        devices=spec['devices'],
        strategy=spec['strategy'],
        batch_size=int(batch_size),
    )

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        trainer = pl.Trainer(
            accelerator=spec['accelerator'],
            devices=spec['devices'],
            strategy=spec['strategy'],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[CollisionExplainWriter(root)],
        )
        _ = trainer.predict(
            CollisionEncodeModule(
                model,
                prefix_mode=prefix_mode,
                inventory=inventory,
                explain=explain,
            ),
            dataloaders=DataLoader(
                CollisionEncodeDataset(items),
                batch_size=min(int(batch_size), max(len(items), 1)),
                shuffle=False,
                collate_fn=CollisionEncodeCollate(collator, inventory),
                num_workers=int(num_workers),
                persistent_workers=int(num_workers) > 0,
            ),
            return_predictions=False,
        )
        states: dict[str, ExplainState] = CollisionExplainWriter.read_all(root)
    _log.info(
        'collision.eval.explain_done',
        apps=len(states),
        devices=spec['devices'],
        strategy=spec['strategy'],
    )
    return states


def encode_shard_root(eval_config: CollisionEvalConfig) -> Path:
    """Shard directory: explicit encode_shards, else output_dir/encode-shards."""
    if eval_config.encode_shards:
        return Path(eval_config.encode_shards)
    return Path(eval_config.output_dir) / 'encode-shards'


def split_collision_pairs(
    eval_config: CollisionEvalConfig,
    *,
    dataset_path: Path,
    pair_source: CitationPairSource,
) -> tuple[tuple[CitationPair, ...], tuple[CitationPair, ...], tuple[CitationPair, ...]] | None:
    """Load citation pairs and cut the locked query-application split."""
    pairs = pair_source.load_pairs(dataset_path)
    if not pairs:
        return None
    train_pairs, eval_pairs, test_pairs = split_pairs_by_query(
        pairs,
        train=eval_config.split.train,
        eval_fraction=eval_config.split.eval,
        test=eval_config.split.test,
        seed=eval_config.split.seed,
        query_limit=eval_config.query_limit,
    )
    return train_pairs, eval_pairs, test_pairs


def resolve_encoded_corpus(
    eval_config: CollisionEvalConfig,
    queued: CollisionEvalQueue,
    *,
    collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch],
    model: SoftTrunkModel,
    inventory: Inventory | None = None,
) -> tuple[PatentEmbeddingRecord, ...] | None:
    """Reuse complete encode shards when present; otherwise run Lightning predict."""
    store = CollisionEncodeShardStore(root=encode_shard_root(eval_config))
    if eval_config.reuse_encode:
        reused = store.try_read_complete()
        if reused is not None:
            _log.info(
                'collision.eval.encode_reuse',
                encoded_apps=len(reused),
                shard_dir=str(store.root),
            )
            return cast(tuple[PatentEmbeddingRecord, ...] | None, reused)
    return cast(
        tuple[PatentEmbeddingRecord, ...] | None,
        encode_collision_records(
            queued.items,
            collator=collator,
            model=model,
            batch_size=int(eval_config.encode_batch_size),
            num_devices=eval_config.num_devices,
            use_gpu=bool(eval_config.use_gpu),
            num_workers=int(eval_config.dataloader_num_workers),
            shard_dir=store.root,
            log_dir=Path(eval_config.output_dir),
            shard_flush_every=int(eval_config.shard_flush_every),
            prefix_mode=eval_config.prefix_mode,
            inventory=inventory,
            disclosure_max_chunks=eval_config.disclosure_max_chunks,
        ),
    )


def rank_split_pairs(
    split_pairs: tuple[CitationPair, ...],
    by_app: Mapping[str, PatentEmbeddingRecord],
    records: tuple[PatentEmbeddingRecord, ...],
    *,
    empty: CollisionEvalResult,
    recall_k: Sequence[int],
    covering: Covering,
    query_tile: int = RANK_QUERY_TILE,
    banks: Data | None = None,
    devices: Sequence[torch.device] | None = None,
    pool: CoveringRankPool | None = None,
) -> CollisionEvalResult:
    """Rank one query split against the encoded n-corpus by saturation covering."""
    eval_pairs = tuple(
        pair
        for pair in split_pairs
        if pair.query_application_number in by_app and pair.partner_application_number in by_app
    )
    if not eval_pairs:
        return empty
    return evaluate_collision_ranking(
        tuple(by_app[pair.query_application_number] for pair in eval_pairs),
        records,
        tuple(pair.partner_application_number for pair in eval_pairs),
        relevances=tuple(float(pair.grade) for pair in eval_pairs),
        marks=tuple(pair.marks for pair in eval_pairs),
        k_values=recall_k,
        covering=covering,
        query_tile=query_tile,
        banks=banks,
        devices=devices,
        pool=pool,
    )


def build_collision_eval_queue(
    eval_config: CollisionEvalConfig,
    *,
    dataset_path: Path,
    pair_source: CitationPairSource,
    hupd_dir: Path,
    index_cache: Path | None = None,
) -> CollisionEvalQueue | None:
    """Load split pairs and the HUPD paths that exist for those applications."""
    resolved_cache = index_cache
    if resolved_cache is None and eval_config.checkpoint:
        ckpt_root = Path(eval_config.checkpoint)
        cached = (ckpt_root if ckpt_root.is_dir() else ckpt_root.parent) / 'hupd_path_index.txt'
        resolved_cache = cached if cached.is_file() else None
    split = split_collision_pairs(
        eval_config,
        dataset_path=dataset_path,
        pair_source=pair_source,
    )
    if split is None:
        return None
    train_pairs, eval_pairs, test_pairs = split
    selected = (*train_pairs, *eval_pairs, *test_pairs)
    apps = tuple(
        sorted(
            {pair.query_application_number for pair in selected}
            | {pair.partner_application_number for pair in selected}
        )
    )
    index = hupd_stem_index(hupd_dir, resolved_cache)
    missing = tuple(app for app in apps if app not in index)
    if missing:
        _log.info('collision.eval.missing_hupd', count=len(missing))
    items = tuple((app, index[app]) for app in apps if app in index)
    if not items:
        return None
    return CollisionEvalQueue(
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        test_pairs=test_pairs,
        items=items,
    )


def persist_rank_covering(
    eval_config: CollisionEvalConfig,
    ranked: Mapping[str, CollisionEvalResult],
    *,
    encoded_apps: int,
) -> Path:
    """Write covering.json as soon as the three split tables exist."""
    written = write_covering_report(
        Path(eval_config.output_dir),
        report_from_splits(
            train=ranked['train'],
            eval_result=ranked['eval'],
            test=ranked['test'],
            eval_config=eval_config,
            checkpoint=eval_config.checkpoint,
            encoded_apps=encoded_apps,
        ),
    )
    path = Path(written)
    _log.info(
        'collision.eval.rank_done',
        covering=str(path),
        eval_unpaid_x=ranked['eval'].unpaid_x,
        eval_unpaid_a=ranked['eval'].unpaid_a,
        eval_unpaid_random=ranked['eval'].unpaid_random,
        eval_queries_x=ranked['eval'].queries_x,
    )
    return path


def rank_covering_from_records(
    eval_config: CollisionEvalConfig,
    records: tuple[PatentEmbeddingRecord, ...],
    *,
    train_pairs: tuple[CitationPair, ...],
    eval_pairs: tuple[CitationPair, ...],
    test_pairs: tuple[CitationPair, ...],
    covering: Covering | None = None,
) -> dict[str, CollisionEvalResult]:
    """Rank three splits against an already-encoded n-corpus. Writes covering.json."""
    empty = empty_collision_eval_result(eval_config.recall_k)
    covering = covering if covering is not None else Covering(eval_config.covering)
    by_app = {record.application_number: record for record in records}
    devices = resolve_rank_devices(
        eval_config.num_devices,
        use_gpu=bool(eval_config.use_gpu),
    )
    banks = corpus_rank_banks(records, devices)
    splits = {
        'train': train_pairs,
        'eval': eval_pairs,
        'test': test_pairs,
    }

    def rank_with(workers: CoveringRankPool | None) -> dict[str, CollisionEvalResult]:
        _log.info(
            'collision.eval.rank_start',
            encoded_apps=len(records),
            train_pairs=len(train_pairs),
            eval_pairs=len(eval_pairs),
            test_pairs=len(test_pairs),
            query_tile=int(eval_config.rank_query_tile),
            rank_devices=tuple(str(device) for device in devices),
            rank_backend='local' if workers is None else workers.backend,
            rank_workers=1 if workers is None else workers.worker_count,
            slot_mass_keep=eval_config.covering.slot_mass_keep,
            slot_top_k=eval_config.covering.slot_top_k,
        )
        return {
            name: rank_split_pairs(
                pairs,
                by_app,
                records,
                empty=empty,
                recall_k=eval_config.recall_k,
                covering=covering,
                query_tile=int(eval_config.rank_query_tile),
                banks=banks,
                devices=devices,
                pool=workers,
            )
            for name, pairs in splits.items()
        }

    if banks is None:
        ranked = rank_with(None)
    else:

        @impure_safe
        def rank_splits(workers: CoveringRankPool) -> dict[str, CollisionEvalResult]:
            return rank_with(workers)

        ranked = CoveringRankPool.run(
            CoveringRankPool.acquire(covering, banks, devices),
            rank_splits,
        )
    _ = persist_rank_covering(eval_config, ranked, encoded_apps=len(records))
    return ranked


def rank_covering_from_shards(
    eval_config: CollisionEvalConfig,
    *,
    dataset_path: Path,
    pair_source: CitationPairSource,
    shard_dir: Path | None = None,
) -> dict[str, CollisionEvalResult]:
    """Load complete encode shards and rank. Does not encode or explain."""
    root = shard_dir if shard_dir is not None else encode_shard_root(eval_config)
    records = CollisionEncodeShardStore(root=root).try_read_complete()
    if records is None:
        msg = f'complete encode shards are required under {root}'
        raise FileNotFoundError(msg)
    split = split_collision_pairs(
        eval_config,
        dataset_path=dataset_path,
        pair_source=pair_source,
    )
    if split is None:
        empty = empty_collision_eval_result(eval_config.recall_k)
        ranked = {'train': empty, 'eval': empty, 'test': empty}
        _ = persist_rank_covering(eval_config, ranked, encoded_apps=0)
        return ranked
    train_pairs, eval_pairs, test_pairs = split
    return rank_covering_from_records(
        eval_config,
        records,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        test_pairs=test_pairs,
    )


def place_corpus_banks(banks: Data | None, home: torch.device) -> None:
    """Move the document replica onto the top-hit device when it is CUDA."""
    if banks is None or home.type != 'cuda':
        return
    banks.entity_shards = tuple(
        shard.to(device=home, non_blocking=True) for shard in banks.entity_shards
    )
    banks.relation_shards = tuple(
        shard.to(device=home, non_blocking=True) for shard in banks.relation_shards
    )


def release_corpus_banks(banks: Data | None, home: torch.device) -> None:
    """Return the document replica to host memory before explain predict."""
    if banks is not None:
        banks.entity_shards = tuple(shard.cpu() for shard in banks.entity_shards)
        banks.relation_shards = tuple(shard.cpu() for shard in banks.relation_shards)
    if home.type == 'cuda':
        torch.cuda.empty_cache()


def run_collision_eval(
    eval_config: CollisionEvalConfig,
    *,
    dataset_path: Path,
    pair_source: CitationPairSource,
    hupd_dir: Path,
    collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch],
    model: SoftTrunkModel,
    index_cache: Path | None = None,
    covering: Covering | None = None,
) -> CollisionJobResult | None:
    """Encode HUPD patents for split pairs and compute ranking metrics.

    Returns None on non-writer DDP ranks so they do not emit empty artefacts.
    """
    empty = empty_collision_eval_result(eval_config.recall_k)
    empty_job = CollisionJobResult(
        train=empty,
        eval=empty,
        test=empty,
        checkpoint=eval_config.checkpoint,
    )
    model.eval()
    queued = build_collision_eval_queue(
        eval_config,
        dataset_path=dataset_path,
        pair_source=pair_source,
        hupd_dir=hupd_dir,
        index_cache=index_cache,
    )
    if queued is None:
        return empty_job

    covering = covering if covering is not None else Covering(eval_config.covering)
    inventory = Inventory(
        occupied_floor=float(model.soft_occupied_floor),
    )
    match resolve_encoded_corpus(
        eval_config,
        queued,
        collator=collator,
        model=model,
        inventory=inventory,
    ):
        case None:
            return None
        case ():
            return empty_job
        case records:
            explainer = Explain(covering)
            ranked = rank_covering_from_records(
                eval_config,
                records,
                train_pairs=queued.train_pairs,
                eval_pairs=queued.eval_pairs,
                test_pairs=queued.test_pairs,
                covering=covering,
            )
            devices = resolve_rank_devices(
                eval_config.num_devices,
                use_gpu=bool(eval_config.use_gpu),
            )
            banks = corpus_rank_banks(records, devices)
            home = devices[0]
            place_corpus_banks(banks, home)
            hit_keys = {
                name: top_hit_keys(
                    pairs,
                    records,
                    covering,
                    top_n=eval_config.explain_top_n,
                    banks=banks,
                )
                for name, pairs in {
                    'train': queued.train_pairs,
                    'eval': queued.eval_pairs,
                    'test': queued.test_pairs,
                }.items()
            }
            release_corpus_banks(banks, home)
            needed = {app for keys in hit_keys.values() for pair in keys for app in pair}
            explain_items = tuple(item for item in queued.items if item[0] in needed)
            states = collect_explain_states(
                explain_items,
                collator=collator,
                model=model,
                inventory=inventory,
                explain=explainer,
                prefix_mode=eval_config.prefix_mode,
                batch_size=int(eval_config.encode_batch_size),
                use_gpu=bool(eval_config.use_gpu),
                num_devices=eval_config.num_devices,
                num_workers=int(eval_config.dataloader_num_workers),
            )
            return CollisionJobResult(
                train=ranked['train'],
                eval=ranked['eval'],
                test=ranked['test'],
                checkpoint=eval_config.checkpoint,
                encoded_apps=len(records),
                explain={
                    split: artefacts_for_keys(
                        keys,
                        states,
                        explainer,
                        tau=eval_config.contour_tau,
                    )
                    for split, keys in hit_keys.items()
                },
            )


__all__ = [
    'CollisionEncodeBatch',
    'CollisionEncodeCollate',
    'CollisionEncodeDataset',
    'CollisionEncodeModule',
    'CollisionEncodeProbe',
    'CollisionEncodeRow',
    'CollisionEncodeShardStore',
    'CollisionEncodeStep',
    'CollisionEncodeWriter',
    'CollisionEvalQueue',
    'CollisionExplainWriter',
    'CollisionJobResult',
    'build_collision_eval_queue',
    'collate_encode_rows',
    'collision_encode_trainer',
    'encode_collision_records',
    'encode_rows_from_hupd',
    'encode_shard_root',
    'encode_trainer_kwargs',
    'explain_predict_strategy',
    'explain_states_from_batch',
    'export_encode_batch',
    'hupd_stem_index',
    'load_collision_trunk',
    'rank_covering_from_records',
    'rank_covering_from_shards',
    'rank_split_pairs',
    'resolve_checkpoint',
    'resolve_encode_devices',
    'resolve_encoded_corpus',
    'resolve_explain_device',
    'resolve_rank_devices',
    'run_collision_eval',
    'split_collision_pairs',
]
