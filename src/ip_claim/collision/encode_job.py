"""Lightning predict job that writes covering intensities to per-rank shards.

Encode is a `Trainer.predict` pass with `BasePredictionWriter`. The module
step returns a `CollisionEncodeStep`. The writer owns shard, progress, CSV,
and JSONL I/O. Rank shards stay `PatentEmbeddingRecord` tuples so existing
shard files remain readable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import reduce
from itertools import batched, chain, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict, cast

import lightning.pytorch as pl
import structlog
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.loggers import CSVLogger, Logger
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from ip_claim.collision.artefacts import ExplainState
from ip_claim.collision.collide import PatentEmbeddingRecord
from ip_claim.collision.config import CollisionPrefixMode
from ip_claim.collision.disclosure import add_chunk_intensities, disclosure_windows
from ip_claim.collision.explain import Explain
from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmExample
from ip_claim.ssv.graph_batch import dest_claim_texts, graph_batch_from_hupd_dict
from ip_claim.ssv.inventory import CoveringInventory, Inventory
from ip_claim.ssv.model import SoftTrunkModel, TrunkExport

_log = structlog.get_logger(__name__)


class CollisionEncodeProbe(BaseModel):
    """Per-batch bank-assignment scalars recorded during collision encode."""

    model_config = ConfigDict(frozen=True)

    row_entropy: float
    assignment_perplexity: float
    entity_batch_utilization: float
    relation_row_entropy: float
    relation_batch_utilization: float
    usage_entropy: float
    prefix_l2: float
    prefix_pairwise_cosine: float
    zd_zg_cosine: float
    apps_step: int = Field(ge=0)

    @classmethod
    def empty(cls, apps_step: int) -> CollisionEncodeProbe:
        """Zeroed probe when a predict step did not emit bank stats."""
        return cls.from_bank_stats(
            row_entropy=0.0,
            batch_usage=torch.zeros(1),
            utilization_eps=1.0,
            relation_row_entropy=0.0,
            relation_batch_usage=None,
            prefix_l2=0.0,
            prefix_pairwise_cosine=0.0,
            zd_zg_cosine=0.0,
            apps_step=apps_step,
        )

    @classmethod
    def from_bank_stats(
        cls,
        *,
        row_entropy: float,
        batch_usage: Tensor,
        utilization_eps: float,
        relation_row_entropy: float,
        relation_batch_usage: Tensor | None,
        prefix_l2: float,
        prefix_pairwise_cosine: float,
        zd_zg_cosine: float,
        apps_step: int,
    ) -> CollisionEncodeProbe:
        """Build probe scalars from assignment usage and prefix geometry."""
        mass = batch_usage.clamp_min(0)
        total = mass.sum()
        live = (batch_usage >= utilization_eps).to(dtype=torch.float32)
        relation_live = (
            None
            if relation_batch_usage is None
            else (relation_batch_usage >= utilization_eps).to(dtype=torch.float32)
        )
        return cls(
            row_entropy=row_entropy,
            assignment_perplexity=math.exp(row_entropy),
            entity_batch_utilization=float(live.mean().item()),
            relation_row_entropy=relation_row_entropy,
            relation_batch_utilization=(
                0.0 if relation_live is None else float(relation_live.mean().item())
            ),
            usage_entropy=(
                0.0
                if float(total.item()) <= 0
                else float(torch.special.entr(mass / total).sum().item())
            ),
            prefix_l2=prefix_l2,
            prefix_pairwise_cosine=prefix_pairwise_cosine,
            zd_zg_cosine=zd_zg_cosine,
            apps_step=apps_step,
        )

    @classmethod
    def from_inventory(
        cls,
        model: object,
        export: TrunkExport,
        inventory: CoveringInventory | None,
        input_ids: Tensor,
    ) -> CollisionEncodeProbe:
        """Measure late-assignment bank stats and prefix variation for one batch."""
        soft_tokens = export.soft_tokens
        graph = export.z_g.detach()
        text = export.z_d.detach()
        prefix_l2 = float(soft_tokens.detach().norm(dim=-1).mean().item())
        prefix_pairwise_cosine = 0.0
        if graph.size(0) >= 2 and bool((graph.norm(dim=-1) > 0).all()):
            grams = pairwise_cosine_similarity(graph, graph)
            n_rows = grams.size(0)
            prefix_pairwise_cosine = float(
                ((grams.sum() - grams.diagonal().sum()) / (n_rows * (n_rows - 1))).item()
            )
        zd_zg_cosine = float(F.cosine_similarity(text, graph, dim=-1).mean().item())
        apps_step = int(input_ids.size(0))
        if inventory is None:
            return cls.from_bank_stats(
                row_entropy=0.0,
                batch_usage=torch.zeros(1),
                utilization_eps=1.0,
                relation_row_entropy=0.0,
                relation_batch_usage=None,
                prefix_l2=prefix_l2,
                prefix_pairwise_cosine=prefix_pairwise_cosine,
                zd_zg_cosine=zd_zg_cosine,
                apps_step=apps_step,
            )
        vocab = getattr(model, 'soft_vocab', None)
        relation_usage = inventory.relation_batch_usage
        return cls.from_bank_stats(
            row_entropy=float(inventory.mean_row_entropy.detach().item()),
            batch_usage=inventory.batch_usage.detach(),
            utilization_eps=float(getattr(vocab, '_utilization_eps', 1e-3))
            if vocab is not None
            else 1e-3,
            relation_row_entropy=inventory.relation_row_entropy,
            relation_batch_usage=relation_usage.detach() if relation_usage is not None else None,
            prefix_l2=prefix_l2,
            prefix_pairwise_cosine=prefix_pairwise_cosine,
            zd_zg_cosine=zd_zg_cosine,
            apps_step=apps_step,
        )


class CollisionEncodeStep(BaseModel):
    """One predict step: embedding records plus the bank probe for that batch."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    records: tuple[PatentEmbeddingRecord, ...]
    probe: CollisionEncodeProbe


class CollisionEncodeRow(BaseModel):
    """One HUPD patent ready for batched trunk export."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    application_number: str
    example: SoftMlmExample
    claim_blob: str = ''
    claim_numbers: tuple[int, ...] = ()
    numbered_claim_texts: tuple[str, ...] = ()
    disclosure: str = ''
    cpc_section: str | None = None

    @classmethod
    def from_hupd(cls, application_number: str, row: Mapping[str, object]) -> CollisionEncodeRow:
        """Parse one HUPD JSON dict once into text, graph, and coarse CPC."""
        graph_batch = graph_batch_from_hupd_dict(dict(row))
        independents = dest_claim_texts(graph_batch)
        return cls(
            application_number=application_number,
            example=SoftMlmExample(
                text=graph_batch.text,
                graph=graph_batch.data,
                claim_text=independents[0] if independents else '',
                disclosure_text=graph_batch.disclosure,
            ),
            claim_blob=graph_batch.claim_blob,
            claim_numbers=graph_batch.claim_numbers,
            numbered_claim_texts=graph_batch.claim_texts,
            disclosure=graph_batch.disclosure,
            cpc_section=graph_batch.main_cpc_section,
        )


class CollisionEncodeBatch(BaseModel):
    """Collated host tensors plus the application ids and CPC labels they encode."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    application_numbers: tuple[str, ...]
    cpc_sections: tuple[str | None, ...]
    claim_mask: Tensor
    claim_numbers: tuple[int | None, ...] = ()
    disclosures: tuple[str, ...] = ()
    mlm: SoftMlmBatch


def encode_rows_from_hupd(
    loaded: Sequence[tuple[str, Mapping[str, object]]],
) -> Iterator[CollisionEncodeRow]:
    """Stream encode rows so graphs exist only for the current batch."""
    return starmap(CollisionEncodeRow.from_hupd, loaded)


class CollisionEncodeDataset(Dataset[CollisionEncodeRow]):
    """Random-access HUPD rows for Lightning predict and DistributedSampler."""

    def __init__(
        self,
        items: Sequence[tuple[str, Path | Mapping[str, object]]],
    ) -> None:
        self._items = tuple(items)

    def __len__(self) -> int:
        """Number of patents queued for trunk export."""
        return len(self._items)

    def __getitem__(self, index: int) -> CollisionEncodeRow:
        """Parse one queued patent when the DataLoader asks for that index."""
        app, source = self._items[index]
        row = (
            json.loads(source.read_text(encoding='utf-8'))
            if isinstance(source, Path)
            else dict(source)
        )
        return CollisionEncodeRow.from_hupd(app, row)


class CollisionEncodeShardStore(BaseModel):
    """Per-rank embedding shards written during Lightning predict."""

    model_config = ConfigDict(frozen=True)

    root: Path

    def shard_path(self, rank: int) -> Path:
        """Path for one rank's embedding tuple."""
        return self.root / f'rank-{rank:02d}.pt'

    def write(self, rank: int, records: tuple[PatentEmbeddingRecord, ...]) -> None:
        """Persist one rank's records for the rank-0 merge."""
        self.root.mkdir(parents=True, exist_ok=True)
        torch.save(records, self.shard_path(rank))

    def load_shard(self, path: Path) -> tuple[PatentEmbeddingRecord, ...]:
        """Validate one rank shard as embedding records."""
        payload = torch.load(path, weights_only=False)
        rows = payload if isinstance(payload, tuple) else ()
        return tuple(
            row
            if isinstance(row, PatentEmbeddingRecord)
            else PatentEmbeddingRecord.model_validate(row)
            for row in rows
        )

    def read_all(self, world_size: int) -> tuple[PatentEmbeddingRecord, ...]:
        """Concatenate shards in rank order after every rank has written."""
        ranks = range(world_size)
        return tuple(chain.from_iterable(self.load_shard(self.shard_path(rank)) for rank in ranks))

    def try_read_complete(self) -> tuple[PatentEmbeddingRecord, ...] | None:
        """Load contiguous rank shards when every file is present and non-empty."""
        paths = tuple(sorted(self.root.glob('rank-*.pt')))
        expected = tuple(self.shard_path(rank) for rank in range(len(paths)))
        if not paths or paths != expected or any(path.stat().st_size <= 0 for path in paths):
            return None
        records = tuple(chain.from_iterable(self.load_shard(path) for path in paths))
        if not records:
            return None
        return records


class CollisionEncodeWriter(BasePredictionWriter):
    """Write progress every batch and flush the rank shard on a configured cadence."""

    def __init__(
        self,
        store: CollisionEncodeShardStore,
        shard_flush_every: int = 50,
        metrics_jsonl: Path | None = None,
    ) -> None:
        super().__init__(write_interval='batch_and_epoch')
        self._store = store
        self._flush_every = max(int(shard_flush_every), 1)
        self._metrics_jsonl = metrics_jsonl
        self._records: list[PatentEmbeddingRecord] = []

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Append this batch, write live metrics, and flush the shard on cadence."""
        _ = (pl_module, batch_indices, batch, dataloader_idx)
        if not isinstance(prediction, CollisionEncodeStep):
            return
        step = prediction
        self._records.extend(step.records)
        records = tuple(self._records)
        probe = step.probe.model_dump()
        self._store.root.mkdir(parents=True, exist_ok=True)
        progress_path = self._store.root / f'progress-rank-{trainer.global_rank:02d}.json'
        _ = progress_path.write_text(
            json.dumps(
                {
                    'rank': trainer.global_rank,
                    'batch': batch_idx,
                    'apps': len(records),
                    **probe,
                },
                indent=2,
                sort_keys=True,
            )
            + '\n',
            encoding='utf-8',
        )
        logger = trainer.logger
        if isinstance(logger, Logger):
            logger.log_metrics(probe, step=batch_idx)
            logger.save()
        if trainer.is_global_zero and self._metrics_jsonl is not None:
            self._metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self._metrics_jsonl.open('a', encoding='utf-8') as handle:
                _ = handle.write(json.dumps({'batch': batch_idx, **probe}, sort_keys=True) + '\n')
        if (batch_idx + 1) % self._flush_every == 0:
            self._store.write(trainer.global_rank, records)
        _log.info(
            'collision.encode.progress',
            rank=trainer.global_rank,
            batch=batch_idx,
            apps=len(records),
            row_entropy=step.probe.row_entropy,
            entity_batch_utilization=step.probe.entity_batch_utilization,
            prefix_l2=step.probe.prefix_l2,
            zd_zg_cosine=step.probe.zd_zg_cosine,
        )

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[object],
        batch_indices: Sequence[object],
    ) -> None:
        """Flush this rank's accumulated embeddings into its shard file."""
        _ = (pl_module, predictions, batch_indices)
        self._store.write(trainer.global_rank, tuple(self._records))


class CollisionExplainWriter(BasePredictionWriter):
    """Flush this rank's explain states to a shard after the predict epoch."""

    def __init__(self, root: Path) -> None:
        super().__init__(write_interval='batch_and_epoch')
        self.root = root
        self._states: list[ExplainState] = []

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Append explain states from one predict batch."""
        _ = (trainer, pl_module, batch_indices, batch, batch_idx, dataloader_idx)
        if isinstance(prediction, tuple):
            self._states.extend(state for state in prediction if isinstance(state, ExplainState))

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[object],
        batch_indices: Sequence[object],
    ) -> None:
        """Write this rank's collected explain states."""
        _ = (pl_module, predictions, batch_indices)
        self.root.mkdir(parents=True, exist_ok=True)
        torch.save(tuple(self._states), self.root / f'explain-rank-{trainer.global_rank:02d}.pt')

    @classmethod
    def read_all(cls, root: Path) -> dict[str, ExplainState]:
        """Merge per-rank explain shards written by spawn workers."""
        shards = tuple(
            cast(tuple[ExplainState, ...], torch.load(path, map_location='cpu', weights_only=False))
            for path in sorted(root.glob('explain-rank-*.pt'))
        )
        return {state.application_number: state for shard in shards for state in shard}


class CollisionEncodeModule(pl.LightningModule):
    """Lightning predict step: one collated batch becomes embedding records."""

    def __init__(
        self,
        trunk: SoftTrunkModel,
        prefix_mode: CollisionPrefixMode = CollisionPrefixMode.trunk,
        inventory: Inventory | None = None,
        explain: Explain | None = None,
        collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch] | None = None,
        disclosure_max_chunks: int | None = None,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.prefix_mode = prefix_mode
        self.inventory = inventory
        self.explain = explain
        self.collator = collator
        self.disclosure_max_chunks = disclosure_max_chunks

    def transfer_batch_to_device(
        self,
        batch: CollisionEncodeBatch,
        device: torch.device,
        dataloader_idx: int = 0,
    ) -> CollisionEncodeBatch:
        """Keep the Pydantic batch intact; export moves tensors onto the device."""
        return batch

    def predict_step(
        self,
        batch: CollisionEncodeBatch,
        batch_idx: int = 0,
    ) -> CollisionEncodeStep | tuple[ExplainState, ...]:
        """Export embeddings, or recompute explain states, for one batch."""
        _ = batch_idx
        if self.explain is not None and self.inventory is not None:
            return explain_states_from_batch(
                batch,
                model=self.trunk,
                device=self.device,
                prefix_mode=self.prefix_mode,
                inventory=self.inventory,
                explain=self.explain,
            )
        return export_encode_batch(
            batch,
            model=self.trunk,
            device=self.device,
            prefix_mode=self.prefix_mode,
            inventory=self.inventory,
            collator=self.collator,
            disclosure_max_chunks=self.disclosure_max_chunks,
        )


class CollisionEncodeTrainerSpec(TypedDict):
    """Named Trainer fields for the collision encode predict pass."""

    accelerator: str
    devices: int | str
    strategy: str


def encode_trainer_kwargs(
    num_devices: int | None,
    *,
    use_gpu: bool,
) -> CollisionEncodeTrainerSpec:
    """Lightning Trainer accelerator, device count, and strategy='auto'."""
    gpu = bool(use_gpu) and torch.cuda.is_available()
    devices: int | str = 1 if not gpu else ('auto' if num_devices is None else int(num_devices))
    return CollisionEncodeTrainerSpec(
        accelerator='gpu' if gpu else 'cpu',
        devices=devices,
        strategy='auto',
    )


def resolve_encode_devices(
    num_devices: int | None,
    *,
    use_gpu: bool,
) -> tuple[str, int | str, str]:
    """Trainer kwargs as a triple for tests that still read the old shape."""
    spec = encode_trainer_kwargs(num_devices, use_gpu=use_gpu)
    return str(spec['accelerator']), spec['devices'], str(spec['strategy'])


def collision_encode_trainer(
    *,
    num_devices: int | None,
    use_gpu: bool,
    shard_store: CollisionEncodeShardStore,
    log_dir: Path,
    shard_flush_every: int,
) -> pl.Trainer:
    """Declare the Lightning predict trainer for single- or multi-GPU encode."""
    spec = encode_trainer_kwargs(num_devices, use_gpu=use_gpu)
    return pl.Trainer(
        accelerator=spec['accelerator'],
        devices=spec['devices'],
        strategy=spec['strategy'],
        logger=CSVLogger(save_dir=str(log_dir), name='lightning'),
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[
            CollisionEncodeWriter(
                shard_store,
                shard_flush_every=shard_flush_every,
                metrics_jsonl=log_dir / 'encode-metrics.jsonl',
            )
        ],
    )


def collate_encode_rows(
    rows: Sequence[CollisionEncodeRow],
    *,
    collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch],
    inventory: Inventory | None = None,
) -> CollisionEncodeBatch:
    """Keep application, CPC, and numbered-claim demand aligned with the MLM batch."""

    def demand_units(
        row: CollisionEncodeRow,
    ) -> tuple[tuple[SoftMlmExample, int | None], ...]:
        texts = row.numbered_claim_texts
        numbers = row.claim_numbers
        host = row.example.text
        locatable = bool(texts) and all(not text or text in host for text in texts)
        if locatable:
            return tuple(
                (row.example.model_copy(update={'claim_text': text}), number)
                for text, number in zip(texts, numbers, strict=True)
            )
        return ((row.example, numbers[0] if numbers else None),)

    units = tuple((row, example, number) for row in rows for example, number in demand_units(row))
    mlm = collator(tuple(example for _, example, _ in units))
    tokenizer = getattr(collator, 'tokenizer', None)
    max_length = int(getattr(collator, 'max_length', mlm.attention_mask.size(-1)))
    texts = tuple(example.text for _, example, _ in units)
    claims = tuple(example.claim_text for _, example, _ in units)
    claim_mask = (
        inventory.claim_mask(
            tokenizer,
            texts,
            claims,
            mlm.attention_mask,
            max_length=max_length,
        )
        if inventory is not None
        else mlm.attention_mask
    )
    return CollisionEncodeBatch(
        application_numbers=tuple(row.application_number for row, _, _ in units),
        cpc_sections=tuple(row.cpc_section for row, _, _ in units),
        claim_mask=claim_mask,
        claim_numbers=tuple(number for _, _, number in units),
        disclosures=tuple(row.disclosure for row, _, _ in units),
        mlm=mlm,
    )


class CollisionEncodeCollate:
    """Picklable DataLoader collate for Lightning spawn workers."""

    def __init__(
        self,
        collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch],
        inventory: Inventory | None = None,
    ) -> None:
        self.collator = collator
        self.inventory = inventory

    def __call__(self, rows: Sequence[CollisionEncodeRow]) -> CollisionEncodeBatch:
        """Collate one DataLoader batch of encode rows."""
        return collate_encode_rows(rows, collator=self.collator, inventory=self.inventory)


def last_layer_for_inventory(
    batch: CollisionEncodeBatch,
    *,
    model: SoftTrunkModel,
    device: torch.device,
    prefix_mode: CollisionPrefixMode,
    inventory: Inventory | None,
    living: Tensor | None = None,
) -> tuple[TrunkExport, Tensor | None]:
    """Last-layer host states after the prefix, or host-only text states.

    Occupy and the host last layer read unmasked token ids. MLM corruption
    stays on the train forward. Trunk export last-layer states already include
    the covering residual; host-only still prepends an empty prefix.
    """

    def host_only_layer(input_ids: Tensor, attention_mask: Tensor) -> tuple[TrunkExport, Tensor]:
        if inventory is None:
            msg = 'host-only covering inventory requires an Inventory'
            raise RuntimeError(msg)
        empty_prefix = model.host.get_input_embeddings()(input_ids[:, :0])
        text_hidden = inventory.last_layer_after_prefix(
            model, empty_prefix, input_ids, attention_mask
        )
        mask = attention_mask.to(dtype=text_hidden.dtype).unsqueeze(-1)
        z_d = (text_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        prefix = z_d.new_zeros(z_d.size(0), int(model.n_soft_tokens), z_d.size(-1))
        return TrunkExport(z_d=z_d, z_g=torch.zeros_like(z_d), soft_tokens=prefix), text_hidden

    mlm = batch.mlm
    input_ids = mlm.unmasked_input_ids.to(device)
    attention_mask = mlm.attention_mask.to(device)
    can_inventory = (
        getattr(model, 'soft_vocab', None) is not None
        and getattr(model, 'host', None) is not None
        and inventory is not None
    )
    if prefix_mode == CollisionPrefixMode.host_only and can_inventory and inventory is not None:
        return host_only_layer(input_ids, attention_mask)
    export = model.export_trunk(
        tuple(graph.to(device) for graph in mlm.graphs),
        input_ids,
        attention_mask,
        texts=mlm.texts,
        claim_texts=mlm.claim_texts,
        living=living,
        claim_mask=batch.claim_mask.to(device),
    )
    if can_inventory and inventory is not None:
        return export, inventory.last_layer_after_prefix(
            model, export.soft_tokens, input_ids, attention_mask
        )
    return export, None


def explain_states_from_batch(
    batch: CollisionEncodeBatch,
    *,
    model: SoftTrunkModel,
    device: torch.device,
    prefix_mode: CollisionPrefixMode,
    inventory: Inventory,
    explain: Explain,
) -> tuple[ExplainState, ...]:
    """Recompute late assignment and slot-graph W for one explain batch."""
    attention_mask = batch.mlm.attention_mask.to(device)
    claim_mask = batch.claim_mask.to(device)
    take_living = getattr(model, 'living_snapshot', None)
    living = take_living() if callable(take_living) else None
    with torch.inference_mode():
        _export, last_layer = last_layer_for_inventory(
            batch,
            model=model,
            device=device,
            prefix_mode=prefix_mode,
            inventory=inventory,
            living=living,
        )
        if last_layer is None:
            return ()
        assignment, _projected = model.soft_vocab.soft_assign(last_layer)
        occupied = model.occupy_assignment(
            assignment,
            attention_mask,
            batch.mlm.unmasked_input_ids.to(device),
            batch.mlm.texts,
            update_stats=False,
        )
        assignment = occupied.assignment
        claim_w = occupied.weights * claim_mask.to(dtype=occupied.weights.dtype)
        full_w = occupied.weights
        vocab = model.soft_vocab
        demand = (
            vocab.claim_span_demand(assignment, claim_mask) if any(batch.mlm.claim_texts) else None
        )
        claim_bundle = inventory.relation_bundle(
            assignment, claim_w, vocab, demand=demand, living=living
        )
        full_bundle = inventory.relation_bundle(
            assignment, full_w, vocab, demand=demand, living=living
        )
        claim_edges = explain.graphs_from_bundle(assignment, claim_w, claim_bundle)
        full_edges = explain.graphs_from_bundle(assignment, full_w, full_bundle)
    return tuple(
        ExplainState(
            application_number=app,
            assignment=assignment[index].detach().cpu(),
            attention_mask=full_w[index].detach().cpu(),
            claim_mask=claim_w[index].detach().cpu(),
            claim_edges=claim_edges[index].detach().cpu(),
            full_edges=full_edges[index].detach().cpu(),
        )
        for index, app in enumerate(batch.application_numbers)
    )


def fold_disclosure_chunks(
    payload: CoveringInventory,
    batch: CollisionEncodeBatch,
    *,
    model: SoftTrunkModel,
    device: torch.device,
    prefix_mode: CollisionPrefixMode,
    inventory: Inventory,
    collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch],
    max_chunks: int,
    last_layer: Callable[..., tuple[object, Tensor | None]] = last_layer_for_inventory,
    living: Tensor | None = None,
) -> CoveringInventory:
    """Add late-A intensity from disclosure windows onto first-window ``n_full``."""
    tokenizer = getattr(collator, 'tokenizer', None)
    max_length = int(getattr(collator, 'max_length', batch.mlm.attention_mask.size(-1)))
    if tokenizer is None:
        return payload
    owned = tuple(
        (owner, window)
        for owner, text in enumerate(batch.disclosures)
        for window in disclosure_windows(
            tokenizer,
            text,
            max_length=max_length,
            max_chunks=max_chunks,
        )
    )
    if not owned:
        return payload
    tile = max(int(batch.mlm.attention_mask.size(0)), 1)

    def add_tile(
        current: CoveringInventory,
        part: tuple[tuple[int, str], ...],
    ) -> CoveringInventory:
        chunk_batch = collate_encode_rows(
            tuple(
                CollisionEncodeRow(
                    application_number=batch.application_numbers[owner],
                    example=SoftMlmExample(text=window, graph=batch.mlm.graphs[owner]),
                    cpc_section=batch.cpc_sections[owner],
                )
                for owner, window in part
            ),
            collator=collator,
        )
        _export, hidden = last_layer(
            chunk_batch,
            model=model,
            device=device,
            prefix_mode=prefix_mode,
            inventory=inventory,
            living=living,
        )
        if hidden is None:
            return current
        attention = chunk_batch.mlm.attention_mask.to(device)
        owners = torch.tensor(
            [owner for owner, _window in part],
            device=current.n_entity_full.device,
            dtype=torch.long,
        )
        extra = inventory(
            hidden,
            attention,
            attention,
            model=model,
            texts=chunk_batch.mlm.texts,
            input_ids=chunk_batch.mlm.unmasked_input_ids.to(device),
            living=living,
            demand=(
                current.claim_demand.index_select(0, owners.to(device=current.claim_demand.device))
                if current.claim_demand is not None
                else None
            ),
        )
        return CoveringInventory(
            n_entity_claim=current.n_entity_claim,
            n_entity_full=add_chunk_intensities(
                current.n_entity_full,
                extra.n_entity_full.to(device=current.n_entity_full.device),
                owners,
            ),
            n_relation_claim=current.n_relation_claim,
            n_relation_full=add_chunk_intensities(
                current.n_relation_full,
                extra.n_relation_full.to(device=current.n_relation_full.device),
                owners,
            ),
            mean_row_entropy=current.mean_row_entropy,
            batch_usage=current.batch_usage,
            relation_row_entropy=current.relation_row_entropy,
            relation_batch_usage=current.relation_batch_usage,
            claim_demand=current.claim_demand,
        )

    return reduce(add_tile, (tuple(part) for part in batched(owned, tile)), payload)


def export_encode_batch(
    batch: CollisionEncodeBatch,
    *,
    model: SoftTrunkModel,
    device: torch.device,
    prefix_mode: CollisionPrefixMode = CollisionPrefixMode.trunk,
    inventory: Inventory | None = None,
    collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch] | None = None,
    disclosure_max_chunks: int | None = None,
) -> CollisionEncodeStep:
    """Run one collated batch through the trunk and write covering intensities."""
    mlm = batch.mlm
    attention_mask = mlm.attention_mask.to(device)
    claim_mask = batch.claim_mask.to(device)
    take_living = getattr(model, 'living_snapshot', None)
    living = take_living() if callable(take_living) else None
    with torch.inference_mode():
        export, last_layer = last_layer_for_inventory(
            batch,
            model=model,
            device=device,
            prefix_mode=prefix_mode,
            inventory=inventory,
            living=living,
        )
        payload = (
            inventory(
                last_layer,
                attention_mask,
                claim_mask,
                model=model,
                texts=mlm.texts,
                input_ids=mlm.unmasked_input_ids.to(device),
                living=living,
                claim_texts=mlm.claim_texts,
            )
            if inventory is not None and last_layer is not None
            else None
        )
        if (
            payload is not None
            and inventory is not None
            and collator is not None
            and disclosure_max_chunks
        ):
            payload = fold_disclosure_chunks(
                payload,
                batch,
                model=model,
                device=device,
                prefix_mode=prefix_mode,
                inventory=inventory,
                collator=collator,
                max_chunks=int(disclosure_max_chunks),
                living=living,
            )
        write_living = getattr(model, 'absorb_living', None)
        if payload is not None and payload.full_labeled is not None and callable(write_living):
            labeled = payload.full_labeled
            write_living(labeled if labeled.ndim == 3 else labeled.sum(dim=-1))
        probe = CollisionEncodeProbe.from_inventory(
            model,
            export,
            payload,
            mlm.input_ids.to(device),
        )
    z_d = export.z_d.detach().cpu()

    def record_at(
        index: int,
        app: str,
        cpc: str | None,
        claim_number: int | None,
    ) -> PatentEmbeddingRecord:
        if payload is None:
            return PatentEmbeddingRecord(
                application_number=app,
                z_d=z_d[index],
                cpc_section=cpc,
                claim_number=claim_number,
            )
        return PatentEmbeddingRecord(
            application_number=app,
            z_d=z_d[index],
            n_entity_claim=payload.n_entity_claim[index].detach().cpu(),
            n_entity_full=payload.n_entity_full[index].detach().cpu(),
            n_relation_claim=payload.n_relation_claim[index].detach().cpu(),
            n_relation_full=payload.n_relation_full[index].detach().cpu(),
            cpc_section=cpc,
            claim_number=claim_number,
        )

    claim_numbers = batch.claim_numbers or (None,) * len(batch.application_numbers)
    records = tuple(
        record_at(index, app, cpc, number)
        for index, (app, cpc, number) in enumerate(
            zip(batch.application_numbers, batch.cpc_sections, claim_numbers, strict=True)
        )
    )
    return CollisionEncodeStep(records=records, probe=probe)


def encode_collision_records(
    items: Sequence[tuple[str, Path | Mapping[str, object]]],
    *,
    collator: Callable[[Sequence[SoftMlmExample]], SoftMlmBatch],
    model: SoftTrunkModel,
    batch_size: int,
    num_devices: int | None = None,
    use_gpu: bool = True,
    num_workers: int = 0,
    shard_dir: Path | None = None,
    log_dir: Path | None = None,
    shard_flush_every: int = 50,
    prefix_mode: CollisionPrefixMode = CollisionPrefixMode.trunk,
    inventory: Inventory | None = None,
    disclosure_max_chunks: int | None = None,
) -> tuple[PatentEmbeddingRecord, ...] | None:
    """Encode HUPD rows with Lightning predict on every visible GPU.

    Non-writer ranks return None so they never synthesize an empty job result.
    """

    def _collate(rows: Sequence[CollisionEncodeRow]) -> CollisionEncodeBatch:
        return collate_encode_rows(rows, collator=collator, inventory=inventory)

    dataset = CollisionEncodeDataset(items)
    step = min(int(batch_size), max(len(dataset), 1))
    loader = DataLoader(
        dataset,
        batch_size=step,
        shuffle=False,
        collate_fn=_collate,
        num_workers=int(num_workers),
        persistent_workers=int(num_workers) > 0,
    )
    with TemporaryDirectory() as tmp:
        store = CollisionEncodeShardStore(root=shard_dir if shard_dir is not None else Path(tmp))
        logs = log_dir if log_dir is not None else store.root
        trainer = collision_encode_trainer(
            num_devices=num_devices,
            use_gpu=use_gpu,
            shard_store=store,
            log_dir=logs,
            shard_flush_every=int(shard_flush_every),
        )
        _log.info(
            'collision.eval.encode_start',
            apps=len(dataset),
            batch_size=step,
            num_workers=int(num_workers),
            world_size=int(trainer.world_size),
            shard_dir=str(store.root),
            prefix_mode=str(prefix_mode),
            shard_flush_every=int(shard_flush_every),
        )
        _ = trainer.predict(
            CollisionEncodeModule(
                model,
                prefix_mode=prefix_mode,
                inventory=inventory,
                collator=collator,
                disclosure_max_chunks=disclosure_max_chunks,
            ),
            dataloaders=loader,
            return_predictions=False,
        )
        if trainer.global_rank != 0:
            return None
        records = store.read_all(int(trainer.world_size))
        _log.info('collision.eval.encode_done', encoded_apps=len(records))
        return records


__all__ = [
    'CollisionEncodeBatch',
    'CollisionEncodeCollate',
    'CollisionEncodeDataset',
    'CollisionEncodeModule',
    'CollisionEncodeProbe',
    'CollisionEncodeRow',
    'CollisionEncodeShardStore',
    'CollisionEncodeStep',
    'CollisionEncodeTrainerSpec',
    'CollisionEncodeWriter',
    'CollisionExplainWriter',
    'collate_encode_rows',
    'collision_encode_trainer',
    'encode_collision_records',
    'encode_rows_from_hupd',
    'encode_trainer_kwargs',
    'explain_states_from_batch',
    'export_encode_batch',
    'fold_disclosure_chunks',
    'last_layer_for_inventory',
    'resolve_encode_devices',
]
