"""Ray inventory encode pool and last-layer session encode.

Owns the covering experiment's GPU encode workers and the single-process
fallback. The pytest session fixture binds this pool; product scoring stays
in ``ip_claim``. Closing the pool releases actors before a residual pin.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from itertools import batched, starmap
from typing import cast

import pytest
import ray
import torch
from patent_ate.nlp import TermSpan
from ray import ObjectRef
from ray.actor import ActorHandle
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from experiments.covering_objective.relational_arms import compact_encode_seams
from ip_claim.collision.config import CollisionPrefixMode
from ip_claim.collision.cover import Covering
from ip_claim.collision.encode_job import (
    CollisionEncodeBatch,
    CollisionEncodeCollate,
    CollisionEncodeRow,
    last_layer_for_inventory,
)
from ip_claim.shared.ray_runtime import ensure_local_ray
from ip_claim.ssv.collate import SoftMlmCollator
from ip_claim.ssv.covering_trace import (
    CoveringTrace,
    covering_adapters_off,
    covering_patches,
    covering_patches_on,
    covering_seams_kept,
    disable_covering_adapters,
    keep_covering_seams,
    overlay_edge_shift,
    patch_covering,
    patches_on_device,
    record_covering,
    shift_overlay_edges,
)
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.model import SoftTrunkModel

_NODE_SEAMS = frozenset({'compose_state', 'mixed_compose'})
PARAM_GROUPS = {
    'bank_assignment': ('soft_vocab.',),
    'graph_encoder': ('encoder.',),
    'projectors': ('projector.', 'slot_projector.'),
    'host_lora': ('host.',),
}
EncodeView = Callable[..., tuple[Tensor, CoveringTrace]]
EncodeShard = tuple[tuple[str, ...], Tensor, dict[str, Tensor], dict[str, Tensor]]


def covering_pair_devices() -> tuple[torch.device, ...]:
    """First two advertised CUDA devices. Physical cards 2 and 3 stay off-limits.

    When ``CUDA_VISIBLE_DEVICES`` is set, the count comes from that list so
    this process does not load the CUDA driver before the JATE draw.
    """
    raw = os.environ.get('CUDA_VISIBLE_DEVICES')
    tokens = () if raw is None else tuple(part.strip() for part in raw.split(',') if part.strip())
    if any(token in {'2', '3'} for token in tokens):
        pytest.fail('covering measurements refuse GPU indices 2 and 3')
    if raw is not None:
        count = 0 if tokens == ('void',) else len(tokens)
    elif not torch.cuda.is_available():
        count = 0
    else:
        count = torch.cuda.device_count()
    devices = tuple(torch.device('cuda', index) for index in range(min(2, count)))
    if os.environ.get('COVERING_REQUIRE_PAIR_DEVICES', '').strip() == '1' and len(devices) < 2:
        pytest.fail('covering live pair needs two visible CUDA devices')
    return devices


def cat_checkpointed_chunks[Piece](
    encode_one: Callable[[Piece], Tensor],
    chunks: Sequence[Piece],
    device: torch.device,
) -> Tensor:
    """Concatenate live chunk encodes and recompute each chunk on backward."""
    dummy = torch.zeros((), device=device, requires_grad=True)

    def rerun(chunk: Piece) -> Tensor:
        def encode(_: Tensor) -> Tensor:
            return encode_one(chunk)

        encoded = checkpoint(encode, dummy, use_reentrant=False)
        if not isinstance(encoded, Tensor):
            msg = 'checkpointed chunk encode must return a tensor'
            raise TypeError(msg)
        return encoded

    return torch.cat(tuple(rerun(chunk) for chunk in chunks), dim=0)


class InventoryForward(nn.Module):
    """Last-layer inventory encode. Device follows ``trunk`` parameters."""

    def __init__(self, trunk: SoftTrunkModel, inventory: Inventory) -> None:
        super().__init__()
        self.trunk = trunk
        self.inventory = inventory

    @property
    def device(self) -> torch.device:
        """Device of registered parameters or buffers."""
        tensor = next(self.parameters(), None)
        if tensor is None:
            tensor = next(self.buffers(), None)
        return tensor.device if tensor is not None else torch.device('cpu')

    def forward(self, batch: CollisionEncodeBatch, kind: str) -> Tensor:
        """Claim or full covering n: occupy mass plus kept-edge addends."""
        device = self.device
        _export, last_layer = last_layer_for_inventory(
            batch,
            model=self.trunk,
            device=device,
            prefix_mode=CollisionPrefixMode.trunk,
            inventory=self.inventory,
        )
        if last_layer is None:
            return torch.zeros(len(batch.application_numbers), 1, device=device)
        payload = self.inventory(
            last_layer,
            batch.mlm.attention_mask.to(device),
            batch.claim_mask.to(device),
            model=self.trunk,
            texts=batch.mlm.texts,
            input_ids=batch.mlm.unmasked_input_ids.to(device),
            living=self.trunk.living_snapshot(),
            claim_texts=batch.mlm.claim_texts,
        )
        intensity = payload.n_entity_claim if kind == 'claim' else payload.n_entity_full
        return cast(Tensor, intensity)

    def pin_session_compute(self) -> torch.device:
        """Release Ray encode actors, then move the session trunk onto cuda:0."""
        close_encode_pool()
        pair = covering_pair_devices()
        if pair and self.device.type != 'cuda':
            self.to(pair[0])
        return self.device

    def residual(
        self,
        param_map: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
        claim_batch: CollisionEncodeBatch,
        disc_batch: CollisionEncodeBatch,
        covering: Covering,
    ) -> Tensor:
        """Negative normalized unpaid gap under a copied parameter map."""
        state = (dict(param_map), dict(buffers))
        claims = torch.func.functional_call(self, state, (claim_batch, 'claim'))
        supply = torch.func.functional_call(self, state, (disc_batch, 'full'))
        return cast(
            Tensor,
            -covering.normalized_unpaid_gap(covering.pair_table(claims, supply)),
        )


def unnamed_tensor(tensor: Tensor) -> Tensor:
    """Drop named dims so shard stacks and patches stay unnamed."""
    names = getattr(tensor, 'names', None)
    return tensor.rename(None) if names and any(names) else tensor


def row_major_patches(
    patches: Mapping[str, Tensor],
    chunk: Sequence[CollisionEncodeRow],
    *,
    listed_n: int,
    order: Mapping[str, int],
) -> dict[str, Tensor]:
    """Slice document-major seam replacements onto ``chunk``."""
    index = [order[row.application_number] for row in chunk]
    return {
        name: (value[index] if name not in _NODE_SEAMS and value.size(0) == listed_n else value)
        for name, value in patches.items()
    }


def node_major_patches(patches: Mapping[str, Tensor]) -> bool:
    """True when a replacement is graph-node-major and must stay one forward."""
    return any(name in _NODE_SEAMS for name in patches)


@ray.remote
class InventoryEncodeActor:
    """One GPU worker that encodes a shard against a full trunk replica."""

    def __init__(self, inner: InventoryForward, collator: CollisionEncodeCollate) -> None:
        self.inner = inner
        self.inner.eval()
        self.collator = collator
        if torch.cuda.is_available() and self.inner.device.type != 'cuda':
            self.inner.to(torch.device('cuda'))

    def encode(
        self,
        rows: Sequence[CollisionEncodeRow],
        kind: str,
        patches: Mapping[str, Tensor],
        width: int,
        *,
        full_trace: bool,
        adapters_off: bool = False,
        keep_seams: frozenset[str] | None = None,
        inference_spans: Mapping[str, Sequence[TermSpan]] | None = None,
        dest_shift: int = 0,
        attr_shift: int = 0,
    ) -> EncodeShard:
        """Host intensities and seam tensors for one shard.

        Graph-node replacements stay one forward. Document-major patches
        still batch at width. Collate stays on the host copy. Term spans
        come from the driver; this worker does not redraw spaCy.
        """
        listed = tuple(rows)
        order = {row.application_number: index for index, row in enumerate(listed)}
        keep_graph = node_major_patches(patches)
        chunks = (
            (listed,)
            if keep_graph
            else tuple(tuple(chunk) for chunk in batched(listed, max(width, 1)))
        )
        if inference_spans is not None:
            self.inner.trunk.graph_ingress.remember_inference_spans(inference_spans)
        texts = tuple(row.example.text for row in listed if row.example.text.strip())
        missing = self.inner.trunk.graph_ingress.missing_inference_texts(texts)
        if missing:
            raise RuntimeError(
                'encode actor refused a spaCy redraw; '
                f'{len(missing)} host strings have no remembered spans'
            )
        if torch.cuda.is_available() and self.inner.device.type != 'cuda':
            self.inner.to(torch.device('cuda'))
        if torch.cuda.is_available() and self.inner.device.type != 'cuda':
            raise RuntimeError('encode actor has a visible GPU but the trunk stayed on CPU')
        collated = tuple(
            (
                chunk,
                self.collator(chunk),
                (
                    patches
                    if keep_graph
                    else row_major_patches(patches, chunk, listed_n=len(listed), order=order)
                ),
            )
            for chunk in chunks
        )

        def run_chunk(
            item: tuple[Sequence[CollisionEncodeRow], CollisionEncodeBatch, Mapping[str, Tensor]],
        ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
            _chunk, batch, payload = item
            moved = patches_on_device(payload, self.inner.device)
            recorder = record_covering() if full_trace else nullcontext()
            with (
                torch.inference_mode(),
                disable_covering_adapters() if adapters_off else nullcontext(),
                keep_covering_seams(keep_seams) if keep_seams is not None else nullcontext(),
                patch_covering(moved) if moved else nullcontext(),
                shift_overlay_edges(dest=dest_shift, attr=attr_shift),
                recorder as local,
            ):
                intensity = self.inner(batch, kind)
            host = intensity.detach().cpu()
            if not full_trace or local is None:
                return host, {}, {}
            captured = compact_encode_seams(
                {name: unnamed_tensor(tensor) for name, tensor in local.tensors.items()},
                keep=keep_seams,
            )
            return (
                host,
                {name: tensor for name, tensor in captured.items() if name not in _NODE_SEAMS},
                {name: tensor for name, tensor in captured.items() if name in _NODE_SEAMS},
            )

        pieces = tuple(run_chunk(item) for item in collated)
        document_names = {name for _intensity, documents, _nodes in pieces for name in documents}
        node_names = {name for _intensity, _documents, nodes in pieces for name in nodes}
        return (
            tuple(row.application_number for row in listed),
            (
                torch.cat(tuple(intensity for intensity, _documents, _nodes in pieces), dim=0)
                if pieces
                else torch.zeros(0, 1)
            ),
            {
                name: torch.cat(
                    tuple(documents[name] for _intensity, documents, _nodes in pieces),
                    dim=0,
                )
                for name in document_names
            },
            {
                name: torch.cat(
                    tuple(nodes[name] for _intensity, _documents, nodes in pieces),
                    dim=0,
                )
                for name in node_names
            },
        )


class InventoryEncodePool:
    """Ray GPU encode workers. Same control plane as ``CoveringRankPool``."""

    def __init__(
        self,
        actors: tuple[ActorHandle[object], ...],
        *,
        owns_ray: bool,
    ) -> None:
        self.actors = actors
        self.owns_ray = owns_ray

    @classmethod
    def open(
        cls,
        exporter: InventoryForward,
        collator: CollisionEncodeCollate,
        count: int,
    ) -> InventoryEncodePool:
        """Start one actor per card. Driver keeps the trunk on CPU."""
        owns_ray = ensure_local_ray(num_gpus=count)
        inner_ref = ray.put(exporter)
        collate_ref = ray.put(collator)
        remote_actor = InventoryEncodeActor
        return cls(
            tuple(
                cast(
                    ActorHandle[object],
                    remote_actor.options(num_gpus=1).remote(inner_ref, collate_ref),
                )
                for _ in range(count)
            ),
            owns_ray=owns_ray,
        )

    def encode(
        self,
        rows: Sequence[CollisionEncodeRow],
        *,
        kind: str,
        full_trace: bool,
        width: int,
        inference_spans: Mapping[str, Sequence[TermSpan]] | None = None,
    ) -> tuple[Tensor, CoveringTrace]:
        """Shard intensity rows. Traced or patched draws run a full replica per card."""
        listed = tuple(rows)
        order = {row.application_number: index for index, row in enumerate(listed)}
        patches = {name: unnamed_tensor(value) for name, value in covering_patches().items()}
        adapters_off = covering_adapters_off()
        keep_seams = covering_seams_kept()
        dest_shift, attr_shift = overlay_edge_shift()
        replicate = node_major_patches(patches)
        workers = min(len(self.actors), max(len(listed), 1))
        size = (len(listed) + workers - 1) // workers if listed else 1
        shards = (
            (listed,)
            if replicate
            else tuple(listed[index : index + size] for index in range(0, len(listed), size))
        )
        targets = self.actors if replicate else self.actors[: len(shards)]

        def remote_encode(index: int, actor: ActorHandle[object]) -> ObjectRef[EncodeShard]:
            shard = listed if replicate else shards[index]
            return cast(
                ObjectRef[EncodeShard],
                actor.encode.remote(
                    shard,
                    kind,
                    row_major_patches(
                        patches,
                        shard,
                        listed_n=len(listed),
                        order=order,
                    ),
                    width,
                    full_trace=full_trace,
                    adapters_off=adapters_off,
                    keep_seams=keep_seams,
                    inference_spans=inference_spans,
                    dest_shift=dest_shift,
                    attr_shift=attr_shift,
                ),
            )

        packed = tuple(
            cast(
                list[EncodeShard],
                ray.get(list(starmap(remote_encode, enumerate(targets)))),
            )
        )
        chosen = packed[:1] if replicate else packed
        apps = tuple(app for shard in chosen for app in shard[0])
        perm = tuple(sorted(range(len(apps)), key=lambda index: order[apps[index]]))
        intensities = tuple(shard[1] for shard in chosen)
        document_names = {name for shard in chosen for name in shard[2]}
        node_names = {name for shard in chosen for name in shard[3]}

        def aligned(payloads: Sequence[Mapping[str, Tensor]], name: str) -> Tensor:
            joined = torch.cat(tuple(payload[name] for payload in payloads), dim=0)
            return joined[list(perm)] if perm else joined

        with record_covering() as trace:
            intensity = (
                torch.cat(intensities, dim=0)[list(perm)]
                if intensities and perm
                else torch.zeros(0, 1)
            )
            _ = tuple(
                trace.record(name, aligned(tuple(shard[2] for shard in chosen), name))
                for name in document_names
            )
            _ = tuple(
                trace.record(name, aligned(tuple(shard[3] for shard in chosen), name))
                for name in node_names
            )
        return intensity, trace

    def close(self) -> None:
        """Release actors and, when this pool started Ray, shut it down."""
        _ = tuple(ray.kill(actor) for actor in self.actors)
        self.actors = ()
        if self.owns_ray and ray.is_initialized():
            ray.shutdown()
            self.owns_ray = False


_ENCODE_SLOT: list[InventoryEncodePool | None] = [None]


def close_encode_pool() -> None:
    """Kill covering encode actors and shut Ray down when this session started it."""
    pool = _ENCODE_SLOT[0]
    if pool is not None:
        pool.close()
        _ENCODE_SLOT[0] = None


def bound_encode_pool(
    exporter: InventoryForward,
    collator: CollisionEncodeCollate,
    count: int,
) -> InventoryEncodePool:
    """Reuse the session Ray encode pool, starting it on the first encode."""
    if _ENCODE_SLOT[0] is None:
        _ENCODE_SLOT[0] = InventoryEncodePool.open(exporter, collator, count)
    return cast(InventoryEncodePool, _ENCODE_SLOT[0])


def session_encode(
    exporter: InventoryForward,
    rows: Sequence[CollisionEncodeRow],
    *,
    kind: str,
    width: int,
    retain_graph: bool,
    full_trace: bool,
    collate: CollisionEncodeCollate,
    inference_spans: Mapping[str, Sequence[TermSpan]] | None = None,
) -> tuple[Tensor, CoveringTrace]:
    """Single-process inventory encode on the session trunk.

    Pre-computed JATE spans are loaded into the ingress cache before the
    first forward so the session path never redraws spaCy.
    """
    if inference_spans is not None:
        exporter.trunk.graph_ingress.remember_inference_spans(inference_spans)
    context = torch.enable_grad() if retain_graph else torch.inference_mode()
    moved = covering_patches_on(exporter.device)
    with (
        context,
        patch_covering(moved) if moved else nullcontext(),
        record_covering() as trace,
    ):
        if full_trace:
            intensity = exporter(collate(rows), kind)
        else:
            batches = tuple(collate(chunk) for chunk in batched(rows, width))
            if retain_graph and len(batches) > 1:
                intensity = cat_checkpointed_chunks(
                    lambda batch: exporter(batch, kind),
                    batches,
                    exporter.device,
                )
            elif batches:
                intensity = torch.cat(
                    tuple(exporter(batch, kind) for batch in batches),
                    dim=0,
                )
            else:
                intensity = torch.zeros(0, 1, device=exporter.device)
    return intensity, trace


def clone_module_state(module: nn.Module) -> dict[str, Tensor]:
    """Clone parameter and buffer tensors. Skip termhood extra state."""
    return {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
        if isinstance(value, Tensor)
    }


def owned_parameters(module: nn.Module, name: str) -> dict[str, Tensor]:
    """Parameters for one architectural group, including ``trunk.`` wrappers."""
    prefixes = PARAM_GROUPS[name]

    def owned(key: str) -> bool:
        matched = any(key.startswith(prefix) or f'.{prefix}' in f'{key}.' for prefix in prefixes)
        return matched and (name != 'host_lora' or 'lora' in key.lower())

    return {key: value for key, value in module.named_parameters() if owned(key)}


def bind_encode_view(
    exporter: InventoryForward,
    collator: SoftMlmCollator,
    inventory: Inventory,
    *,
    width: int,
    seed: int,
) -> EncodeView:
    """Last-layer inventory encode. Each visible card uses one Ray actor."""
    pair = covering_pair_devices()
    pair_collate = CollisionEncodeCollate(collator, inventory)

    def encode(
        rows: Sequence[CollisionEncodeRow],
        *,
        as_claim: bool,
        retain_graph: bool = False,
        full_trace: bool = False,
    ) -> tuple[Tensor, CoveringTrace]:
        torch.manual_seed(seed)
        kind = 'claim' if as_claim else 'full'
        texts = tuple(row.example.text for row in rows if row.example.text.strip())
        unique = tuple(dict.fromkeys(texts))
        span_map = (
            dict(zip(unique, exporter.trunk.graph_ingress.candidates(unique), strict=True))
            if unique
            else {}
        )
        pinned = exporter.device.type == 'cuda'
        if retain_graph or pinned:
            if retain_graph:
                exporter.pin_session_compute()
            return session_encode(
                exporter,
                rows,
                kind=kind,
                width=width,
                retain_graph=retain_graph,
                full_trace=full_trace,
                collate=pair_collate,
                inference_spans=span_map,
            )
        if pair:
            return bound_encode_pool(exporter, pair_collate, len(pair)).encode(
                rows,
                kind=kind,
                full_trace=full_trace,
                width=width,
                inference_spans=span_map,
            )
        return session_encode(
            exporter,
            rows,
            kind=kind,
            width=width,
            retain_graph=False,
            full_trace=full_trace,
            collate=pair_collate,
            inference_spans=span_map,
        )

    return encode
