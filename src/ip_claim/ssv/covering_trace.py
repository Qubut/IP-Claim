"""Off-by-default named tensors at covering-relevant forward seams.

Production callers leave the context unset. Capture stores a named view and
returns the original unnamed tensor so host and GNN modules never see names.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import Protocol

import torch
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer
from pydantic import BaseModel, ConfigDict, Field
from returns.maybe import Maybe
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.hooks import RemovableHandle


class OverlayFeatures(Protocol):
    """Occupied-code rows from one document overlay."""

    soft_x: Tensor


class CoveringTrace(BaseModel):
    """Named covering-boundary tensors collected during one active context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensors: dict[str, Tensor] = Field(default_factory=dict)

    def record(self, name: str, tensor: Tensor, *names: str) -> None:
        """Store ``tensor`` under ``name``, with dimension names when they fit."""
        if not names or tensor.ndim != len(names):
            self.tensors[name] = tensor
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Named tensors and all their associated APIs are an experimental feature',
            )
            self.tensors[name] = tensor.refine_names(*names)  # type: ignore[no-untyped-call]


_ACTIVE: ContextVar[CoveringTrace | None] = ContextVar('covering_trace', default=None)
_PATCH: ContextVar[dict[str, Tensor] | None] = ContextVar('covering_patch', default=None)
_ADAPTERS_OFF: ContextVar[bool] = ContextVar('covering_adapters_off', default=False)
_KEEP: ContextVar[frozenset[str] | None] = ContextVar('covering_trace_keep', default=None)
_OVERLAY_EDGE_SHIFT: ContextVar[tuple[int, int]] = ContextVar(
    'covering_overlay_edge_shift',
    default=(0, 0),
)


@contextmanager
def record_covering() -> Iterator[CoveringTrace]:
    """Activate diagnostic capture for the current task. Nested calls replace."""
    trace = CoveringTrace()
    token = _ACTIVE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE.reset(token)


@contextmanager
def patch_covering(replacements: Mapping[str, Tensor]) -> Iterator[None]:
    """Replace named seam tensors for the rest of the current forward."""
    token = _PATCH.set(dict(replacements))
    try:
        yield
    finally:
        _PATCH.reset(token)


def trace_tensor(name: str, tensor: Tensor, *names: str) -> Tensor:
    """Record a named view when a covering trace is active. Identity otherwise.

    An active same-shape patch swaps ``tensor`` for the replacement so the
    remainder of the forward reads the patched activation. A zero
    replacement of another rank becomes zeros on the live shape. A
    nonzero replacement of another rank is ignored.
    """

    def same_rank(replacement: Tensor) -> Tensor | None:
        return replacement if replacement.shape == tensor.shape else None

    def zero_on_live(replacement: Tensor) -> Tensor | None:
        return torch.zeros_like(tensor) if int(torch.count_nonzero(replacement)) == 0 else None

    def aligned_patch(replacement: Tensor) -> Maybe[Tensor]:
        return Maybe.from_optional(same_rank(replacement)).lash(
            lambda _: Maybe.from_optional(zero_on_live(replacement))
        )

    value = Maybe.do(
        aligned
        for table in Maybe.from_optional(_PATCH.get())
        for replacement in Maybe.from_optional(table.get(name))
        for aligned in aligned_patch(replacement)
    ).value_or(tensor)
    active = _ACTIVE.get()
    keep = _KEEP.get()
    if active is not None and (keep is None or name in keep):
        active.record(name, value, *names)
    return value


def covering_trace_active() -> bool:
    """True when ``record_covering`` is collecting named seam tensors."""
    return _ACTIVE.get() is not None


@contextmanager
def disable_covering_adapters() -> Iterator[None]:
    """Run the next host forwards on the frozen base, with LoRA skipped."""
    token = _ADAPTERS_OFF.set(True)
    try:
        yield
    finally:
        _ADAPTERS_OFF.reset(token)


def covering_adapters_off() -> bool:
    """True when the adapter-off covering condition is active."""
    return _ADAPTERS_OFF.get()


@contextmanager
def keep_covering_seams(names: frozenset[str]) -> Iterator[None]:
    """Record only the named seams for the next traced forwards."""
    token = _KEEP.set(names)
    try:
        yield
    finally:
        _KEEP.reset(token)


def covering_seams_kept() -> frozenset[str] | None:
    """Active seam allowlist, or None when every traced name is kept."""
    return _KEEP.get()


@contextmanager
def shift_overlay_edges(*, dest: int = 0, attr: int = 0) -> Iterator[None]:
    """Roll overlay destination slots or relation vectors for the next encode."""
    token = _OVERLAY_EDGE_SHIFT.set((int(dest), int(attr)))
    try:
        yield
    finally:
        _OVERLAY_EDGE_SHIFT.reset(token)


def overlay_edge_shift() -> tuple[int, int]:
    """Active overlay destination and relation-vector rolls, or zeros."""
    return _OVERLAY_EDGE_SHIFT.get()


@contextmanager
def capture_adapter_delta(host: nn.Module) -> Iterator[None]:
    """Record per-module LoRA deltas on the next host forward.

    Each PEFT ``LoraLayer`` hook subtracts a fresh ``base_layer`` call from
    the module output and keeps the token mean on CPU. The stacked table is
    ``(batch, module, hidden)``. Hooks install only when an active trace
    keeps ``adapter_delta``. Adapter-off still wraps ``disable_adapter`` so
    the frozen base runs without a second host copy.
    """
    disabler = getattr(host, 'disable_adapter', None)
    opened = disabler() if covering_adapters_off() and callable(disabler) else nullcontext()
    disable = opened if isinstance(opened, AbstractContextManager) else nullcontext()
    keep = _KEEP.get()
    if _ACTIVE.get() is None or (keep is not None and 'adapter_delta' not in keep):
        with disable:
            yield
        return
    layers = tuple(module for module in host.modules() if isinstance(module, LoraLayer))
    captured: list[Tensor] = []

    def hook(module: nn.Module, args: tuple[object, ...], output: object) -> None:
        inbound = args[0] if args else None
        getter = getattr(module, 'get_base_layer', None)
        base = None if getter is None else getter()
        if (
            not isinstance(inbound, Tensor)
            or not isinstance(output, Tensor)
            or not isinstance(base, nn.Module)
        ):
            return
        delta = output - base(inbound)
        pooled = delta.mean(dim=-2) if delta.ndim >= 3 else delta
        captured.append(pooled.detach().to('cpu'))

    def stack_deltas(rows: Sequence[Tensor]) -> Tensor:
        width = max(row.size(-1) for row in rows)
        padded = tuple(
            row if row.size(-1) == width else F.pad(row, (0, width - row.size(-1)))
            for row in rows
        )
        return torch.stack(padded, dim=1)

    def drop(handle: RemovableHandle) -> None:
        handle.remove()

    handles = tuple(layer.register_forward_hook(hook) for layer in layers)
    try:
        with disable:
            yield
    finally:
        _ = tuple(map(drop, handles))
        if captured:
            stacked = stack_deltas(captured)
            trace_tensor('adapter_delta', stacked, 'batch', 'module', 'hidden')


def covering_patches() -> Mapping[str, Tensor]:
    """Active seam replacements for the current forward, or empty."""
    return _PATCH.get() or {}


def patches_on_device(
    patches: Mapping[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    """Copy seam replacements onto the encode worker device."""
    return {name: value.to(device) for name, value in patches.items()}


def covering_patches_on(device: torch.device) -> dict[str, Tensor]:
    """Active seam replacements copied onto ``device``."""
    return patches_on_device(covering_patches(), device)


def trace_overlay(name: str, overlays: Sequence[OverlayFeatures]) -> None:
    """Pad occupied-code features and record them when a trace is active."""
    if _ACTIVE.get() is None:
        return
    rows = tuple(overlay.soft_x for overlay in overlays)
    padded = pad_sequence(list(rows), batch_first=True) if rows else torch.zeros(0, 0, 0)
    trace_tensor(name, padded, 'batch', 'occupied', 'hidden')


__all__ = [
    'CoveringTrace',
    'capture_adapter_delta',
    'covering_adapters_off',
    'covering_patches',
    'covering_patches_on',
    'covering_seams_kept',
    'covering_trace_active',
    'disable_covering_adapters',
    'keep_covering_seams',
    'overlay_edge_shift',
    'patch_covering',
    'patches_on_device',
    'record_covering',
    'shift_overlay_edges',
    'trace_overlay',
    'trace_tensor',
]
