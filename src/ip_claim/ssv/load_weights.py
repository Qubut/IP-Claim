"""Load Lightning checkpoints for the SSV trunk."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import structlog
import torch
from returns.result import Failure, Result, Success, safe
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from torch import Tensor, nn

_CONTRACT_EMA_KEY = 'inventory_entropy_ema'
_BANK_PARTS = frozenset({'entity_bank', 'relation_bank'})
_log = structlog.get_logger(__name__)


def align_hgt_key(key: str) -> str:
    """Insert ``conv`` after HGT ``convs.N`` when a checkpoint predates that nest."""
    parts = key.split('.')
    for index, part in enumerate(parts[:-2]):
        insert_at = index + 3
        if (
            part == 'encoder'
            and parts[index + 1] == 'convs'
            and parts[index + 2].isdigit()
            and insert_at < len(parts)
            and parts[insert_at] != 'conv'
        ):
            return '.'.join([*parts[:insert_at], 'conv', *parts[insert_at:]])
    return key


def take_model_prefix(state: Mapping[str, object]) -> dict[str, object]:
    """Keep Lightning ``model.*`` rows and strip that prefix."""
    return {
        key.removeprefix('model.'): value
        for key, value in state.items()
        if key.startswith('model.')
    }


def load_lightning_payload(path: Path) -> Result[dict[str, Tensor], str]:
    """Read a Lightning ``state_dict`` from disk; missing files are a Failure."""
    if not path.is_file():
        return Failure(f'checkpoint missing: {path}')

    @safe(exceptions=(OSError,))
    @retry(
        retry=retry_if_exception_type(OSError),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.1, max=2.0),
        reraise=True,
    )
    def read_payload() -> object:
        return torch.load(path, map_location='cpu', weights_only=False)

    payload = read_payload()
    match payload:
        case Failure(error):
            return Failure(str(error))
        case Success(s) if isinstance(s, dict):
            state = s.get('state_dict')
            if isinstance(state, dict):
                return Success(state)
            return Failure('Lightning checkpoint has no state_dict mapping')
        case _:
            return Failure('Lightning checkpoint is not a mapping')


def load_model_weights(trunk: nn.Module, path: Path) -> Result[None, str]:
    """Load matching ``model.*`` tensors into the trunk.

    Steer and optimizer stay untouched. Checkpoint keys whose shape differs
    from the current trunk stay at current init. Module extra-state is not a
    weight; termhood bind stays on the train config path or covering attach.

    When the prefix includes entity or relation bank rows, mark the vocab banks
    as already seeded so the first-batch k-means path does not overwrite them.
    """
    loaded = load_lightning_payload(path)
    match loaded:
        case Failure() as fail:
            return fail
        case Success(state):
            taken = {align_hgt_key(key): value for key, value in take_model_prefix(state).items()}
            host = trunk.state_dict()

            def compatible_row(current: object, value: object) -> bool:
                if isinstance(current, Tensor) and isinstance(value, Tensor):
                    return current.shape == value.shape
                return not isinstance(current, Tensor) and not isinstance(value, Tensor)

            compatible = {
                key: value
                for key, value in taken.items()
                if not key.endswith('_extra_state')
                and (key not in host or compatible_row(host[key], value))
            }
            dropped = tuple(sorted(key for key in taken if key not in compatible))
            if dropped:
                _log.info(
                    'ssv.load_weights.dropped_shape_mismatch',
                    count=len(dropped),
                    keys=dropped,
                )
            trunk.load_state_dict(compatible, strict=False)
            has_banks = any(part in _BANK_PARTS for key in compatible for part in key.split('.'))
            vocab = getattr(trunk, 'soft_vocab', None)
            seeded = getattr(vocab, '_banks_seeded', None)
            if has_banks and seeded is not None:
                _ = seeded.fill_(True)
            return Success(None)
        case _:
            return Failure('Lightning checkpoint is not a mapping')


def resolve_fit_ckpt(*, last_ckpt: Path, init_weights: Path | None) -> Path | None:
    """Return ``last.ckpt`` only when it is this contract and may resume.

    This contract is a Lightning payload that stores ``inventory_entropy_ema``.
    A file without that buffer is ignored (weights-only from ``init_weights``).
    When ``init_weights`` is set, ``last.ckpt`` must also be newer than that file.
    """
    if not last_ckpt.is_file():
        return None
    loaded = load_lightning_payload(last_ckpt)
    match loaded:
        case Success(state) if _CONTRACT_EMA_KEY in state:
            pass
        case _:
            return None
    if init_weights is None or not init_weights.is_file():
        return last_ckpt
    if last_ckpt.stat().st_mtime > init_weights.stat().st_mtime:
        return last_ckpt
    return None
