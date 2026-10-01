"""Lightning distributed strategy selection for SSV training."""

from __future__ import annotations

from enum import StrEnum
from functools import partial
from typing import Any

import torch
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy, Strategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from ip_claim.ssv.config import SsvTrainConfig

_FSDP_WRAP_MIN_PARAMS = 100_000


class DistributedStrategy(StrEnum):
    """Multi-GPU training strategy keys (YAML ``distributed_strategy``)."""

    AUTO = 'auto'
    DDP = 'ddp'
    FSDP_GRAD_OP = 'fsdp_grad_op'
    FSDP_FULL = 'fsdp_full'


def _gpu_process_count(job: SsvTrainConfig) -> int:
    if not job.runtime.use_gpu or not torch.cuda.is_available():
        return 1
    if job.fit.num_devices is not None:
        return int(job.fit.num_devices)
    return max(torch.cuda.device_count(), 1)


def _fsdp_strategy(
    *,
    sharding: str,
    activation_checkpoint: bool,
) -> FSDPStrategy:
    wrap = partial(size_based_auto_wrap_policy, min_num_params=_FSDP_WRAP_MIN_PARAMS)
    kwargs: dict[str, Any] = {
        'sharding_strategy': sharding,
        'state_dict_type': 'full',
        'auto_wrap_policy': wrap,
    }
    if activation_checkpoint:
        kwargs['activation_checkpointing_policy'] = wrap
    return FSDPStrategy(**kwargs)


def resolve_lightning_strategy(job: SsvTrainConfig) -> str | Strategy:
    """Pick a Lightning strategy from job config and visible GPU count.

    ``ddp`` minimizes communication overhead when each microbatch fits in device
    memory. ``fsdp_grad_op`` shards optimizer states and gradients. ``fsdp_full``
    shards parameters when VRAM is tight or sequences are long.
    """
    if _gpu_process_count(job) <= 1:
        return 'auto'

    key = DistributedStrategy(job.fit.distributed_strategy)
    match key:
        case DistributedStrategy.AUTO:
            return 'auto'
        case DistributedStrategy.DDP:
            return DDPStrategy(
                static_graph=bool(job.fit.ddp_static_graph),
                find_unused_parameters=bool(job.fit.ddp_find_unused_parameters),
            )
        case DistributedStrategy.FSDP_GRAD_OP:
            return _fsdp_strategy(
                sharding='SHARD_GRAD_OP',
                activation_checkpoint=bool(job.fit.fsdp_activation_checkpointing),
            )
        case DistributedStrategy.FSDP_FULL:
            return _fsdp_strategy(
                sharding='FULL_SHARD',
                activation_checkpoint=bool(job.fit.fsdp_activation_checkpointing),
            )


__all__ = ['DistributedStrategy', 'resolve_lightning_strategy']
