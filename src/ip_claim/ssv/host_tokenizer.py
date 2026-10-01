"""Host tokenizer loading for SSV train and eval."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ip_claim.ssv.config import SsvTrainConfig


def batch_encoding_tensor(encoded: object, key: str) -> Tensor:
    """Return the torch tensor stored under ``key`` after ``return_tensors='pt'``."""
    if not isinstance(encoded, Mapping):
        msg = 'tokenizer must return a mapping'
        raise TypeError(msg)
    value = encoded[key]
    if isinstance(value, Tensor):
        return value
    msg = f'tokenizer must return a tensor for {key}'
    raise TypeError(msg)


def load_fast_host_tokenizer(
    tokenizer_id: str,
    *,
    token: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load a fast tokenizer so occupy can use character offsets without spaCy."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, token=token)
    if not getattr(tokenizer, 'is_fast', False):
        raise RuntimeError(f'host tokenizer {tokenizer_id} has no fast offset mapping')
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_host_tokenizer(config: SsvTrainConfig) -> PreTrainedTokenizerBase:
    """Load the host tokenizer; pad token falls back to eos when missing."""
    token = None
    if config.runtime.hf_token is not None:
        token = config.runtime.hf_token.get_secret_value().strip() or None
    tokenizer_id = config.host.tokenizer_id or config.host.name
    return load_fast_host_tokenizer(tokenizer_id, token=token)


__all__ = [
    'batch_encoding_tensor',
    'load_fast_host_tokenizer',
    'load_host_tokenizer',
]
