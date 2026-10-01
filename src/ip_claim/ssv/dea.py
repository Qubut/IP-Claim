"""Entity denoising: score remasked post-compose states against the clean bank assignment.

The query is the compose state pulled back onto each token. Hidden-token
identities never enter this head. Teacher codes are the detached clean
assignment. The token grid and the pullback grid share attention-true
positions packed in document order.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from ip_claim.ssv.config import SsvTrainConfig


class EntityDenoiseTerms(BaseModel):
    """Soft cross-entropy at hidden nodes and the modal-code gap."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dea_loss: Tensor
    """Mean soft cross-entropy on hidden compose nodes."""

    dea_gap: Tensor
    """Hidden top-1 agreement minus the batch-modal teacher-code rate."""


class EntityDenoiseHead(nn.Module):
    """Score entity-bank codes for remasked positions from post-compose node states."""

    def __init__(self, config: SsvTrainConfig) -> None:
        super().__init__()
        self.to_bank = nn.Linear(int(config.arch.gnn_hidden), int(config.arch.soft_dim))
        self._temperature = float(config.arch.dea_temperature)

    def forward(self, node_states: Tensor, bank: Tensor) -> Tensor:
        """Log-probabilities over bank codes.

        Args:
            node_states: Post-compose states ``(..., gnn_hidden)``.
            bank: Entity codebook ``(K_e, soft_dim)``.

        Returns:
            Log-softmax over bank codes ``(..., K_e)``.
        """
        queries = F.normalize(self.to_bank(node_states), dim=-1)
        codes = F.normalize(bank, dim=-1)
        logits = torch.einsum('...d,kd->...k', queries, codes) / self._temperature
        return F.log_softmax(logits, dim=-1)


def aligned_token_node_index(
    attention_mask: Tensor,
    node_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map attention-true tokens onto the dense compose grid in document order.

    Returns batch index, token index, and packed node index for each real token.
    A count mismatch between real tokens and compose nodes raises; the grids
    are never cropped.
    """
    real = attention_mask.bool()
    n_real = real.sum(dim=-1)
    n_nodes = node_mask.to(dtype=torch.long).sum(dim=-1)
    if not torch.equal(n_real, n_nodes):
        msg = (
            f'attention-true token counts {n_real.tolist()} do not match '
            f'compose node counts {n_nodes.tolist()}'
        )
        raise ValueError(msg)
    node_index = real.to(dtype=torch.long).cumsum(dim=-1) - 1
    batch_index, token_index = real.nonzero(as_tuple=True)
    return batch_index, token_index, node_index[batch_index, token_index]


def pack_token_rows_to_nodes(token_rows: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Pack attention-true token rows onto the dense compose grid in document order."""
    batch_index, token_index, packed_index = aligned_token_node_index(
        attention_mask,
        attention_mask,
    )
    n_max = int(attention_mask.sum(dim=-1).max().item()) if attention_mask.numel() else 0
    packed = token_rows.new_zeros((token_rows.size(0), n_max, token_rows.size(-1)))
    pack_mask = torch.zeros(
        token_rows.size(0),
        n_max,
        dtype=torch.bool,
        device=token_rows.device,
    )
    if n_max > 0:
        packed[batch_index, packed_index] = token_rows[batch_index, token_index]
        pack_mask[batch_index, packed_index] = True
    return packed, pack_mask


def scatter_node_rows_to_tokens(
    node_rows: Tensor,
    attention_mask: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Place dense compose rows onto the attention-true token grid."""
    batch_index, token_index, packed_node_index = aligned_token_node_index(
        attention_mask,
        node_mask,
    )
    batch, token_length = attention_mask.shape
    scattered = node_rows.new_zeros((batch, token_length, *node_rows.shape[2:]))
    scattered[batch_index, token_index] = node_rows[batch_index, packed_node_index]
    return scattered


def entity_denoise_terms(
    head: EntityDenoiseHead,
    node_states: Tensor,
    node_mask: Tensor,
    bank: Tensor,
    teacher: Tensor,
    labels: Tensor,
    attention_mask: Tensor,
) -> EntityDenoiseTerms:
    """Soft CE and modal gap at hidden compose nodes.

    Hidden flags and teacher rows live on the token grid. They are packed onto
    the dense node grid by the attention-true token order.
    """
    batch_index, token_index, packed_node_index = aligned_token_node_index(
        attention_mask,
        node_mask,
    )

    def pack_to_nodes(token_rows: Tensor) -> Tensor:
        """Place attention-true token rows onto the dense compose grid."""
        batch, node_length = node_mask.shape
        packed = token_rows.new_zeros((batch, node_length, *token_rows.shape[2:]))
        packed[batch_index, packed_node_index] = token_rows[batch_index, token_index]
        return packed

    teacher_nodes = pack_to_nodes(teacher.detach())
    hidden_nodes = pack_to_nodes(labels.ge(0)) & node_mask
    log_probs = head(node_states, bank)
    nll = -(teacher_nodes * log_probs).sum(dim=-1)
    hidden_nll = nll[hidden_nodes]
    empty_hidden = hidden_nll.numel() == 0
    if empty_hidden and not node_mask.any():
        return EntityDenoiseTerms(dea_loss=nll.new_zeros(()), dea_gap=nll.new_zeros(()))
    if empty_hidden:
        msg = 'entity denoise received compose nodes but no hidden positions'
        raise RuntimeError(msg)

    pred_codes = log_probs.argmax(dim=-1)[hidden_nodes]
    gold_codes = teacher_nodes.argmax(dim=-1)[hidden_nodes]
    top1 = pred_codes.eq(gold_codes).to(dtype=nll.dtype).mean()
    modal = gold_codes.mode().values
    baseline = gold_codes.eq(modal).to(dtype=nll.dtype).mean()
    return EntityDenoiseTerms(dea_loss=hidden_nll.mean(), dea_gap=(top1 - baseline).detach())


__all__ = [
    'EntityDenoiseHead',
    'EntityDenoiseTerms',
    'aligned_token_node_index',
    'entity_denoise_terms',
    'pack_token_rows_to_nodes',
    'scatter_node_rows_to_tokens',
]
