"""Project GNN soft tokens into the host LM embedding width.

Used for ``inputs_embeds`` prefixing. Host weights and LoRA stay outside this
module.
"""

from __future__ import annotations

import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from ip_claim.ssv.config import SsvTrainConfig


class SoftProjectOutput(BaseModel):
    """Host-width soft tokens ready for embedder-side attachment."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tokens: Tensor
    """Projected soft tokens ``(B, n_soft, d_model)``."""


class SoftTokenProjector(nn.Module):
    """Two-layer MLP from GNN hidden width to host ``d_model``."""

    def __init__(self, config: SsvTrainConfig) -> None:
        super().__init__()
        hidden = int(config.arch.gnn_hidden)
        d_model = int(config.host.d_model)
        mid = max(hidden, d_model)
        self.fc1 = nn.Linear(hidden, mid)
        self.fc2 = nn.Linear(mid, d_model)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, soft_tokens: Tensor) -> SoftProjectOutput:
        """Project ``(B, n_soft, gnn_hidden)`` into host embedding width."""
        hidden = F.gelu(self.fc1(soft_tokens))
        return SoftProjectOutput(tokens=self.fc2(hidden))


__all__ = [
    'SoftProjectOutput',
    'SoftTokenProjector',
]
