"""Paired eval-mode forwards: real graph prefix versus a zeroed prefix.

Positive delta (host-only NLL minus trunk NLL) means the prefix lowers
masked-LM NLL. Lives here so the Lightning probe and the ablation runner share
the comparison without importing the SSV container from the training module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from ip_claim.ssv.collate import SoftMlmBatch
from ip_claim.ssv.model import SoftTrunkModel, SoftTrunkOutput

INJECT_ABLATION_CAVEAT = (
    'A nonzero inject delta shows the residual reaches host logits. '
    'Garbage in the same slots would also move the number, so it is a '
    'wiring check, not a quality score.'
)


@dataclass(frozen=True)
class GraphPrefixDelta:
    """One batch's trunk NLL, host-only NLL, and host-only-minus-trunk delta."""

    trunk_nll: float
    host_only_nll: float

    @property
    def delta(self) -> float:
        """Ablated NLL minus intact NLL. Positive means the intact channel lowers NLL."""
        return self.host_only_nll - self.trunk_nll


@dataclass(frozen=True)
class GraphPrefixTokenSample:
    """One batch's aggregate delta plus its per-masked-token deltas and mass."""

    aggregate: GraphPrefixDelta

    token_delta: Tensor
    """Host-only minus trunk NLL at each masked position, shape ``(n_masked,)``."""

    assignment_mass: Tensor
    """Entity-assignment peakiness at the same positions, shape ``(n_masked,)``."""


def _paired_forward(
    model: nn.Module,
    batch: SoftMlmBatch,
    *,
    zero_prefix: bool = False,
    zero_inject: bool = False,
) -> tuple[SoftTrunkOutput, SoftTrunkOutput]:
    """Eval-mode paired forward: intact trunk, then one ablated channel.

    Restores the model's prior train/eval mode. ``SoftVocabModule.forward``
    gates usage-EMA and dead-row restarts on ``self.training``, so this uses
    ``eval()`` rather than only ``no_grad()``. The ablated forward zeroes the
    prefix, the injected slots, or both, as requested. A nonzero inject delta
    proves the residual is wired into host logits. It does not prove the
    residual carries information the host lacks.
    """
    was_training = model.training
    model.eval()

    def run(*, ablate_prefix: bool, ablate_inject: bool) -> SoftTrunkOutput:
        return cast(
            SoftTrunkOutput,
            model(
                graphs=batch.graphs,
                input_ids=batch.input_ids,
                unmasked_input_ids=batch.unmasked_input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                texts=batch.texts,
                zero_prefix=ablate_prefix,
                zero_inject=ablate_inject,
            ),
        )

    with torch.no_grad():
        trunk_out = run(ablate_prefix=False, ablate_inject=False)
        host_only_out = run(ablate_prefix=zero_prefix, ablate_inject=zero_inject)
    model.train(was_training)
    return trunk_out, host_only_out


def measure_graph_prefix_delta(model: nn.Module, batch: SoftMlmBatch) -> GraphPrefixDelta:
    """Aggregate host-only-vs-trunk masked-LM NLL delta for one held-out batch."""
    trunk_out, host_only_out = _paired_forward(model, batch, zero_prefix=True)
    return GraphPrefixDelta(
        trunk_nll=float(trunk_out.mlm_loss),
        host_only_nll=float(host_only_out.mlm_loss),
    )


def measure_graph_prefix_token_deltas(
    model: SoftTrunkModel,
    batch: SoftMlmBatch,
    *,
    assignment_top_k: int = 1,
) -> GraphPrefixTokenSample:
    """Per-masked-token host-only-vs-trunk NLL delta, paired with assignment mass.

    Assignment mass is the same soft-vocab peakiness score
    ``SoftMlmCollator`` uses to build ``rho``-weighted mask rates, read from
    the unmasked embeddings so it does not depend on which prefix is zeroed.
    """
    trunk_out, host_only_out = _paired_forward(model, batch, zero_prefix=True)
    aggregate = GraphPrefixDelta(
        trunk_nll=float(trunk_out.mlm_loss),
        host_only_nll=float(host_only_out.mlm_loss),
    )
    masked = trunk_out.mlm_label_ids.ne(-100)
    token_delta = (host_only_out.mlm_token_nll - trunk_out.mlm_token_nll)[masked]

    embed = model.host.get_input_embeddings()
    with torch.no_grad():
        assignment, _ = model.soft_vocab.soft_assign(embed(batch.unmasked_input_ids))
        mass = model.soft_vocab.assignment_mass(assignment, top_k=assignment_top_k)
    text_masked = masked[:, model.n_soft_tokens :]
    return GraphPrefixTokenSample(
        aggregate=aggregate,
        token_delta=token_delta,
        assignment_mass=mass[text_masked],
    )


def measure_graph_inject_delta(model: nn.Module, batch: SoftMlmBatch) -> GraphPrefixDelta:
    """Aggregate NLL delta from zeroing injected slots, prefix left intact."""
    trunk_out, inject_zero_out = _paired_forward(model, batch, zero_inject=True)
    return GraphPrefixDelta(
        trunk_nll=float(trunk_out.mlm_loss),
        host_only_nll=float(inject_zero_out.mlm_loss),
    )


measure_graph_inject_delta.__doc__ = (
    'Aggregate NLL delta from zeroing injected slots, prefix left intact.\n\n'
    + INJECT_ABLATION_CAVEAT
)


__all__ = [
    'INJECT_ABLATION_CAVEAT',
    'GraphPrefixDelta',
    'GraphPrefixTokenSample',
    'measure_graph_inject_delta',
    'measure_graph_prefix_delta',
    'measure_graph_prefix_token_deltas',
]
