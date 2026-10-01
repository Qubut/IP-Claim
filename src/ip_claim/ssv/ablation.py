from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal, TypeVar

import polars as pl
import torch
from returns.result import Failure, Success
from torch.utils.data import DataLoader

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ssv.collate import SoftMlmBatch
from ip_claim.ssv.config import DEFAULT_SSV_TRAIN_PATH, SsvTrainConfig
from ip_claim.ssv.dataset import LazyHupdMlmDataset
from ip_claim.ssv.graph_prefix import (
    INJECT_ABLATION_CAVEAT,
    GraphPrefixDelta,
    GraphPrefixTokenSample,
    measure_graph_inject_delta,
    measure_graph_prefix_delta,
    measure_graph_prefix_token_deltas,
)

__doc__ = (
    'Paired graph-channel ablation: prefix versus inject.\n\n'
    'One checkpoint, one held-out collated stream, two eval-mode forwards. '
    'The prefix channel differs in ``zero_prefix``. The inject channel differs '
    'in ``zero_inject``. ' + INJECT_ABLATION_CAVEAT
)


def _markdown_table(rows: Sequence[Mapping[str, str]]) -> str:
    """Render pre-formatted string cells as a GFM pipe table via one Polars frame."""
    frame = pl.DataFrame(rows)
    with pl.Config(
        tbl_formatting='MARKDOWN',
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_rows=-1,
        fmt_str_lengths=200,
        tbl_width_chars=400,
    ):
        return f'{frame}'


_DEFAULT_RHO = 0.31
_DEFAULT_N_BATCHES = 4
_DEFAULT_BATCH_SIZE = 4

T = TypeVar('T')


@dataclass(frozen=True)
class GraphPrefixAblationReport:
    """Per-batch deltas plus mean, sample sd, and sign-flip count."""

    batches: tuple[GraphPrefixDelta, ...]
    mean_delta: float
    sd_delta: float
    n_sign_flips: int
    n_positive: int
    n_negative: int
    trunk_column: str = 'trunk NLL (real prefix)'
    host_column: str = 'host-only NLL (zero prefix)'
    caveat: str = ''

    def as_table(self) -> str:
        """Render the per-batch NLL table plus the aggregate footer row."""
        n = len(self.batches)
        mean_trunk = statistics.fmean(row.trunk_nll for row in self.batches)
        mean_host = statistics.fmean(row.host_only_nll for row in self.batches)
        rows = [
            {
                'batch': str(index),
                self.trunk_column: f'{row.trunk_nll:.3f}',
                self.host_column: f'{row.host_only_nll:.3f}',
                'delta': f'{row.delta:+.3f}',
            }
            for index, row in enumerate(self.batches)
        ]
        rows.append({
            'batch': 'mean',
            self.trunk_column: f'{mean_trunk:.4f}',
            self.host_column: f'{mean_host:.4f}',
            'delta': (
                f'{self.mean_delta:+.4f} (sd {self.sd_delta:.4f}, '
                f'{self.n_sign_flips} sign flips / {n})'
            ),
        })
        table = f'{_markdown_table(rows)}'
        if not self.caveat:
            return table
        return f'{table}\n{self.caveat}'


@dataclass(frozen=True)
class MassStratumSummary:
    """Mean, sample sd, and token count for one assignment-mass stratum."""

    mean_delta: float
    sd_delta: float
    n_tokens: int


@dataclass(frozen=True)
class GraphPrefixMassStratificationReport:
    """Held-out per-token deltas split at the pooled assignment-mass median."""

    threshold: float
    high_mass: MassStratumSummary
    low_mass: MassStratumSummary

    def as_table(self) -> str:
        """Render the two-stratum mean/sd/count table."""
        rows = [
            {
                'stratum': f'high mass (>= {self.threshold:.4f})',
                'n tokens': str(self.high_mass.n_tokens),
                'mean delta': f'{self.high_mass.mean_delta:+.4f}',
                'sd delta': f'{self.high_mass.sd_delta:.4f}',
            },
            {
                'stratum': f'low mass (< {self.threshold:.4f})',
                'n tokens': str(self.low_mass.n_tokens),
                'mean delta': f'{self.low_mass.mean_delta:+.4f}',
                'sd delta': f'{self.low_mass.sd_delta:.4f}',
            },
        ]
        return f'{_markdown_table(rows)}'


@dataclass(frozen=True)
class GraphPrefixAblationResult:
    """Aggregate ablation report plus an optional assignment-mass stratification."""

    ablation: GraphPrefixAblationReport
    mass_stratification: GraphPrefixMassStratificationReport | None = None


def stratify_graph_prefix_token_deltas(
    rows: Sequence[GraphPrefixTokenSample],
) -> GraphPrefixMassStratificationReport:
    """Split pooled masked-token deltas at the median assignment mass."""
    delta = torch.cat([row.token_delta for row in rows])
    mass = torch.cat([row.assignment_mass for row in rows])
    finite = torch.isfinite(delta) & torch.isfinite(mass)
    delta = delta[finite]
    mass = mass[finite]
    if delta.numel() == 0:
        msg = 'graph-prefix mass stratification produced no finite masked-token deltas'
        raise ValueError(msg)
    threshold = float(mass.median())
    high = delta[mass >= threshold]
    low = delta[mass < threshold]

    def summarize(values: torch.Tensor) -> MassStratumSummary:
        count = int(values.numel())
        return MassStratumSummary(
            mean_delta=float(values.mean()) if count else float('nan'),
            sd_delta=float(values.std()) if count >= 2 else 0.0,
            n_tokens=count,
        )

    return GraphPrefixMassStratificationReport(
        threshold=threshold,
        high_mass=summarize(high),
        low_mass=summarize(low),
    )


def summarize_graph_prefix_deltas(
    rows: Sequence[GraphPrefixDelta],
    *,
    trunk_column: str = 'trunk NLL (real prefix)',
    host_column: str = 'host-only NLL (zero prefix)',
    caveat: str = '',
) -> GraphPrefixAblationReport:
    """Mean, sample sd, and sign-flip count over finite paired-forward deltas."""
    finite = tuple(row for row in rows if math.isfinite(row.delta))
    if not finite:
        msg = 'graph-prefix ablation produced no finite NLL deltas'
        raise ValueError(msg)
    deltas = tuple(row.delta for row in finite)
    mean_delta = statistics.fmean(deltas)
    sd_delta = statistics.stdev(deltas) if len(deltas) >= 2 else 0.0
    n_positive = sum(delta > 0.0 for delta in deltas)
    n_negative = sum(delta < 0.0 for delta in deltas)
    n_sign_flips = n_negative if mean_delta > 0.0 else n_positive if mean_delta < 0.0 else 0
    return GraphPrefixAblationReport(
        batches=finite,
        mean_delta=mean_delta,
        sd_delta=sd_delta,
        n_sign_flips=n_sign_flips,
        n_positive=n_positive,
        n_negative=n_negative,
        trunk_column=trunk_column,
        host_column=host_column,
        caveat=caveat,
    )


def run_graph_prefix_ablation(  # noqa: C901
    *,
    checkpoint: Path,
    config_path: Path | None = None,
    hupd_dir: Path | None = None,
    hupd_limit: int | None = None,
    n_batches: int = _DEFAULT_N_BATCHES,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    rho: float = _DEFAULT_RHO,
    stratify_by_mass: bool = False,
    device: torch.device | None = None,
    channel: Literal['prefix', 'inject'] = 'prefix',
) -> GraphPrefixAblationResult:
    """Load a trunk checkpoint, collate held-out HUPD rows, and measure deltas.

    ``channel='prefix'`` zeroes the soft prefix. ``channel='inject'`` zeroes
    the hidden-slot residual and leaves the prefix intact.

    ``stratify_by_mass`` additionally splits the same held-out batches' masked-
    token deltas by the collator's own assignment-mass peakiness score. It
    applies only to the prefix channel.
    """

    def require_args() -> None:
        if n_batches < 1:
            msg = f'n_batches must be >= 1; got {n_batches}'
            raise ValueError(msg)
        if batch_size < 1:
            msg = f'batch_size must be >= 1; got {batch_size}'
            raise ValueError(msg)
        if not 0.0 <= rho <= 1.0:
            msg = f'rho must be in [0, 1]; got {rho}'
            raise ValueError(msg)
        if not checkpoint.is_file():
            msg = f'checkpoint missing: {checkpoint}'
            raise FileNotFoundError(msg)
        if channel == 'inject' and stratify_by_mass:
            msg = 'inject ablation has no assignment-mass token pairing'
            raise ValueError(msg)

    require_args()
    yaml_path = config_path if config_path is not None else DEFAULT_SSV_TRAIN_PATH
    job = SsvTrainConfig.from_yaml(yaml_path)
    if hupd_dir is not None:
        job = job.overlay({'hupd_dir': str(hupd_dir)})
    row_cap = hupd_limit if hupd_limit is not None else n_batches * batch_size
    job = job.overlay({'hupd_limit': row_cap, 'batch_size': batch_size})

    container = SsvContainer(config=job)
    module = container.lightning_module()
    collator = container.train_collator()
    module.bind_mask_collator(collator)
    collator.set_rho(rho)
    match container.init_weights(path=checkpoint):
        case Failure(message):
            raise RuntimeError(message)
        case Success():
            pass

    resolved_device = (
        device
        if device is not None
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    _ = module.to(resolved_device)
    dataset = LazyHupdMlmDataset(
        Path(job.runtime.hupd_dir) if job.runtime.hupd_dir else None,
        limit=job.runtime.hupd_limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), max(len(dataset), 1)),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    def collect(measure: Callable[[SoftMlmBatch], T]) -> tuple[T, ...]:
        rows = tuple(measure(batch) for batch in islice(loader, n_batches))
        if len(rows) < n_batches:
            msg = f'held-out stream yielded {len(rows)} batches; need {n_batches}'
            raise ValueError(msg)
        return rows

    if stratify_by_mass:

        def token_sample_for_batch(batch: SoftMlmBatch) -> GraphPrefixTokenSample:
            moved = module.transfer_batch_to_device(batch, resolved_device)
            return measure_graph_prefix_token_deltas(
                module.model,
                moved,
                assignment_top_k=collator.assignment_top_k,
            )

        samples = collect(token_sample_for_batch)
        return GraphPrefixAblationResult(
            ablation=summarize_graph_prefix_deltas([sample.aggregate for sample in samples]),
            mass_stratification=stratify_graph_prefix_token_deltas(samples),
        )

    columns = {
        'prefix': ('trunk NLL (real prefix)', 'host-only NLL (zero prefix)'),
        'inject': ('trunk NLL (inject on)', 'host-only NLL (zero inject)'),
    }
    measures = {
        'prefix': measure_graph_prefix_delta,
        'inject': measure_graph_inject_delta,
    }
    trunk_column, host_column = columns[channel]
    measure = measures[channel]

    def delta_for_batch(batch: SoftMlmBatch) -> GraphPrefixDelta:
        moved = module.transfer_batch_to_device(batch, resolved_device)
        return measure(module.model, moved)

    report = summarize_graph_prefix_deltas(
        collect(delta_for_batch),
        trunk_column=trunk_column,
        host_column=host_column,
        caveat=INJECT_ABLATION_CAVEAT if channel == 'inject' else '',
    )
    return GraphPrefixAblationResult(ablation=report)


__all__ = [
    'INJECT_ABLATION_CAVEAT',
    'GraphPrefixAblationReport',
    'GraphPrefixAblationResult',
    'GraphPrefixDelta',
    'GraphPrefixMassStratificationReport',
    'GraphPrefixTokenSample',
    'MassStratumSummary',
    'measure_graph_inject_delta',
    'measure_graph_prefix_delta',
    'measure_graph_prefix_token_deltas',
    'run_graph_prefix_ablation',
    'stratify_graph_prefix_token_deltas',
    'summarize_graph_prefix_deltas',
]
