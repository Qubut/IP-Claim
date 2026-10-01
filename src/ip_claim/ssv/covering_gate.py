"""Covering-ready train reading: bound directions, not bare scalars.

Document inventory entropy is an upper bound. Covering wants a sparse
filing bag, so H(μ) should fall well below ln K. Corpus usage entropy is
a lower bound: sparse filings must still union across the bank. Host NLL
is host health, not a covering score. The dEA gap is graph soundness, not
a covering verdict. Token perplexity and row entropy are hygiene.
Terminal acceptance stays unpaid X below A below random on exported
inventories.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

INVENTORY_BOUND = 'upper: fall well below ln K (sparse filing)'
USAGE_BOUND = 'lower: hold near target (sparse filings still union the bank)'
HOST_BOUND = 'host health: stay at or below the NLL ceiling'
DEA_BOUND = 'graph soundness only: not a covering verdict'
ROW_HYGIENE = 'hygiene: peaked tokens already held when covering failed'
TERMINAL_ACCEPTANCE = 'terminal: unpaid X below A below random on exported inventories'


class CoveringTrainReading(BaseModel):
    """One-step reading of the covering-ready train arms.

    Slack signs are load-bearing. ``inventory_below_ln_k`` grows as the
    document bag sparsifies. ``usage_above_target`` grows as the corpus
    still covers the bank. A falling document entropy is health on the
    inventory arm, not degradation.
    """

    model_config = ConfigDict(frozen=True)

    inventory_entropy: float
    ln_k: float
    usage_entropy: float
    usage_target: float
    row_entropy: float
    host_nll: float | None = None
    host_nll_ceiling: float | None = None
    dea_gap: float | None = None

    @property
    def inventory_below_ln_k(self) -> float:
        """Positive when document entropy sits below uniform ln K."""
        return self.ln_k - self.inventory_entropy

    @property
    def usage_above_target(self) -> float:
        """Positive when corpus usage entropy meets the lower bound."""
        return self.usage_entropy - self.usage_target

    @property
    def assignment_perplexity(self) -> float:
        """Token-assignment perplexity; hygiene, not a covering arm."""
        return math.exp(self.row_entropy)

    @property
    def host_nll_above_ceiling(self) -> float | None:
        """Positive when host NLL sits above the seeded ceiling."""
        if self.host_nll is None or self.host_nll_ceiling is None:
            return None
        return self.host_nll - self.host_nll_ceiling

    def log_scalars(self) -> dict[str, float]:
        """Directional slacks for the CSV. Existing entropy keys stay elsewhere."""
        values = {
            'inventory_ln_k': self.ln_k,
            'inventory_below_ln_k': self.inventory_below_ln_k,
            'usage_above_target': self.usage_above_target,
        }
        if self.host_nll_above_ceiling is not None:
            values['host_nll_above_ceiling'] = self.host_nll_above_ceiling
        return values

    def health_cells(self) -> dict[str, object]:
        """Inspect health columns with bound direction on every leading arm."""
        return {
            'inventory_entropy': round(self.inventory_entropy, 4),
            'inventory_bound': INVENTORY_BOUND,
            'inventory_below_ln_k': round(self.inventory_below_ln_k, 4),
            'usage_entropy': round(self.usage_entropy, 4),
            'usage_bound': USAGE_BOUND,
            'usage_above_target': round(self.usage_above_target, 4),
            'row_entropy_hygiene': round(self.row_entropy, 4),
            'row_entropy_role': ROW_HYGIENE,
        }

    def report_rows(self, label: str) -> tuple[dict[str, object], ...]:
        """One report row per arm so bound direction cannot be inverted."""

        def row(
            arm: str,
            value: float | None,
            bound: str,
            slack: float | None,
            role: Literal['leading', 'host_health', 'graph_soundness', 'hygiene', 'terminal'],
        ) -> dict[str, object]:
            return {
                'label': label,
                'arm': arm,
                'value': '' if value is None else round(value, 4),
                'bound': bound,
                'slack': '' if slack is None else round(slack, 4),
                'role': role,
            }

        return (
            row(
                'inventory_entropy',
                self.inventory_entropy,
                INVENTORY_BOUND,
                self.inventory_below_ln_k,
                'leading',
            ),
            row(
                'usage_entropy',
                self.usage_entropy,
                USAGE_BOUND,
                self.usage_above_target,
                'leading',
            ),
            row(
                'host_nll',
                self.host_nll,
                HOST_BOUND,
                self.host_nll_above_ceiling,
                'host_health',
            ),
            row('dea_gap', self.dea_gap, DEA_BOUND, None, 'graph_soundness'),
            row(
                'row_entropy',
                self.row_entropy,
                ROW_HYGIENE,
                None,
                'hygiene',
            ),
            row('covering_eval', None, TERMINAL_ACCEPTANCE, None, 'terminal'),
        )


__all__ = [
    'DEA_BOUND',
    'HOST_BOUND',
    'INVENTORY_BOUND',
    'ROW_HYGIENE',
    'TERMINAL_ACCEPTANCE',
    'USAGE_BOUND',
    'CoveringTrainReading',
]
