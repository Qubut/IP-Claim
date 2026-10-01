"""Session gate cache, measurement stash, and covering ledger writers.

Live items announce measurements on the pytest item. The report hook writes
JSON and cache; the terminal hook writes the Polars ledger. Tests do not
write artefacts.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from itertools import chain, starmap
from pathlib import Path

import polars as pl
import pytest
from great_tables import GT, loc, style

from ip_claim.collision.artefacts import write_json

_LEDGER_ROOT = Path('experiments/runs/covering_objective')
_CALL_OUTCOMES = ('passed', 'failed', 'error')
_GATE_ORDER = (
    'test_algebraic_oracle.py',
    'test_mask_schedule.py',
    'test_paired_identity.py',
    'test_inventory_shape.py',
    'test_activation_patch.py',
    'test_scale_occupancy.py',
    'test_letter_separability.py',
    'test_letter_graph_contribution.py',
    'test_functional_steps.py',
    'test_isolated_pilot.py',
    'test_held_out_pilot.py',
)
_LEDGERS = {
    'test_algebraic_oracle': 'algebraic_oracle',
    'test_mask_schedule': 'mask_schedule',
    'test_paired_identity': 'paired_identity',
    'test_inventory_shape': 'inventory_shape',
    'test_activation_patch': 'activation_patch',
    'test_scale_occupancy': 'scale_occupancy',
    'test_letter_separability': 'letter_separability',
    'test_letter_graph_contribution': 'letter_graph_contribution',
    'test_functional_steps': 'functional_steps',
    'test_isolated_pilot': 'isolated_pilot',
    'test_held_out_pilot': 'held_out_pilot',
}
GateRecorder = Callable[..., None]
GATE_SLUG = pytest.StashKey[str]()
MEASUREMENT = pytest.StashKey[dict[str, object]]()
CACHE_EXTRAS = pytest.StashKey[dict[str, object]]()
_SESSION_LIVE_GATES: set[str] = set()
_SESSION_ANNOUNCED: dict[str, bool] = {}
_SESSION_EXTRAS: dict[str, object] = {}


def live_ledger_slug(item: pytest.Item) -> str | None:
    """Ledger slug for a live covering item, or None for host-free tests."""
    slug = _LEDGERS.get(Path(str(item.path)).stem)
    if slug is None or item.get_closest_marker('live') is None:
        return None
    return slug


def session_live_gate(slug: str) -> bool:
    """True when this session collected a live item for ``slug``."""
    return slug in _SESSION_LIVE_GATES


def session_gate_extra(key: str) -> object | None:
    """Extra cache payload announced by a live gate in this session."""
    return _SESSION_EXTRAS.get(key)


def ledger_dir(rootpath: Path, slug: str) -> Path:
    """Ledger directory. Absolute COVERING_LEDGER_ROOT wins over the tree path."""
    override = os.environ.get('COVERING_LEDGER_ROOT', '').strip()
    root = Path(override) if override else Path(rootpath) / _LEDGER_ROOT
    return root / slug


def begin_session(items: Sequence[pytest.Item]) -> None:
    """Reset session gate sets from the collected live items."""
    _SESSION_LIVE_GATES.clear()
    _SESSION_ANNOUNCED.clear()
    _SESSION_EXTRAS.clear()
    _SESSION_LIVE_GATES.update(
        slug for item in items if (slug := live_ledger_slug(item)) is not None
    )


def prepare_collection(items: list[pytest.Item], *, live_corpus: bool) -> None:
    """Skip live encodes without HUPD. Run hygiene before later gates."""
    skip_live = pytest.mark.skip(reason='live SSV encode needs COVERING_HUPD_DIR')

    def mark_live(item: pytest.Item) -> None:
        if not live_corpus and item.get_closest_marker('live') is not None:
            item.add_marker(skip_live)

    _ = tuple(mark_live(item) for item in items)
    rank = {name: index for index, name in enumerate(_GATE_ORDER)}
    items.sort(key=lambda item: rank.get(item.path.name, len(_GATE_ORDER)))
    begin_session(items)


def stash_gate_measurement(
    item: pytest.Item,
    slug: str,
    measurement: dict[str, object],
    extras: Mapping[str, object] | None,
    properties: Sequence[tuple[str, object]],
) -> None:
    """Attach one announced measurement to the pytest item and session extras."""
    item.stash[GATE_SLUG] = slug
    item.stash[MEASUREMENT] = measurement
    if extras is not None:
        payload = dict(extras)
        item.stash[CACHE_EXTRAS] = payload
        _SESSION_EXTRAS.update(payload)
    item.user_properties.extend(properties)


def bind_require_gate(request: pytest.FixtureRequest) -> Callable[..., None]:
    """Skip when a collected live gate has not run, or a cached prior gate failed.

    Collected live prerequisites are session-honest: a skip does not inherit a
    stale disk pass. ``ran=True`` accepts an announced measurement even when
    that item failed. Uncollected prerequisites still read the pytest cache.
    """

    def require(slug: str, reason: str, *, ran: bool = False) -> None:
        if slug in _SESSION_LIVE_GATES:
            if slug not in _SESSION_ANNOUNCED:
                pytest.skip(reason)
            if not ran and not _SESSION_ANNOUNCED[slug]:
                pytest.skip(reason)
            return
        cache = request.config.cache
        if cache is None or not cache.get(f'covering/{slug}', False):
            pytest.skip(reason)

    return require


def record_call_report(
    item: pytest.Item,
    call: pytest.CallInfo[None],
    report: pytest.TestReport,
) -> pytest.TestReport:
    """Write measurements and gate cache after the call, not from the test body."""
    cache = item.config.cache
    live = live_ledger_slug(item)
    if (
        live is not None
        and GATE_SLUG not in item.stash
        and call.when in {'setup', 'call'}
        and report.outcome in {'skipped', 'failed'}
    ):
        if live not in _SESSION_ANNOUNCED and cache is not None:
            cache.set(f'covering/{live}', False)
        longrepr = str(getattr(report, 'longrepr', '') or '')
        ledger_path = ledger_dir(Path(item.config.rootpath), live)
        write_json(
            ledger_path / 'measurements.json',
            {
                'culprit': 'SKIP' if report.skipped else 'ERROR',
                'culprit_name': 'NOT_ANNOUNCED',
                'when': call.when,
                'outcome': report.outcome,
                'longrepr': longrepr,
            },
        )
        (ledger_path / 'traceback.txt').write_text(longrepr, encoding='utf-8')
        return report
    if call.when != 'call' or GATE_SLUG not in item.stash:
        return report
    slug = item.stash[GATE_SLUG]
    _SESSION_ANNOUNCED[slug] = bool(report.passed)

    def store(key: str, value: object) -> str:
        if cache is not None:
            cache.set(key, value)
        return key

    extras = item.stash.get(CACHE_EXTRAS, {})
    _SESSION_EXTRAS.update(extras)
    _ = (
        store(f'covering/{slug}', bool(report.passed)),
        *tuple(starmap(store, extras.items())),
    )
    write_json(
        ledger_dir(Path(item.config.rootpath), slug) / 'measurements.json',
        item.stash.get(MEASUREMENT, {}),
    )
    return report


def write_session_ledgers(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Write Polars/Great Tables ledger from this pytest session's reports."""

    def ledger_row(report: pytest.TestReport) -> dict[str, str]:
        props = dict(getattr(report, 'user_properties', ()))
        scores = ', '.join(f'{key}={value}' for key, value in props.items())
        nodeid = str(getattr(report, 'nodeid', ''))
        return {
            'claim': nodeid.rsplit('::', maxsplit=1)[-1],
            'verdict': 'PASS' if getattr(report, 'passed', False) else 'FAIL',
            'scores': scores,
        }

    stats = getattr(terminalreporter, 'stats', {})
    reports = tuple(
        report
        for report in chain.from_iterable(stats.get(outcome, ()) for outcome in _CALL_OUTCOMES)
        if isinstance(report, pytest.TestReport) and getattr(report, 'when', 'call') == 'call'
    )

    def write_ledger(slug: str, chosen: tuple[pytest.TestReport, ...]) -> str:
        rows = tuple(ledger_row(report) for report in chosen)
        frame = (
            pl.DataFrame(rows)
            if rows
            else pl.DataFrame(
                schema={'claim': pl.String, 'verdict': pl.String, 'scores': pl.String}
            )
        )
        output_dir = ledger_dir(Path(config.rootpath), slug)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / 'ledger.json'
        write_json(
            json_path,
            {'experiment': slug, 'exitstatus': exitstatus, 'rows': list(rows)},
        )
        table = GT(frame).tab_header(
            title=f'{slug.replace("_", " ")}  exit={exitstatus}',
            subtitle=str(json_path),
        )
        painted = table.tab_style(
            style.fill(color='#f8d7da'),
            loc.body(rows=pl.col('verdict') == 'FAIL'),
        )
        _ = (output_dir / 'ledger.html').write_text(
            painted.as_raw_html(make_page=True),
            encoding='utf-8',
        )
        write_line = getattr(terminalreporter, 'write_line', None)
        if callable(write_line):
            _ = write_line(f'{slug} ledger: {json_path}')
        return slug

    grouped = {
        slug: tuple(report for report in reports if key in str(getattr(report, 'nodeid', '')))
        for key, slug in _LEDGERS.items()
    }
    _ = tuple(write_ledger(slug, chosen) for slug, chosen in grouped.items() if chosen)
