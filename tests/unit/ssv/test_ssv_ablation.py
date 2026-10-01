"""Graph-prefix ablation: shared paired-forward delta, report shape, CLI args."""

from __future__ import annotations

import ast
import inspect
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from tests._ssv_fixtures import ssv_fixture_batch, ssv_smoke_module
from torch import nn

from ip_claim.ssv import SsvLightningModule
from ip_claim.ssv import __main__ as ssv_main
from ip_claim.ssv import ablation as ablation_mod
from ip_claim.ssv import module as module_mod
from ip_claim.ssv.ablation import (
    INJECT_ABLATION_CAVEAT,
    GraphPrefixAblationReport,
    GraphPrefixAblationResult,
    GraphPrefixDelta,
    GraphPrefixMassStratificationReport,
    GraphPrefixTokenSample,
    MassStratumSummary,
    measure_graph_prefix_delta,
    stratify_graph_prefix_token_deltas,
    summarize_graph_prefix_deltas,
)
from ip_claim.ssv.graph_prefix import (
    measure_graph_inject_delta,
    measure_graph_prefix_token_deltas,
)


class _NllStub(nn.Module):
    """Return distinct NLLs for the intact trunk versus one ablated channel."""

    def __init__(
        self,
        host_only_nll: float,
        trunk_nll: float,
        *,
        flag: str = 'zero_prefix',
    ) -> None:
        super().__init__()
        self._host_only_nll = host_only_nll
        self._trunk_nll = trunk_nll
        self._flag = flag

    def forward(self, *args: object, **kwargs: object) -> SimpleNamespace:
        del args
        nll = self._host_only_nll if kwargs.get(self._flag) else self._trunk_nll
        return SimpleNamespace(mlm_loss=torch.tensor(nll))


def _dummy_batch() -> SimpleNamespace:
    return SimpleNamespace(
        graphs=None,
        input_ids=None,
        unmasked_input_ids=None,
        attention_mask=None,
        labels=None,
        texts=(),
    )


def test_measure_delta_is_host_only_minus_trunk() -> None:
    model = _NllStub(host_only_nll=3.0, trunk_nll=1.0)
    model.train()
    measured = measure_graph_prefix_delta(model, _dummy_batch())  # type: ignore[arg-type]
    assert measured.trunk_nll == pytest.approx(1.0)
    assert measured.host_only_nll == pytest.approx(3.0)
    assert measured.delta == pytest.approx(2.0)
    assert model.training is True


def test_measure_restores_eval_mode_when_already_eval() -> None:
    model = _NllStub(host_only_nll=1.0, trunk_nll=1.0)
    model.eval()
    _ = measure_graph_prefix_delta(model, _dummy_batch())  # type: ignore[arg-type]
    assert model.training is False


def test_measure_inject_delta_is_zero_inject_minus_trunk() -> None:
    model = _NllStub(host_only_nll=2.5, trunk_nll=1.0, flag='zero_inject')
    model.train()
    measured = measure_graph_inject_delta(model, _dummy_batch())  # type: ignore[arg-type]
    assert measured.trunk_nll == pytest.approx(1.0)
    assert measured.host_only_nll == pytest.approx(2.5)
    assert measured.delta == pytest.approx(1.5)
    assert model.training is True


def test_prefix_measure_ignores_zero_inject_flag() -> None:
    model = _NllStub(host_only_nll=3.0, trunk_nll=1.0, flag='zero_inject')
    measured = measure_graph_prefix_delta(model, _dummy_batch())  # type: ignore[arg-type]
    assert measured.delta == pytest.approx(0.0)


def test_summarize_mean_sd_and_sign_flips() -> None:
    rows = (
        GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.014),
        GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.003),
        GraphPrefixDelta(trunk_nll=1.0, host_only_nll=0.994),
        GraphPrefixDelta(trunk_nll=1.0, host_only_nll=0.996),
    )
    expected_sd = statistics.stdev((0.014, 0.003, -0.006, -0.004))
    report = summarize_graph_prefix_deltas(rows)
    assert report.mean_delta == pytest.approx(0.00175)
    assert report.sd_delta == pytest.approx(expected_sd)
    assert report.n_positive == 2
    assert report.n_negative == 2
    assert report.n_sign_flips == 2
    table = report.as_table()
    assert 'trunk NLL (real prefix)' in table
    assert 'host-only NLL (zero prefix)' in table
    assert 'sign flips / 4' in table
    assert table.splitlines()[0].startswith('| batch |')
    assert INJECT_ABLATION_CAVEAT not in table


def test_ablation_module_doc_names_both_channels() -> None:
    assert ablation_mod.__doc__ is not None
    assert 'zero_prefix' in ablation_mod.__doc__
    assert 'zero_inject' in ablation_mod.__doc__
    assert INJECT_ABLATION_CAVEAT in ablation_mod.__doc__


def test_inject_report_table_states_wiring_check() -> None:
    rows = (GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.2),)
    table = summarize_graph_prefix_deltas(rows, caveat=INJECT_ABLATION_CAVEAT).as_table()
    assert INJECT_ABLATION_CAVEAT in table
    assert table.endswith(INJECT_ABLATION_CAVEAT)


def test_summarize_rejects_empty_finite_rows() -> None:
    rows = (GraphPrefixDelta(trunk_nll=float('nan'), host_only_nll=float('nan')),)
    with pytest.raises(ValueError, match='no finite NLL deltas'):
        summarize_graph_prefix_deltas(rows)


def test_measure_token_deltas_shapes_and_aggregate_agree(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    batch = ssv_fixture_batch(module)
    sample = measure_graph_prefix_token_deltas(module.model, batch)
    n_masked = int(batch.labels.ne(-100).sum())
    assert sample.token_delta.shape == (n_masked,)
    assert sample.assignment_mass.shape == (n_masked,)
    assert torch.isfinite(sample.assignment_mass).all()
    assert sample.aggregate.delta == pytest.approx(float(sample.token_delta.mean()), abs=1e-4)


def test_measure_token_deltas_respects_assignment_top_k(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    batch = ssv_fixture_batch(module)
    top_1 = measure_graph_prefix_token_deltas(module.model, batch, assignment_top_k=1)
    top_4 = measure_graph_prefix_token_deltas(module.model, batch, assignment_top_k=4)
    assert bool((top_4.assignment_mass >= top_1.assignment_mass - 1e-6).all())


def test_stratify_splits_at_median_and_summarizes() -> None:
    rows = (
        GraphPrefixTokenSample(
            aggregate=GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.0),
            token_delta=torch.tensor([0.10, 0.02]),
            assignment_mass=torch.tensor([0.9, 0.5]),
        ),
        GraphPrefixTokenSample(
            aggregate=GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.0),
            token_delta=torch.tensor([0.08]),
            assignment_mass=torch.tensor([0.1]),
        ),
    )
    report = stratify_graph_prefix_token_deltas(rows)
    assert report.threshold == pytest.approx(0.5)
    assert report.high_mass.n_tokens == 2
    assert report.high_mass.mean_delta == pytest.approx(0.06)
    assert report.low_mass.n_tokens == 1
    assert report.low_mass.mean_delta == pytest.approx(0.08)
    table = report.as_table()
    assert 'high mass' in table
    assert 'low mass' in table


def test_stratify_drops_non_finite_before_splitting() -> None:
    rows = (
        GraphPrefixTokenSample(
            aggregate=GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.0),
            token_delta=torch.tensor([float('nan'), 0.02, 0.04]),
            assignment_mass=torch.tensor([0.9, 0.1, 0.5]),
        ),
    )
    report = stratify_graph_prefix_token_deltas(rows)
    assert report.high_mass.n_tokens + report.low_mass.n_tokens == 2


def test_stratify_rejects_all_non_finite_rows() -> None:
    rows = (
        GraphPrefixTokenSample(
            aggregate=GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.0),
            token_delta=torch.tensor([float('nan')]),
            assignment_mass=torch.tensor([0.5]),
        ),
    )
    with pytest.raises(ValueError, match='no finite masked-token deltas'):
        stratify_graph_prefix_token_deltas(rows)


def test_ablate_cli_requires_checkpoint() -> None:
    assert ssv_main.main(['ablate-graph-prefix']) != 0
    assert ssv_main.main(['ablate-graph-inject']) != 0


def test_ablate_cli_forwards_args_and_prints_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ckpt = tmp_path / 'last.ckpt'
    ckpt.write_bytes(b'not-a-real-ckpt')
    captured: dict[str, Any] = {}
    result = GraphPrefixAblationResult(
        ablation=GraphPrefixAblationReport(
            batches=(GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.2),),
            mean_delta=0.2,
            sd_delta=0.0,
            n_sign_flips=0,
            n_positive=1,
            n_negative=0,
        ),
    )

    def fake_run(**kwargs: Any) -> GraphPrefixAblationResult:
        captured.update(kwargs)
        return result

    with patch.object(ssv_main, 'run_graph_prefix_ablation', fake_run):
        code = ssv_main.main([
            'ablate-graph-prefix',
            '--checkpoint',
            str(ckpt),
            '--config',
            str(Path(__file__).resolve().parents[3] / 'configs' / 'ssv_train.smoke.yaml'),
            '--n-batches',
            '4',
            '--batch-size',
            '4',
            '--rho',
            '0.31',
        ])

    assert code == 0
    assert captured['checkpoint'] == ckpt
    assert captured['n_batches'] == 4
    assert captured['batch_size'] == 4
    assert captured['rho'] == pytest.approx(0.31)
    printed = capsys.readouterr().out
    assert 'trunk NLL (real prefix)' in printed
    assert '+0.200' in printed or '+0.2000' in printed
    assert captured.get('channel', 'prefix') in {None, 'prefix'}


def test_ablate_inject_cli_forwards_channel(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ckpt = tmp_path / 'last.ckpt'
    ckpt.write_bytes(b'not-a-real-ckpt')
    captured: dict[str, Any] = {}
    result = GraphPrefixAblationResult(
        ablation=GraphPrefixAblationReport(
            batches=(GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.1),),
            mean_delta=0.1,
            sd_delta=0.0,
            n_sign_flips=0,
            n_positive=1,
            n_negative=0,
            trunk_column='trunk NLL (inject on)',
            host_column='host-only NLL (zero inject)',
            caveat=INJECT_ABLATION_CAVEAT,
        ),
    )
    logged: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> GraphPrefixAblationResult:
        captured.update(kwargs)
        return result

    def capture_info(event: str, **kwargs: object) -> None:
        logged['event'] = event
        logged.update(kwargs)

    with (
        patch.object(ssv_main, 'run_graph_prefix_ablation', fake_run),
        patch.object(ssv_main._log, 'info', capture_info),
    ):
        code = ssv_main.main([
            'ablate-graph-inject',
            '--checkpoint',
            str(ckpt),
            '--config',
            str(Path(__file__).resolve().parents[3] / 'configs' / 'ssv_train.smoke.yaml'),
        ])

    assert code == 0
    assert captured['channel'] == 'inject'
    printed = capsys.readouterr().out
    assert 'host-only NLL (zero inject)' in printed
    assert INJECT_ABLATION_CAVEAT in printed
    assert logged['event'] == 'ssv.ablate.graph_inject'
    assert logged['caveat'] == INJECT_ABLATION_CAVEAT


def test_ablate_cli_stratify_by_mass_forwards_flag_and_prints_stratum_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ckpt = tmp_path / 'last.ckpt'
    ckpt.write_bytes(b'not-a-real-ckpt')
    captured: dict[str, Any] = {}
    result = GraphPrefixAblationResult(
        ablation=GraphPrefixAblationReport(
            batches=(GraphPrefixDelta(trunk_nll=1.0, host_only_nll=1.2),),
            mean_delta=0.2,
            sd_delta=0.0,
            n_sign_flips=0,
            n_positive=1,
            n_negative=0,
        ),
        mass_stratification=GraphPrefixMassStratificationReport(
            threshold=0.5,
            high_mass=MassStratumSummary(mean_delta=0.3, sd_delta=0.01, n_tokens=10),
            low_mass=MassStratumSummary(mean_delta=0.1, sd_delta=0.01, n_tokens=10),
        ),
    )

    def fake_run(**kwargs: Any) -> GraphPrefixAblationResult:
        captured.update(kwargs)
        return result

    with patch.object(ssv_main, 'run_graph_prefix_ablation', fake_run):
        code = ssv_main.main([
            'ablate-graph-prefix',
            '--checkpoint',
            str(ckpt),
            '--config',
            str(Path(__file__).resolve().parents[3] / 'configs' / 'ssv_train.smoke.yaml'),
            '--stratify-by-mass',
        ])

    assert code == 0
    assert captured['stratify_by_mass'] is True
    printed = capsys.readouterr().out
    assert 'high mass' in printed
    assert 'low mass' in printed


def test_lightning_module_does_not_import_ablation() -> None:
    """The training module must not import the ablation runner (container cycle)."""
    tree = ast.parse(inspect.getsource(module_mod))
    imported = tuple(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert 'ip_claim.ssv.ablation' not in imported
    assert SsvLightningModule.__name__ == 'SsvLightningModule'


def test_train_cli_duplicate_checkpoint_dir_last_wins(tmp_path: Path) -> None:
    first = tmp_path / 'ssv'
    second = tmp_path / 'ssv-graphfix-smoke'
    captured: dict[str, Any] = {}

    def fake_train_func(config: Any) -> Path:
        captured['checkpoint_dir'] = config.fit.checkpoint_dir
        return tmp_path / 'ssv.ckpt'

    smoke = Path(__file__).resolve().parents[3] / 'configs' / 'ssv_train.smoke.yaml'
    with patch.object(ssv_main, 'train_func', fake_train_func):
        code = ssv_main.main([
            'train',
            '--local',
            '--config',
            str(smoke),
            '--checkpoint-dir',
            str(first),
            '--checkpoint-dir',
            str(second),
        ])

    assert code == 0
    assert captured['checkpoint_dir'] == str(second)
