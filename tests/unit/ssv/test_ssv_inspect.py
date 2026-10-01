"""Offline overlay inspector: spans, defect signatures, evolution, CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from tests._ssv_fixtures import SSV_HUPD_FIXTURES, ssv_smoke_config, ssv_smoke_module

from ip_claim.ssv import __main__ as ssv_main
from ip_claim.ssv.covering_gate import INVENTORY_BOUND, USAGE_BOUND, CoveringTrainReading
from ip_claim.ssv.host_tokenizer import load_host_tokenizer
from ip_claim.ssv.inspect import (
    DefectFinding,
    DocumentInspection,
    EntitySpan,
    GraphInspectionReport,
    InspectThresholds,
    MaskedCodeView,
    RelationArc,
    SnapshotInspection,
    collapse_spans,
    compare_snapshots,
    induce_snapshot,
    run_graph_inspect,
    score_defects,
    write_inspection,
)
from ip_claim.ssv.soft_graph import SoftRelationBundle
from ip_claim.ssv.soft_vocab import SoftVocabModule


class _PieceTok:
    """Decode integer ids to fixed surface pieces."""

    all_special_ids = (0,)

    def __init__(self, table: dict[int, str]) -> None:
        self._table = table

    def decode(self, ids: list[int] | int, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(ids, int):
            return self._table.get(ids, '')
        return ' '.join(self._table[i] for i in ids if i in self._table)


def _span(
    application: str = '1',
    start: int = 0,
    end: int = 2,
    code: int = 1,
    text: str = 'pump',
) -> EntitySpan:
    return EntitySpan(
        application_number=application,
        start=start,
        end=end,
        code=code,
        mass=0.9,
        text=text,
    )


def _snapshot(
    documents: tuple[DocumentInspection, ...],
    *,
    checkpoint: str = 'a.ckpt',
    defects: tuple[DefectFinding, ...] = (),
) -> SnapshotInspection:
    return SnapshotInspection(
        checkpoint=checkpoint,
        global_step=1,
        documents=documents,
        defects=defects,
        entity_occupancy=1.0,
        relation_occupancy=1.0,
        mean_row_entropy=1.0,
        n_dead_entity=0,
        n_dead_relation=0,
    )


def test_collapse_spans_merges_same_code_and_skips_specials() -> None:
    tok = _PieceTok({1: 'fluid', 2: 'pump', 3: 'housing'})
    spans = collapse_spans(
        application_number='13817165',
        codes=(4, 7, 7, 9),
        masses=(0.4, 0.8, 0.9, 0.5),
        token_ids=(0, 1, 2, 3),
        special_ids=frozenset({0}),
        tokenizer=tok,  # type: ignore[arg-type]
    )
    assert [(s.start, s.end, s.code, s.text) for s in spans] == [
        (1, 3, 7, 'fluid pump'),
        (3, 4, 9, 'housing'),
    ]


def test_score_defects_fires_collapse_high_df_and_dead(tmp_path: Path) -> None:
    vocab = SoftVocabModule(ssv_smoke_config(tmp_path))
    vocab.entity_usage_ema.fill_(0.0)
    vocab.relation_usage_ema.fill_(0.0)
    assignment = torch.zeros(2, 4, vocab.entity_bank_size)
    assignment[..., 0] = 1.0
    rel_usage = torch.zeros(vocab.relation_bank_size)
    rel_usage[0] = 0.95
    bundle = SoftRelationBundle(
        overlays=(),
        soft_ke_loss=torch.tensor(0.0),
        pair_count=0,
        relation_batch_usage=rel_usage,
    )
    findings = {
        row.name: row
        for row in score_defects(
            assignment=assignment,
            token_mask=torch.ones(2, 4),
            token_ids=torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5]]),
            vocab=vocab,
            bundle=bundle,
            thresholds=InspectThresholds(bank_collapse_top_k=1, min_code_tokens=3),
            utilization_eps=1e-3,
        )
    }
    assert findings['bank_collapse'].fired
    assert findings['relation_collapse'].fired
    assert findings['degenerate_assignment'].fired
    assert findings['dead_codes'].fired


def test_score_defects_stays_quiet_on_spread_technical_codes(tmp_path: Path) -> None:
    vocab = SoftVocabModule(ssv_smoke_config(tmp_path))
    vocab.entity_usage_ema.fill_(0.1)
    vocab.relation_usage_ema.fill_(0.1)
    assignment = torch.zeros(1, 4, vocab.entity_bank_size)
    for index in range(4):
        assignment[0, index, index] = 1.0
    usage = torch.full((vocab.relation_bank_size,), 1.0 / vocab.relation_bank_size)
    bundle = SoftRelationBundle(
        overlays=(),
        soft_ke_loss=torch.tensor(0.0),
        pair_count=0,
        relation_batch_usage=usage,
    )
    findings = score_defects(
        assignment=assignment,
        token_mask=torch.ones(1, 4),
        token_ids=torch.tensor([[1, 2, 3, 4]]),
        vocab=vocab,
        bundle=bundle,
        thresholds=InspectThresholds(bank_collapse_top_k=1),
        utilization_eps=1e-3,
    )
    assert all(not row.fired for row in findings)


def test_compare_snapshots_marks_code_changes() -> None:
    left = _snapshot((
        DocumentInspection(
            application_number='1',
            text='pump housing',
            spans=(_span(code=3, text='pump'), _span(start=2, end=4, code=5, text='housing')),
        ),
    ))
    right = _snapshot((
        DocumentInspection(
            application_number='1',
            text='pump housing',
            spans=(_span(code=8, text='pump'), _span(start=2, end=4, code=5, text='housing')),
        ),
    ))
    rows, stable = compare_snapshots(left, right)
    assert len(rows) == 2
    changed = {row.text: row.changed for row in rows}
    assert changed['pump'] is True
    assert changed['housing'] is False
    assert stable == pytest.approx(0.5)


def test_write_inspection_includes_defects_spans_and_reserved_masked_codes(
    tmp_path: Path,
) -> None:
    document = DocumentInspection(
        application_number='13817165',
        text='A pump.',
        spans=(_span(application='13817165', text='pump'),),
        relations=(
            RelationArc(
                application_number='13817165',
                src_start=0,
                src_end=2,
                dst_start=0,
                dst_end=2,
                src_code=1,
                dst_code=1,
                rel_code=2,
                mass=0.6,
                residual=1.25,
                src_text='pump',
                dst_text='pump',
            ),
        ),
        masked_codes=(MaskedCodeView(token_index=3, teacher_code=1, predicted_code=4),),
    )
    covering = CoveringTrainReading(
        inventory_entropy=3.04,
        ln_k=5.545,
        usage_entropy=4.59,
        usage_target=5.20,
        row_entropy=2.85,
        host_nll=1.022,
        host_nll_ceiling=0.912,
        dea_gap=-0.0027,
    )
    earlier = _snapshot(
        (document,),
        defects=(DefectFinding(name='bank_collapse', fired=True, metric=0.9, detail='collapsed'),),
    )
    report = GraphInspectionReport(earlier=earlier.model_copy(update={'covering': covering}))
    html_path = write_inspection(report, tmp_path / 'out')
    page = html_path.read_text(encoding='utf-8')
    assert '13817165' in page
    assert 'bank_collapse' in page
    assert 'Covering-ready train gate' in page
    assert INVENTORY_BOUND in page
    assert USAGE_BOUND in page
    assert 'Assigned reading' in page
    assert 'Kept-edge leftover residual' in page
    assert '1.25' in page
    assert 'Masked-position codes' in page
    assert 'pump' in page
    assert not (tmp_path / 'out' / 'inspect.md').exists()


def test_induce_snapshot_on_smoke_trunk_has_spans_and_defects(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    snapshot = induce_snapshot(
        trunk=module.model,
        tokenizer=load_host_tokenizer(module.config),
        texts=('A fluid pump comprising a housing.',),
        application_numbers=('13817165',),
        max_length=48,
        thresholds=InspectThresholds(),
        utilization_eps=float(module.config.bank.utilization_eps),
        usage_target=float(module.config.resolved_usage_entropy_target()),
        checkpoint=tmp_path / 'none.ckpt',
        global_step=0,
    )
    assert snapshot.documents[0].application_number == '13817165'
    assert snapshot.documents[0].spans
    assert snapshot.covering is not None
    row = snapshot.health_row('earlier')
    assert row['inventory_bound'] == INVENTORY_BOUND
    assert row['usage_bound'] == USAGE_BOUND
    assert 'fall well below ln K' in str(row['inventory_bound'])
    assert 'hold near target' in str(row['usage_bound'])
    assert {row.name for row in snapshot.defects} == {
        'bank_collapse',
        'relation_collapse',
        'degenerate_assignment',
        'dead_codes',
    }


def _lightning_payload(module: object, step: int) -> dict[str, object]:
    model = module.model  # type: ignore[attr-defined]
    return {
        'state_dict': {
            f'model.{key}': (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in model.state_dict().items()
        },
        'global_step': step,
    }


def test_run_graph_inspect_writes_report_for_two_checkpoints(tmp_path: Path) -> None:
    module = ssv_smoke_module(tmp_path)
    ckpt_a = tmp_path / 'a.ckpt'
    ckpt_b = tmp_path / 'b.ckpt'
    torch.save(_lightning_payload(module, 10), ckpt_a)
    flipped = _lightning_payload(module, 80)
    state = flipped['state_dict']
    assert isinstance(state, dict)
    state['model.soft_vocab.entity_bank'] = -state['model.soft_vocab.entity_bank']
    torch.save(flipped, ckpt_b)
    out = tmp_path / 'peek'
    report = run_graph_inspect(
        checkpoint=ckpt_a,
        checkpoint_b=ckpt_b,
        output_dir=out,
        job=ssv_smoke_config(tmp_path),
        hupd_dir=SSV_HUPD_FIXTURES,
        hupd_limit=2,
        device=torch.device('cpu'),
    )
    assert (out / 'inspect.html').is_file()
    assert not (out / 'inspect.md').exists()
    assert report.earlier.global_step == 10
    assert report.later is not None
    assert report.later.global_step == 80
    assert report.stable_fraction is not None
    assert report.earlier.documents


def test_inspect_cli_requires_checkpoint() -> None:
    assert ssv_main.main(['inspect-graph']) != 0


def test_inspect_cli_forwards_args(tmp_path: Path) -> None:
    ckpt = tmp_path / 'last.ckpt'
    ckpt.write_bytes(b'not-a-real-ckpt')
    captured: dict[str, Any] = {}
    report = GraphInspectionReport(earlier=_snapshot(()))

    def fake_run(**kwargs: Any) -> GraphInspectionReport:
        captured.update(kwargs)
        return report

    with patch.object(ssv_main, 'run_graph_inspect', fake_run):
        code = ssv_main.main([
            'inspect-graph',
            '--checkpoint',
            str(ckpt),
            '--output',
            str(tmp_path / 'out'),
            '--hupd-limit',
            '3',
        ])

    assert code == 0
    assert captured['checkpoint'] == ckpt
    assert captured['hupd_limit'] == 3
    assert captured['output_dir'] == tmp_path / 'out'
