"""Offline inspector for the soft-entity overlay a trunk checkpoint induces.

Loads one or two Lightning checkpoints through the same container and weight
init path as graph-prefix ablation, assigns entities on clean host embeds
without remask, then writes one Great Tables HTML report. Leftover unpaid of
occupy against Phi is attributed to kept typed edges; consumed refuse is
not drawn. The covering-ready gate table states bound direction on inventory
(upper) and usage (lower). Bank occupancy and dead-row counts come from the
vocab usage EMA already maintained in training. Row entropy on that occupancy
table is hygiene. Reserved ``masked_codes`` stays empty until a denoise head
exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import accumulate
from pathlib import Path
from typing import Any, NoReturn, Self

import polars as pl
import torch
from great_tables import GT, loc, style
from pydantic import BaseModel, ConfigDict, Field, model_validator
from returns.maybe import Maybe, maybe
from returns.result import Failure
from torch import Tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.collision.explain import Explain
from ip_claim.ingestion.adapters.hupd_json.paths import load_hupd_patents
from ip_claim.ssv.config import DEFAULT_SSV_TRAIN_PATH, SsvTrainConfig
from ip_claim.ssv.covering_gate import CoveringTrainReading
from ip_claim.ssv.graph_batch import graph_batch_from_patent
from ip_claim.ssv.host_tokenizer import batch_encoding_tensor
from ip_claim.ssv.model import SoftTrunkModel
from ip_claim.ssv.soft_graph import SoftRelationBundle, build_soft_relation_bundle
from ip_claim.ssv.soft_vocab import SoftVocabModule

_DEFAULT_LIMIT = 4


class InspectThresholds(BaseModel):
    """Cutoffs for automatic overlay defect signatures."""

    model_config = ConfigDict(frozen=True)

    bank_collapse_top_k: int = Field(default=8, ge=1)
    bank_collapse_share: float = Field(default=0.80, ge=0.0, le=1.0)
    relation_collapse_share: float = Field(default=0.50, ge=0.0, le=1.0)
    boilerplate_share: float = Field(default=0.70, ge=0.0, le=1.0)
    min_code_tokens: int = Field(default=3, ge=1)


class MaskedCodeView(BaseModel):
    """Reserved per-position denoise view; empty until a denoise head exists."""

    model_config = ConfigDict(frozen=True)

    token_index: int = Field(ge=0)
    teacher_code: int | None = None
    predicted_code: int | None = None
    teacher_mass: float | None = None


class EntitySpan(BaseModel):
    """Consecutive live tokens that share one argmax entity code."""

    model_config = ConfigDict(frozen=True)

    application_number: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    code: int = Field(ge=0)
    mass: float
    text: str


class RelationArc(BaseModel):
    """One occupied-code pair labeled by its argmax relation code."""

    model_config = ConfigDict(frozen=True)

    application_number: str
    src_start: int = Field(ge=0)
    src_end: int = Field(ge=0)
    dst_start: int = Field(ge=0)
    dst_end: int = Field(ge=0)
    src_code: int = Field(ge=0)
    dst_code: int = Field(ge=0)
    rel_code: int = Field(ge=0)
    mass: float
    residual: float = 0.0
    src_text: str
    dst_text: str


class DefectFinding(BaseModel):
    """One named overlay defect and whether it fired on this peek."""

    model_config = ConfigDict(frozen=True)

    name: str
    fired: bool
    metric: float
    detail: str


class PeekMetrics(BaseModel):
    """Sample and EMA numbers that the four named defect checks read."""

    model_config = ConfigDict(frozen=True)

    top_k: int
    top_share: float
    bank_cut: float
    rel_top: float
    has_relations: bool
    rel_cut: float
    boilerplate_codes: tuple[int, ...]
    dead_entity: int
    dead_relation: int
    utilization_eps: float

    @classmethod
    def measure(
        cls,
        *,
        assignment: Tensor,
        token_mask: Tensor,
        token_ids: Tensor,
        vocab: SoftVocabModule,
        bundle: SoftRelationBundle,
        thresholds: InspectThresholds,
        utilization_eps: float,
    ) -> PeekMetrics:
        """Collect peek mass, pair usage, high-DF codes, and EMA dead rows."""
        live = token_mask.bool()
        n_live = int(live.sum())
        usage = (
            assignment[live].mean(dim=0) if n_live else assignment.new_zeros(assignment.size(-1))
        )
        top_k = min(int(thresholds.bank_collapse_top_k), int(usage.numel()))
        rel_usage = bundle.relation_batch_usage
        if rel_usage is not None and bool(rel_usage.numel()):
            rel_top = float(rel_usage.max())
            has_relations = True
        else:
            rel_top = 0.0
            has_relations = False
        return cls(
            top_k=top_k,
            top_share=float(usage.topk(top_k).values.sum()) if top_k else 0.0,
            bank_cut=float(thresholds.bank_collapse_share),
            rel_top=rel_top,
            has_relations=has_relations,
            rel_cut=float(thresholds.relation_collapse_share),
            boilerplate_codes=cls.high_df_codes(
                assignment=assignment,
                token_mask=token_mask,
                token_ids=token_ids,
                share_cut=float(thresholds.boilerplate_share),
                min_tokens=int(thresholds.min_code_tokens),
            ),
            dead_entity=int((vocab.entity_usage_ema < utilization_eps).sum()),
            dead_relation=int((vocab.relation_usage_ema < utilization_eps).sum()),
            utilization_eps=utilization_eps,
        )

    @staticmethod
    def high_df_codes(
        *,
        assignment: Tensor,
        token_mask: Tensor,
        token_ids: Tensor,
        share_cut: float,
        min_tokens: int,
    ) -> tuple[int, ...]:
        """Entity codes whose tokens have high document frequency on this peek."""
        live = token_mask.bool()
        n_docs = int(assignment.size(0))
        vacant = n_docs < 2 or int(live.sum()) == 0
        frame = pl.DataFrame({
            'doc': torch.nonzero(live, as_tuple=True)[0].tolist(),
            'code': assignment.argmax(dim=-1)[live].tolist(),
            'token': token_ids[live].tolist(),
        })
        type_df = frame.group_by('token').agg(
            (pl.col('doc').n_unique() / max(n_docs, 1)).alias('df'),
        )
        flagged = (
            frame
            .join(type_df, on='token')
            .group_by('code')
            .agg(
                pl.len().alias('n'),
                (pl.col('df') >= share_cut).mean().alias('high_df_share'),
            )
            .filter((pl.col('n') >= min_tokens) & (pl.col('high_df_share') >= share_cut))
            .sort('code')
        )
        return () if vacant else tuple(int(code) for code in flagged['code'].to_list())

    def findings(self) -> tuple[DefectFinding, ...]:
        """Fold the four named checks over this peek's metrics."""
        rel_detail = (
            'no occupied-code relations in this peek',
            (
                f'dominant relation code holds {self.rel_top:.3f} of pair mass '
                f'(cut {self.rel_cut:.2f})'
            ),
        )[int(self.has_relations)]
        deg_detail = (
            'no entity code tracks high document-frequency tokens on this peek',
            f'high document-frequency codes: {self.boilerplate_codes}',
        )[int(bool(self.boilerplate_codes))]
        checks = (
            (
                'bank_collapse',
                self.top_share >= self.bank_cut,
                self.top_share,
                (
                    f'top {self.top_k} entity codes hold {self.top_share:.3f} of sample '
                    f'assignment mass (cut {self.bank_cut:.2f})'
                ),
            ),
            (
                'relation_collapse',
                self.has_relations and self.rel_top >= self.rel_cut,
                self.rel_top,
                rel_detail,
            ),
            (
                'degenerate_assignment',
                bool(self.boilerplate_codes),
                float(len(self.boilerplate_codes)),
                deg_detail,
            ),
            (
                'dead_codes',
                (self.dead_entity + self.dead_relation) > 0,
                float(self.dead_entity + self.dead_relation),
                (
                    f'{self.dead_entity} entity and {self.dead_relation} relation rows have '
                    f'training-run usage EMA below {self.utilization_eps:g}'
                ),
            ),
        )
        return tuple(
            DefectFinding(name=name, fired=fired, metric=metric, detail=detail)
            for name, fired, metric, detail in checks
        )


class DocumentInspection(BaseModel):
    """One patent's induced spans, pair edges, and optional denoise slots."""

    model_config = ConfigDict(frozen=True)

    application_number: str
    text: str
    spans: tuple[EntitySpan, ...]
    relations: tuple[RelationArc, ...] = ()
    masked_codes: tuple[MaskedCodeView, ...] = ()


class SnapshotInspection(BaseModel):
    """Overlay induced by one checkpoint on a fixed patent sample."""

    model_config = ConfigDict(frozen=True)

    checkpoint: str
    global_step: int | None
    documents: tuple[DocumentInspection, ...]
    defects: tuple[DefectFinding, ...]
    entity_occupancy: float
    relation_occupancy: float
    mean_row_entropy: float
    n_dead_entity: int = Field(ge=0)
    n_dead_relation: int = Field(ge=0)
    covering: CoveringTrainReading | None = None

    def health_row(self, label: str) -> dict[str, object]:
        """Occupancy plus covering-arm cells with bound direction stated."""
        cells: dict[str, object] = {
            'label': label,
            'checkpoint': self.checkpoint,
            'step': Maybe.from_optional(self.global_step).value_or(''),
            'entity_occupancy': round(self.entity_occupancy, 4),
            'relation_occupancy': round(self.relation_occupancy, 4),
            'mean_row_entropy_hygiene': round(self.mean_row_entropy, 4),
            'dead_entity': self.n_dead_entity,
            'dead_relation': self.n_dead_relation,
        }
        return (
            Maybe
            .from_optional(self.covering)
            .map(lambda covering: {**cells, **covering.health_cells()})
            .value_or(cells)
        )

    def defect_rows(self, label: str) -> tuple[dict[str, object], ...]:
        """Defect-signature rows tagged with this snapshot's label."""
        return tuple(
            {
                'label': label,
                'name': row.name,
                'fired': row.fired,
                'metric': round(row.metric, 4),
                'detail': row.detail,
            }
            for row in self.defects
        )

    def reading_rows(self) -> tuple[dict[str, object], ...]:
        """Assigned-reading rows: span text plus entity-code labels."""
        return tuple(
            {
                'application': document.application_number,
                'reading': ' '.join(f'{span.text} [e{span.code}]' for span in document.spans),
            }
            for document in self.documents
        )

    def span_rows(self) -> tuple[dict[str, object], ...]:
        """Per-span token interval, code, mass, and surface text."""
        return tuple(
            {
                'application': span.application_number,
                'start': span.start,
                'end': span.end,
                'code': span.code,
                'mass': round(span.mass, 4),
                'text': span.text,
            }
            for document in self.documents
            for span in document.spans
        )

    def relation_rows(self) -> tuple[dict[str, object], ...]:
        """Occupied-code pair arcs labeled by argmax relation code."""
        return tuple(
            {
                'application': arc.application_number,
                'src': arc.src_text,
                'dst': arc.dst_text,
                'src_code': arc.src_code,
                'dst_code': arc.dst_code,
                'rel_code': arc.rel_code,
                'mass': round(arc.mass, 4),
            }
            for document in self.documents
            for arc in document.relations
        )

    def residual_rows(self) -> tuple[dict[str, object], ...]:
        """Leftover unpaid of occupy against Phi, on kept typed edges."""
        return tuple(
            {
                'application': arc.application_number,
                'src': arc.src_text,
                'dst': arc.dst_text,
                'src_code': arc.src_code,
                'dst_code': arc.dst_code,
                'rel_code': arc.rel_code,
                'leftover': round(arc.residual, 4),
            }
            for document in self.documents
            for arc in document.relations
        )

    def masked_rows(self) -> tuple[dict[str, object], ...]:
        """Reserved denoise slots; empty until a denoise head exists."""
        return tuple(
            {
                'application': document.application_number,
                'token_index': view.token_index,
                'teacher_code': view.teacher_code,
                'predicted_code': view.predicted_code,
                'teacher_mass': view.teacher_mass,
            }
            for document in self.documents
            for view in document.masked_codes
        )


class EvolutionRow(BaseModel):
    """Same surface span under two checkpoints."""

    model_config = ConfigDict(frozen=True)

    application_number: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    code_a: int = Field(ge=0)
    code_b: int = Field(ge=0)
    changed: bool


class ReportTable(BaseModel):
    """One Great Tables fragment: title, rows, optional fired-row paint."""

    model_config = ConfigDict(frozen=True)

    title: str
    rows: tuple[Mapping[str, object], ...]
    mark_fired: bool = False

    def as_html(self) -> str:
        """Render this spec through Great Tables."""
        cells = self.rows or ({'note': '(none)'},)
        frame = pl.DataFrame(list(cells))
        table = GT(frame).tab_header(title=self.title)
        painted = (
            table.tab_style(
                style.fill(color='#f8d7da'),
                loc.body(rows=pl.col('fired')),
            )
            if self.mark_fired and 'fired' in frame.columns
            else table
        )
        return painted.as_raw_html()


class GraphInspectionReport(BaseModel):
    """One or two checkpoint peeks plus an optional span-code evolution table."""

    model_config = ConfigDict(frozen=True)

    earlier: SnapshotInspection
    later: SnapshotInspection | None = None
    evolution: tuple[EvolutionRow, ...] = ()
    stable_fraction: float | None = None

    def labeled_snapshots(self) -> tuple[tuple[str, SnapshotInspection], ...]:
        """Earlier snapshot, plus later when a second checkpoint was loaded."""
        extra = Maybe.from_optional(self.later).map(lambda snap: (('later', snap),)).value_or(())
        return (('earlier', self.earlier), *extra)

    def evolution_specs(self) -> tuple[ReportTable, ...]:
        """Evolution table when a later snapshot exists; otherwise empty."""
        stable = (
            Maybe
            .from_optional(self.stable_fraction)
            .map(lambda fraction: f'{fraction:.3f}')
            .value_or('no aligned spans')
        )
        rows = tuple(
            {
                'application': row.application_number,
                'start': row.start,
                'end': row.end,
                'text': row.text,
                'code_a': row.code_a,
                'code_b': row.code_b,
                'changed': row.changed,
            }
            for row in self.evolution
        ) or ({'note': f'stable fraction {stable}'},)
        return (
            Maybe
            .from_optional(self.later)
            .map(lambda _: (ReportTable(title=f'Evolution (stable fraction {stable})', rows=rows),))
            .value_or(())
        )

    def covering_gate_table(self) -> ReportTable:
        """Covering-ready arms with bound direction on every row."""
        rows = tuple(
            row
            for label, snap in self.labeled_snapshots()
            for row in (
                Maybe
                .from_optional(snap.covering)
                .map(lambda covering, tag=label: covering.report_rows(tag))
                .value_or(())
            )
        )
        return ReportTable(
            title='Covering-ready train gate (inventory upper, usage lower)',
            rows=rows or ({'note': 'no covering-ready reading on this peek'},),
        )

    def bank_health_table(self) -> ReportTable:
        """Occupancy and dead-row counts; row entropy here is hygiene."""
        return ReportTable(
            title='Bank occupancy (not the covering gate)',
            rows=tuple(snap.health_row(label) for label, snap in self.labeled_snapshots()),
        )

    def defect_table(self) -> ReportTable:
        """Named defect signatures with fired rows marked."""
        return ReportTable(
            title='Defect signatures',
            rows=tuple(
                row for label, snap in self.labeled_snapshots() for row in snap.defect_rows(label)
            ),
            mark_fired=True,
        )

    def reading_table(self) -> ReportTable:
        """Assigned reading: span text plus entity-code labels."""
        return ReportTable(
            title='Assigned reading',
            rows=tuple(row for _, snap in self.labeled_snapshots() for row in snap.reading_rows()),
        )

    def span_table(self) -> ReportTable:
        """Token-interval spans and their argmax entity codes."""
        return ReportTable(
            title='Entity spans',
            rows=tuple(row for _, snap in self.labeled_snapshots() for row in snap.span_rows()),
        )

    def relation_table(self) -> ReportTable:
        """Occupied-code pair arcs labeled by argmax relation code."""
        return ReportTable(
            title='Relation arcs',
            rows=tuple(row for _, snap in self.labeled_snapshots() for row in snap.relation_rows()),
        )

    def residual_table(self) -> ReportTable:
        """Leftover unpaid of occupy against Phi on kept typed edges."""
        rows = tuple(row for _, snap in self.labeled_snapshots() for row in snap.residual_rows())
        return ReportTable(
            title='Kept-edge leftover residual',
            rows=rows or ({'note': 'no kept typed edges on this peek'},),
        )

    def masked_table(self) -> ReportTable:
        """Reserved denoise slots; empty until a denoise head exists."""
        return ReportTable(
            title='Masked-position codes',
            rows=tuple(row for _, snap in self.labeled_snapshots() for row in snap.masked_rows()),
        )

    def table_specs(self) -> tuple[ReportTable, ...]:
        """Declared report sections plus optional evolution."""
        return (
            self.covering_gate_table(),
            self.bank_health_table(),
            self.defect_table(),
            self.reading_table(),
            self.span_table(),
            self.relation_table(),
            self.residual_table(),
            self.masked_table(),
            *self.evolution_specs(),
        )


class InspectRequest(BaseModel):
    """Validated inspect-graph envelope: paths, peek size, and optional job."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    checkpoint: Path
    output_dir: Path
    checkpoint_b: Path | None = None
    config_path: Path | None = None
    job: SsvTrainConfig | None = None
    hupd_dir: Path | None = None
    hupd_limit: int = Field(default=_DEFAULT_LIMIT, ge=1)
    thresholds: InspectThresholds = Field(default_factory=InspectThresholds)
    device: torch.device | None = None

    @model_validator(mode='after')
    def existing_checkpoints(self) -> Self:
        """Fail at the I/O boundary when a named checkpoint file is absent."""

        def require_present(path: Path) -> NoReturn:
            msg = f'checkpoint missing: {path}'
            raise FileNotFoundError(msg)

        return (
            Maybe
            .from_optional(
                next(
                    (
                        path
                        for path in (self.checkpoint, self.checkpoint_b)
                        if path is not None and not path.is_file()
                    ),
                    None,
                )
            )
            .map(require_present)
            .value_or(self)
        )

    def resolved_job(self) -> SsvTrainConfig:
        """YAML or injected job, with peek directory and limit overlaid."""

        def load_yaml() -> SsvTrainConfig:
            return SsvTrainConfig.from_yaml(
                Maybe.from_optional(self.config_path).value_or(DEFAULT_SSV_TRAIN_PATH)
            )

        base = Maybe.from_optional(self.job).or_else_call(load_yaml)
        updates: dict[str, Any] = {'hupd_limit': self.hupd_limit}
        updates |= (
            Maybe
            .from_optional(self.hupd_dir)
            .map(lambda path: {'hupd_dir': str(path)})
            .value_or({})
        )
        return base.overlay(updates)

    def resolved_device(self) -> torch.device:
        """Caller device, else CUDA when present."""
        return self.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def peek(self, job: SsvTrainConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Load the peek patents already used by graph-batch construction."""
        listed = job.runtime.hupd_dir
        root = (
            Path(listed)
            if listed
            else Path(__file__).resolve().parents[3] / 'tests' / 'fixtures' / 'hupd'
        )
        batches = tuple(
            graph_batch_from_patent(patent)
            for patent in load_hupd_patents(root, limit=self.hupd_limit)
        )
        return tuple(batch.text for batch in batches), tuple(
            batch.application_number for batch in batches
        )


class InspectSession:
    """Loaded trunk, tokenizer, and peek texts for one or two checkpoints."""

    def __init__(self, request: InspectRequest) -> None:
        self.request = request
        self.job = request.resolved_job()
        self.texts, self.numbers = request.peek(self.job)
        self.container = SsvContainer(config=self.job)
        self.module = self.container.lightning_module()
        self.tokenizer = self.container.host_tokenizer()
        _ = self.module.to(request.resolved_device())

    @maybe
    def checkpoint_step(self, path: Path) -> int | None:
        """Read Lightning ``global_step`` when the payload is a mapping."""
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(payload, dict) and 'global_step' in payload:
            return int(payload['global_step'])
        return None

    def induce(self, path: Path) -> SnapshotInspection:
        """Init weights from ``path`` and induce the overlay on the fixed peek."""
        match self.container.init_weights(path=path):
            case Failure(message):
                raise RuntimeError(message)
            case _:
                return induce_snapshot(
                    trunk=self.module.model,
                    tokenizer=self.tokenizer,
                    texts=self.texts,
                    application_numbers=self.numbers,
                    max_length=int(self.job.arch.max_length),
                    thresholds=self.request.thresholds,
                    utilization_eps=float(self.job.bank.utilization_eps),
                    usage_target=float(self.job.resolved_usage_entropy_target()),
                    checkpoint=path,
                    global_step=self.checkpoint_step(path).value_or(None),
                )

    def report(self) -> GraphInspectionReport:
        """Induce earlier, optionally later, and join span codes."""
        earlier = self.induce(self.request.checkpoint)
        later = Maybe.from_optional(self.request.checkpoint_b).map(self.induce).value_or(None)
        evolution, stable = compare_snapshots(earlier, later) if later is not None else ((), None)
        return GraphInspectionReport(
            earlier=earlier,
            later=later,
            evolution=evolution,
            stable_fraction=stable,
        )


def collapse_spans(
    *,
    application_number: str,
    codes: Sequence[int],
    masses: Sequence[float],
    token_ids: Sequence[int],
    special_ids: frozenset[int],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[EntitySpan, ...]:
    """Merge consecutive non-special tokens that share an argmax entity code."""
    runs: list[list[int]] = []
    for index, token_id in enumerate(token_ids):
        if token_id in special_ids:
            continue
        if runs and runs[-1][1] == index and codes[runs[-1][0]] == codes[index]:
            runs[-1][1] = index + 1
            continue
        runs.append([index, index + 1])

    def piece(start: int, end: int) -> str:
        decoded = tokenizer.decode(list(token_ids[start:end]), skip_special_tokens=True)
        return decoded.strip() if isinstance(decoded, str) else ' '.join(decoded).strip()

    return tuple(
        EntitySpan(
            application_number=application_number,
            start=start,
            end=end,
            code=int(codes[start]),
            mass=float(sum(masses[start:end]) / max(end - start, 1)),
            text=text,
        )
        for start, end in runs
        if (text := piece(start, end))
    )


def score_defects(
    *,
    assignment: Tensor,
    token_mask: Tensor,
    token_ids: Tensor,
    vocab: SoftVocabModule,
    bundle: SoftRelationBundle,
    thresholds: InspectThresholds,
    utilization_eps: float,
) -> tuple[DefectFinding, ...]:
    """Score bank collapse, relation collapse, high-DF codes, and dead EMA rows."""
    return PeekMetrics.measure(
        assignment=assignment,
        token_mask=token_mask,
        token_ids=token_ids,
        vocab=vocab,
        bundle=bundle,
        thresholds=thresholds,
        utilization_eps=utilization_eps,
    ).findings()


def compare_snapshots(
    earlier: SnapshotInspection,
    later: SnapshotInspection,
) -> tuple[tuple[EvolutionRow, ...], float]:
    """Join spans on application number and token interval; report code changes."""

    def span_frame(snapshot: SnapshotInspection, side: str) -> pl.DataFrame:
        rows = [
            {
                'application_number': span.application_number,
                'start': span.start,
                'end': span.end,
                f'text_{side}': span.text,
                f'code_{side}': span.code,
            }
            for document in snapshot.documents
            for span in document.spans
        ]
        return (
            pl.DataFrame(rows)
            if rows
            else pl.DataFrame({
                'application_number': [],
                'start': [],
                'end': [],
                f'text_{side}': [],
                f'code_{side}': [],
            })
        )

    joined = span_frame(earlier, 'a').join(
        span_frame(later, 'b'),
        on=['application_number', 'start', 'end'],
        how='inner',
    )
    empty = joined.is_empty()
    evolved = joined.with_columns((pl.col('code_a') != pl.col('code_b')).alias('changed'))
    rows = tuple(
        EvolutionRow(
            application_number=str(row['application_number']),
            start=int(row['start']),
            end=int(row['end']),
            text=str(row['text_a']),
            code_a=int(row['code_a']),
            code_b=int(row['code_b']),
            changed=bool(row['changed']),
        )
        for row in evolved.iter_rows(named=True)
    )
    return ((), 0.0) if empty else (rows, 1.0 - int(evolved['changed'].sum()) / len(rows))


def induce_snapshot(
    *,
    trunk: SoftTrunkModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    application_numbers: Sequence[str],
    max_length: int,
    thresholds: InspectThresholds,
    utilization_eps: float,
    usage_target: float,
    checkpoint: Path,
    global_step: int | None,
) -> SnapshotInspection:
    """Assign entities and relations on clean token embeds; do not remask."""

    def documents_from_batch(
        input_ids: Tensor,
        attention_mask: Tensor,
        assignment: Tensor,
        bundle: SoftRelationBundle,
        labels: tuple[tuple[str, ...], ...],
        leftover: Tensor,
    ) -> tuple[DocumentInspection, ...]:
        special_ids = frozenset(int(token) for token in tokenizer.all_special_ids)
        lengths = [int(n) for n in attention_mask.gt(0).sum(dim=-1).tolist()]
        counts = tuple(int(overlay.edge_index.size(1)) for overlay in bundle.overlays)
        starts = (0, *tuple(accumulate(counts))[:-1]) if counts else ()

        def span_for_code(spans: Sequence[EntitySpan], code: int) -> EntitySpan | None:
            matches = tuple(span for span in spans if span.code == code)
            return max(matches, key=lambda span: span.mass) if matches else None

        def relations_for(row: int, spans: Sequence[EntitySpan]) -> tuple[RelationArc, ...]:
            assignment_rel = bundle.relation_assignment
            if assignment_rel is None or row >= len(bundle.overlays) or row >= len(starts):
                return ()
            overlay = bundle.overlays[row]
            count = int(overlay.edge_index.size(1))
            if count == 0:
                return ()
            rel = assignment_rel[starts[row] : starts[row] + count]
            code_ids = overlay.code_ids.tolist()
            application = application_numbers[row]
            return tuple(
                RelationArc(
                    application_number=application,
                    src_start=head.start,
                    src_end=head.end,
                    dst_start=tail.start,
                    dst_end=tail.end,
                    src_code=head.code,
                    dst_code=tail.code,
                    rel_code=int(rel_code),
                    mass=float(rel_mass),
                    residual=float(leftover[row, int(code_ids[src]), int(code_ids[dst])]),
                    src_text=head.text,
                    dst_text=tail.text,
                )
                for src, dst, rel_code, rel_mass in zip(
                    overlay.edge_index[0].tolist(),
                    overlay.edge_index[1].tolist(),
                    rel.argmax(dim=-1).tolist(),
                    rel.max(dim=-1).values.tolist(),
                    strict=True,
                )
                if src < len(code_ids)
                and dst < len(code_ids)
                and (head := span_for_code(spans, int(code_ids[src]))) is not None
                and (tail := span_for_code(spans, int(code_ids[dst]))) is not None
            )

        def document_for(row: int) -> DocumentInspection:
            keyed = tuple(
                EntitySpan(
                    application_number=application_numbers[row],
                    start=index,
                    end=index + 1,
                    code=int(assignment[row, index].argmax()),
                    mass=float(assignment[row, index].max()),
                    text=label,
                )
                for index, label in enumerate(labels[row] if row < len(labels) else ())
                if label
            )
            spans = keyed or collapse_spans(
                application_number=application_numbers[row],
                codes=assignment[row, : lengths[row]].argmax(dim=-1).tolist(),
                masses=assignment[row, : lengths[row]].max(dim=-1).values.tolist(),
                token_ids=input_ids[row, : lengths[row]].tolist(),
                special_ids=special_ids,
                tokenizer=tokenizer,
            )
            return DocumentInspection(
                application_number=application_numbers[row],
                text=texts[row],
                spans=spans,
                relations=relations_for(row, spans),
            )

        return tuple(document_for(row) for row in range(len(texts)))

    trunk.eval()
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors='pt',
        return_attention_mask=True,
    )
    device = next(trunk.parameters()).device
    input_ids = batch_encoding_tensor(encoded, 'input_ids').to(device)
    attention_mask = batch_encoding_tensor(encoded, 'attention_mask').to(device)
    with torch.no_grad():
        embeds = trunk.host.get_input_embeddings()(input_ids)
        assignment, projected = trunk.soft_vocab.soft_assign(embeds)
        occupied = trunk.occupy_assignment(
            assignment,
            attention_mask,
            input_ids,
            texts,
            tokenizer=tokenizer,
            update_stats=False,
        )
        assignment = occupied.assignment
        bundle = build_soft_relation_bundle(
            trunk.soft_vocab,
            assignment,
            occupied.weights,
            mass_floor=trunk.soft_occupied_floor,
            demand=assignment.new_zeros(assignment.size(0), assignment.size(-1)),
            living=getattr(trunk, 'living_pair_mass', None),
        )
        terms, mean_row_entropy, batch_usage = trunk.soft_vocab.diversity_loss(
            assignment,
            projected,
            token_mask=attention_mask,
        )
        usage_mass = batch_usage.clamp_min(0)
        usage_p = usage_mass / usage_mass.sum().clamp_min(1e-12)
        usage_entropy = -(usage_p * usage_p.clamp_min(1e-12).log()).sum()
        occupy = trunk.soft_vocab.masked_intensity(assignment, occupied.weights)
        explain = Explain()
        leftover = explain.leftover_on_kept(
            occupy,
            occupy,
            explain.graphs_from_bundle(assignment, occupied.weights, bundle),
        )[2]
    vocab = trunk.soft_vocab
    entity_occupancy, n_dead_entity = vocab.occupancy_and_dead(vocab.entity_usage_ema)
    relation_occupancy, n_dead_relation = vocab.occupancy_and_dead(vocab.relation_usage_ema)
    return SnapshotInspection(
        checkpoint=str(checkpoint),
        global_step=global_step,
        documents=documents_from_batch(
            input_ids, attention_mask, assignment, bundle, occupied.labels, leftover
        ),
        defects=score_defects(
            assignment=assignment,
            token_mask=occupied.weights,
            token_ids=input_ids,
            vocab=vocab,
            bundle=bundle,
            thresholds=thresholds,
            utilization_eps=utilization_eps,
        ),
        entity_occupancy=float(entity_occupancy.detach()),
        relation_occupancy=float(relation_occupancy.detach()),
        mean_row_entropy=float(mean_row_entropy.detach()),
        n_dead_entity=int(n_dead_entity.detach()),
        n_dead_relation=int(n_dead_relation.detach()),
        covering=CoveringTrainReading(
            inventory_entropy=float(terms.inventory.detach()),
            ln_k=math.log(int(vocab.entity_bank_size)),
            usage_entropy=float(usage_entropy.detach()),
            usage_target=usage_target,
            row_entropy=float(mean_row_entropy.detach()),
        ),
    )


def write_inspection(report: GraphInspectionReport, output_dir: Path) -> Path:
    """Write one Great Tables HTML report under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / 'inspect.html'
    body = ''.join(spec.as_html() for spec in report.table_specs())
    _ = html_path.write_text(
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        f'<title>SSV overlay inspection</title></head><body>{body}</body></html>',
        encoding='utf-8',
    )
    return html_path


def run_graph_inspect(
    *,
    checkpoint: Path,
    output_dir: Path,
    checkpoint_b: Path | None = None,
    config_path: Path | None = None,
    job: SsvTrainConfig | None = None,
    hupd_dir: Path | None = None,
    hupd_limit: int = _DEFAULT_LIMIT,
    thresholds: InspectThresholds | None = None,
    device: torch.device | None = None,
) -> GraphInspectionReport:
    """Load one or two checkpoints, induce overlays on a HUPD peek, and write the report."""
    request = InspectRequest(
        checkpoint=checkpoint,
        output_dir=output_dir,
        checkpoint_b=checkpoint_b,
        config_path=config_path,
        job=job,
        hupd_dir=hupd_dir,
        hupd_limit=hupd_limit,
        thresholds=thresholds or InspectThresholds(),
        device=device,
    )
    report = InspectSession(request).report()
    _ = write_inspection(report, request.output_dir)
    return report


__all__ = [
    'DefectFinding',
    'DocumentInspection',
    'EntitySpan',
    'EvolutionRow',
    'GraphInspectionReport',
    'InspectThresholds',
    'MaskedCodeView',
    'RelationArc',
    'SnapshotInspection',
    'collapse_spans',
    'compare_snapshots',
    'induce_snapshot',
    'run_graph_inspect',
    'score_defects',
    'write_inspection',
]
