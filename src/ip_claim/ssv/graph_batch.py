"""Build CPC-prefix and claim-dependency HeteroData from HUPD patents.

Pydantic tables hold structure; HeteroData is projected at the boundary with
PyG node/edge type keys.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from returns.maybe import Maybe
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from ip_claim.ingestion.adapters.hupd_json.loader import patent_from_hupd_dict
from ip_claim.ingestion.models import Claim, Patent

_COMPACT_CPC = re.compile(r'^([A-HY])(\d{2})([A-Z])', re.IGNORECASE)
CPC_SECTIONS = 'ABCDEFGHY'
CPC_SECTION_COUNT = len(CPC_SECTIONS)
CPC_CLASS_COUNT = CPC_SECTION_COUNT * 100
CPC_SUBCLASS_COUNT = CPC_CLASS_COUNT * 26
CPC_IDENTITY_UNKNOWN = 0
CPC_IDENTITY_VOCAB_SIZE = 1 + CPC_SECTION_COUNT + CPC_CLASS_COUNT + CPC_SUBCLASS_COUNT


class EdgeTable(BaseModel):
    """Directed edge list as parallel index columns."""

    model_config = ConfigDict(frozen=True)

    src: tuple[int, ...] = Field(default_factory=tuple)
    dst: tuple[int, ...] = Field(default_factory=tuple)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, int]]) -> EdgeTable:
        """Build an edge table from ``(src, dst)`` pairs."""
        materialized = tuple(pairs)
        return cls(
            src=tuple(s for s, _ in materialized),
            dst=tuple(d for _, d in materialized),
        )

    def edge_index(self) -> torch.Tensor:
        """``[2, E]`` long tensor; empty when there are no edges."""
        if not self.src:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor([self.src, self.dst], dtype=torch.long)


class CpcTable(BaseModel):
    """Ordered CPC-prefix nodes and section/class/subclass parent edges."""

    model_config = ConfigDict(frozen=True)

    labels: tuple[str, ...]
    parents: EdgeTable


class ClaimTable(BaseModel):
    """Claim nodes indexed in patent order and depends-on edges."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=0)
    depends: EdgeTable


class StructuralTables(BaseModel):
    """Node/edge tables ready for HeteroData projection."""

    model_config = ConfigDict(frozen=True)

    cpc: CpcTable
    claims: ClaimTable


class PatentGraphBatch(BaseModel):
    """One patent's structural graph plus SOH-stripped training and disclosure text."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    application_number: str
    text: str
    claim_blob: str = ''
    claim_numbers: tuple[int, ...] = ()
    claim_texts: tuple[str, ...] = ()
    claim_is_independent: tuple[bool, ...] = ()
    disclosure: str = ''
    data: HeteroData
    main_cpc_section: str | None = None


def strip_soh_markup(text: str) -> str:
    """Remove HUPD ``<SOH>`` / ``<EOH>`` wrappers before tokenization."""
    return text.replace('<SOH>', '').replace('<EOH>', '').strip()


def cpc_prefix_labels(code: str) -> tuple[str, ...]:
    """Section / class / subclass prefixes from a compact HUPD CPC string.

    Compact codes like ``A61M51723`` do not encode unambiguous group/subgroup
    slash boundaries without a CPC scheme, so only coarse prefixes are returned.
    """
    cleaned = code.strip().upper()

    def prefixes(match: re.Match[str]) -> tuple[str, ...]:
        section, clazz, subclass = match.group(1), match.group(2), match.group(3)
        return (section, f'{section}{clazz}', f'{section}{clazz}{subclass}')

    return Maybe.from_optional(_COMPACT_CPC.match(cleaned)).map(prefixes).value_or(())


def patent_claim_blob(patent: Patent) -> str:
    """SOH-stripped claim texts joined in filing order."""
    return strip_soh_markup(' '.join(claim.text for claim in patent.claims))


def dest_claim_texts(batch: PatentGraphBatch) -> tuple[str, ...]:
    """Independent numbered claim texts in filing order."""
    return tuple(
        text
        for text, independent in zip(
            batch.claim_texts,
            batch.claim_is_independent,
            strict=True,
        )
        if independent
    )


def patent_training_text(patent: Patent) -> str:
    """Concatenate claims and abstract (plus non-empty summary) with SOH stripped."""
    claim_blob = ' '.join(claim.text for claim in patent.claims)
    parts = (claim_blob, patent.abstract, patent.summary)
    return strip_soh_markup(' '.join(part for part in parts if part.strip()))


def patent_disclosure_text(patent: Patent) -> str:
    """Background and full description with SOH stripped. Collision supply only."""
    parts = (patent.background, patent.full_description)
    return strip_soh_markup(' '.join(part for part in parts if part.strip()))


def patent_cpc_codes(patent: Patent) -> tuple[str, ...]:
    """Main CPC first, then remaining classification codes without duplicates."""
    rest = tuple(patent.classification.cpc_codes)
    return (
        Maybe
        .from_optional(patent.classification.main_cpc)
        .map(lambda code: (code, *tuple(item for item in rest if item != code)))
        .value_or(rest)
    )


def ordered_cpc_prefixes(codes: Sequence[str]) -> tuple[str, ...]:
    """Unique coarse prefixes in first-seen order; ``_UNC`` if none parse."""
    labels = tuple(dict.fromkeys(label for code in codes for label in cpc_prefix_labels(code)))
    return labels or ('_UNC',)


def cpc_identity_index(label: str) -> int:
    """Map a coarse CPC prefix string onto the closed identity table.

    Index 0 is the unknown bucket (``_UNC``, empty, or a string that is not
    a section, class, or subclass prefix). The remaining rows enumerate the
    grammatical coarse space so each official prefix has a stable row.
    """
    cleaned = label.strip().upper()
    match cleaned:
        case '' | '_UNC':
            return CPC_IDENTITY_UNKNOWN
        case section if len(section) == 1 and section in CPC_SECTIONS:
            return 1 + CPC_SECTIONS.index(section)
        case clazz if len(clazz) == 3 and clazz[0] in CPC_SECTIONS and clazz[1:].isdigit():
            return 1 + CPC_SECTION_COUNT + CPC_SECTIONS.index(clazz[0]) * 100 + int(clazz[1:])
        case subclass if (
            len(subclass) == 4
            and subclass[0] in CPC_SECTIONS
            and subclass[1:3].isdigit()
            and 'A' <= subclass[3] <= 'Z'
        ):
            class_base = CPC_SECTIONS.index(subclass[0]) * 100 + int(subclass[1:3])
            letter = ord(subclass[3]) - ord('A')
            return 1 + CPC_SECTION_COUNT + CPC_CLASS_COUNT + class_base * 26 + letter
        case _:
            return CPC_IDENTITY_UNKNOWN


def cpc_identity_depth(index: int) -> float:
    """Tree depth implied by an identity index: section 0, class 1, subclass 2."""
    if index <= CPC_SECTION_COUNT:
        return 0.0
    if index <= CPC_SECTION_COUNT + CPC_CLASS_COUNT:
        return 1.0
    return 2.0


def cpc_identity_ids(labels: Sequence[str]) -> torch.Tensor:
    """Long identity indices for one document's CPC-prefix nodes."""
    if not labels:
        return torch.zeros((0,), dtype=torch.long)
    return torch.tensor([cpc_identity_index(label) for label in labels], dtype=torch.long)


def cpc_parent_edge_table(labels: Sequence[str]) -> EdgeTable:
    """Parent edges along each label's section / class / subclass chain."""

    def chain_steps(label: str) -> tuple[tuple[str, str], ...]:
        if len(label) == 1:
            return ()
        if len(label) == 3:
            return ((label[0], label),)
        if len(label) == 4:
            return ((label[0], label[:3]), (label[:3], label))
        return ()

    index = {label: i for i, label in enumerate(labels)}
    pairs = (
        (index[parent], index[child])
        for label in labels
        if label != '_UNC'
        for parent, child in chain_steps(label)
        if parent in index and child in index
    )
    return EdgeTable.from_pairs(pairs)


def claim_depends_edge_table(claims: Sequence[Claim]) -> EdgeTable:
    """Dependent-to-parent edges using parser ``parent_number`` only."""
    index = {claim.number: i for i, claim in enumerate(claims)}
    pairs = (
        (index[claim.number], index[claim.parent_number])
        for claim in claims
        if claim.parent_number is not None and claim.parent_number in index
    )
    return EdgeTable.from_pairs(pairs)


def structural_tables_from_patent(patent: Patent) -> StructuralTables:
    """Derive immutable CPC and claim tables from a Patent value object."""
    labels = ordered_cpc_prefixes(patent_cpc_codes(patent))
    return StructuralTables(
        cpc=CpcTable(labels=labels, parents=cpc_parent_edge_table(labels)),
        claims=ClaimTable(
            count=len(patent.claims),
            depends=claim_depends_edge_table(patent.claims),
        ),
    )


def project_hetero(tables: StructuralTables) -> HeteroData:
    """Project node/edge tables into a ``HeteroData`` batch."""
    cpc: NodeType = 'cpc'
    claim: NodeType = 'claim'
    cpc_parent: EdgeType = (cpc, 'parent_of', cpc)
    claim_depends: EdgeType = (claim, 'depends_on', claim)

    n_cpc = len(tables.cpc.labels)
    n_claim = tables.claims.count
    graph = HeteroData()
    graph[cpc].num_nodes = n_cpc
    graph[cpc].x = torch.zeros((n_cpc, 1), dtype=torch.float)
    graph[cpc].cpc_id = cpc_identity_ids(tables.cpc.labels)
    graph[cpc].label = tuple(tables.cpc.labels)
    graph[claim].num_nodes = n_claim
    graph[claim].x = torch.zeros((n_claim, 1), dtype=torch.float)
    graph[cpc_parent].edge_index = tables.cpc.parents.edge_index()
    graph[claim_depends].edge_index = tables.claims.depends.edge_index()
    return graph


def build_hetero_from_patent(patent: Patent) -> HeteroData:
    """Build CPC-prefix and claim-dependency ``HeteroData`` from a Patent VO."""
    return project_hetero(structural_tables_from_patent(patent))


def graph_batch_from_patent(patent: Patent) -> PatentGraphBatch:
    """Map one domain patent into a structural graph batch."""
    main = patent.classification.main_cpc
    prefixes = cpc_prefix_labels(main) if main else ()
    return PatentGraphBatch(
        application_number=str(patent.application_number),
        text=patent_training_text(patent),
        claim_blob=patent_claim_blob(patent),
        claim_numbers=tuple(claim.number for claim in patent.claims),
        claim_texts=tuple(strip_soh_markup(claim.text) for claim in patent.claims),
        claim_is_independent=tuple(claim.is_independent for claim in patent.claims),
        disclosure=patent_disclosure_text(patent),
        data=build_hetero_from_patent(patent),
        main_cpc_section=prefixes[0] if prefixes else None,
    )


def graph_batch_from_hupd_dict(raw: dict[str, Any]) -> PatentGraphBatch:
    """Map one HUPD JSON dict into a structural graph batch."""
    return graph_batch_from_patent(patent_from_hupd_dict(raw))
