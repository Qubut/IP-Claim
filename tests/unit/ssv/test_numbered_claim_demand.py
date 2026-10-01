"""Numbered filing claims are demand rows; the joined blob mixes them."""

from __future__ import annotations

import inspect

import pytest
import torch

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.collision.encode_job import CollisionEncodeRow, collate_encode_rows
from ip_claim.ingestion.adapters.hupd_json.claim_parser import parse_claims
from ip_claim.ingestion.adapters.hupd_json.loader import patent_from_hupd_dict
from ip_claim.ingestion.models import Claim, Patent
from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmExample
from ip_claim.ssv.dataset import dest_examples_from_graph_batch, example_from_graph_batch
from ip_claim.ssv.graph_batch import (
    ClaimTable,
    dest_claim_texts,
    graph_batch_from_patent,
    patent_claim_blob,
    structural_tables_from_patent,
)

_CLAIMS_BLOB = (
    '1. A widget comprising a latchbolt. '
    '2. A gadget comprising a photodiode. '
    '3. The widget of claim 1, wherein the apparatus is an apparatus '
    'and the apparatus includes an apparatus.'
)
_HUPD_ROW = {
    'application_number': '99000001',
    'decision': 'PENDING',
    'claims': _CLAIMS_BLOB,
}
_SLOTS = ('latchbolt', 'photodiode', 'apparatus')
_CITATION_STEMS = ('cite', 'cited', 'categor', 'grade', 'mark')


def _slot_demand(text: str) -> torch.Tensor:
    lowered = text.lower()
    return torch.tensor([float(lowered.count(slot)) for slot in _SLOTS], dtype=torch.float64)


def _unpaid(covering: Covering, demand: torch.Tensor, supply: torch.Tensor) -> torch.Tensor:
    return covering.unpaid_fraction(covering.pair_table(demand, supply)).squeeze()


@pytest.fixture
def covering() -> Covering:
    return Covering(CoveringKnobs(sigma=1.0, sigma_edge=1.0))


@pytest.fixture
def claims() -> tuple[Claim, ...]:
    parsed = parse_claims(_CLAIMS_BLOB)
    assert tuple(claim.number for claim in parsed) == (1, 2, 3)
    return parsed


@pytest.fixture
def patent(claims: tuple[Claim, ...]) -> Patent:
    loaded = patent_from_hupd_dict(_HUPD_ROW)
    assert loaded.claims == claims
    return loaded


def test_parse_and_claim_table_yield_three_numbered_demand_rows(
    claims: tuple[Claim, ...],
    patent: Patent,
) -> None:
    """Two independents and one shared-field dependent become three claim rows."""
    independents = tuple(claim for claim in claims if claim.is_independent)
    dependents = tuple(claim for claim in claims if not claim.is_independent)
    assert tuple(claim.number for claim in independents) == (1, 2)
    assert tuple(claim.number for claim in dependents) == (3,)
    assert dependents[0].parent_number == 1
    assert 'apparatus' in dependents[0].text.lower()
    assert 'apparatus' not in independents[0].text.lower()
    assert 'apparatus' not in independents[1].text.lower()

    tables = structural_tables_from_patent(patent)
    assert isinstance(tables.claims, ClaimTable)
    assert tables.claims.count == 3
    assert tables.claims.depends.src == (2,)
    assert tables.claims.depends.dst == (0,)
    batch = graph_batch_from_patent(patent)
    assert batch.data['claim'].num_nodes == 3


def test_independent_claims_are_the_dest_demand_objects(
    claims: tuple[Claim, ...],
) -> None:
    """Dest leftover unpaid is each independent claim. Eval keeps every number."""
    dest_rows = tuple(claim for claim in claims if claim.is_independent)
    eval_numbers = tuple(claim.number for claim in claims)
    assert tuple(claim.number for claim in dest_rows) == (1, 2)
    assert eval_numbers == (1, 2, 3)
    assert all(claim.is_independent for claim in dest_rows)
    assert dest_rows[0].text != dest_rows[1].text


def test_blob_demand_is_not_a_linear_image_of_the_claim_rows(
    covering: Covering,
    claims: tuple[Claim, ...],
    patent: Patent,
) -> None:
    """Joined-blob leftover unpaid mixes rows; dest independents stay unpaid."""
    rows = tuple(_slot_demand(claim.text) for claim in claims)
    blob = patent_claim_blob(patent)
    blob_demand = _slot_demand(blob)
    assert blob == ' '.join(claim.text for claim in patent.claims)
    assert all(text in blob for text in (claims[0].text, claims[1].text, claims[2].text))
    assert rows[0][0] > 0
    assert rows[0][2] == 0
    assert rows[1][1] > 0
    assert rows[1][2] == 0
    assert rows[2][2] > 0
    assert rows[2][0] == 0
    assert rows[2][1] == 0
    assert blob_demand[2] == rows[2][2]
    independents = torch.stack(rows[:2])
    # Shared-field mass on the blob is outside the span of dest-default rows.
    assert torch.count_nonzero(independents[:, 2]) == 0
    assert blob_demand[2] > 0

    shared_supply = torch.tensor([0.0, 0.0, 100.0], dtype=torch.float64)
    unpaid_rows = tuple(_unpaid(covering, row, shared_supply) for row in rows)
    unpaid_blob = _unpaid(covering, blob_demand, shared_supply)
    unpaid_independents = unpaid_rows[:2]
    unweighted = torch.stack(unpaid_rows).mean()
    assert torch.allclose(torch.stack(unpaid_independents), torch.ones(2, dtype=torch.float64))
    assert unpaid_rows[2].item() < 0.05
    assert unpaid_blob.item() < min(float(u.item()) for u in unpaid_independents)
    assert not torch.allclose(unpaid_blob, unweighted)
    batch = graph_batch_from_patent(patent)
    example = example_from_graph_batch(batch)
    dest_rows = dest_examples_from_graph_batch(batch)
    assert example.claim_text == claims[0].text
    assert example.claim_text != blob
    assert dest_claim_texts(batch) == (claims[0].text, claims[1].text)
    assert tuple(row.claim_text for row in dest_rows) == (claims[0].text, claims[1].text)
    assert all(row.text == batch.text for row in dest_rows)


def test_empty_claim_text_unpaid_fraction_is_nan(covering: Covering) -> None:
    """Empty claim text is empty demand; leftover unpaid is NaN."""
    assert parse_claims('') == ()
    assert parse_claims('   ') == ()
    empty = _slot_demand('')
    assert torch.count_nonzero(empty) == 0
    unpaid = _unpaid(covering, empty, torch.ones(len(_SLOTS), dtype=torch.float64))
    assert torch.isnan(unpaid).all()


def test_numbered_claim_demand_does_not_read_citation_fields(patent: Patent) -> None:
    """Filing claim numbers come from parse and Claim fields, not cite tables."""
    assert set(_HUPD_ROW) == {'application_number', 'decision', 'claims'}
    names = frozenset(
        name for model in (Claim, Patent, SoftMlmExample) for name in model.model_fields
    )
    assert not any(any(stem in name.lower() for stem in _CITATION_STEMS) for name in names)
    parse_params = inspect.signature(parse_claims).parameters
    blob_params = inspect.signature(patent_claim_blob).parameters
    assert tuple(parse_params) == ('blob',)
    assert tuple(blob_params) == ('patent',)
    sources = (
        inspect.getsource(parse_claims),
        inspect.getsource(patent_claim_blob),
        inspect.getsource(structural_tables_from_patent),
        inspect.getsource(example_from_graph_batch),
        inspect.getsource(dest_examples_from_graph_batch),
        inspect.getsource(dest_claim_texts),
        inspect.getsource(CollisionEncodeRow.from_hupd),
        inspect.getsource(collate_encode_rows),
    )
    joined = ' '.join(sources).lower()
    assert 'cited_hupd' not in joined
    assert 'cited_id' not in joined
    assert 'categories' not in joined
    assert {claim.number for claim in patent.claims} == {1, 2, 3}


def test_collate_and_encode_use_numbered_claim_demand(
    claims: tuple[Claim, ...],
    patent: Patent,
) -> None:
    """Collate and encode demand rows are numbered claims, not the joined blob."""
    batch = graph_batch_from_patent(patent)
    blob = patent_claim_blob(patent)
    dest_rows = dest_examples_from_graph_batch(batch)
    collated_batch = SoftMlmBatch(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        unmasked_input_ids=torch.ones(2, 4, dtype=torch.long),
        attention_mask=torch.ones(2, 4),
        labels=torch.full((2, 4), -100, dtype=torch.long),
        graphs=tuple(row.graph for row in dest_rows),
        texts=tuple(row.text for row in dest_rows),
        claim_texts=tuple(row.claim_text for row in dest_rows),
        disclosure_texts=tuple(row.disclosure_text for row in dest_rows),
        require_supervised=False,
    )
    assert collated_batch.claim_texts == (claims[0].text, claims[1].text)
    assert blob not in collated_batch.claim_texts

    encode_row = CollisionEncodeRow.from_hupd(str(patent.application_number), _HUPD_ROW)
    assert encode_row.example.claim_text == claims[0].text
    assert encode_row.numbered_claim_texts == tuple(claim.text for claim in claims)
    assert encode_row.claim_numbers == (1, 2, 3)
    assert encode_row.claim_blob == blob
    assert encode_row.example.claim_text != encode_row.claim_blob

    def collator(examples: tuple[SoftMlmExample, ...]) -> SoftMlmBatch:
        n_rows = len(examples)
        return SoftMlmBatch(
            input_ids=torch.ones(n_rows, 4, dtype=torch.long),
            unmasked_input_ids=torch.ones(n_rows, 4, dtype=torch.long),
            attention_mask=torch.ones(n_rows, 4),
            labels=torch.full((n_rows, 4), -100, dtype=torch.long),
            graphs=tuple(example.graph for example in examples),
            texts=tuple(example.text for example in examples),
            claim_texts=tuple(example.claim_text for example in examples),
            disclosure_texts=tuple(example.disclosure_text for example in examples),
            require_supervised=False,
        )

    encoded = collate_encode_rows((encode_row,), collator=collator)
    assert encoded.application_numbers == (str(patent.application_number),) * 3
    assert encoded.claim_numbers == (1, 2, 3)
    assert encoded.mlm.claim_texts == tuple(claim.text for claim in claims)
    assert blob not in encoded.mlm.claim_texts
