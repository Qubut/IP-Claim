"""Unit tests for graph-guided mask mix, KE/align joint loss, and trunk export."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import HeteroData

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ingestion.adapters.hupd_json.paths import patent_from_hupd_path
from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmCollator
from ip_claim.ssv.config import (
    ArchSpec,
    FitSpec,
    HostSpec,
    KendallSpec,
    MlmSpec,
    RuntimeSpec,
    SsvTrainConfig,
)
from ip_claim.ssv.dataset import examples_from_patents
from ip_claim.ssv.graph_ingress import OccupancyMap
from ip_claim.ssv.host_tokenizer import batch_encoding_tensor, load_host_tokenizer
from ip_claim.ssv.model import SoftTrunkOutput, TrunkExport, build_soft_trunk
from ip_claim.ssv.module import SsvLightningModule
from ip_claim.ssv.soft_graph import SoftRelationBundle, build_soft_relation_bundle
from ip_claim.ssv.soft_vocab import SoftVocabModule

_FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'hupd'


def _pt_ids_and_special(encoded: object) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch_encoding_tensor(encoded, 'input_ids')
    attention = batch_encoding_tensor(encoded, 'attention_mask')
    special = batch_encoding_tensor(encoded, 'special_tokens_mask').bool() | (attention == 0)
    return input_ids, special


def _smoke_config(tmp_path: Path) -> SsvTrainConfig:
    return SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        arch=ArchSpec(
            gnn_hidden=32,
            gnn_heads=4,
            gnn_layers=1,
            n_soft_tokens=4,
            entity_bank_size=16,
            soft_dim=32,
            max_length=48,
            memory_queue_size=32,
        ),
        fit=FitSpec(
            max_steps=1,
            batch_size=2,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        ),
        mlm=MlmSpec(mlm_probability=0.15, rho_max=0.5, rho_warmup_steps=10),
        kendall=KendallSpec(beta_div=0.1),
        runtime=RuntimeSpec(ray=False),
    )


def _fixture_examples(*names: str):
    return examples_from_patents(tuple(patent_from_hupd_path(_FIXTURES / name) for name in names))


def _collator_with_assignment(
    config: SsvTrainConfig,
    tokenizer,
    *,
    rho: float,
    mlm_probability: float = 0.5,
) -> SoftMlmCollator:
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=mlm_probability,
        max_length=int(config.arch.max_length),
        rho=rho,
        cpc_aux_weight=0.0,
        assignment_top_k=int(config.mlm.assignment_top_k),
    )
    soft_vocab = SoftVocabModule(config)
    embed = torch.nn.Embedding(len(tokenizer), int(config.host.d_model))
    collator.bind_assignment_source(soft_vocab, embed)
    return collator


def test_graph_guided_collator_masks_with_rho(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = _collator_with_assignment(config, tokenizer, rho=0.6)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    assert batch.rho == pytest.approx(0.6)
    assert (batch.labels != -100).any()
    assert batch.input_ids.shape == batch.labels.shape
    assert batch.unmasked_input_ids.shape == batch.input_ids.shape
    hidden = batch.labels.ge(0)
    assert torch.equal(batch.unmasked_input_ids[hidden], batch.labels[hidden])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_graph_guided_collator_rho_with_cuda_embed(tmp_path: Path) -> None:
    """rho>0 collation must not embed CPU input_ids against CUDA host weights."""
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = _collator_with_assignment(config, tokenizer, rho=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config).cuda()
    collator.bind_assignment_source(model.soft_vocab, model.host.get_input_embeddings())
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    assert batch.rho == pytest.approx(0.6)
    assert (batch.labels != -100).any()


def test_assignment_mass_boosts_high_mass_positions(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.2,
        max_length=48,
        rho=1.0,
        cpc_aux_weight=0.0,
    )
    examples = _fixture_examples('13817165.json')
    encoded = tokenizer(
        examples[0].text,
        truncation=True,
        max_length=48,
        padding=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors='pt',
    )
    input_ids, special = _pt_ids_and_special(encoded)
    mass = torch.ones_like(input_ids, dtype=torch.float)
    coords = (~special).nonzero(as_tuple=False)
    assert coords.shape[0] >= 2
    hot = coords[0]
    cold = coords[1]
    mass[hot[0], hot[1]] = 100.0
    mass[cold[0], cold[1]] = 0.01
    collator.set_assignment_mass(mass)
    rates = collator._graph_mix_probability(input_ids, special, examples)
    assert rates[hot[0], hot[1]] > rates[cold[0], cold[1]]
    assert rates[hot[0], hot[1]] > float(collator.mlm_probability or 0.0)


def test_cpc_token_overlap_does_not_drive_without_aux(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.2,
        max_length=48,
        rho=1.0,
        cpc_boost=100.0,
        cpc_aux_weight=0.0,
    )
    examples = _fixture_examples('13817165.json')
    encoded = tokenizer(
        examples[0].text,
        truncation=True,
        max_length=48,
        padding=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors='pt',
    )
    input_ids, special = _pt_ids_and_special(encoded)
    uniform = torch.ones_like(input_ids, dtype=torch.float)
    collator.set_assignment_mass(uniform)
    rates_no_aux = collator._graph_mix_probability(input_ids, special, examples)
    maskable = ~special
    flat = rates_no_aux[maskable]
    assert torch.allclose(flat, flat[0].expand_as(flat), atol=1e-5)

    collator.cpc_aux_weight = 1.0
    rates_cpc_only = collator._graph_mix_probability(input_ids, special, examples)
    # With aux off, CPC token-id hits must not reshape the mixture.
    # With aux=1, CPC path is free to deviate when hits exist.
    if rates_cpc_only[maskable].std() > 1e-6:
        assert not torch.allclose(rates_no_aux[maskable], rates_cpc_only[maskable], atol=1e-4)


def test_rho_without_assignment_source_raises(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.5, max_length=48, rho=0.5)
    with pytest.raises(ValueError, match='soft-assignment mass'):
        collator(_fixture_examples('13817165.json'))


def test_mlm_requires_mask_token_id(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    broken = MagicMock(wraps=tokenizer)
    broken.mask_token_id = None
    broken.pad_token_id = tokenizer.pad_token_id
    with pytest.raises(ValueError, match='mask_token_id'):
        SoftMlmCollator(broken, mlm=True, mlm_probability=0.15, max_length=48, rho=0.0)


def test_soft_trunk_exports_ke_and_trunk_vectors(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = _collator_with_assignment(config, tokenizer, rho=0.3, mlm_probability=0.4)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.train()
        out = model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
        )
    assert isinstance(out, SoftTrunkOutput)
    assert torch.isfinite(out.mlm_loss)
    assert torch.isfinite(out.diversity_loss)
    assert torch.isfinite(out.ke_loss)
    assert out.z_d.shape[0] == batch.input_ids.shape[0]
    assert out.z_g.shape == out.z_d.shape
    assert out.soft_tokens.shape[0] == batch.input_ids.shape[0]
    export = out.as_trunk_export()
    assert isinstance(export, TrunkExport)
    assert export.z_d.shape == out.z_d.shape
    loss = out.mlm_loss + config.kendall.beta_div * out.diversity_loss + out.ke_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_export_trunk_without_labels(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.0, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        with torch.no_grad():
            export = model.export_trunk(batch.graphs, batch.input_ids, batch.attention_mask)
    assert isinstance(export, TrunkExport)
    assert export.z_d.shape[0] == 1
    assert export.soft_tokens.shape[1] == config.arch.n_soft_tokens


def test_remask_entities_replaces_hidden_codes(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    vocab = SoftVocabModule(config)
    entities = torch.randn(2, 5, config.arch.soft_dim)
    hide = torch.tensor([
        [False, True, False, False, True],
        [True, False, False, True, False],
    ])
    remasked = vocab.remask_entities(entities, hide)
    assert torch.allclose(remasked[hide], vocab.entity_dmask.expand_as(remasked[hide]))
    assert torch.equal(remasked[~hide], entities[~hide])


def test_overlay_assigns_on_unmasked_and_drops_hidden_occupancy(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.5, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    assert batch.labels.ge(0).any()
    first_states: dict[str, torch.Tensor] = {}
    overlay_masks: list[torch.Tensor] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.train()
        real_assign = model.soft_vocab.soft_assign
        real_bundle = build_soft_relation_bundle

        def _capture(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if 'states' not in first_states:
                first_states['states'] = states.detach().clone()
            return real_assign(states)

        def _bundle(
            vocab: SoftVocabModule,
            assignment: torch.Tensor,
            token_mask: torch.Tensor,
            *,
            mass_floor: float = 0.0,
            demand: torch.Tensor | None = None,
            pair_mass: torch.Tensor | None = None,
            living: torch.Tensor | None = None,
        ) -> SoftRelationBundle:
            overlay_masks.append(token_mask.detach().clone())
            return real_bundle(
                vocab,
                assignment,
                token_mask,
                mass_floor=mass_floor,
                demand=demand,
                pair_mass=pair_mass,
                living=living,
            )

        def occupy(
            assignment: torch.Tensor,
            live_mask: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> OccupancyMap:
            del args, kwargs
            width = int(assignment.size(1))
            rows = int(assignment.size(0))
            return OccupancyMap(
                assignment=assignment,
                weights=live_mask.to(dtype=assignment.dtype),
                labels=tuple(tuple('' for _ in range(width)) for _ in range(rows)),
            )

        with (
            patch.object(model, 'occupy_assignment', side_effect=occupy),
            patch.object(model.soft_vocab, 'soft_assign', side_effect=_capture),
            patch('ip_claim.ssv.model.build_soft_relation_bundle', side_effect=_bundle),
        ):
            out = model(
                batch.graphs,
                batch.input_ids,
                batch.unmasked_input_ids,
                batch.attention_mask,
                batch.labels,
            )
        clean = model.host.get_input_embeddings()(batch.unmasked_input_ids)
        assert torch.allclose(first_states['states'], clean)
        out.mlm_loss.backward()
    hidden = batch.labels.ge(0)
    visible = batch.attention_mask * (~hidden).to(dtype=batch.attention_mask.dtype)
    assert overlay_masks
    assert torch.equal(overlay_masks[0], visible)
    assert model.soft_vocab.entity_dmask.grad is None


def test_joint_loss_rho_ramp_and_align(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.4, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json', '14111139.json'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        container = SsvContainer(config=config)
        module = container.lightning_module()
    assert isinstance(module, SsvLightningModule)
    module.bind_mask_collator(collator)
    module.on_train_batch_start(batch, 0)
    assert collator.rho >= 0.0
    assert collator.soft_vocab is module.model.soft_vocab
    out = module(batch)
    align = module._align_loss(out.z_d, out.z_g)
    loss = out.mlm_loss + config.kendall.beta_div * out.diversity_loss + out.ke_loss + align
    assert torch.isfinite(loss)
    assert torch.isfinite(align)


def _span_run_lengths(hidden: torch.Tensor) -> list[int]:
    lengths: list[int] = []
    for row in hidden.tolist():
        run = 0
        for flag in row:
            if flag:
                run += 1
                continue
            if run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return lengths


def _span_run_starts(hidden: torch.Tensor) -> torch.Tensor:
    prev = torch.zeros_like(hidden)
    prev[:, 1:] = hidden[:, :-1]
    return hidden & ~prev


def test_span_mask_runs_are_contiguous_geo_bounded(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.3,
        max_length=48,
        rho=0.0,
        mlm_span_mask=True,
        mlm_span_geo_p=0.2,
        mlm_span_max_length=10,
        seed=0,
    )
    examples = _fixture_examples('13817165.json', '14111139.json')
    run_lengths: list[int] = []
    coverages: list[float] = []
    for _ in range(32):
        batch = collator(examples)
        special = batch.attention_mask == 0
        hidden = batch.labels.ge(0)
        assert not hidden[special].any()
        maskable = (~special).float()
        coverages.append(float((hidden.float() * maskable).sum() / maskable.sum()))
        run_lengths.extend(_span_run_lengths(hidden & ~special))
    assert run_lengths
    assert min(run_lengths) >= 1
    assert any(length >= 2 for length in run_lengths)
    assert any(length > 5 for length in run_lengths)
    assert sum(coverages) / len(coverages) == pytest.approx(0.3, abs=0.12)


def test_span_mask_skips_right_edge_pad(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.5,
        max_length=48,
        rho=0.0,
        mlm_span_mask=True,
        mlm_span_max_length=10,
    )
    examples = _fixture_examples('13817165.json')
    encoded = tokenizer(
        examples[0].text,
        truncation=True,
        max_length=48,
        padding='max_length',
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors='pt',
    )
    input_ids, special = _pt_ids_and_special(encoded)
    content = (~special).nonzero(as_tuple=False)
    last = content[-1]
    rates = torch.zeros(input_ids.shape, dtype=torch.float)
    rates[last[0], last[1]] = 8.0
    _, labels = collator._mask_from_span_rates(input_ids.clone(), special, rates)
    assert (labels[special] == -100).all()
    assert int(labels[last[0], last[1]].item()) != -100
    isolated = _span_run_lengths(labels.ge(0) & ~special)
    assert isolated
    assert max(isolated) <= 10


def test_isolated_span_length_matches_truncated_geo(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.15,
        max_length=48,
        rho=0.0,
        mlm_span_mask=True,
        mlm_span_geo_p=0.2,
        mlm_span_max_length=10,
        seed=2,
    )
    examples = _fixture_examples('13817165.json')
    encoded = tokenizer(
        examples[0].text,
        truncation=True,
        max_length=48,
        padding=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors='pt',
    )
    input_ids, special = _pt_ids_and_special(encoded)
    coords = (~special).nonzero(as_tuple=False)
    assert coords.shape[0] >= 20
    start_at = coords[2]
    sampled: list[int] = []
    for _ in range(64):
        rates = torch.zeros(input_ids.shape, dtype=torch.float)
        rates[start_at[0], start_at[1]] = 8.0
        _, labels = collator._mask_from_span_rates(input_ids.clone(), special, rates)
        sampled.extend(_span_run_lengths(labels.ge(0) & ~special))
    assert sampled
    assert min(sampled) >= 1
    assert max(sampled) <= 10
    expected = (1.0 - (1.0 - 0.2) ** 10) / 0.2
    assert sum(sampled) / len(sampled) == pytest.approx(expected, rel=0.35)
    assert any(length > 5 for length in sampled)


def test_span_starts_follow_assignment_mass(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.25,
        max_length=48,
        rho=1.0,
        cpc_aux_weight=0.0,
        mlm_span_mask=True,
        mlm_span_geo_p=0.2,
        mlm_span_max_length=10,
        seed=1,
    )
    examples = _fixture_examples('13817165.json')
    encoded = tokenizer(
        examples[0].text,
        truncation=True,
        max_length=48,
        padding=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors='pt',
    )
    input_ids, special = _pt_ids_and_special(encoded)
    coords = (~special).nonzero(as_tuple=False)
    hot = coords[0]
    cold = coords[min(4, coords.shape[0] - 1)]
    mass = torch.full(input_ids.shape, 0.02, dtype=torch.float)
    mass[hot[0], hot[1]] = 50.0
    mass[cold[0], cold[1]] = 0.02
    mass[special] = 0.0
    collator.set_assignment_mass(mass)
    hot_starts = 0
    cold_starts = 0
    for _ in range(48):
        batch = collator(examples)
        hidden = batch.labels.ge(0)
        starts = _span_run_starts(hidden)
        hot_starts += int(starts[hot[0], hot[1]].item())
        cold_starts += int(starts[cold[0], cold[1]].item())
    assert hot_starts > cold_starts


def test_span_geo_p_rejects_closed_unit_interval(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    with pytest.raises(ValueError, match='mlm_span_geo_p'):
        SoftMlmCollator(tokenizer, mlm_probability=0.15, mlm_span_geo_p=0.0)
    with pytest.raises(ValueError, match='mlm_span_geo_p'):
        SoftMlmCollator(tokenizer, mlm_probability=0.15, mlm_span_geo_p=1.0)


def test_mlm_collator_forces_one_supervised_on_empty_draw(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.15,
        max_length=48,
        rho=0.0,
        mlm_span_mask=True,
    )
    examples = _fixture_examples('13817165.json', '14111139.json')

    def empty_draw(rates: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        return torch.zeros(rates.shape, dtype=rates.dtype, device=rates.device)

    with patch.object(torch, 'bernoulli', side_effect=empty_draw):
        batch = collator(examples)
    assert batch.labels.ge(0).sum() == 1
    assert batch.require_supervised is True
    row, col = batch.labels.ge(0).nonzero(as_tuple=True)
    mask_id = tokenizer.mask_token_id
    assert isinstance(mask_id, int)
    assert batch.input_ids[row, col].eq(mask_id).all()
    assert torch.equal(
        batch.labels[row, col],
        batch.unmasked_input_ids[row, col],
    )


def test_supervised_batch_rejects_empty_labels() -> None:
    graph = HeteroData()
    with pytest.raises(ValueError, match='supervised position'):
        SoftMlmBatch(
            input_ids=torch.ones(1, 4, dtype=torch.long),
            unmasked_input_ids=torch.ones(1, 4, dtype=torch.long),
            attention_mask=torch.ones(1, 4),
            labels=torch.full((1, 4), -100),
            graphs=(graph,),
            require_supervised=True,
        )


def test_eval_collator_allows_empty_labels(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(tokenizer, mlm_probability=0.0, max_length=48, rho=0.0)
    batch = collator(_fixture_examples('13817165.json'))
    assert not batch.labels.ge(0).any()
    assert batch.require_supervised is False


def test_nan_span_rates_fail_closed(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.15,
        max_length=48,
        rho=0.0,
        mlm_span_mask=True,
    )
    examples = _fixture_examples('13817165.json')
    encoded = tokenizer(
        examples[0].text,
        truncation=True,
        max_length=48,
        padding=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors='pt',
    )
    input_ids, special = _pt_ids_and_special(encoded)
    nan_rates = torch.full(input_ids.shape, float('nan'))
    with pytest.raises(RuntimeError, match='Bernoulli rates'):
        collator._mask_from_span_rates(input_ids, special, nan_rates)
