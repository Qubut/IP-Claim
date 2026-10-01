"""Hidden-slot additive injection and the fixed inject-scale ramp."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import PropertyMock, patch

import pytest
import torch
from tests._ssv_fixtures import attach_cpu_trainer
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ingestion.adapters.hupd_json.paths import patent_from_hupd_path
from ip_claim.ssv.collate import SoftMlmBatch, SoftMlmCollator
from ip_claim.ssv.config import ArchSpec, FitSpec, HostSpec, MlmSpec, RuntimeSpec, SsvTrainConfig
from ip_claim.ssv.dataset import examples_from_patents
from ip_claim.ssv.dea import pack_token_rows_to_nodes, scatter_node_rows_to_tokens
from ip_claim.ssv.encode import SoftEncodeOutput
from ip_claim.ssv.graph_ingress import OccupancyMap
from ip_claim.ssv.host_tokenizer import load_host_tokenizer
from ip_claim.ssv.model import SoftTrunkModel, build_soft_trunk
from ip_claim.ssv.project import SoftTokenProjector
from ip_claim.ssv.soft_graph import (
    SoftRelationBundle,
    build_soft_relation_bundle,
    pullback_compose_to_tokens,
)

_FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'hupd'


def _smoke_config(tmp_path: Path, **overrides: Any) -> SsvTrainConfig:
    base = SsvTrainConfig(
        host=HostSpec(name='hf-internal-testing/tiny-random-bert', d_model=32),
        arch=ArchSpec(
            gnn_hidden=32,
            gnn_heads=4,
            gnn_layers=1,
            n_soft_tokens=4,
            entity_bank_size=16,
            soft_dim=32,
            max_length=48,
        ),
        fit=FitSpec(
            max_steps=1,
            batch_size=2,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        ),
        mlm=MlmSpec(mlm_probability=0.4),
        runtime=RuntimeSpec(ray=False),
    )
    return base.overlay(overrides) if overrides else base


def _fixture_batch(config: SsvTrainConfig) -> SoftMlmBatch:
    tokenizer = load_host_tokenizer(config)
    collator = SoftMlmCollator(
        tokenizer,
        mlm_probability=0.5,
        max_length=48,
        rho=0.0,
    )
    names = ('13817165.json', '14111139.json')
    examples = examples_from_patents(
        tuple(patent_from_hupd_path(_FIXTURES / name) for name in names)
    )
    return collator(examples)


def _occupy_visible(model: SoftTrunkModel):
    """Keep assignment mass as occupy weights so tiny-host tests still mix."""

    def occupy(
        assignment: Tensor,
        live_mask: Tensor,
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

    return patch.object(model, 'occupy_assignment', side_effect=occupy)


def _spy_host_forward(
    host_fwd: Callable[..., Any],
    captured: dict[str, Tensor],
) -> Callable[..., Any]:
    def capture_host(
        *,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> Any:
        if inputs_embeds is not None:
            captured['inputs_embeds'] = inputs_embeds.detach().clone()
        host_out = host_fwd(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )
        captured['logits'] = host_out.logits.detach().clone()
        return host_out

    return capture_host


def _text_embeds_and_nodes(
    model: SoftTrunkModel,
    batch: SoftMlmBatch,
    living: Tensor | None = None,
) -> tuple[Tensor, SoftEncodeOutput, SoftRelationBundle]:
    encoded_box: dict[str, SoftEncodeOutput] = {}
    embed_box: dict[str, Tensor] = {}
    bundle_box: dict[str, SoftRelationBundle] = {}
    encode_fwd = model.encoder.forward
    bundle_fwd = build_soft_relation_bundle

    def capture_encode(
        graphs: HeteroData | Sequence[HeteroData],
        text_query: Tensor | None = None,
    ) -> SoftEncodeOutput:
        encoded = encode_fwd(graphs, text_query)
        encoded_box['out'] = encoded
        return encoded

    def capture_bundle(*args: object, **kwargs: object) -> SoftRelationBundle:
        bundle = bundle_fwd(*args, **kwargs)
        bundle_box['bundle'] = bundle
        return bundle

    with (
        _occupy_visible(model),
        patch.object(model.encoder, 'forward', side_effect=capture_encode),
        patch('ip_claim.ssv.model.build_soft_relation_bundle', side_effect=capture_bundle),
        patch.object(
            model.host,
            'forward',
            side_effect=_spy_host_forward(model.host.forward, embed_box),
        ),
    ):
        _ = model(
            batch.graphs,
            batch.input_ids,
            batch.unmasked_input_ids,
            batch.attention_mask,
            batch.labels,
            batch.texts,
            claim_texts=batch.claim_texts,
            living=model.living_snapshot() if living is None else living,
        )
    text = embed_box['inputs_embeds'][:, model.n_soft_tokens :, :]
    return text, encoded_box['out'], bundle_box['bundle']


def _pulled_token_slots(
    model: SoftTrunkModel,
    batch: SoftMlmBatch,
    encoded: SoftEncodeOutput,
    bundle: SoftRelationBundle,
) -> Tensor:
    """Project compose states onto the token grid the same way the trunk injects."""
    clean = model.host.get_input_embeddings()(batch.unmasked_input_ids)
    assignment, _ = model.soft_vocab.soft_assign(clean)
    token_states = pullback_compose_to_tokens(
        encoded.node_states,
        encoded.node_mask,
        bundle.overlays,
        assignment,
    )
    packed, packed_mask = pack_token_rows_to_nodes(token_states, batch.attention_mask)
    return scatter_node_rows_to_tokens(
        model.slot_projector(packed).tokens,
        batch.attention_mask,
        packed_mask,
    )


def test_inject_ramp_fields_live_on_arch_spec() -> None:
    assert SsvTrainConfig().arch.inject_ramp_steps == 100
    assert SsvTrainConfig().arch.inject_ramp_end == pytest.approx(0.5)
    config = SsvTrainConfig(arch=ArchSpec(inject_ramp_steps=20, inject_ramp_end=0.4))
    assert config.arch.inject_ramp_steps == 20
    assert config.arch.inject_ramp_end == pytest.approx(0.4)


def test_build_soft_trunk_owns_distinct_slot_projector(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
    assert isinstance(model.slot_projector, SoftTokenProjector)
    assert model.slot_projector is not model.projector
    assert model.inject_scale == pytest.approx(0.0)


def test_no_learnable_inject_gate_parameter(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
    names = [name for name, _ in model.named_parameters()]
    assert not any('inject_scale' in name or 'inject_gate' in name for name in names)
    assert isinstance(model.inject_scale, float)
    assert not isinstance(model._inject_scale, nn.Parameter)


def test_set_inject_scale_rejects_out_of_range(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
    with pytest.raises(ValueError, match='inject scale'):
        model.set_inject_scale(-0.1)
    with pytest.raises(ValueError, match='inject scale'):
        model.set_inject_scale(1.1)


def _rms_matched_residual(slots: Tensor, host: Tensor) -> Tensor:
    def rms(values: Tensor) -> Tensor:
        return (
            values
            .square()
            .mean(dim=-1, keepdim=True)
            .sqrt()
            .clamp_min(torch.finfo(values.dtype).eps)
        )

    return slots * (rms(host) / rms(slots))


def test_hidden_slot_injection_preserves_mask_embedding(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    assert batch.labels.ge(0).any()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        masked = model.host.get_input_embeddings()(batch.input_ids)
        hidden = batch.labels.ge(0)
        visible = batch.attention_mask.bool() & ~hidden
        with torch.no_grad():
            living = model.living_snapshot()
            model.set_inject_scale(0.0)
            text_zero, _, _ = _text_embeds_and_nodes(model, batch, living)
            model.set_inject_scale(1.0)
            text_ceiling, encoded, bundle = _text_embeds_and_nodes(model, batch, living)
            slots = _pulled_token_slots(model, batch, encoded, bundle)
            aligned = _rms_matched_residual(slots, masked)
            model.set_inject_scale(0.4)
            text_mid, encoded_mid, bundle_mid = _text_embeds_and_nodes(model, batch, living)
            slots_mid = _pulled_token_slots(model, batch, encoded_mid, bundle_mid)
            aligned_mid = _rms_matched_residual(slots_mid, masked)

    live = slots[hidden].square().mean(dim=-1) > torch.finfo(slots.dtype).eps
    live_mid = slots_mid[hidden].square().mean(dim=-1) > torch.finfo(slots_mid.dtype).eps
    assert torch.allclose(text_zero, masked)
    assert live.any()
    assert torch.allclose(text_ceiling[hidden][live], (masked + aligned)[hidden][live])
    assert torch.allclose(text_ceiling[visible], masked[visible])
    assert torch.allclose(
        text_mid[hidden][live_mid],
        (masked + 0.4 * aligned_mid)[hidden][live_mid],
    )
    assert torch.allclose(text_mid[visible], masked[visible])
    residual_at_one = text_ceiling[hidden] - masked[hidden]
    assert residual_at_one.abs().max() > 0
    assert torch.allclose(text_ceiling[hidden] - residual_at_one, masked[hidden])


def test_injected_residual_rms_matches_host_embedding(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    hidden = batch.labels.ge(0)
    assert hidden.any()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        masked = model.host.get_input_embeddings()(batch.input_ids)
        with torch.no_grad():
            model.set_inject_scale(0.5)
            living = model.living_snapshot()
            text_half, encoded, bundle = _text_embeds_and_nodes(model, batch, living)
            slots = _pulled_token_slots(model, batch, encoded, bundle)
    aligned = _rms_matched_residual(slots, masked)
    live = slots[hidden].square().mean(dim=-1) > torch.finfo(slots.dtype).eps
    assert live.any()
    host_rms = masked[hidden][live].square().mean(dim=-1).sqrt()
    aligned_rms = aligned[hidden][live].square().mean(dim=-1).sqrt()
    residual_rms = (text_half[hidden][live] - masked[hidden][live]).square().mean(dim=-1).sqrt()
    assert torch.allclose(aligned_rms, host_rms, rtol=1e-5, atol=1e-6)
    assert torch.allclose(residual_rms, 0.5 * host_rms, rtol=1e-5, atol=1e-6)
    ratio = residual_rms / host_rms.clamp_min(torch.finfo(host_rms.dtype).eps)
    assert torch.all((ratio > 0.2) & (ratio < 0.8))


def test_covering_inject_scale_uses_ramp_ceiling(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, inject_ramp_end=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
    assert model.inject_scale == pytest.approx(0.0)
    assert model.covering_inject_scale == pytest.approx(0.4)
    model.set_inject_scale(0.8)
    assert model.covering_inject_scale == pytest.approx(0.8)


def test_export_trunk_injects_covering_residual(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    captured: dict[str, Tensor] = {}
    mixed: list[Tensor] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        mix = model.mix_slot_residual

        def capture_mix(
            text_embeds: Tensor,
            packed_states: Tensor,
            packed_mask: Tensor,
            attention_mask: Tensor,
            *,
            scale: float,
            slot_mask: Tensor | None = None,
            zero_inject: bool = False,
        ) -> Tensor:
            assert scale == pytest.approx(model.covering_inject_scale)
            out = mix(
                text_embeds,
                packed_states,
                packed_mask,
                attention_mask,
                scale=scale,
                slot_mask=slot_mask,
                zero_inject=zero_inject,
            )
            mixed.append(out.detach().clone())
            return out

        with (
            torch.no_grad(),
            patch.object(model, 'mix_slot_residual', side_effect=capture_mix),
            patch.object(
                model.host,
                'forward',
                side_effect=_spy_host_forward(model.host.forward, captured),
            ),
        ):
            export = model.export_trunk(
                batch.graphs,
                batch.input_ids,
                batch.attention_mask,
            )
    assert mixed
    text = captured['inputs_embeds'][:, model.n_soft_tokens :, :]
    assert torch.equal(text, mixed[0])
    assert export.z_d.shape[0] == batch.input_ids.shape[0]


def test_inject_scale_starts_at_zero_on_first_batch(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, inject_ramp_steps=10, inject_ramp_end=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    module.on_train_batch_start(None, 0)
    assert module.model.inject_scale == pytest.approx(0.0)


def test_inject_scale_follows_arch_ramp(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, inject_ramp_steps=10, inject_ramp_end=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    with patch.object(type(module), 'global_step', new_callable=PropertyMock, return_value=5):
        module.on_train_batch_start(None, 0)
    assert module.model.inject_scale == pytest.approx(0.25)


def test_inject_scale_caps_at_ramp_end(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, inject_ramp_steps=10, inject_ramp_end=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    with patch.object(type(module), 'global_step', new_callable=PropertyMock, return_value=50):
        module.on_train_batch_start(None, 0)
    assert module.model.inject_scale == pytest.approx(0.6)


def test_inject_scale_is_logged_with_rho(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, inject_ramp_steps=8, inject_ramp_end=0.8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = SsvContainer(config=config).lightning_module()
    logged: dict[str, float] = {}

    def capture(name: str, value: Tensor | float, **_kwargs: object) -> None:
        logged[name] = float(value.detach()) if isinstance(value, Tensor) else value

    attach_cpu_trainer(module)
    with (
        patch.object(type(module), 'global_step', new_callable=PropertyMock, return_value=4),
        patch.object(module, 'log', side_effect=capture),
    ):
        module.on_train_batch_start(None, 0)
    assert 'rho' in logged
    assert logged['inject_scale'] == pytest.approx(0.4)


def test_node_states_align_to_token_grid(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        with torch.no_grad():
            _, encoded, _ = _text_embeds_and_nodes(model, batch)
    clean = model.host.get_input_embeddings()(batch.unmasked_input_ids)
    assignment, _ = model.soft_vocab.soft_assign(clean)
    visible = batch.attention_mask * (~batch.labels.ge(0)).to(dtype=batch.attention_mask.dtype)
    bundle = build_soft_relation_bundle(
        model.soft_vocab,
        assignment,
        visible,
        mass_floor=model.soft_occupied_floor,
        demand=model.soft_vocab.masked_intensity(assignment, visible),
    )
    token_states = pullback_compose_to_tokens(
        encoded.node_states,
        encoded.node_mask,
        bundle.overlays,
        assignment,
    )
    packed, packed_mask = pack_token_rows_to_nodes(token_states, batch.attention_mask)
    assert encoded.node_states.shape[0] == batch.labels.shape[0]
    assert encoded.node_states.shape[-1] == config.arch.gnn_hidden
    assert token_states.shape[:2] == batch.labels.shape
    assert torch.equal(packed_mask.sum(dim=-1), batch.attention_mask.sum(dim=-1))
    slots = scatter_node_rows_to_tokens(packed, batch.attention_mask, packed_mask)
    assert slots.shape[:2] == batch.labels.shape


def test_inject_scale_moves_host_logits(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    hidden = batch.labels.ge(0)
    assert hidden.any()
    captured: dict[str, Tensor] = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        with (
            torch.no_grad(),
            _occupy_visible(model),
            patch.object(
                model.host,
                'forward',
                side_effect=_spy_host_forward(model.host.forward, captured),
            ),
        ):
            living = model.living_snapshot()
            model.set_inject_scale(0.0)
            out_zero = model(
                batch.graphs,
                batch.input_ids,
                batch.unmasked_input_ids,
                batch.attention_mask,
                batch.labels,
                batch.texts,
                claim_texts=batch.claim_texts,
                living=living,
            )
            logits_zero = captured['logits']
            model.set_inject_scale(1.0)
            out_one = model(
                batch.graphs,
                batch.input_ids,
                batch.unmasked_input_ids,
                batch.attention_mask,
                batch.labels,
                batch.texts,
                claim_texts=batch.claim_texts,
                living=living,
            )
            logits_one = captured['logits']

    text_zero = logits_zero[:, model.n_soft_tokens :, :]
    text_one = logits_one[:, model.n_soft_tokens :, :]
    hidden_delta = (text_one[hidden] - text_zero[hidden]).abs().max()
    assert hidden_delta > 0
    nll_hidden_zero = out_zero.mlm_token_nll[:, model.n_soft_tokens :][hidden]
    nll_hidden_one = out_one.mlm_token_nll[:, model.n_soft_tokens :][hidden]
    assert not torch.equal(nll_hidden_zero, nll_hidden_one)


def test_zero_inject_moves_host_logits_and_leaves_prefix(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    hidden = batch.labels.ge(0)
    assert hidden.any()
    captured: dict[str, Tensor] = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        model.set_inject_scale(0.5)
        with (
            torch.no_grad(),
            _occupy_visible(model),
            patch.object(
                model.host,
                'forward',
                side_effect=_spy_host_forward(model.host.forward, captured),
            ),
        ):
            living = model.living_snapshot()
            _ = model(
                batch.graphs,
                batch.input_ids,
                batch.unmasked_input_ids,
                batch.attention_mask,
                batch.labels,
                batch.texts,
                claim_texts=batch.claim_texts,
                living=living,
                zero_inject=False,
            )
            embeds_on = captured['inputs_embeds']
            _ = model(
                batch.graphs,
                batch.input_ids,
                batch.unmasked_input_ids,
                batch.attention_mask,
                batch.labels,
                batch.texts,
                claim_texts=batch.claim_texts,
                living=living,
                zero_inject=True,
            )
            embeds_off = captured['inputs_embeds']

    n_soft = model.n_soft_tokens
    assert torch.allclose(embeds_on[:, :n_soft], embeds_off[:, :n_soft])
    text_on = embeds_on[:, n_soft:, :]
    text_off = embeds_off[:, n_soft:, :]
    assert (text_on[hidden] - text_off[hidden]).abs().max() > 0
    visible = batch.attention_mask.bool() & ~hidden
    assert torch.allclose(text_on[visible], text_off[visible])


def test_zero_prefix_leaves_injected_slots(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    batch = _fixture_batch(config)
    hidden = batch.labels.ge(0)
    assert hidden.any()
    captured: dict[str, Tensor] = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = build_soft_trunk(config)
        model.eval()
        model.set_inject_scale(0.5)
        with (
            torch.no_grad(),
            _occupy_visible(model),
            patch.object(
                model.host,
                'forward',
                side_effect=_spy_host_forward(model.host.forward, captured),
            ),
        ):
            living = model.living_snapshot()
            _ = model(
                batch.graphs,
                batch.input_ids,
                batch.unmasked_input_ids,
                batch.attention_mask,
                batch.labels,
                batch.texts,
                claim_texts=batch.claim_texts,
                living=living,
                zero_prefix=False,
            )
            embeds_on = captured['inputs_embeds']
            _ = model(
                batch.graphs,
                batch.input_ids,
                batch.unmasked_input_ids,
                batch.attention_mask,
                batch.labels,
                batch.texts,
                claim_texts=batch.claim_texts,
                living=living,
                zero_prefix=True,
            )
            embeds_off = captured['inputs_embeds']

    n_soft = model.n_soft_tokens
    assert torch.allclose(embeds_off[:, :n_soft], torch.zeros_like(embeds_off[:, :n_soft]))
    assert not torch.allclose(embeds_on[:, :n_soft], embeds_off[:, :n_soft])
    text_on = embeds_on[:, n_soft:, :]
    text_off = embeds_off[:, n_soft:, :]
    assert torch.allclose(text_on[hidden], text_off[hidden])
