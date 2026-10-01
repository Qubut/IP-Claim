"""Trace-off equivalence for covering-boundary named tensors."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear

from ip_claim.ssv.config import ArchSpec, HostSpec, SsvTrainConfig
from ip_claim.ssv.covering_trace import (
    capture_adapter_delta,
    covering_patches_on,
    disable_covering_adapters,
    keep_covering_seams,
    patch_covering,
    patches_on_device,
    record_covering,
    trace_tensor,
)
from ip_claim.ssv.encode import SoftGraphEncoder
from ip_claim.ssv.graph_batch import graph_batch_from_hupd_dict
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.soft_vocab import SoftVocabModule
from tests._ssv_fixtures import (
    SSV_HUPD_FIXTURES,
    ssv_fixture_batch,
    ssv_smoke_module,
    ssv_tiny_vocab_config,
)


def test_encode_trace_off_matches_named_capture() -> None:
    encoder = SoftGraphEncoder(
        SsvTrainConfig(
            host=HostSpec(d_model=24),
            arch=ArchSpec(
                n_soft_tokens=4,
                gnn_hidden=32,
                gnn_heads=4,
                gnn_layers=1,
                soft_dim=16,
            ),
        )
    )
    encoder.eval()
    graphs = tuple(
        graph_batch_from_hupd_dict(
            json.loads((SSV_HUPD_FIXTURES / name).read_text(encoding='utf-8'))
        ).data
        for name in ('13817165.json', '14111139.json')
    )
    with torch.no_grad():
        quiet = encoder(graphs)
        with record_covering() as trace:
            traced = encoder(graphs)
    assert torch.allclose(quiet.soft_tokens, traced.soft_tokens)
    assert torch.allclose(quiet.node_states, traced.node_states)
    assert set(trace.tensors) >= {'compose_state', 'mixed_compose', 'graph_readout'}
    assert trace.tensors['graph_readout'].names == ('batch', 'slot', 'hidden')


def test_inventory_trace_off_matches_named_capture() -> None:
    vocab = SoftVocabModule(ssv_tiny_vocab_config())
    model = SimpleNamespace(soft_vocab=vocab)
    last_layer = torch.randn(2, 6, 32)
    attention = torch.ones(2, 6)
    claim = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    inventory = Inventory()
    quiet = inventory(
        last_layer,
        attention,
        claim,
        model=model,
        claim_texts=('1. A photodiode.', '1. A housing.'),
    )
    with record_covering() as trace:
        traced = inventory(
            last_layer,
            attention,
            claim,
            model=model,
            claim_texts=('1. A photodiode.', '1. A housing.'),
        )
    assert torch.allclose(quiet.n_entity_claim, traced.n_entity_claim)
    assert torch.allclose(quiet.n_entity_full, traced.n_entity_full)
    assert set(trace.tensors) >= {
        'late_assignment',
        'claim_intensity',
        'disclosure_intensity',
        'labeled_endpoint',
        'collapsed_endpoint',
    }
    assert trace.tensors['claim_intensity'].names == ('batch', 'bank')
    labeled = trace.tensors['labeled_endpoint'].rename(None)
    collapsed = trace.tensors['collapsed_endpoint'].rename(None)
    assert labeled.shape == (2, 2, 8, 8)
    assert collapsed.shape == (2, 2, 8, 8)
    torch.testing.assert_close(collapsed, labeled)


def test_export_trunk_trace_off_matches_named_capture(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        module = ssv_smoke_module(tmp_path)
        model = module.model
        model.eval()
        batch = ssv_fixture_batch(module)
        with torch.no_grad():
            quiet = model.export_trunk(batch.graphs, batch.input_ids, batch.attention_mask)
            with record_covering() as trace:
                traced = model.export_trunk(batch.graphs, batch.input_ids, batch.attention_mask)
    assert torch.allclose(quiet.z_d, traced.z_d)
    assert torch.allclose(quiet.z_g, traced.z_g)
    assert torch.allclose(quiet.soft_tokens, traced.soft_tokens)
    assert set(trace.tensors) >= {
        'early_assignment',
        'termhood_weights',
        'projected_prefix',
        'token_residual',
        'host_text_state',
        'graph_readout',
    }
    assert trace.tensors['projected_prefix'].names == ('batch', 'slot', 'hidden')


def test_keep_covering_seams_records_only_named_tensors() -> None:
    """An allowlist still applies patches but drops unlisted captures."""
    original = torch.ones(2, 3)
    replacement = torch.full((2, 3), 4.0)
    with record_covering() as trace, keep_covering_seams(frozenset({'token_residual'})):
        kept = trace_tensor('token_residual', original, 'batch', 'token')
        with patch_covering({'host_text_state': replacement}):
            skipped = trace_tensor('host_text_state', original, 'batch', 'token')
    assert torch.equal(kept, original)
    assert torch.equal(skipped, replacement)
    assert set(trace.tensors) == {'token_residual'}


def test_patch_covering_replaces_named_seam_without_names() -> None:
    original = torch.ones(2, 3)
    replacement = torch.full((2, 3), 4.0)
    quiet = trace_tensor('host_text_state', original, 'batch', 'token')
    with patch_covering({'host_text_state': replacement}):
        patched = trace_tensor('host_text_state', original, 'batch', 'token')
    assert torch.equal(quiet, original)
    assert torch.equal(patched, replacement)


def test_zero_patch_of_compacted_rank_zeros_live_token_residual() -> None:
    """Residual-off zeros the live addend, not a compacted matching seam."""
    live = torch.ones(2, 256, 8)
    prefix_wide = torch.zeros(2, 16, 8)
    token_mean = torch.zeros(2, 8)
    scalar = torch.zeros(1)
    with patch_covering({'token_residual': prefix_wide}):
        from_prefix = trace_tensor('token_residual', live, 'batch', 'token', 'hidden')
    with patch_covering({'token_residual': token_mean}):
        from_mean = trace_tensor('token_residual', live, 'batch', 'token', 'hidden')
    with patch_covering({'token_residual': scalar}):
        from_scalar = trace_tensor('token_residual', live, 'batch', 'token', 'hidden')
    expected = torch.zeros_like(live)
    assert from_prefix.shape == live.shape
    assert from_mean.shape == live.shape
    assert from_scalar.shape == live.shape
    assert torch.equal(from_prefix, expected)
    assert torch.equal(from_mean, expected)
    assert torch.equal(from_scalar, expected)
    torch.testing.assert_close(live + from_prefix, live)


def test_nonzero_wrong_shape_patch_keeps_live_tensor() -> None:
    live = torch.ones(2, 256, 8)
    with patch_covering({'token_residual': torch.ones(2, 16, 8)}):
        kept = trace_tensor('token_residual', live, 'batch', 'token', 'hidden')
    assert torch.equal(kept, live)


def test_patches_on_device_copies_replacements() -> None:
    replacement = torch.ones(2, 3)
    moved = patches_on_device({'early_assignment': replacement}, torch.device('cpu'))
    assert moved['early_assignment'].device.type == 'cpu'
    assert torch.equal(moved['early_assignment'], replacement)
    with patch_covering({'early_assignment': replacement}):
        pinned = covering_patches_on(torch.device('cpu'))
    assert torch.equal(pinned['early_assignment'], replacement)


def test_capture_adapter_delta_is_output_minus_base() -> None:
    """LoRA hook stores enabled output minus the unwrapped base layer."""
    torch.manual_seed(0)
    layer = LoraLinear(
        torch.nn.Linear(4, 6, bias=False),
        'default',
        LoraConfig(r=2, lora_alpha=4, target_modules=['proj']),
        r=2,
        lora_alpha=4,
    )
    host = torch.nn.Sequential(layer)
    inbound = torch.randn(2, 3, 4)
    layer.eval()
    with torch.no_grad():
        enabled = layer(inbound)
        base_out = layer.get_base_layer()(inbound)
        with record_covering() as trace, capture_adapter_delta(host):
            again = host(inbound)
    captured = trace.tensors['adapter_delta'].rename(None)
    assert torch.equal(again, enabled)
    torch.testing.assert_close(captured[:, 0], (enabled - base_out).mean(dim=-2))
    assert captured.shape == (2, 1, 6)
    assert captured.device.type == 'cpu'


def test_disable_covering_adapters_zeros_recorded_delta() -> None:
    """Adapter-off runs the frozen base and records a zero LoRA delta."""
    torch.manual_seed(0)
    layer = LoraLinear(
        torch.nn.Linear(4, 6, bias=False),
        'default',
        LoraConfig(r=2, lora_alpha=4, target_modules=['proj']),
        r=2,
        lora_alpha=4,
    )

    class Host(torch.nn.Module):
        def __init__(self, inner: LoraLinear) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, inbound: torch.Tensor) -> torch.Tensor:
            return self.inner(inbound)

        @contextmanager
        def disable_adapter(self) -> Iterator[None]:
            self.inner.enable_adapters(enabled=False)
            try:
                yield
            finally:
                self.inner.enable_adapters(enabled=True)

    host = Host(layer)
    inbound = torch.randn(2, 3, 4)
    layer.eval()
    with torch.no_grad():
        base_out = layer.get_base_layer()(inbound)
        with record_covering() as trace, disable_covering_adapters(), capture_adapter_delta(host):
            again = host(inbound)
    captured = trace.tensors['adapter_delta'].rename(None)
    torch.testing.assert_close(again, base_out)
    torch.testing.assert_close(captured[:, 0], torch.zeros_like(base_out.mean(dim=-2)))


def test_capture_adapter_delta_skips_when_seam_not_kept() -> None:
    """Hooks stay off when the active trace does not keep adapter_delta."""
    torch.manual_seed(0)
    layer = LoraLinear(
        torch.nn.Linear(4, 6, bias=False),
        'default',
        LoraConfig(r=2, lora_alpha=4, target_modules=['proj']),
        r=2,
        lora_alpha=4,
    )
    host = torch.nn.Sequential(layer)
    inbound = torch.randn(2, 3, 4)
    layer.eval()
    with (
        torch.no_grad(), record_covering() as trace,
        keep_covering_seams(frozenset({'host_text_state'})),
        capture_adapter_delta(host),
    ):
        _ = host(inbound)
    assert 'adapter_delta' not in trace.tensors
