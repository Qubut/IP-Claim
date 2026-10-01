"""Nested SsvTrainConfig specs; flat keys still lift for CLI and tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from patent_ate.spec import AteSpec

from ip_claim.ssv.config import (
    ArchSpec,
    BankHealthSpec,
    DestComparisonSpec,
    DualSpec,
    FitSpec,
    GraphProbeSpec,
    KendallSpec,
    LogSpec,
    MlmSpec,
    OverlayShapeSpec,
    RuntimeSpec,
    SsvTrainConfig,
)

_NEST_TYPES = (
    ArchSpec,
    KendallSpec,
    DualSpec,
    MlmSpec,
    BankHealthSpec,
    FitSpec,
    LogSpec,
    RuntimeSpec,
    GraphProbeSpec,
    AteSpec,
    DestComparisonSpec,
    OverlayShapeSpec,
)
_SHIPPED_NEST_KEYS = frozenset({
    'host',
    'tes_sac',
    'arch',
    'kendall',
    'dual',
    'mlm',
    'bank',
    'fit',
    'log',
    'runtime',
    'graph_probe',
    'ate',
    'dest_comparison',
    'overlay_shape',
})
_CONFIGS = Path(__file__).resolve().parents[3] / 'configs'


def test_nest_field_names_are_unique() -> None:
    names = [name for spec in _NEST_TYPES for name in spec.model_fields]
    assert len(names) == len(set(names))


def test_flat_and_nested_payloads_validate_equal() -> None:
    flat = SsvTrainConfig.model_validate(
        {'entity_bank_size': 12, 'max_steps': 9, 'beta_div': 0.3},
    )
    nested = SsvTrainConfig.model_validate(
        {
            'arch': {'entity_bank_size': 12},
            'fit': {'max_steps': 9},
            'kendall': {'beta_div': 0.3},
        },
    )
    assert flat.arch.entity_bank_size == nested.arch.entity_bank_size == 12
    assert flat.fit.max_steps == nested.fit.max_steps == 9
    assert flat.kendall.beta_div == nested.kendall.beta_div == pytest.approx(0.3)


def test_flat_keys_override_nested_dump() -> None:
    job = SsvTrainConfig.model_validate(
        {'arch': {'entity_bank_size': 12}, 'entity_bank_size': 20},
    )
    assert job.arch.entity_bank_size == 20


def test_overlay_lifts_flat_cli_overrides() -> None:
    job = SsvTrainConfig().overlay({'max_steps': 7, 'ray': True, 'use_gpu': True})
    assert job.fit.max_steps == 7
    assert job.runtime.ray is True
    assert job.runtime.use_gpu is True


def test_overlay_lifts_init_weights() -> None:
    path = 'artifacts/ssv/ssv-step009000.ckpt'
    job = SsvTrainConfig().overlay({'init_weights': path})
    assert job.runtime.init_weights == path
    cleared = job.overlay({'init_weights': ''})
    assert cleared.runtime.init_weights is None


def test_overlay_clears_termhood_store_path() -> None:
    job = SsvTrainConfig.model_validate(
        {'runtime': {'termhood_store_path': '/outputs/ate-ray-full-hupd'}},
    )
    assert job.runtime.termhood_store_path == '/outputs/ate-ray-full-hupd'
    cleared = job.overlay({'termhood_store_path': None})
    assert cleared.runtime.termhood_store_path is None


def test_ate_spacy_model_lifts_from_flat_key() -> None:
    job = SsvTrainConfig.model_validate({'spacy_model': 'en_core_web_sm'})
    assert job.ate.spacy_model == 'en_core_web_sm'
    assert SsvTrainConfig().ate.spacy_model == AteSpec().spacy_model


def test_ate_extract_knobs_lift_from_nested_and_flat() -> None:
    nested = SsvTrainConfig.model_validate({
        'ate': {
            'spacy_model': 'en_core_web_lg',
            'cpu_job_width': 12,
            'extract_block_rows': 32,
            'pipe_docs': 32,
            'sentence_group': 16,
            'duckdb_memory': '4GB',
            'duckdb_threads': 4,
            'parent_buckets': 8,
            'tail_window_cap': 1000,
            'tail_weighted_bytes_cap': 2000,
            'base_window_cap': 3000,
            'base_weighted_bytes_cap': 4000,
            'max_parent_units': 16,
            'max_candidate_scans': 16,
            'tail_candidate_row_cap': 100,
            'tail_compare_cap': 200,
        }
    })
    assert nested.ate.cpu_job_width == 12
    assert nested.ate.extract_block_rows == 32
    assert nested.ate.pipe_docs == 32
    assert nested.ate.sentence_group == 16
    assert nested.ate.duckdb_memory == '4GB'
    assert nested.ate.duckdb_threads == 4
    assert nested.ate.parent_buckets == 8
    assert nested.ate.tail_window_cap == 1000
    assert nested.ate.tail_weighted_bytes_cap == 2000
    assert nested.ate.base_window_cap == 3000
    assert nested.ate.base_weighted_bytes_cap == 4000
    assert nested.ate.max_parent_units == 16
    assert nested.ate.max_candidate_scans == 16
    assert nested.ate.tail_candidate_row_cap == 100
    assert nested.ate.tail_compare_cap == 200
    flat = SsvTrainConfig.model_validate({
        'cpu_job_width': 6,
        'extract_block_rows': 64,
        'sentence_group': 8,
        'parent_buckets': 16,
        'tail_window_cap': 3000,
        'tail_weighted_bytes_cap': 4000,
        'base_window_cap': 5000,
        'base_weighted_bytes_cap': 6000,
        'max_parent_units': 20,
        'max_candidate_scans': 20,
        'tail_candidate_row_cap': 300,
        'tail_compare_cap': 400,
    })
    assert flat.ate.cpu_job_width == 6
    assert flat.ate.extract_block_rows == 64
    assert flat.ate.sentence_group == 8
    assert flat.ate.parent_buckets == 16
    assert flat.ate.tail_window_cap == 3000
    assert flat.ate.tail_weighted_bytes_cap == 4000
    assert flat.ate.base_window_cap == 5000
    assert flat.ate.base_weighted_bytes_cap == 6000
    assert flat.ate.max_parent_units == 20
    assert flat.ate.max_candidate_scans == 20
    assert flat.ate.tail_candidate_row_cap == 300
    assert flat.ate.tail_compare_cap == 400
    gpu = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.gpu_dea_smoke.yaml')
    assert gpu.ate.cpu_job_width == 8
    assert gpu.ate.extract_block_rows == 256
    assert gpu.ate.pipe_docs == 8
    assert gpu.ate.sentence_group == 32
    assert gpu.ate.duckdb_memory == '128GB'
    assert gpu.ate.duckdb_threads == 8
    assert gpu.ate.parent_buckets == 32
    assert gpu.ate.tail_window_cap == 400_000
    assert gpu.ate.tail_weighted_bytes_cap == 400_000_000
    assert gpu.ate.base_window_cap == 25_000_000
    assert gpu.ate.base_weighted_bytes_cap == 2_500_000_000
    assert gpu.ate.max_parent_units == 64
    assert gpu.ate.max_candidate_scans == 64
    assert gpu.ate.tail_candidate_row_cap == 50_000
    assert gpu.ate.tail_compare_cap == 50_000
    assert AteSpec().duckdb_memory == '8GB'
    assert AteSpec().duckdb_threads == 8
    assert AteSpec().parent_buckets == 32
    assert AteSpec().tail_window_cap == 400_000
    assert AteSpec().tail_weighted_bytes_cap == 400_000_000
    assert AteSpec().base_window_cap == 25_000_000
    assert AteSpec().base_weighted_bytes_cap == 2_500_000_000
    assert AteSpec().max_parent_units == 64
    assert AteSpec().max_candidate_scans == 64
    assert AteSpec().tail_candidate_row_cap == 50_000
    assert AteSpec().tail_compare_cap == 50_000
    assert AteSpec().containment == 'window'
    assert AteSpec().indexed is None
    assert SsvTrainConfig().ate.duckdb_memory == '8GB'
    assert SsvTrainConfig().ate.duckdb_threads == 8
    assert SsvTrainConfig().ate.parent_buckets == 32
    assert SsvTrainConfig().ate.tail_window_cap == 400_000
    assert SsvTrainConfig().ate.tail_weighted_bytes_cap == 400_000_000
    assert SsvTrainConfig().ate.base_window_cap == 25_000_000
    assert SsvTrainConfig().ate.max_parent_units == 64


def test_shipped_train_yaml_uses_nested_sections() -> None:
    paths = tuple(_CONFIGS / name for name in ('ssv_train.prod.yaml', 'ssv_train.smoke.yaml'))
    raws = tuple(OmegaConf.to_container(OmegaConf.load(str(path)), resolve=False) for path in paths)
    assert all(
        isinstance(raw, dict)
        and set(raw) <= _SHIPPED_NEST_KEYS
        and all(isinstance(raw[key], dict) for key in raw)
        for raw in raws
    )
    assert all(SsvTrainConfig.from_yaml(path).host.name for path in paths)
    prod = SsvTrainConfig.from_yaml(_CONFIGS / 'ssv_train.prod.yaml')
    assert prod.runtime.init_weights == '/outputs/ssv/ssv-step009000.ckpt'
