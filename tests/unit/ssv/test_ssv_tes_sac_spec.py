"""TesSacSpec Field defaults and envelope type."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from ip_claim.ssv.config import SsvTrainConfig, TesSacSpec

_SSV = Path(__file__).resolve().parents[3] / 'src' / 'ip_claim' / 'ssv'


def test_tes_sac_spec_defaults_match_xu_lock() -> None:
    spec = TesSacSpec()
    assert spec.band == pytest.approx(0.01)
    assert spec.std_max == pytest.approx(0.05)
    assert spec.ema_decay == pytest.approx(0.999)
    assert spec.ratchet == pytest.approx(0.9)
    assert spec.lock_steps == 20


def test_ssv_train_config_nests_tes_sac_defaults() -> None:
    job = SsvTrainConfig()
    assert job.tes_sac == TesSacSpec()
    assert issubclass(TesSacSpec, BaseModel)
    assert not dataclasses.is_dataclass(TesSacSpec)


def test_tes_sac_spec_is_frozen() -> None:
    spec = TesSacSpec()
    with pytest.raises(ValidationError):
        spec.band = 0.2  # type: ignore[misc]


def test_steer_modules_have_no_stdlib_dataclass() -> None:
    paths = (
        _SSV / 'config.py',
        _SSV / 'load_weights.py',
        _SSV / 'steer' / 'tes_sac.py',
        _SSV / 'steer' / 'duals.py',
        _SSV / 'steer' / 'cheng.py',
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == 'dataclasses'
            for alias in node.names
        }
        assert 'dataclass' not in imported, path
