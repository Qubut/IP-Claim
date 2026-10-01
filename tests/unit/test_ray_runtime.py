"""Local Ray session UV-hook disable."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ip_claim.shared.ray_runtime import ensure_local_ray


def test_ensure_local_ray_clears_captured_uv_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    constants = SimpleNamespace()
    inits: list[object] = []
    monkeypatch.setattr('ip_claim.shared.ray_runtime.ray.is_initialized', lambda: False)
    monkeypatch.setattr(
        'ip_claim.shared.ray_runtime.ray.init',
        lambda **kwargs: inits.append(kwargs),
    )
    monkeypatch.setattr(
        'ip_claim.shared.ray_runtime.importlib.import_module',
        lambda _name: constants,
    )
    monkeypatch.setattr('ip_claim.shared.ray_runtime.RAY_TEMP_DIR', tmp_path)
    assert ensure_local_ray(num_gpus=0) is True
    assert constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is False
    assert inits
