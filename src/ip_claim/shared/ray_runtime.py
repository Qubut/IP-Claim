"""Local Ray session with a short AF_UNIX temp dir."""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

import torch

_UV_RUNTIME_ENV = 'RAY_ENABLE_UV_RUN_RUNTIME_ENV'
os.environ[_UV_RUNTIME_ENV] = '0'

import ray  # noqa: E402

RAY_TEMP_DIR = Path(os.environ.get('IP_CLAIM_RAY_TEMP', '/tmp/ip-claim-ray'))  # noqa: S108


def ensure_local_ray(*, num_gpus: int, num_cpus: int | None = None) -> bool:
    """Start a local Ray runtime if one is not already up.

    Returns True when this call started Ray and the caller must shut it down.
    Session sockets are AF_UNIX and cannot exceed 107 bytes, so the temp dir
    is short rather than devenv TMPDIR. ``uv run`` would otherwise package this
    tree and spawn workers with ``uv run --python``. The UV hook constant is
    captured at import, so it is cleared here before ``ray.init``.

    Stale session state from a previous run is removed before init so a fresh
    start does not inherit dead-session sockets or logs. Worker logs stream
    to the driver so an actor death is visible in the container output even
    when the worker process is killed before flushing its own log file.
    """
    if ray.is_initialized():
        return False
    if RAY_TEMP_DIR.exists():
        shutil.rmtree(RAY_TEMP_DIR)
    RAY_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ[_UV_RUNTIME_ENV] = '0'
    setattr(importlib.import_module('ray._private.ray_constants'), _UV_RUNTIME_ENV, False)
    library_path = os.environ.get('LD_LIBRARY_PATH')
    runtime_env = {'env_vars': {'LD_LIBRARY_PATH': library_path}} if library_path else None
    cpu_slots = {} if num_cpus is None else {'num_cpus': num_cpus}
    common = {
        'ignore_reinit_error': True,
        'num_gpus': num_gpus,
        'log_to_driver': True,
        '_temp_dir': str(RAY_TEMP_DIR),
        'runtime_env': runtime_env,
        '_skip_env_hook': True,
        **cpu_slots,
    }
    try:
        ray.init(
            include_dashboard=True,
            dashboard_host='127.0.0.1',
            dashboard_port=8265,
            **common,
        )
        print('ray dashboard: http://127.0.0.1:8265', flush=True)
    except Exception as exc:
        print(f'ray dashboard unavailable ({exc}); encode continues without it', flush=True)
        if ray.is_initialized():
            ray.shutdown()
        ray.init(include_dashboard=False, **common)
    print(f'ray session temp dir: {RAY_TEMP_DIR}', flush=True)
    return True


def visible_cuda_count() -> int:
    """Count cards from ``CUDA_VISIBLE_DEVICES`` without allocating tensors."""
    listed = os.environ.get('CUDA_VISIBLE_DEVICES')
    if listed is not None:
        names = tuple(part.strip() for part in listed.split(',') if part.strip())
        if not names or names == ('void',):
            return 0
        return len(names)
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
