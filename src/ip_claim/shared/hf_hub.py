"""Hugging Face Hub upload for local SSV / collision run directories.

Creates a dataset repo when missing, then uploads the run folder under
``runs/{name}`` via ``HfApi.upload_folder``. Checkpoints and Ray train storage
are ignored by default; metrics and run metadata are the publish surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from pydantic import SecretStr

__all__ = ['PublishResult', 'publish_local_run_to_hf']

_DEFAULT_IGNORE = (
    '**/*.ckpt',
    '**/ray_train_storage/**',
    '**/.git/**',
)


@dataclass(frozen=True, slots=True)
class PublishResult:
    """Outcome of a Hub folder upload."""

    run_uri: str
    files_uploaded: int


def publish_local_run_to_hf(
    run_dir: Path,
    *,
    repo_id: str,
    run_name: str | None = None,
    hf_token: SecretStr,
    private: bool = True,
    ignore_patterns: tuple[str, ...] = _DEFAULT_IGNORE,
) -> PublishResult:
    """Upload an on-disk run folder to a Hugging Face dataset repo."""
    run_path = run_dir.resolve()
    if not run_path.is_dir():
        msg = f'Missing run directory: {run_path}'
        raise FileNotFoundError(msg)
    token = hf_token.get_secret_value().strip()
    if not token:
        msg = 'hf_token is empty; set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) at config load'
        raise ValueError(msg)
    name = run_name or run_path.name
    api = HfApi(token=token)
    _ = api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True, private=private)
    result = api.upload_folder(
        folder_path=str(run_path),
        path_in_repo=f'runs/{name}',
        repo_id=repo_id,
        repo_type='dataset',
        ignore_patterns=list(ignore_patterns),
    )

    def is_published(path: Path) -> bool:
        if not path.is_file():
            return False
        rel = path.relative_to(run_path).as_posix()
        if rel.endswith('.ckpt') or 'ray_train_storage' in path.parts:
            return False
        return '.git' not in path.parts

    published = tuple(path for path in run_path.rglob('*') if is_published(path))
    commit = getattr(result, 'commit_url', None) or getattr(result, 'url', str(result))
    return PublishResult(
        run_uri=f'hf://{repo_id}/runs/{name} ({commit})',
        files_uploaded=len(published),
    )
