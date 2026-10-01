"""Unit tests for Hugging Face Hub run publish helper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from ip_claim.shared.hf_hub import publish_local_run_to_hf


def test_publish_local_run_to_hf_uploads_metrics_not_ckpt(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    metrics = run_dir / 'lightning' / 'version_0' / 'metrics.csv'
    metrics.parent.mkdir(parents=True)
    metrics.write_text('step,loss\n0,1.0\n', encoding='utf-8')
    (run_dir / 'ssv.ckpt').write_bytes(b'ckpt')

    api = MagicMock()
    api.upload_folder.return_value = MagicMock(commit_url='https://hf.example/commit/1')

    with patch('ip_claim.shared.hf_hub.HfApi', return_value=api) as hf_api:
        result = publish_local_run_to_hf(
            run_dir,
            repo_id='org/ip-claim-runs',
            run_name='smoke',
            hf_token=SecretStr('hf_test_token'),
            private=True,
        )

    hf_api.assert_called_once_with(token='hf_test_token')
    api.create_repo.assert_called_once_with(
        repo_id='org/ip-claim-runs',
        repo_type='dataset',
        exist_ok=True,
        private=True,
    )
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs['path_in_repo'] == 'runs/smoke'
    assert kwargs['repo_type'] == 'dataset'
    assert '**/*.ckpt' in kwargs['ignore_patterns']
    assert result.files_uploaded == 1
    assert 'hf://org/ip-claim-runs/runs/smoke' in result.run_uri


def test_publish_rejects_empty_token(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    with pytest.raises(ValueError, match='hf_token is empty'):
        publish_local_run_to_hf(
            run_dir,
            repo_id='org/ip-claim-runs',
            hf_token=SecretStr(''),
        )
