"""Collision eval CLI for ``python -m ip_claim.collision``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import structlog
import typer
from dependency_injector import providers
from omegaconf import OmegaConf
from pydantic import SecretStr

from ip_claim.app.container.collision import CollisionContainer
from ip_claim.collision.artefacts import report_from_splits, write_collision_artefacts
from ip_claim.collision.config import DEFAULT_COLLISION_EVAL_PATH, CollisionEvalConfig
from ip_claim.collision.data import EpoProcessorCitationPairSource
from ip_claim.collision.diagnose import OVERLAP_PER_LETTER, judge_from_paths
from ip_claim.collision.disclosure import DEFAULT_MEASURE_SAMPLE, measure_disclosure_from_paths
from ip_claim.collision.eval import (
    encode_shard_root,
    rank_covering_from_shards,
    run_collision_eval,
)
from ip_claim.collision.rank_grid import rank_keep_grid_from_paths
from ip_claim.shared.hf_hub import publish_local_run_to_hf
from ip_claim.ssv.config import ArchSpec, HostSpec, RuntimeSpec, SsvTrainConfig
from ip_claim.ssv.host_tokenizer import load_host_tokenizer

_log = structlog.get_logger(__name__)

app = typer.Typer(
    name='collision',
    help='Patent collision detection eval CLI.',
    no_args_is_help=True,
    add_completion=False,
)

_OPT_DATASET = typer.Option('--dataset', help='Parquet from epo-processor analyze --dataset.')
_OPT_HUPD_DIR = typer.Option('--hupd-dir', help='HUPD JSON root for pair application numbers.')
_OPT_HOST = typer.Option('--host', help='HF masked-LM id (must match host.d_model).')
_OPT_D_MODEL = typer.Option('--d-model', help='Host hidden size.')
_OPT_RECALL_K = typer.Option(
    '--recall-k',
    help='Comma-separated Recall@K cutoffs (default: 5,10,20,50).',
)
_OPT_CONFIG = typer.Option(
    '--config',
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    help='OmegaConf YAML (defaults: package collision_eval.yaml).',
)
_OPT_OUTPUT_DIR = typer.Option('--output-dir', help='Directory for covering.json / explain/.')
_OPT_PUBLISH = typer.Option('--publish-to-hub', help='Upload eval artefacts after ranking.')
_OPT_HUB_REPO = typer.Option('--hub-repo-id', help='HF dataset repo id for runs/.')
_OPT_HUB_RUN = typer.Option('--hub-run-name', help='Path segment under runs/.')
_OPT_HUB_PRIVATE = typer.Option('--hub-private/--hub-public', help='Create private Hub repo.')
_OPT_SHARD_DIR = typer.Option('--shard-dir', help='Existing encode-shards directory.')
_OPT_OVERLAP = typer.Option('--overlap-per-letter', help='Partner sample size per letter.')
_OPT_SAMPLE = typer.Option('--sample', help='HUPD JSON files to tokenize for the histogram.')


def _parse_recall_k(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(',') if part.strip()]
    if not parts:
        raise typer.BadParameter('at least one --recall-k value is required')
    return tuple(int(part) for part in parts)


def _host_from_yaml(path: Path) -> HostSpec:
    raw: Any = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(raw, dict):
        return HostSpec()
    host_raw = raw.get('host')
    if not isinstance(host_raw, dict):
        return HostSpec()
    return HostSpec.model_validate(host_raw)


def _optional_updates(**kwargs: object) -> dict[str, object]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _maybe_publish_eval(eval_config: CollisionEvalConfig, out: Path) -> str | None:
    if not eval_config.publish_to_hub:
        return None
    repo_id = (eval_config.hf_hub_repo_id or '').strip()
    if not repo_id:
        raise typer.BadParameter('publish_to_hub requires --hub-repo-id or YAML hf_hub_repo_id')
    token = eval_config.hf_token if eval_config.hf_token is not None else SecretStr('')
    published = publish_local_run_to_hf(
        out,
        repo_id=repo_id,
        run_name=eval_config.hf_hub_run_name,
        hf_token=token,
        private=bool(eval_config.hf_hub_private),
    )
    return str(published.run_uri)


@app.callback()
def _cli() -> None:
    """Patent collision detection evaluation entry."""


@app.command('eval')
def run_eval(
    *,
    dataset: Annotated[Path | None, _OPT_DATASET] = None,
    hupd_dir: Annotated[Path | None, _OPT_HUPD_DIR] = None,
    host: Annotated[str | None, _OPT_HOST] = None,
    d_model: Annotated[int | None, _OPT_D_MODEL] = None,
    recall_k: Annotated[str | None, _OPT_RECALL_K] = None,
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    output_dir: Annotated[Path | None, _OPT_OUTPUT_DIR] = None,
    publish_to_hub: Annotated[bool | None, _OPT_PUBLISH] = None,
    hub_repo_id: Annotated[str | None, _OPT_HUB_REPO] = None,
    hub_run_name: Annotated[str | None, _OPT_HUB_RUN] = None,
    hub_private: Annotated[bool | None, _OPT_HUB_PRIVATE] = None,
) -> None:
    """Rank the n-corpus by saturation covering and write covering.json."""
    config_path = config if config is not None else DEFAULT_COLLISION_EVAL_PATH
    updates = _optional_updates(
        recall_k=_parse_recall_k(recall_k) if recall_k is not None else None,
        dataset=str(dataset) if dataset is not None else None,
        hupd_dir=str(hupd_dir) if hupd_dir is not None else None,
        output_dir=str(output_dir) if output_dir is not None else None,
        publish_to_hub=publish_to_hub,
        hf_hub_repo_id=hub_repo_id,
        hf_hub_run_name=hub_run_name,
        hf_hub_private=hub_private,
    )
    eval_config = CollisionEvalConfig.from_yaml(config_path)
    if updates:
        eval_config = eval_config.model_copy(update=updates)

    dataset_path = Path(eval_config.dataset) if eval_config.dataset else None
    hupd_path = Path(eval_config.hupd_dir) if eval_config.hupd_dir else None
    if dataset_path is None or not dataset_path.is_file():
        raise typer.BadParameter('pass --dataset or set dataset in YAML to an existing parquet')
    if hupd_path is None or not hupd_path.is_dir():
        raise typer.BadParameter('pass --hupd-dir or set hupd_dir in YAML to an existing directory')

    wiring = CollisionContainer(eval_config=eval_config)
    if eval_config.ssv_config or eval_config.checkpoint:
        ssv_config, model = wiring.trunk()
        wiring.ssv.config.override(providers.Object(ssv_config))
    else:
        yaml_host = _host_from_yaml(config_path)
        ssv_config = SsvTrainConfig(
            host=HostSpec(
                name=host if host is not None else yaml_host.name,
                d_model=d_model if d_model is not None else yaml_host.d_model,
                tokenizer_id=yaml_host.tokenizer_id,
                lora_target_modules=yaml_host.lora_target_modules,
            ),
            arch=ArchSpec(
                gnn_hidden=32,
                gnn_heads=4,
                gnn_layers=1,
                n_soft_tokens=4,
                entity_bank_size=16,
                soft_dim=32,
                max_length=64,
            ),
            runtime=RuntimeSpec(hf_token=eval_config.hf_token),
        )
        wiring.ssv.config.override(providers.Object(ssv_config))
        model = wiring.ssv.soft_trunk()
    result = run_collision_eval(
        eval_config,
        dataset_path=dataset_path,
        pair_source=EpoProcessorCitationPairSource(),
        hupd_dir=hupd_path,
        collator=wiring.ssv.eval_collator(),
        model=model,
        covering=wiring.covering(),
    )
    if result is None:
        raise typer.Exit(0)
    out = Path(eval_config.output_dir)
    report = write_collision_artefacts(
        out,
        report_from_splits(
            train=result.train,
            eval_result=result.eval,
            test=result.test,
            eval_config=eval_config,
            checkpoint=result.checkpoint,
            encoded_apps=result.encoded_apps,
        ),
        eval_config,
        explain=result.explain,
    )
    hub_uri = _maybe_publish_eval(eval_config, out)
    _log.info(
        'collision.eval.finished',
        hub_uri=hub_uri,
        detector=report.detector,
        encoded_apps=result.encoded_apps,
    )
    raise typer.Exit(0)


@app.command('rank')
def run_rank(
    *,
    shard_dir: Annotated[Path | None, _OPT_SHARD_DIR] = None,
    dataset: Annotated[Path | None, _OPT_DATASET] = None,
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    output_dir: Annotated[Path | None, _OPT_OUTPUT_DIR] = None,
) -> None:
    """Rank existing encode shards. Sweeps keep_grid when the YAML declares one."""
    eval_config = CollisionEvalConfig.from_yaml(
        config if config is not None else DEFAULT_COLLISION_EVAL_PATH
    )
    updates = _optional_updates(
        dataset=str(dataset) if dataset is not None else None,
        output_dir=str(output_dir) if output_dir is not None else None,
        encode_shards=str(shard_dir) if shard_dir is not None else None,
    )
    if updates:
        eval_config = eval_config.model_copy(update=updates)
    pairs = Path(eval_config.dataset) if eval_config.dataset else None
    shards = encode_shard_root(eval_config)
    if pairs is None or not pairs.is_file():
        raise typer.BadParameter('pass --dataset or set dataset in YAML to an existing parquet')
    if not shards.is_dir():
        raise typer.BadParameter('pass --shard-dir or set encode_shards to an existing directory')
    if eval_config.keep_grid:
        report = rank_keep_grid_from_paths(
            eval_config,
            dataset_path=pairs,
            pair_source=EpoProcessorCitationPairSource(),
            shard_dir=shards,
        )
        _log.info(
            'collision.rank.finished',
            cells=len(report.cells),
            encoded_apps=report.encoded_apps,
        )
        raise typer.Exit(0)
    ranked = rank_covering_from_shards(
        eval_config,
        dataset_path=pairs,
        pair_source=EpoProcessorCitationPairSource(),
        shard_dir=shards,
    )
    _log.info(
        'collision.rank.finished',
        eval_unpaid_x=ranked['eval'].unpaid_x,
        eval_unpaid_a=ranked['eval'].unpaid_a,
        eval_queries_x=ranked['eval'].queries_x,
    )
    raise typer.Exit(0)


@app.command('diagnose')
def run_diagnose(
    *,
    shard_dir: Annotated[Path | None, _OPT_SHARD_DIR] = None,
    dataset: Annotated[Path | None, _OPT_DATASET] = None,
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    output_dir: Annotated[Path | None, _OPT_OUTPUT_DIR] = None,
    overlap_per_letter: Annotated[int, _OPT_OVERLAP] = OVERLAP_PER_LETTER,
) -> None:
    """Score existing shards for paired X/A deltas and union-Y. No encode."""
    eval_config = CollisionEvalConfig.from_yaml(
        config if config is not None else DEFAULT_COLLISION_EVAL_PATH
    )
    shards = shard_dir if shard_dir is not None else Path(eval_config.output_dir) / 'encode-shards'
    pairs = (
        dataset
        if dataset is not None
        else (Path(eval_config.dataset) if eval_config.dataset else None)
    )
    out = output_dir if output_dir is not None else Path(eval_config.output_dir)
    if pairs is None or not pairs.is_file():
        raise typer.BadParameter('pass --dataset or set dataset in YAML to an existing parquet')
    if not shards.is_dir():
        raise typer.BadParameter('pass --shard-dir to an existing encode-shards directory')
    written = judge_from_paths(
        shard_dir=shards,
        dataset=pairs,
        output_dir=out,
        config=eval_config,
        overlap_per_letter=int(overlap_per_letter),
    )
    _log.info('collision.diagnose.finished', path=str(written))
    raise typer.Exit(0)


@app.command('measure-disclosure')
def run_measure_disclosure(
    *,
    hupd_dir: Annotated[Path | None, _OPT_HUPD_DIR] = None,
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    output_dir: Annotated[Path | None, _OPT_OUTPUT_DIR] = None,
    sample: Annotated[int, _OPT_SAMPLE] = DEFAULT_MEASURE_SAMPLE,
) -> None:
    """Histogram HUPD disclosure token length. Does not encode or train."""
    eval_config = CollisionEvalConfig.from_yaml(
        config if config is not None else DEFAULT_COLLISION_EVAL_PATH
    )
    hupd = (
        hupd_dir
        if hupd_dir is not None
        else (Path(eval_config.hupd_dir) if eval_config.hupd_dir else None)
    )
    out = output_dir if output_dir is not None else Path(eval_config.output_dir)
    if hupd is None or not hupd.is_dir():
        raise typer.BadParameter('pass --hupd-dir or set hupd_dir in YAML to an existing directory')
    ssv_path = Path(eval_config.ssv_config) if eval_config.ssv_config else None
    ssv = SsvTrainConfig.from_yaml(ssv_path)
    if eval_config.hf_token is not None:
        ssv = ssv.overlay({'hf_token': eval_config.hf_token})
    cache = None
    if eval_config.checkpoint:
        ckpt = Path(eval_config.checkpoint)
        candidate = (ckpt if ckpt.is_dir() else ckpt.parent) / 'hupd_path_index.txt'
        cache = candidate if candidate.is_file() else None
    written = measure_disclosure_from_paths(
        hupd_dir=hupd,
        output_dir=out,
        tokenizer=load_host_tokenizer(ssv),
        max_length=int(ssv.arch.max_length),
        sample=int(sample),
        seed=eval_config.split.seed,
        index_cache=cache,
    )
    _log.info('collision.disclosure.finished', path=str(written))
    raise typer.Exit(0)


@app.command('publish')
def publish_run(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    *,
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    hub_repo_id: Annotated[str | None, _OPT_HUB_REPO] = None,
    hub_run_name: Annotated[str | None, _OPT_HUB_RUN] = None,
    hub_private: Annotated[bool, _OPT_HUB_PRIVATE] = True,
) -> None:
    """Upload an on-disk collision eval directory to a Hugging Face dataset repo."""
    job = CollisionEvalConfig.from_yaml(
        config if config is not None else DEFAULT_COLLISION_EVAL_PATH,
    )
    repo_id = (hub_repo_id or job.hf_hub_repo_id or '').strip()
    if not repo_id:
        raise typer.BadParameter('pass --hub-repo-id or set hf_hub_repo_id in YAML')
    token = job.hf_token if job.hf_token is not None else SecretStr('')
    result = publish_local_run_to_hf(
        run_dir,
        repo_id=repo_id,
        run_name=hub_run_name or job.hf_hub_run_name,
        hf_token=token,
        private=hub_private,
    )
    _log.info(
        'collision.publish.finished',
        files_uploaded=result.files_uploaded,
        run_uri=result.run_uri,
    )
    raise typer.Exit(0)


def main(argv: list[str] | None = None) -> int:
    """Invoke the Typer app; return a process exit code."""
    try:
        app(prog_name='python -m ip_claim.collision', args=argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
