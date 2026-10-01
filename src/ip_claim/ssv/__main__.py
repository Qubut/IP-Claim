"""SSV train and ablation CLI for ``python -m ip_claim.ssv``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import structlog
import typer
from pydantic import SecretStr
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from ip_claim.shared.hf_hub import publish_local_run_to_hf
from ip_claim.ssv.ablation import INJECT_ABLATION_CAVEAT, run_graph_prefix_ablation
from ip_claim.ssv.config import (
    DEFAULT_RAY_SCALING_PATH,
    DEFAULT_SSV_TRAIN_PATH,
    HostSpec,
    RayScalingSpec,
    SsvTrainConfig,
)
from ip_claim.ssv.ingress_probe import IngressProbeRequest, run_ingress_probe, write_probe
from ip_claim.ssv.inspect import run_graph_inspect
from ip_claim.ssv.train import train_func

_log = structlog.get_logger(__name__)

app = typer.Typer(
    name='ssv',
    help='SSV train and graph-prefix ablation CLI.',
    no_args_is_help=True,
    add_completion=False,
)

_OPT_LOCAL = typer.Option('--local', help='Single-process Lightning fit.')
_OPT_RAY = typer.Option(
    '--ray',
    help='Ray TorchTrainer (ScalingConfig from package ssv_ray.yaml).',
)
_OPT_MAX_STEPS = typer.Option(
    '--max-steps',
    help='Safety step ceiling; fit may stop earlier on inventory track.',
)
_OPT_CHECKPOINT_DIR = typer.Option('--checkpoint-dir', help='Directory for ssv.ckpt.')
_OPT_BATCH_SIZE = typer.Option('--batch-size')
_OPT_HOST = typer.Option('--host', help='HF masked-LM id (must match host.d_model).')
_OPT_D_MODEL = typer.Option('--d-model', help='Host hidden size.')
_OPT_CONFIG = typer.Option(
    '--config',
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    help='OmegaConf YAML (defaults: package ssv_train.yaml).',
)
_OPT_HUPD_DIR = typer.Option(
    '--hupd-dir',
    help='HUPD JSON root (default: fixtures, or YAML hupd_dir).',
)
_OPT_HUPD_LIMIT = typer.Option('--hupd-limit', help='Max JSON files to load.')
_OPT_PUBLISH = typer.Option('--publish-to-hub', help='Upload run artefacts after fit.')
_OPT_HUB_REPO = typer.Option('--hub-repo-id', help='HF dataset repo id for runs/.')
_OPT_HUB_RUN = typer.Option('--hub-run-name', help='Path segment under runs/.')
_OPT_HUB_PRIVATE = typer.Option('--hub-private/--hub-public', help='Create private Hub repo.')


@app.callback()
def _cli() -> None:
    """SSV foundation trunk training and graph-prefix ablation entry."""


def _optional_updates(**kwargs: object) -> dict[str, object]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _merge_train_config(
    *,
    config_path: Path | None,
    ray_mode: bool,
    use_gpu: bool | None,
    overrides: dict[str, object],
) -> SsvTrainConfig:
    base = SsvTrainConfig.from_yaml(
        config_path if config_path is not None else DEFAULT_SSV_TRAIN_PATH
    )
    host_name = overrides.pop('host_name', None)
    host_d_model = overrides.pop('host_d_model', None)
    updates: dict[str, object] = {'ray': ray_mode, **overrides}
    if use_gpu is not None:
        updates['use_gpu'] = use_gpu
    if host_name is not None or host_d_model is not None:
        name = str(host_name) if isinstance(host_name, str) else base.host.name
        dim = host_d_model if isinstance(host_d_model, int) else base.host.d_model
        updates['host'] = HostSpec(
            name=name,
            d_model=dim,
            tokenizer_id=base.host.tokenizer_id,
            lora_target_modules=base.host.lora_target_modules,
        )
    return base.overlay(updates)


@app.command()
def train(
    *,
    local: Annotated[bool, _OPT_LOCAL] = False,
    ray: Annotated[bool, _OPT_RAY] = False,
    max_steps: Annotated[int | None, _OPT_MAX_STEPS] = None,
    checkpoint_dir: Annotated[Path | None, _OPT_CHECKPOINT_DIR] = None,
    batch_size: Annotated[int | None, _OPT_BATCH_SIZE] = None,
    host: Annotated[str | None, _OPT_HOST] = None,
    d_model: Annotated[int | None, _OPT_D_MODEL] = None,
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    hupd_dir: Annotated[Path | None, _OPT_HUPD_DIR] = None,
    hupd_limit: Annotated[int | None, _OPT_HUPD_LIMIT] = None,
    publish_to_hub: Annotated[bool | None, _OPT_PUBLISH] = None,
    hub_repo_id: Annotated[str | None, _OPT_HUB_REPO] = None,
    hub_run_name: Annotated[str | None, _OPT_HUB_RUN] = None,
    hub_private: Annotated[bool | None, _OPT_HUB_PRIVATE] = None,
) -> None:
    """Run SSV training locally or on Ray."""
    if local == ray:
        raise typer.BadParameter('exactly one of --local or --ray is required')

    overrides = _optional_updates(
        max_steps=max_steps,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        batch_size=batch_size,
        host_name=host,
        host_d_model=d_model,
        hupd_dir=str(hupd_dir) if hupd_dir is not None else None,
        hupd_limit=hupd_limit,
        publish_to_hub=publish_to_hub,
        hf_hub_repo_id=hub_repo_id,
        hf_hub_run_name=hub_run_name,
        hf_hub_private=hub_private,
    )
    if local:
        path = train_func(
            _merge_train_config(
                config_path=config,
                ray_mode=False,
                use_gpu=None,
                overrides=overrides,
            ),
        )
        _log.info('ssv.train.local_finished', checkpoint=str(path))
        raise typer.Exit(0)

    scaling = RayScalingSpec.from_yaml(DEFAULT_RAY_SCALING_PATH)
    train_config = _merge_train_config(
        config_path=config,
        ray_mode=True,
        use_gpu=bool(scaling.use_gpu),
        overrides=overrides,
    )
    result = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config.model_dump(mode='json'),
        scaling_config=ScalingConfig(
            num_workers=int(scaling.num_workers),
            use_gpu=bool(scaling.use_gpu),
        ),
    ).fit()
    _log.info('ssv.train.ray_finished', metrics=result.metrics)
    raise typer.Exit(0)


@app.command('ablate-graph-prefix')
def ablate_graph_prefix(
    *,
    checkpoint: Annotated[
        Path,
        typer.Option(
            '--checkpoint',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help='Lightning checkpoint whose model.* tensors are probed.',
        ),
    ],
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    hupd_dir: Annotated[Path | None, _OPT_HUPD_DIR] = None,
    hupd_limit: Annotated[int | None, _OPT_HUPD_LIMIT] = None,
    n_batches: Annotated[
        int,
        typer.Option('--n-batches', min=1, help='Held-out batches to average.'),
    ] = 4,
    batch_size: Annotated[int, typer.Option('--batch-size', min=1)] = 4,
    rho: Annotated[
        float,
        typer.Option(
            '--rho',
            min=0.0,
            max=1.0,
            help='Graph-guided mask mix for the probe batches.',
        ),
    ] = 0.31,
    stratify_by_mass: Annotated[
        bool,
        typer.Option(
            '--stratify-by-mass',
            help='Also split masked-token deltas by assignment-mass median.',
        ),
    ] = False,
) -> None:
    """Paired eval-mode MLM NLL: real graph prefix versus a zeroed prefix."""
    result = run_graph_prefix_ablation(
        checkpoint=checkpoint,
        config_path=config,
        hupd_dir=hupd_dir,
        hupd_limit=hupd_limit,
        n_batches=n_batches,
        batch_size=batch_size,
        rho=rho,
        stratify_by_mass=stratify_by_mass,
    )
    typer.echo(result.ablation.as_table())
    if result.mass_stratification is not None:
        typer.echo(result.mass_stratification.as_table())
    _log.info(
        'ssv.ablate.graph_prefix',
        checkpoint=str(checkpoint),
        n_batches=len(result.ablation.batches),
        mean_delta=result.ablation.mean_delta,
        sd_delta=result.ablation.sd_delta,
        n_sign_flips=result.ablation.n_sign_flips,
        n_positive=result.ablation.n_positive,
        n_negative=result.ablation.n_negative,
    )
    raise typer.Exit(0)


@app.command('ablate-graph-inject')
def ablate_graph_inject(
    *,
    checkpoint: Annotated[
        Path,
        typer.Option(
            '--checkpoint',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help='Lightning checkpoint whose model.* tensors are probed.',
        ),
    ],
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    hupd_dir: Annotated[Path | None, _OPT_HUPD_DIR] = None,
    hupd_limit: Annotated[int | None, _OPT_HUPD_LIMIT] = None,
    n_batches: Annotated[
        int,
        typer.Option('--n-batches', min=1, help='Held-out batches to average.'),
    ] = 4,
    batch_size: Annotated[int, typer.Option('--batch-size', min=1)] = 4,
    rho: Annotated[
        float,
        typer.Option(
            '--rho',
            min=0.0,
            max=1.0,
            help='Graph-guided mask mix for the probe batches.',
        ),
    ] = 0.31,
) -> None:
    """Paired eval-mode MLM NLL: injected slots versus the same slots zeroed."""
    result = run_graph_prefix_ablation(
        checkpoint=checkpoint,
        config_path=config,
        hupd_dir=hupd_dir,
        hupd_limit=hupd_limit,
        n_batches=n_batches,
        batch_size=batch_size,
        rho=rho,
        channel='inject',
    )
    typer.echo(result.ablation.as_table())
    _log.info(
        'ssv.ablate.graph_inject',
        checkpoint=str(checkpoint),
        n_batches=len(result.ablation.batches),
        mean_delta=result.ablation.mean_delta,
        sd_delta=result.ablation.sd_delta,
        n_sign_flips=result.ablation.n_sign_flips,
        n_positive=result.ablation.n_positive,
        n_negative=result.ablation.n_negative,
        caveat=INJECT_ABLATION_CAVEAT,
    )
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
    """Upload an on-disk SSV run directory to a Hugging Face dataset repo."""
    job = SsvTrainConfig.from_yaml(config if config is not None else DEFAULT_SSV_TRAIN_PATH)
    repo_id = (hub_repo_id or job.runtime.hf_hub_repo_id or '').strip()
    if not repo_id:
        raise typer.BadParameter('pass --hub-repo-id or set hf_hub_repo_id in YAML')
    token = job.runtime.hf_token if job.runtime.hf_token is not None else SecretStr('')
    result = publish_local_run_to_hf(
        run_dir,
        repo_id=repo_id,
        run_name=hub_run_name or job.runtime.hf_hub_run_name,
        hf_token=token,
        private=hub_private,
    )
    _log.info(
        'ssv.publish.finished',
        files_uploaded=result.files_uploaded,
        run_uri=result.run_uri,
    )
    raise typer.Exit(0)


@app.command('inspect-graph')
def inspect_graph(
    *,
    checkpoint: Annotated[
        Path,
        typer.Option(
            '--checkpoint',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help='Lightning checkpoint whose overlay graph is rendered.',
        ),
    ],
    output: Annotated[
        Path,
        typer.Option('--output', help='Directory for inspect.html.'),
    ],
    checkpoint_b: Annotated[
        Path | None,
        typer.Option(
            '--checkpoint-b',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help='Later checkpoint; same patents are compared for code drift.',
        ),
    ] = None,
    config: Annotated[Path | None, _OPT_CONFIG] = None,
    hupd_dir: Annotated[Path | None, _OPT_HUPD_DIR] = None,
    hupd_limit: Annotated[
        int,
        typer.Option('--hupd-limit', min=1, help='How many HUPD JSON files to peek.'),
    ] = 4,
) -> None:
    """Render the soft-entity overlay a checkpoint induces on a small HUPD peek."""
    report = run_graph_inspect(
        checkpoint=checkpoint,
        checkpoint_b=checkpoint_b,
        output_dir=output,
        config_path=config,
        hupd_dir=hupd_dir,
        hupd_limit=hupd_limit,
    )
    fired = tuple(row.name for row in report.earlier.defects if row.fired)
    typer.echo(str(output / 'inspect.html'))
    _log.info(
        'ssv.inspect.graph',
        checkpoint=str(checkpoint),
        checkpoint_b=None if checkpoint_b is None else str(checkpoint_b),
        n_documents=len(report.earlier.documents),
        defects=fired,
        stable_fraction=report.stable_fraction,
    )
    raise typer.Exit(0)


@app.command('probe-ingress')
def probe_ingress(
    *,
    output: Annotated[
        Path,
        typer.Option('--output', help='Directory for probe.json and probe.html.'),
    ],
    occupy_limit: Annotated[
        int,
        typer.Option('--occupy-limit', min=1, help='How many extract filings to occupy.'),
    ] = 1000,
    seed: Annotated[
        int,
        typer.Option('--seed', help='Seed stored on the occupancy report.'),
    ] = 20260830,
    termhood: Annotated[
        Path,
        typer.Option(
            '--termhood',
            help='Existing termhood artifact directory or legacy JSON. Required for occupy.',
        ),
    ],
    termhood_docs: Annotated[
        int,
        typer.Option('--termhood-docs', min=1, help='Document count for a product dump.'),
    ] = 100_000,
    extract_dir: Annotated[
        Path | None,
        typer.Option(
            '--extract',
            help='Compact extract Parquet directory. Defaults to extract/ next to termhood.',
        ),
    ] = None,
) -> None:
    """Join extract parquet to committed termhood. Writes probe.json and probe.html."""
    request = IngressProbeRequest(
        output_dir=output,
        occupy_limit=occupy_limit,
        seed=seed,
        termhood_path=termhood,
        termhood_docs=termhood_docs,
        extract_dir=extract_dir,
    )
    try:
        report = run_ingress_probe(request)
    except ValueError as exc:
        _log.error('ssv.probe.ingress.failed', error=str(exc))
        raise typer.Exit(1) from exc
    html_path = write_probe(report, output)
    typer.echo(str(html_path))
    _log.info(
        'ssv.probe.ingress',
        output=str(html_path),
        n_patents=report.n_patents,
        n_occupy=report.n_occupy,
        n_pool=report.n_pool,
        n_empty=report.n_empty,
        occupy_rate=report.occupy_rate,
        n_unique_labels=report.n_unique_labels,
        n_stop_leaks=report.n_stop_leaks,
    )
    raise typer.Exit(0)


def main(argv: list[str] | None = None) -> int:
    """Invoke the Typer app; return a process exit code."""
    try:
        app(prog_name='python -m ip_claim.ssv', args=argv)
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
