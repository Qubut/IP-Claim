"""Pytest collection for covering-objective experiments."""

from __future__ import annotations

import os
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
from dependency_injector import providers
from hypothesis import HealthCheck, settings
from patent_ate.termhood import TermhoodStore
from returns.result import Failure, Success
from torch import Tensor

from experiments.covering_objective.corpus import (
    HupdLoader,
    bind_termhood_attach,
    hupd_loader,
)
from experiments.covering_objective.encode_pool import (
    EncodeView,
    InventoryForward,
    bind_encode_view,
    clone_module_state,
    close_encode_pool,
    covering_pair_devices,
)
from experiments.covering_objective.ledgers import (
    GateRecorder,
    bind_require_gate,
    prepare_collection,
    record_call_report,
    stash_gate_measurement,
    write_session_ledgers,
)
from experiments.covering_objective.pilot import (
    CoveringPilotSpec,
    covering_cli_flag,
    covering_env_name,
)
from experiments.covering_objective.relational_arms import (
    endpoint_shuffle,
    endpoint_tables,
    relation_label_shuffle,
)
from experiments.covering_objective.verdicts import CoveringSchedule, PairGap, PairScore
from ip_claim.app.container.collision import CollisionContainer
from ip_claim.app.container.ssv import SsvContainer
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.cover import Covering
from ip_claim.collision.data.citation_pairs import CitationPair, EpoProcessorCitationPairSource
from ip_claim.collision.encode_job import CollisionEncodeRow
from ip_claim.collision.eval import resolve_explain_device
from ip_claim.ssv.collate import SoftMlmCollator
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.model import SoftTrunkModel

_REPO = Path(__file__).resolve().parents[2]
_PROD_YAML = _REPO / 'configs' / 'ssv_train.prod.yaml'


@pytest.fixture
def endpoint_counterfactuals() -> dict[str, Callable[..., object]]:
    """Deterministic endpoint and relation-label shuffles for five-arm readout."""
    return {
        'endpoint_shuffle': endpoint_shuffle,
        'relation_label_shuffle': relation_label_shuffle,
        'endpoint_tables': endpoint_tables,
    }


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose CoveringPilotSpec fields as ``--covering-*`` flags."""
    group = parser.getgroup('covering', 'covering objective knobs')
    _ = tuple(
        group.addoption(
            covering_cli_flag(path),
            default=None,
            help=f'Override covering pilot {path} ({covering_env_name(path)})',
        )
        for path in CoveringPilotSpec.option_paths()
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip live encodes without HUPD. Run hygiene before later gates."""
    prepare_collection(
        items,
        live_corpus=bool(os.environ.get('COVERING_HUPD_DIR', '').strip()),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Load the Hypothesis profile from the covering pilot envelope."""
    config.addinivalue_line('markers', 'experiment: covering-objective collection')
    config.addinivalue_line('markers', 'live: real SSV encode needs COVERING_HUPD_DIR')
    settings.register_profile(
        'algebraic_oracle',
        max_examples=CoveringPilotSpec.from_yaml(config=config).algebraic_examples,
        derandomize=True,
        database=None,
        deadline=None,
        suppress_health_check=(HealthCheck.function_scoped_fixture,),
    )
    settings.load_profile('algebraic_oracle')


@pytest.fixture(scope='session')
def covering_pilot(pytestconfig: pytest.Config) -> CoveringPilotSpec:
    """Validated draw and step budgets for this collection."""
    return CoveringPilotSpec.from_yaml(config=pytestconfig)


@pytest.fixture(scope='module')
def collision_container() -> CollisionContainer:
    """Eval composition root. Covering knobs stay on CollisionEvalConfig."""
    return CollisionContainer()


@pytest.fixture(scope='module')
def covering(collision_container: CollisionContainer) -> Covering:
    """Product covering Factory, not a hand-built CoveringKnobs graph."""
    return collision_container.covering()


@pytest.fixture(scope='session')
def ssv_job(covering_pilot: CoveringPilotSpec) -> SsvTrainConfig:
    """Production train contract at init: real host, no Lightning dump, no Ray."""
    yaml_env = os.environ.get('COVERING_TRAIN_YAML')
    listed = Path(yaml_env) if yaml_env else _PROD_YAML
    hupd_env = os.environ.get('COVERING_HUPD_DIR')
    hupd = Path(hupd_env) if hupd_env else None
    job = SsvTrainConfig.from_yaml_or_job(config_path=listed, hupd_dir=hupd)
    if 'tiny-random' in job.host.name or 'hf-internal-testing' in job.host.name:
        pytest.fail(f'non-SSV smoke host is not the unit: {job.host.name}')
    devices = covering_pair_devices()
    return job.overlay({
        'max_length': covering_pilot.encode_max_length,
        'ray': False,
        'batch_size': covering_pilot.batch_size,
        'num_devices': max(len(devices), 1),
        'use_gpu': len(devices) > 0 or torch.cuda.is_available(),
        'termhood_store_path': None,
    })


@pytest.fixture(scope='session')
def ssv_device(ssv_job: SsvTrainConfig) -> torch.device:
    """Keep the pytest parent off CUDA so the JATE draw never forks after a CUDA context."""
    if covering_pair_devices():
        return torch.device('cpu')
    return cast(torch.device, resolve_explain_device(use_gpu=bool(ssv_job.runtime.use_gpu)))


@pytest.fixture(scope='session')
def ssv_container(ssv_job: SsvTrainConfig) -> SsvContainer:
    """DI graph for the initialized trunk."""
    return SsvContainer(config=ssv_job)


@pytest.fixture(scope='session')
def ssv_model(
    ssv_container: SsvContainer,
    ssv_job: SsvTrainConfig,
    ssv_device: torch.device,
) -> SoftTrunkModel:
    """SoftTrunk with the job warm-start weights on the measurement device."""
    model = ssv_container.soft_trunk()
    init_path = Path(ssv_job.runtime.init_weights) if ssv_job.runtime.init_weights else None
    if init_path is not None:
        match ssv_container.init_weights(path=init_path):
            case Failure(message):
                raise RuntimeError(message)
            case Success():
                pass
    model.eval()
    model.to(ssv_device)
    return model


@pytest.fixture(scope='session')
def ssv_inventory() -> Inventory:
    """Shared late-assignment inventory for encode and functional steps."""
    return Inventory()


@pytest.fixture(scope='session')
def covering_collator(
    ssv_container: SsvContainer,
    ssv_model: SoftTrunkModel,
    covering_pilot: CoveringPilotSpec,
    ssv_job: SsvTrainConfig,
) -> Iterator[SoftMlmCollator]:
    """Train MLM collator Factory with rho overridden to the pilot operating point."""
    scheduled = covering_pilot.mask_schedule == 'scheduled'
    rho = float(ssv_job.mlm.rho_max) if scheduled else 0.0
    entropy = 1.0 if scheduled else 0.0
    with ssv_container.collator_rho.override(providers.Object(rho)):
        collator = ssv_container.train_collator()
        CoveringSchedule.refuse_inert_scheduled({
            'mask_schedule': covering_pilot.mask_schedule,
            'rho': float(collator.rho),
            'mlm_probability': float(collator.mlm_probability or 0.0),
        })
        prior = float(ssv_model.soft_vocab.entropy_scale)
        ssv_model.soft_vocab.set_entropy_scale(entropy)
        collator.bind_assignment_source(
            ssv_model.soft_vocab,
            ssv_model.host.get_input_embeddings(),
        )
        yield collator
        ssv_model.soft_vocab.set_entropy_scale(prior)


@pytest.fixture(scope='session')
def covering_schedule(
    covering_collator: SoftMlmCollator,
    ssv_model: SoftTrunkModel,
    covering_pilot: CoveringPilotSpec,
    ssv_job: SsvTrainConfig,
) -> CoveringSchedule:
    """Ledger snapshot of the overridden collator, not the operating-point API."""
    schedule = CoveringSchedule(
        kind=covering_pilot.mask_schedule,
        rho=float(covering_collator.rho),
        entropy_scale=float(ssv_model.soft_vocab.entropy_scale),
        mlm_probability=float(covering_collator.mlm_probability or 0.0),
        inject_scale=float(ssv_model.inject_scale),
        global_step=0,
        rho_warmup_steps=int(ssv_job.mlm.rho_warmup_steps),
    )
    schedule.require_honest()
    return schedule


@pytest.fixture(scope='session')
def termhood_store() -> TermhoodStore | None:
    """Existing committed store, or None. Never starts an occupy crawl."""
    env = os.environ.get('COVERING_TERMHOOD', '').strip()
    sibling = _REPO.parent.parent / 'patent-ate' / '.cache' / 'hupd-termhood'
    root = next(
        (
            path
            for path in ((Path(env) if env else None), Path('/data/termhood'), sibling)
            if path is not None and (path / 'termhood.meta.json').is_file()
        ),
        None,
    )
    return None if root is None else TermhoodStore.open(root)


@pytest.fixture(scope='session')
def inventory_exporter(
    ssv_model: SoftTrunkModel,
    ssv_inventory: Inventory,
) -> InventoryForward:
    """Shared last-layer encode on the pinned trunk device."""
    return InventoryForward(ssv_model, ssv_inventory)


@pytest.fixture(scope='session')
def encode_view(
    covering_collator: SoftMlmCollator,
    inventory_exporter: InventoryForward,
    ssv_inventory: Inventory,
    covering_pilot: CoveringPilotSpec,
) -> EncodeView:
    """Last-layer inventory encode. Two visible cards use Ray actors."""
    return bind_encode_view(
        inventory_exporter,
        covering_collator,
        ssv_inventory,
        width=covering_pilot.batch_size,
        seed=covering_pilot.seed,
    )


@pytest.fixture(scope='session')
def load_hupd_draw(
    ssv_job: SsvTrainConfig,
    covering_pilot: CoveringPilotSpec,
) -> HupdLoader:
    """Seeded HUPD draw. Termhood scores land later via attach, not here."""
    return hupd_loader(
        ssv_job,
        hygiene_n=covering_pilot.hygiene_n,
        seed=covering_pilot.seed,
    )


@pytest.fixture(scope='session')
def attach_termhood(
    ssv_model: SoftTrunkModel,
    termhood_store: TermhoodStore | None,
) -> Callable[[Sequence[str]], int]:
    """Publish requested termhood scores on the session ingress. Zero when no store."""
    publish = bind_termhood_attach(ssv_model, termhood_store)

    def attach(keys: Sequence[str]) -> int:
        published = publish(keys)
        close_encode_pool()
        return published

    return attach


@pytest.fixture
def require_gate(request: pytest.FixtureRequest) -> Callable[..., None]:
    """Skip when a collected live gate has not run, or a cached prior gate failed."""
    return bind_require_gate(request)


@pytest.fixture
def announce_gate(
    request: pytest.FixtureRequest,
    covering_schedule: CoveringSchedule,
) -> GateRecorder:
    """Attach measurements to the pytest item. The report hook writes the artefact."""

    def announce(
        slug: str,
        payload: Mapping[str, object],
        *,
        culprit: str,
        cache: Mapping[str, object] | None = None,
        **props: object,
    ) -> None:
        measurement = {
            **covering_schedule.reading(),
            **dict(payload),
            'culprit': culprit or 'none',
        }
        CoveringSchedule.refuse_inert_scheduled(measurement)
        stash_gate_measurement(
            request.node,
            slug,
            measurement,
            cache,
            (
                *covering_schedule.reading().items(),
                *props.items(),
                ('culprit', culprit or 'none'),
            ),
        )

    return announce


@pytest.fixture
def restored_trunk(ssv_model: SoftTrunkModel) -> Iterator[SoftTrunkModel]:
    """Restore session trunk weights and grad flags after an optimizing gate."""
    snapshot = clone_module_state(ssv_model)
    flags = tuple(parameter.requires_grad for parameter in ssv_model.parameters())
    yield ssv_model
    close_encode_pool()
    ssv_model.load_state_dict(snapshot, strict=False)
    _ = tuple(
        parameter.requires_grad_(flag)
        for parameter, flag in zip(ssv_model.parameters(), flags, strict=True)
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Write measurements and gate cache after the call, not from the test body."""
    report = yield
    return record_call_report(item, call, report)


@pytest.fixture
def pair_score(
    encode_view: EncodeView,
    covering: Covering,
) -> PairScore:
    """Differentiable normalized unpaid gap on product claim and disclosure rows."""

    def score(
        claim_rows: Sequence[CollisionEncodeRow],
        disc_rows: Sequence[CollisionEncodeRow],
        *,
        retain_graph: bool = False,
    ) -> Tensor:
        claims, _trace = encode_view(claim_rows, as_claim=True, retain_graph=retain_graph)
        supply, _trace = encode_view(disc_rows, as_claim=False, retain_graph=retain_graph)
        return cast(
            Tensor,
            covering.normalized_unpaid_gap(covering.pair_table(claims, supply)),
        )

    return score


@pytest.fixture
def pair_gap(pair_score: PairScore) -> PairGap:
    """Finite float reading of ``pair_score``, or None when the reducer is NaN."""

    def gap(
        claim_rows: Sequence[CollisionEncodeRow],
        disc_rows: Sequence[CollisionEncodeRow],
        *,
        retain_graph: bool = False,
    ) -> float | None:
        value = pair_score(claim_rows, disc_rows, retain_graph=retain_graph)
        return None if torch.isnan(value) else float(value.item())

    return gap


@pytest.fixture
def citation_pairs() -> tuple[CitationPair, ...]:
    """ST.14 pairs from COVERING_CITATIONS or the eval dataset path."""
    env = os.environ.get('COVERING_CITATIONS', '').strip()
    configured = CollisionEvalConfig.from_yaml().dataset
    path = Path(env) if env else Path(configured or '')
    return cast(tuple[CitationPair, ...], EpoProcessorCitationPairSource().load_pairs(path))


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Write Polars/Great Tables ledger from this pytest session's reports."""
    write_session_ledgers(terminalreporter, exitstatus, config)
