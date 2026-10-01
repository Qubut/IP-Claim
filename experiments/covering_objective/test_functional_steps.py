"""Backward reachability and one-group ``functional_call`` steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import pytest
import torch
from ray.actor import ActorHandle
from torch import Tensor, nn
from torch.nn.parallel import replicate
from torch_geometric.data import HeteroData

from experiments.covering_objective.corpus import HupdLoader
from experiments.covering_objective.encode_pool import (
    PARAM_GROUPS,
    InventoryEncodePool,
    InventoryForward,
    bind_encode_view,
    cat_checkpointed_chunks,
    close_encode_pool,
    covering_pair_devices,
    owned_parameters,
)
from experiments.covering_objective.ledgers import GateRecorder
from experiments.covering_objective.pilot import CoveringPilotSpec
from ip_claim.collision.cover import Covering
from ip_claim.collision.encode_job import CollisionEncodeRow, collate_encode_rows
from ip_claim.ssv.collate import SoftMlmCollator, SoftMlmExample
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.model import SoftTrunkModel

pytestmark = pytest.mark.experiment


def directional_step_report(
    name: str,
    *,
    owned: Mapping[str, Tensor],
    params: Mapping[str, Tensor],
    base: Tensor,
    residual: Callable[[Mapping[str, Tensor]], Tensor],
    steps: Sequence[float],
) -> dict[str, float | bool | str | None]:
    """One concatenated unit step per size; keep the first same-sign size."""
    if not owned:
        return {'name': name, 'reachable': False, 'same_sign': False, 'grad_norm': 0.0}
    grads = cast(
        tuple[Tensor | None, ...],
        torch.autograd.grad(
            base,
            tuple(owned.values()),
            allow_unused=True,
            retain_graph=True,
        ),
    )
    used = tuple(
        (key, value, grad)
        for (key, value), grad in zip(owned.items(), grads, strict=True)
        if grad is not None
    )
    if not used:
        return {'name': name, 'reachable': False, 'same_sign': False, 'grad_norm': 0.0}
    scale = torch.linalg.vector_norm(
        torch.cat([grad.reshape(-1) for _key, _param, grad in used])
    ).clamp_min(1e-8)
    grad_norm = float(scale.item())

    def at_step(step: float) -> dict[str, float | bool | str | None]:
        stepped = {
            **params,
            **{key: value - step * grad / scale for key, value, grad in used},
        }
        exact = residual(stepped)
        predicted = -step * scale
        same_sign = bool(
            torch.isfinite(exact)
            and torch.isfinite(predicted)
            and float((exact - base).item()) * float(predicted.item()) >= 0.0
        )
        return {
            'name': name,
            'reachable': grad_norm > 0.0,
            'same_sign': same_sign,
            'grad_norm': grad_norm,
            'step': step,
            'delta': None if not torch.isfinite(exact) else float((exact - base).item()),
        }

    rows = tuple(at_step(step) for step in steps)
    return next((row for row in rows if row['same_sign']), rows[-1])


def group_step_report(
    name: str,
    *,
    exporter: InventoryForward,
    params: Mapping[str, Tensor],
    base: Tensor,
    residual: Callable[[Mapping[str, Tensor]], Tensor],
    steps: Sequence[float],
) -> dict[str, float | bool | str | None]:
    """One-group ``autograd.grad`` plus concatenated ``functional_call`` steps."""
    return directional_step_report(
        name,
        owned=owned_parameters(exporter, name),
        params=params,
        base=base,
        residual=residual,
        steps=steps,
    )


class TestFunctionalSteps:
    """Host-free pair-device and functional-call identities."""

    def test_concatenated_unit_step_agrees_on_two_tensor_quadratic(self) -> None:
        """Joint unit step descends a two-weight quadratic; per-tensor unit steps climb."""

        class Pair(nn.Module):
            a: Tensor
            b: Tensor

            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Parameter(torch.tensor([3.0, 0.0]))
                self.b = nn.Parameter(torch.tensor([0.0, 4.0]))

        pair = Pair()
        params = dict(pair.named_parameters())

        def residual(param_map: Mapping[str, Tensor]) -> Tensor:
            return cast(Tensor, 0.5 * sum(value.square().sum() for value in param_map.values()))

        base = residual(params)
        step = 8.0
        report = directional_step_report(
            'pair',
            owned=params,
            params=params,
            base=base,
            residual=residual,
            steps=(step,),
        )
        replay = residual(params)
        grads = torch.autograd.grad(replay, tuple(params.values()))
        used = tuple(
            (key, value, grad)
            for (key, value), grad in zip(params.items(), grads, strict=True)
            if grad is not None
        )
        per_tensor = {
            **params,
            **{key: value - step * grad / grad.norm().clamp_min(1e-8) for key, value, grad in used},
        }
        per_tensor_delta = float((residual(per_tensor) - replay).item())
        assert per_tensor_delta > 0.0
        assert report['reachable']
        assert report['same_sign']
        assert report['delta'] is not None
        assert float(report['delta']) < 0.0

    def test_replicate_requires_every_tensor_on_home_device(self) -> None:
        """``replicate`` rejects a CPU buffer; ``Module.to`` on the home device clears it."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip('needs two CUDA devices')
        home, other = torch.device('cuda', 0), torch.device('cuda', 1)

        class Pair(nn.Module):
            weight: Tensor
            occupied_floor: Tensor

            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, device=home))
                self.occupied_floor = nn.Buffer(torch.tensor(3))

            def forward(self, dummy: Tensor) -> Tensor:
                return dummy * self.weight.sum() + self.occupied_floor

        module = Pair()
        with pytest.raises(RuntimeError, match='devices\\[0\\]'):
            _ = replicate(module, [home, other], detach=False)
        _ = module.to(home)
        replicas = replicate(module, [home, other], detach=False)
        assert len(replicas) == 2
        assert replicas[1].occupied_floor.device == other

    def test_covering_pair_devices_stay_on_first_two_cards(self) -> None:
        """Live pair scoring may use cuda:0 and cuda:1 only."""
        devices = covering_pair_devices()
        assert len(devices) <= 2
        assert all(device.index in {0, 1, None} for device in devices)

    def test_covering_pair_devices_refuse_card_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A visible list that names physical card 2 fails closed."""
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,2')
        with pytest.raises(pytest.fail.Exception, match='2 and 3'):
            covering_pair_devices()

    def test_covering_pair_devices_require_two_cards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The covering job fails closed when only one CUDA device is visible."""
        monkeypatch.setenv('COVERING_REQUIRE_PAIR_DEVICES', '1')
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
        with pytest.raises(pytest.fail.Exception, match='two visible'):
            covering_pair_devices()

    def test_one_visible_card_opens_one_actor_on_encode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A letter arm with one remapped card still uses the Ray encode pool."""
        monkeypatch.delenv('COVERING_REQUIRE_PAIR_DEVICES', raising=False)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
        opened: list[int] = []

        class FakePool:
            def encode(self, *_args: object, **_kwargs: object) -> str:
                return 'pooled'

        def fake_bound(_exporter: object, _collate: object, count: int) -> FakePool:
            opened.append(count)
            return FakePool()

        monkeypatch.setattr(
            'experiments.covering_objective.encode_pool.bound_encode_pool',
            fake_bound,
        )

        class FakeDevice:
            type = 'cpu'

        class FakeIngress:
            def candidates(self, texts: object) -> tuple[()]:
                return ()

        class FakeTrunk:
            graph_ingress = FakeIngress()

        class FakeExporter:
            device = FakeDevice()
            trunk = FakeTrunk()

        encode = bind_encode_view(
            cast(InventoryForward, FakeExporter()),
            cast(SoftMlmCollator, object()),
            Inventory(),
            width=2,
            seed=0,
        )
        assert encode((), as_claim=True) == 'pooled'
        assert opened == [1]
        assert [device.index for device in covering_pair_devices()] == [0]

    def test_covering_pair_devices_read_env_without_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An advertised pair is counted from the env, not by loading the CUDA driver."""
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
        monkeypatch.delenv('COVERING_REQUIRE_PAIR_DEVICES', raising=False)

        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError('cuda driver')

        monkeypatch.setattr(torch.cuda, 'is_available', boom)
        monkeypatch.setattr(torch.cuda, 'device_count', boom)
        devices = covering_pair_devices()
        assert [device.index for device in devices] == [0, 1]

    def test_bind_encode_view_keeps_required_pair_lazy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A required pair still waits for the first encode before Ray starts."""
        monkeypatch.setenv('COVERING_REQUIRE_PAIR_DEVICES', '1')
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
        opened: list[int] = []

        def fake_bound(_exporter: object, _collate: object, count: int) -> object:
            opened.append(count)
            return object()

        monkeypatch.setattr(
            'experiments.covering_objective.encode_pool.bound_encode_pool',
            fake_bound,
        )

        def unused_collate(examples: object) -> object:
            raise AssertionError(examples)

        bind_encode_view(
            cast(InventoryForward, object()),
            cast(SoftMlmCollator, unused_collate),
            Inventory(),
            width=2,
            seed=0,
        )
        assert opened == []

    def test_bind_encode_view_keeps_optional_pair_lazy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the pair requirement the pool waits for the first encode."""
        monkeypatch.delenv('COVERING_REQUIRE_PAIR_DEVICES', raising=False)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
        opened: list[int] = []

        def fake_bound(_exporter: object, _collate: object, count: int) -> object:
            opened.append(count)
            return object()

        monkeypatch.setattr(
            'experiments.covering_objective.encode_pool.bound_encode_pool',
            fake_bound,
        )

        def unused_collate(examples: object) -> object:
            raise AssertionError(examples)

        bind_encode_view(
            cast(InventoryForward, object()),
            cast(SoftMlmCollator, unused_collate),
            Inventory(),
            width=2,
            seed=0,
        )
        assert opened == []

    def test_close_encode_pool_before_first_encode_is_idle(self) -> None:
        """Attach may drop actors that were never started."""
        close_encode_pool()
        close_encode_pool()

    def test_encode_pool_waits_on_all_actor_refs_together(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both encode actors are submitted before the driver waits."""
        submitted: list[int] = []

        class FakeRemote:
            def __init__(self, index: int) -> None:
                self.index = index

            def remote(self, *_args: object, **_kwargs: object) -> int:
                submitted.append(self.index)
                return self.index

        class FakeActor:
            def __init__(self, index: int) -> None:
                self.encode = FakeRemote(index)

        def fake_get(refs: object) -> list[object]:
            listed = list(cast(list[int], refs))
            assert listed == [0, 1]
            assert submitted == [0, 1]
            return [
                (('a',), torch.zeros(1, 1), {}, {}),
                (('b',), torch.zeros(1, 1), {}, {}),
            ]

        monkeypatch.setattr('experiments.covering_objective.encode_pool.ray.get', fake_get)
        pool = InventoryEncodePool(
            actors=cast(
                tuple[ActorHandle[object], ...],
                (FakeActor(0), FakeActor(1)),
            ),
            owns_ray=False,
        )
        rows = (
            CollisionEncodeRow(
                application_number='a',
                example=SoftMlmExample(text='a', graph=HeteroData()),
                claim_blob='a',
            ),
            CollisionEncodeRow(
                application_number='b',
                example=SoftMlmExample(text='b', graph=HeteroData()),
                claim_blob='b',
            ),
        )
        intensity, _trace = pool.encode(rows, kind='full', full_trace=False, width=1)
        assert intensity.size(0) == 2

    def test_checkpointed_chunks_match_full_cat_and_reach_weight(self) -> None:
        """Chunk concat equals the full stack; backward recomputes and reaches the weight."""
        weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        chunks = ((0, 1), (2, 3), (4,))
        calls = {'n': 0}

        def encode_one(chunk: Sequence[int]) -> Tensor:
            calls['n'] += 1
            scale = torch.tensor([float(index + 1) for index in chunk]).unsqueeze(-1)
            return scale * weight

        full = torch.cat(tuple(encode_one(chunk) for chunk in chunks), dim=0)
        calls['n'] = 0
        joined = cat_checkpointed_chunks(encode_one, chunks, torch.device('cpu'))
        assert torch.equal(joined, full)
        assert calls['n'] == len(chunks)
        joined.square().sum().backward()
        assert calls['n'] == 2 * len(chunks)
        assert weight.grad is not None
        assert float(weight.grad.abs().sum().item()) > 0.0


@pytest.mark.live
class TestFunctionalStepsLive:
    """One-group functional_call steps on the covering residual."""

    def test_one_group_functional_steps(
        self,
        covering_pilot: CoveringPilotSpec,
        ssv_model: SoftTrunkModel,
        covering_collator: SoftMlmCollator,
        ssv_inventory: Inventory,
        covering: Covering,
        load_hupd_draw: HupdLoader,
        attach_termhood: Callable[[Sequence[str]], int],
        require_gate: Callable[..., None],
        announce_gate: GateRecorder,
    ) -> None:
        """``autograd.grad`` plus ``functional_call`` on one responsibility group."""
        require_gate('paired_identity', 'paired-identity hygiene did not pass')
        draw = load_hupd_draw(covering_pilot.functional_n)
        termhood_n = attach_termhood(draw.lemma_keys(ssv_model))
        collator = covering_collator
        claim_batch = collate_encode_rows(
            draw.claim_rows, collator=collator, inventory=ssv_inventory
        )
        disc_batch = collate_encode_rows(draw.disc_rows, collator=collator, inventory=ssv_inventory)
        exporter = InventoryForward(ssv_model, ssv_inventory)
        exporter.pin_session_compute()
        params = {key: value for key, value in exporter.named_parameters() if value.requires_grad}
        buffers = dict(exporter.named_buffers())

        def residual(param_map: Mapping[str, Tensor]) -> Tensor:
            return exporter.residual(param_map, buffers, claim_batch, disc_batch, covering)

        base = residual(params)
        if not torch.isfinite(base):
            demand = exporter(claim_batch, 'claim').detach()
            occupy_mass = float(demand.abs().sum().item())
            culprit = 'DATA' if termhood_n == 0 or occupy_mass <= 0.0 else 'BACKWARD_SEAM'
            announce_gate(
                'functional_steps',
                {
                    'base': None,
                    'termhood_n': termhood_n,
                    'occupy_mass': occupy_mass,
                },
                culprit=culprit,
            )
            pytest.fail(
                'covering residual is not finite at init '
                f'(termhood_n={termhood_n} occupy_mass={occupy_mass} culprit={culprit})'
            )
        reports = {
            name: group_step_report(
                name,
                exporter=exporter,
                params=params,
                base=base,
                residual=residual,
                steps=covering_pilot.step_ladder,
            )
            for name in PARAM_GROUPS
        }
        reachable = tuple(name for name, row in reports.items() if row['reachable'])
        effective = tuple(name for name in reachable if reports[name]['same_sign'])
        culprit = (
            ''
            if reachable and len(effective) == len(reachable)
            else ('BACKWARD_SEAM' if not reachable else 'OPTIMIZER')
        )
        announce_gate(
            'functional_steps',
            {
                'groups': reports,
                'culprit': culprit or 'none',
                'base': float(base.item()),
                'termhood_n': termhood_n,
            },
            culprit=culprit,
            cache={'covering/effective_groups': effective[:2]},
            reachable=','.join(reachable),
        )
        assert reachable
        assert all(reports[name]['same_sign'] for name in reachable)
