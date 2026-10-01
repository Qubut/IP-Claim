"""Pinned graph-mask rho and entropy scale for covering encodes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from dependency_injector import providers
from torch_geometric.data import HeteroData

from experiments.covering_objective.pilot import CoveringPilotSpec
from experiments.covering_objective.verdicts import CoveringSchedule
from ip_claim.app.container.ssv import SsvContainer
from ip_claim.ssv.collate import SoftMlmCollator, SoftMlmExample
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.soft_vocab import SoftVocabModule

pytestmark = pytest.mark.experiment


class TestMaskSchedule:
    """Operating-point rho through ``SsvContainer.collator_rho`` override."""

    @pytest.fixture(scope='class')
    @classmethod
    def train_job(cls) -> SsvTrainConfig:
        """Default train contract. Does not load the covering host."""
        return SsvTrainConfig()

    @pytest.fixture(scope='class')
    @classmethod
    def host_container(cls, train_job: SsvTrainConfig) -> SsvContainer:
        """Tiny-host composition root. Singleton trunk, Factory collators."""
        return SsvContainer(config=train_job)

    @pytest.fixture
    def train_collator(
        self,
        host_container: SsvContainer,
    ) -> Iterator[tuple[SoftMlmCollator, SoftVocabModule]]:
        """Train collator Factory plus the Singleton vocab, reset after each override."""
        model = host_container.soft_trunk()
        vocab = model.soft_vocab
        scale = float(vocab.entropy_scale)
        yield host_container.train_collator(), vocab
        vocab.set_entropy_scale(scale)

    @pytest.mark.parametrize('kind', ['floor', 'scheduled'])
    def test_override_sets_floor_and_scheduled_rho(
        self,
        host_container: SsvContainer,
        train_job: SsvTrainConfig,
        kind: str,
    ) -> None:
        """The train YAML supplies rho_max; override does not walk warmup steps."""
        rho = 0.0 if kind == 'floor' else float(train_job.mlm.rho_max)
        entropy = 0.0 if kind == 'floor' else 1.0
        with host_container.collator_rho.override(providers.Object(rho)):
            collator = host_container.train_collator()
            vocab = host_container.soft_trunk().soft_vocab
            vocab.set_entropy_scale(entropy)
            CoveringSchedule.refuse_inert_scheduled({
                'mask_schedule': kind,
                'rho': float(collator.rho),
                'mlm_probability': float(collator.mlm_probability or 0.0),
            })
            assert collator.rho == pytest.approx(rho)
            assert vocab.entropy_scale == pytest.approx(entropy)
            assert collator.mlm_probability == pytest.approx(train_job.mlm.mlm_probability)
            assert train_job.mlm.rho_warmup_steps >= 0
            reading = {
                'mask_schedule': kind,
                'rho': float(collator.rho),
                'entropy_scale': float(vocab.entropy_scale),
                'mlm_probability': float(collator.mlm_probability or 0.0),
                'inject_scale': 0.0,
                'global_step': 0,
                'rho_warmup_steps': int(train_job.mlm.rho_warmup_steps),
            }
            assert reading['inject_scale'] == pytest.approx(0.0)
            assert reading['rho'] != reading['inject_scale'] or kind == 'floor'

    def test_pilot_defaults_to_scheduled_mask(self, covering_pilot: CoveringPilotSpec) -> None:
        """Covering encodes consume the scheduled pin unless the envelope says floor."""
        assert covering_pilot.mask_schedule == 'scheduled'

    def test_env_overlays_letter_noise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """COVERING_* overlays the YAML letter envelope used by host-free oracles."""
        monkeypatch.setenv('COVERING_LETTER_NOISE', '0.03')
        spec = CoveringPilotSpec.from_yaml()
        assert spec.letter.noise == pytest.approx(0.03)
        assert spec.hygiene_n >= 2

    def test_scheduled_override_writes_collator_rho_and_entropy_scale(
        self,
        train_job: SsvTrainConfig,
        host_container: SsvContainer,
        train_collator: tuple[SoftMlmCollator, SoftVocabModule],
    ) -> None:
        """``collator_rho`` override plus ``set_entropy_scale`` are the covering hooks."""
        _collator, vocab = train_collator
        with host_container.collator_rho.override(providers.Object(float(train_job.mlm.rho_max))):
            collator = host_container.train_collator()
            vocab.set_entropy_scale(1.0)
            assert collator.rho == pytest.approx(float(train_job.mlm.rho_max))
            assert vocab.entropy_scale == pytest.approx(1.0)
            assert collator.mlm_probability == pytest.approx(train_job.mlm.mlm_probability)
            assert float(collator.mlm_probability or 0.0) > 0.0

    def test_floor_override_keeps_rho_zero(
        self,
        host_container: SsvContainer,
        train_collator: tuple[SoftMlmCollator, SoftVocabModule],
    ) -> None:
        """rho=0 remains a labeled baseline, not the scheduled ledger."""
        _collator, vocab = train_collator
        vocab.set_entropy_scale(1.0)
        with host_container.collator_rho.override(providers.Object(0.0)):
            collator = host_container.train_collator()
            vocab.set_entropy_scale(0.0)
            assert collator.rho == pytest.approx(0.0)
            assert vocab.entropy_scale == pytest.approx(0.0)

    def test_scheduled_ledger_refuses_zero_rho(self, train_job: SsvTrainConfig) -> None:
        """A scheduled artefact that still encodes at rho=0 fails closed."""
        with pytest.raises(pytest.fail.Exception, match='inert'):
            CoveringSchedule.refuse_inert_scheduled({
                'mask_schedule': 'scheduled',
                'rho': 0.0,
                'mlm_probability': float(train_job.mlm.mlm_probability),
            })

    def test_scheduled_ledger_refuses_zero_mlm_probability(
        self,
        train_job: SsvTrainConfig,
        host_container: SsvContainer,
    ) -> None:
        """Eval collate (mlm_probability=0) cannot claim a scheduled graph-mask mix."""
        eval_collator = host_container.eval_collator()
        assert eval_collator.rho == pytest.approx(0.0)
        assert float(eval_collator.mlm_probability or 0.0) == pytest.approx(0.0)
        with pytest.raises(pytest.fail.Exception, match='inert'):
            CoveringSchedule(
                kind='scheduled',
                rho=float(train_job.mlm.rho_max),
                entropy_scale=1.0,
                mlm_probability=float(eval_collator.mlm_probability or 0.0),
                global_step=0,
                rho_warmup_steps=int(train_job.mlm.rho_warmup_steps),
            ).require_honest()

    def test_floor_ledger_may_record_zero_rho(self) -> None:
        """Floor baseline JSON is allowed to sit at rho=0."""
        CoveringSchedule.refuse_inert_scheduled({
            'mask_schedule': 'floor',
            'rho': 0.0,
            'mlm_probability': 0.15,
        })

    def test_collated_batch_records_overridden_rho(
        self,
        train_job: SsvTrainConfig,
        host_container: SsvContainer,
        train_collator: tuple[SoftMlmCollator, SoftVocabModule],
    ) -> None:
        """``SoftMlmBatch.rho`` follows the Factory override used by covering encode."""
        _collator, vocab = train_collator
        model = host_container.soft_trunk()
        with host_container.collator_rho.override(providers.Object(float(train_job.mlm.rho_max))):
            collator = host_container.train_collator()
            vocab.set_entropy_scale(1.0)
            collator.bind_assignment_source(vocab, model.host.get_input_embeddings())
            row = SoftMlmExample(text='one two three four five six seven eight', graph=HeteroData())
            batch = collator((row,))
            assert batch.rho == pytest.approx(float(train_job.mlm.rho_max))
            assert bool(batch.labels.ge(0).any().item())

    def test_graph_mix_labels_differ_from_floor(
        self,
        train_job: SsvTrainConfig,
        host_container: SsvContainer,
        train_collator: tuple[SoftMlmCollator, SoftVocabModule],
    ) -> None:
        """Assignment-mass mix at rho_max is not the uniform rho=0 mask draw."""
        _unused, vocab = train_collator
        model = host_container.soft_trunk()
        row = SoftMlmExample(text='one two three four five six seven eight', graph=HeteroData())
        embeddings = model.host.get_input_embeddings()
        with host_container.collator_rho.override(providers.Object(0.0)):
            probe = host_container.train_collator()
            probe.bind_assignment_source(vocab, embeddings)
            torch.manual_seed(0)
            attention = probe((row,)).attention_mask
        mass = torch.zeros_like(attention, dtype=torch.float)
        mass[:, 1] = 1.0

        def labels_at(rho: float) -> torch.Tensor:
            with host_container.collator_rho.override(providers.Object(rho)):
                collator = host_container.train_collator()
                collator.bind_assignment_source(vocab, embeddings)
                collator.set_assignment_mass(mass)
                torch.manual_seed(0)
                return collator((row,)).labels

        floor = labels_at(0.0)
        scheduled = labels_at(float(train_job.mlm.rho_max))
        assert not torch.equal(floor, scheduled)
