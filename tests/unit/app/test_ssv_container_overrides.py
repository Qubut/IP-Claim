"""Composition-root overrides replace tokenizer, trunk, covering, and load."""

from __future__ import annotations

import pytest
from dependency_injector import providers

from ip_claim.app.container.collision import CollisionContainer
from ip_claim.app.container.ssv import SsvContainer
from ip_claim.collision import eval as collision_eval
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.load_weights import align_hgt_key, load_lightning_payload


def test_ssv_container_override_soft_trunk_and_tokenizer() -> None:
    fake_trunk = object()
    fake_tokenizer = object()
    with (
        SsvContainer.soft_trunk.override(providers.Object(fake_trunk)),
        SsvContainer.host_tokenizer.override(providers.Object(fake_tokenizer)),
    ):
        container = SsvContainer(config=SsvTrainConfig())
        assert container.soft_trunk() is fake_trunk
        assert container.soft_trunk() is container.soft_trunk()
        assert container.host_tokenizer() is fake_tokenizer
        assert container.host_tokenizer() is container.host_tokenizer()


def test_collision_container_override_covering_and_trunk() -> None:
    fake_covering = object()
    fake_trunk = object()
    with (
        CollisionContainer.covering.override(providers.Object(fake_covering)),
        CollisionContainer.trunk.override(providers.Object(fake_trunk)),
    ):
        container = CollisionContainer(
            ssv_config=SsvTrainConfig(),
            eval_config=CollisionEvalConfig(),
        )
        assert container.covering() is fake_covering
        assert container.trunk() is fake_trunk


def test_ssv_container_override_collator_rho() -> None:
    with SsvContainer.collator_rho.override(providers.Object(0.5)):
        container = SsvContainer(config=SsvTrainConfig())
        assert container.collator_rho() == pytest.approx(0.5)


def test_collision_eval_imports_ssv_load_not_shared() -> None:
    assert collision_eval.align_hgt_key is align_hgt_key
    assert collision_eval.load_lightning_payload is load_lightning_payload
    assert 'ip_claim.shared' not in collision_eval.align_hgt_key.__module__
