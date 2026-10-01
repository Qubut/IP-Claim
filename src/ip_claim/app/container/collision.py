"""DI factories for patent collision detection (depends on SSV trunk wiring)."""

from __future__ import annotations

from typing import final

from dependency_injector import containers, providers

from ip_claim.app.container.ssv import SsvContainer
from ip_claim.collision.config import CollisionEvalConfig
from ip_claim.collision.cover import Covering
from ip_claim.collision.eval import load_collision_trunk
from ip_claim.ssv.config import SsvTrainConfig


@final
class CollisionContainer(containers.DeclarativeContainer):
    """Composition root for collision eval (SSV trunk injected at call site)."""

    ssv_config = providers.Object(SsvTrainConfig())
    eval_config = providers.Object(CollisionEvalConfig())
    ssv = providers.Container(SsvContainer, config=ssv_config)
    covering = providers.Factory(Covering, knobs=eval_config.provided.covering)
    trunk = providers.Callable(load_collision_trunk, eval_config=eval_config)


__all__ = [
    'CollisionContainer',
]
