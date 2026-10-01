"""Composition root for the SSV foundation trunk and collision eval stack."""

from __future__ import annotations

from typing import final

from dependency_injector import containers, providers

from ip_claim.app.container.collision import CollisionContainer
from ip_claim.app.container.ssv import SsvContainer


@final
class Container(containers.DeclarativeContainer):
    """ip_claim DI container (SSV trunk + collision use case)."""

    ssv = providers.Container(SsvContainer)
    collision = providers.Container(CollisionContainer)
