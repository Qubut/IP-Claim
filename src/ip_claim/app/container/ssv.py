"""DI factories for the Soft Structural Vocabulary training stack."""

from __future__ import annotations

from typing import final

from dependency_injector import containers, providers

from ip_claim.collision.cover import Covering, CoveringKnobs
from ip_claim.ssv.collate import SoftMlmCollator, build_eval_collator
from ip_claim.ssv.config import SsvTrainConfig
from ip_claim.ssv.host_tokenizer import load_host_tokenizer
from ip_claim.ssv.inventory import Inventory
from ip_claim.ssv.load_weights import load_model_weights
from ip_claim.ssv.model import build_soft_trunk
from ip_claim.ssv.module import SsvLightningModule


@final
class SsvContainer(containers.DeclarativeContainer):
    """Composition root for SSV train objects (config injected at call site)."""

    config = providers.Object(SsvTrainConfig())

    host_tokenizer = providers.Singleton(load_host_tokenizer, config=config)
    soft_trunk = providers.Singleton(build_soft_trunk, config=config)
    covering = providers.Factory(Covering, knobs=providers.Factory(CoveringKnobs))
    inventory = providers.Factory(
        Inventory,
        occupied_floor=config.provided.arch.soft_occupied_floor,
    )
    lightning_module = providers.Factory(
        SsvLightningModule,
        config=config,
        model=soft_trunk,
        covering=covering,
        inventory=inventory,
    )
    init_weights = providers.Callable(load_model_weights, trunk=soft_trunk)
    collator_rho = providers.Object(0.0)
    train_collator = providers.Factory(
        SoftMlmCollator,
        host_tokenizer,
        mlm_probability=config.provided.mlm.mlm_probability,
        max_length=config.provided.arch.max_length,
        rho=collator_rho,
        cpc_boost=config.provided.mlm.cpc_boost,
        cpc_aux_weight=config.provided.mlm.cpc_aux_weight,
        assignment_top_k=config.provided.mlm.assignment_top_k,
        mlm_span_mask=config.provided.mlm.mlm_span_mask,
        mlm_span_geo_p=config.provided.mlm.mlm_span_geo_p,
        mlm_span_max_length=config.provided.mlm.mlm_span_max_length,
        prime_jate_spans=config.provided.fit.prime_jate_spans,
        spacy_model=config.provided.ate.spacy_model,
    )
    eval_collator = providers.Factory(build_eval_collator, config=config, tokenizer=host_tokenizer)


__all__ = [
    'SsvContainer',
]
