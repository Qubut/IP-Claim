"""Patent claim collision detection: saturation covering on SSV inventories."""

from ip_claim.collision.artefacts import CoveringReport, ExplainArtefact, ExplainState
from ip_claim.collision.collide import (
    CollisionEvalRequest,
    CollisionEvalResult,
    CorpusIndex,
    PatentEmbeddingRecord,
    RankingBatch,
    cpc_hard_pool_mask,
    cpc_hard_rank_positions,
    empty_collision_eval_result,
    evaluate_collision_ranking,
    partner_rank_positions,
    prepare_ranking_batch,
)
from ip_claim.collision.config import CollisionEvalConfig, CollisionSplitSpec
from ip_claim.collision.cover import Covering, CoveringKnobs, CoveringScore, CoveringTable
from ip_claim.collision.data import (
    HUPD_APPLICATION,
    AnalyzeCited,
    AnalyzeRecord,
    CitationPair,
    CitationPairSource,
    EpoProcessorCitationPairSource,
    StubCitationPairSource,
)
from ip_claim.collision.eval import CollisionJobResult, run_collision_eval
from ip_claim.collision.explain import CollisionCommunity, CoveringContour, Explain

__all__ = [
    'HUPD_APPLICATION',
    'AnalyzeCited',
    'AnalyzeRecord',
    'CitationPair',
    'CitationPairSource',
    'CollisionCommunity',
    'CollisionEvalConfig',
    'CollisionEvalRequest',
    'CollisionEvalResult',
    'CollisionJobResult',
    'CollisionSplitSpec',
    'CorpusIndex',
    'Covering',
    'CoveringContour',
    'CoveringKnobs',
    'CoveringReport',
    'CoveringScore',
    'CoveringTable',
    'EpoProcessorCitationPairSource',
    'Explain',
    'ExplainArtefact',
    'ExplainState',
    'PatentEmbeddingRecord',
    'RankingBatch',
    'StubCitationPairSource',
    'cpc_hard_pool_mask',
    'cpc_hard_rank_positions',
    'empty_collision_eval_result',
    'evaluate_collision_ranking',
    'partner_rank_positions',
    'prepare_ranking_batch',
    'run_collision_eval',
]
