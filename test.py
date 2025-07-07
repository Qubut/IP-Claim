import asyncio
import json
import operator
from functools import reduce
from pathlib import Path
from pprint import pp

import aiofiles
from beanie import init_beanie
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from omegaconf import DictConfig, ListConfig, OmegaConf

from models.hupd import (
    ApplicationDates,
    ApplicationMetadata,
    ClassificationInfo,
    ExaminerInfo,
    PatentApplication,
    PatentContent,
)
from utils.chunker import PatentChunker, PatentChunkerConfig
from utils.loader import PatentConverter


def load_config() -> DictConfig | ListConfig:
    """Load and merge configuration from CLI and config file."""
    user_config = OmegaConf.from_cli()
    return OmegaConf.merge(
        OmegaConf.load(user_config.get('config', 'config/hupd_importer.yml')),
        OmegaConf.load(user_config.get('db-config', 'config/mongodb.yml')),
        user_config,
    )


async def init_db(config: DictConfig | ListConfig) -> AsyncIOMotorClient[PatentApplication]:
    """Initialize database connection and indexes."""
    config = config.db
    client: AsyncIOMotorClient[PatentApplication] = AsyncIOMotorClient(
        config.uri, maxPoolSize=config.max_pool_size, serverSelectionTimeoutMS=config.timeout_ms
    )
    await init_beanie(
        database=client[config.db_name],
        document_models=[PatentApplication],
        allow_index_dropping=config.index_options.allow_dropping,
    )
    return client


async def main():
    config = load_config()
    await init_db(config)
    docs = []
    async for batch in PatentConverter.batch_convert(batch_size=10):
        docs.extend(batch)
        break
    chunker_config = PatentChunkerConfig()
    chunker = PatentChunker(chunker_config)
    chunks = chunker.chunk_document(docs[9])
    pp(chunks)


if __name__ == '__main__':
    asyncio.run(main())
