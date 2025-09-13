import asyncio
import json
from pathlib import Path

import aiofiles
from beanie import init_beanie
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from omegaconf import DictConfig, ListConfig, OmegaConf
from returns.future import FutureResult, future_safe
from returns.io import IO, IOResult, impure

from models.hupd import (
    ApplicationDates,
    ApplicationMetadata,
    ClassificationInfo,
    ExaminerInfo,
    PatentApplication,
    PatentContent,
)


@impure
def load_config() -> DictConfig | ListConfig:
    """Load and merge configuration from CLI and config file."""
    user_config = OmegaConf.from_cli()
    return OmegaConf.merge(
        OmegaConf.load(user_config.get('config', 'config/hupd_importer.yml')),
        OmegaConf.load(user_config.get('db-config', 'config/mongodb.yml')),
        user_config,
    )


class Importer:
    """Class responsible for importing patent data into the database.

    Args:
        config (DictConfig | ListConfig): Configuration object containing settings for the importer.
    """

    def __init__(self, config: DictConfig | ListConfig):
        """Class constructor."""
        self.config = config

    @future_safe
    async def init_db(self) -> AsyncIOMotorClient[PatentApplication]:
        """Initialize database connection and indexes."""
        config = self.config.db
        client: AsyncIOMotorClient[PatentApplication] = AsyncIOMotorClient(
            config.uri, maxPoolSize=config.max_pool_size, serverSelectionTimeoutMS=config.timeout_ms
        )
        await init_beanie(
            database=client[config.db_name],
            document_models=[PatentApplication],
            allow_index_dropping=config.index_options.allow_dropping,
        )
        return client

    @staticmethod
    @future_safe
    async def _insert_into_db(patent_application: PatentApplication) -> None:
        await PatentApplication.insert(patent_application)

    @future_safe
    async def process_file(self, file_path: Path) -> None:
        """Process a single JSON file and create a PatentApplication object.

        Args:
            file_path (Path): Path to the JSON file to process.

        Returns:
            PatentApplication: The parsed patent application data.
        """
        async with aiofiles.open(file_path, encoding='utf-8') as f:
            data = json.loads(await f.read())
            publication_number = data.get('publication_number')
            if not publication_number:
                logger.warning(f'No publication_number found in {file_path}, skipping.')
                return
            existing = await PatentApplication.find_one(
                PatentApplication.metadata.publication_number == publication_number
            )
            if existing:
                logger.info(f'⏭️ Patent application {publication_number} already exists, skipping.')
                return
        patent_application = PatentApplication(
            metadata=ApplicationMetadata(
                **{
                    k: data.get(k)
                    for k in [
                        'application_number',
                        'publication_number',
                        'patent_number',
                        'title',
                        'decision',
                    ]
                }
            ),
            dates=ApplicationDates(
                **{
                    k: data.get(k)
                    for k in [
                        'date_produced',
                        'date_published',
                        'filing_date',
                        'patent_issue_date',
                        'abandon_date',
                    ]
                }
            ),
            classification=ClassificationInfo(
                **{
                    k: data.get(k)
                    for k in [
                        'main_cpc_label',
                        'cpc_labels',
                        'main_ipcr_label',
                        'ipcr_labels',
                        'uspc_class',
                        'uspc_subclass',
                    ]
                }
            ),
            examiner=ExaminerInfo(
                **{
                    k: data.get(k)
                    for k in [
                        'examiner_id',
                        'examiner_name_last',
                        'examiner_name_first',
                        'examiner_name_middle',
                    ]
                }
            ),
            inventors=data.get('inventor_list', []),
            content=PatentContent(
                **{
                    k: data.get(k)
                    for k in ['abstract', 'claims', 'background', 'summary', 'full_description']
                }
            ),
        )
        task: FutureResult[None, Exception] = self._insert_into_db(patent_application)
        result = await task
        result.alt(lambda e: logger.error(f'Error processing file: {e}'))
        result.map(lambda _: logger.info(f'✅ Inserted {file_path} documents into the database'))

    @staticmethod
    @future_safe
    async def _create_indexes() -> None:
        await PatentApplication.get_motor_collection().create_index(
            [
                ('metadata.application_number', 1),
            ]
        )
        await PatentApplication.get_motor_collection().create_index(
            [
                ('metadata.publication_number', 'text'),
            ]
        )
        logger.info('✅ Created full-text indexes')

    @future_safe
    async def import_patents(self):
        """Import patent data from JSON files into the database.

        This method reads all JSON files from the specified directory,
        processes them into PatentApplication objects,
        and inserts them into the database.
        It also creates text indexes on certain fields after the import.
        """
        logger.info('Starting patent import process')
        importer = self.config.importer
        files = [f for f in Path(importer.data_dir).glob('*.json') if f.is_file()]
        logger.info(f'Found {len(files)} JSON files to import')
        step = importer.batch_size
        results: list[IOResult[PatentApplication, Exception]] = [
            result
            for i in range(0, len(files), step)
            for result in await asyncio.gather(
                *[self.process_file(file) for file in files[i : i + step]]
            )
        ]
        successful_applications = sum(result.value_or(False) == IO(None) for result in results)

        logger.info(f'Successfully processed {len(successful_applications)} patent applications')
        await self._create_indexes().alt(logger.error)


async def main() -> None:
    """Main function."""
    config: IO[DictConfig | ListConfig] = load_config()

    importer = config.map(Importer)._inner_value  # noqa: SLF001
    result = await importer.init_db().bind_awaitable(lambda _: importer.import_patents())
    result.alt(logger.error)


def run():
    """Runs the script."""
    asyncio.run(main())


if __name__ == '__main__':
    run()
