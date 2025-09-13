import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf
from returns.future import FutureResult
from returns.io import IO

from models.hupd import PatentApplication
from scripts.hupd_importer import Importer, main

SAMPLE_CONFIG = OmegaConf.create(
    {
        'db': {
            'uri': 'mongodb://mock:27017',
            'max_pool_size': 10,
            'timeout_ms': 5000,
            'db_name': 'test_db',
            'index_options': {'allow_dropping': True},
        },
        'importer': {'data_dir': '/fake/data/dir'},
    }
)

SAMPLE_PATENT_JSON = {
    'application_number': '14112715',
    'publication_number': 'US20140221868A1-20140807',
    'title': 'Test Patent',
    'decision': 'PENDING',
    'date_produced': '20140723',
    'date_published': '20140807',
    'main_cpc_label': 'A61B5128',
    'cpc_labels': ['A61B5128', 'A61B50059'],
    'main_ipcr_label': 'A61B512',
    'ipcr_labels': ['A61B512', 'A61B500'],
    'patent_number': 'None',
    'filing_date': '20140326',
    'patent_issue_date': '',
    'abandon_date': '',
    'uspc_class': '600',
    'uspc_subclass': '558000',
    'examiner_id': '75147.0',
    'examiner_name_last': 'SMITH',
    'examiner_name_first': 'JOHN',
    'examiner_name_middle': '',
    'inventor_list': [{'inventor_name_last': 'Doe', 'inventor_name_first': 'Jane'}],
    'abstract': 'Test abstract',
    'background': 'Test background',
    'claims': 'Test claims',
    'summary': 'Test summary',
    'full_description': 'Test description',
}


@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration loading."""
    monkeypatch.setattr('scripts.hupd_importer.load_config', lambda: IO(SAMPLE_CONFIG))


@pytest.fixture
def importer():
    """Create Importer instance w/ the mocked config."""
    return Importer(SAMPLE_CONFIG)


@pytest.mark.asyncio
async def test_init_db_success(importer):
    """Test successful database initialization."""
    with (
        patch('scripts.hupd_importer.AsyncIOMotorClient', autospec=True) as mock_client,
        patch('scripts.hupd_importer.init_beanie', AsyncMock()) as mock_init_beanie,
    ):
        result = await importer.init_db()
        result.unwrap()

        mock_client.assert_called_once_with(
            SAMPLE_CONFIG.db.uri,
            maxPoolSize=SAMPLE_CONFIG.db.max_pool_size,
            serverSelectionTimeoutMS=SAMPLE_CONFIG.db.timeout_ms,
        )
        mock_init_beanie.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_file_success(importer, tmp_path):
    """Test processing a valid JSON file."""
    file_path = tmp_path / 'test.json'
    file_path.write_text(json.dumps(SAMPLE_PATENT_JSON))

    # Mock database queries and insertion
    with (
        patch.object(PatentApplication, 'find_one', AsyncMock(return_value=None)),
        patch.object(Importer, '_insert_into_db', AsyncMock()) as mock_insert,
    ):
        print(await importer.process_file(file_path))
        # Verify insertion was called with expected data
        mock_insert.assert_awaited_once()
        inserted_patent = mock_insert.call_args.args[0]
        assert inserted_patent.metadata.title == 'Test Patent'
        assert inserted_patent.examiner.examiner_name_last == 'SMITH'
        assert len(inserted_patent.inventors) == 1
        assert inserted_patent.content.abstract == 'Test abstract'


@pytest.mark.asyncio
async def test_process_file_invalid(importer, tmp_path):
    """Test handling of invalid JSON file."""
    file_path = tmp_path / 'invalid.json'
    file_path.write_text('{invalid: json}')  # Invalid JSON format

    result = await importer.process_file(file_path)
    assert result.failure()  # Check that the result is a failure due to JSONDecodeError


@pytest.mark.asyncio
async def test_insert_into_db_success(importer):
    """Test successful database insertion."""
    mock_patent = MagicMock(spec=PatentApplication)
    with patch('scripts.hupd_importer.PatentApplication.insert', AsyncMock()) as mock_insert:
        await Importer._insert_into_db(mock_patent)
        mock_insert.assert_awaited_once_with(mock_patent)


@pytest.mark.asyncio
async def test_import_patents(importer, tmp_path):
    """Test full import workflow."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    for i in range(3):
        (data_dir / f'patent_{i}.json').write_text(json.dumps(SAMPLE_PATENT_JSON))

    importer.config.importer.data_dir = str(data_dir)

    with (
        patch.object(
            Importer, 'process_file', AsyncMock(return_value=FutureResult.from_value(None))
        ) as mock_process,
        patch.object(
            Importer, '_create_indexes', MagicMock(return_value=FutureResult.from_value(None))
        ),
    ):
        await importer.import_patents()
        assert mock_process.call_count == 3


@pytest.mark.asyncio
async def test_main_execution(mock_config, monkeypatch):
    mock_importer = MagicMock()
    mock_client = MagicMock()
    mock_importer.init_db.return_value = FutureResult.from_value(mock_client)
    mock_importer.import_patents = AsyncMock()
    monkeypatch.setattr('scripts.hupd_importer.Importer', MagicMock(return_value=mock_importer))
    await main()
    assert mock_importer.init_db.called
    assert mock_importer.import_patents.await_count == 1
