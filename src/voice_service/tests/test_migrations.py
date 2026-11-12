import pytest
import pytest_asyncio
from voice_service.store import VectorIndexedBase


TEST_DB_DSN = "postgresql://voice_app:postgres@localhost:5433/voice_db"
TEST_MODEL_VERSION = "speaker-model-v1"

@pytest_asyncio.fixture
async def store():
    db = VectorIndexedBase(
        dsn=TEST_DB_DSN,
        model_version=TEST_MODEL_VERSION,
    )
    await db.open()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_migrations_are_idempotent(store: VectorIndexedBase):
    await store.initialize()

    await store.initialize()
    
    await store.initialize()

    assert True