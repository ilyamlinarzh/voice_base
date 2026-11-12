import pytest
import pytest_asyncio
from voice_service.store import VectorIndexedBase

# Настройки (возьмите свои)
TEST_DB_DSN = "postgresql://voice_app:postgres@localhost:5433/voice_db"
TEST_MODEL_VERSION = "speaker-model-v1"

@pytest_asyncio.fixture
async def store():
    db = VectorIndexedBase(
        dsn=TEST_DB_DSN,
        model_version=TEST_MODEL_VERSION,
    )
    await db.open()
    await db.initialize()
    yield db
    
    async with db.pool.connection() as conn:
        await conn.execute("TRUNCATE TABLE persons, voices RESTART IDENTITY CASCADE;")
    
    await db.close()

@pytest.mark.asyncio
async def test_float32_precision_loss(store: VectorIndexedBase):
    original_vector = [0.1, 0.123456789, 1e-8] + [0.5] * 509
    
    voice_id, person_id = await store.register_voice(
        embedding=original_vector, first_name="Тест", last_name="Точности", middle_name=None
    )

    saved_embedding = await store.get_person_embedding(person_id)
    
    print(f"\nОтправлено: {original_vector[:3]}")
    print(f"Получено:   {saved_embedding.embedding[:3]}")
    
    for orig, saved in zip(original_vector[:3], saved_embedding.embedding[:3]):
        assert abs(orig - saved) < 1e-6