import pytest
import pytest_asyncio
import numpy as np
import time
import random
import string

from voice_service.store import VectorIndexedBase
from voice_service.schemas import DIMENSION

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


def generate_random_name():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(10))


@pytest.mark.asyncio
async def test_spam_registration_performance(store: VectorIndexedBase):
    num_requests = 10000 
    print(f"\nНачинаем спам-регистрацию ({num_requests} записей)...")
    
    start_time = time.time()
    
    for i in range(num_requests):
        rng = np.random.default_rng(i)
        embedding = rng.random(DIMENSION).astype(np.float32).tolist()
        
        await store.register_voice(
            embedding=embedding,
            first_name=generate_random_name(),
            last_name=generate_random_name(),
            middle_name=None,
        )
    
    spam_time = time.time() - start_time
    print(f"Среднее время на запись: {spam_time / num_requests:.4f} сек.")
    
    test_embedding = np.random.default_rng(9999).random(DIMENSION).astype(np.float32).tolist()
    
    start_search = time.time()
    await store.identify(test_embedding, threshold=0.5)
    search_time = time.time() - start_search
    
    print(f"Время поиска по базе (в которой {num_requests} мусорных векторов): {search_time:.4f} сек.")

    assert True