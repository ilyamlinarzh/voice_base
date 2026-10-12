from importlib.resources import files
from uuid import UUID, uuid4

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .schemas import (
    DIMENSION,
    EmbeddingResponse,
    PersonResponse,
)


class VectorIndexedBase:
    def __init__(
        self,
        dsn: str,
        model_version: str,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        if not model_version.strip():
            raise ValueError("model_version cannot be empty")
        if max_pool_size < min_pool_size:
            raise ValueError("Invalid pool size")

        self.dsn = dsn
        self.model_version = model_version.strip()
        self._initialized = False

        self.pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_pool_size,
            max_size=max_pool_size,
            open=False,
            kwargs={"row_factory": dict_row},
            configure=self._configure_connection,
        )

    @staticmethod
    async def _configure_connection(conn: AsyncConnection) -> None:
        await register_vector_async(conn)

    async def open(self) -> None:
        # Ensure pgvector exists before registering its type
        # on connections created by the pool.
        async with await AsyncConnection.connect(
            self.dsn,
            autocommit=True,
        ) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await self.pool.open()
        await self.pool.wait()

    async def close(self) -> None:
        await self.pool.close()
        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Database has not been initialized")

    @staticmethod
    def _validate_embedding(embedding: list[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)

        if vector.shape != (DIMENSION,):
            raise ValueError(f"Embedding must have {DIMENSION} dimensions")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains NaN or infinity")
        if np.linalg.norm(vector) == 0:
            raise ValueError("Zero embedding is not supported")

        return vector

    async def initialize(self) -> None:
        migration_dir = files("voice_service").joinpath("migrations")
        migrations = sorted(
            (item for item in migration_dir.iterdir() if item.name.endswith(".sql")),
            key=lambda item: item.name,
        )

        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (742019381,),
                )
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """)

                cursor = await conn.execute("SELECT version FROM schema_migrations")
                applied = {row["version"] for row in await cursor.fetchall()}

                for migration in migrations:
                    version = migration.name.split("_", maxsplit=1)[0]
                    if version in applied:
                        continue

                    # Project migrations contain simple DDL statements.
                    for statement in migration.read_text(encoding="utf-8").split(";"):
                        if statement.strip():
                            await conn.execute(statement.strip())

                    await conn.execute(
                        """
                        INSERT INTO schema_migrations (version)
                        VALUES (%s)
                        """,
                        (version,),
                    )

                cursor = await conn.execute("""
                    SELECT embedding_dimension, model_version, distance_metric
                    FROM vector_index_config
                    WHERE id = 1
                    FOR UPDATE
                    """)
                config = await cursor.fetchone()

                if config is None:
                    cursor = await conn.execute("SELECT COUNT(*) AS count FROM voices")
                    if (await cursor.fetchone())["count"] > 0:
                        raise RuntimeError(
                            "Missing embedding configuration in non-empty DB"
                        )

                    await conn.execute(
                        """
                        INSERT INTO vector_index_config (
                            id,
                            embedding_dimension,
                            model_version,
                            distance_metric
                        )
                        VALUES (1, %s, %s, 'cosine')
                        """,
                        (DIMENSION, self.model_version),
                    )
                elif (
                    config["embedding_dimension"] != DIMENSION
                    or config["model_version"] != self.model_version
                    or config["distance_metric"] != "cosine"
                ):
                    raise RuntimeError(
                        "Database embedding configuration mismatch. "
                        "An explicit migration is required."
                    )

                # Recover if an administrator removed the index.
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS voices_embedding_hnsw_idx
                    ON voices USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                    """)

        self._initialized = True

    async def register_voice(
        self,
        embedding: list[float],
        first_name: str,
        last_name: str,
        middle_name: str | None,
    ) -> tuple[UUID, UUID]:
        self._require_initialized()
        vector = self._validate_embedding(embedding)
        voice_id = uuid4()
        person_id = uuid4()

        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO voices (id, embedding, model_version)
                    VALUES (%s, %s, %s)
                    """,
                    (voice_id, vector, self.model_version),
                )
                await conn.execute(
                    """
                    INSERT INTO persons (
                        id, first_name, last_name, middle_name, voice_id
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        person_id,
                        first_name,
                        last_name,
                        middle_name,
                        voice_id,
                    ),
                )

        return voice_id, person_id

    async def identify(
        self,
        embedding: list[float],
        threshold: float,
    ) -> tuple[PersonResponse, float] | None:
        self._require_initialized()
        vector = self._validate_embedding(embedding)

        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    p.id,
                    p.first_name,
                    p.last_name,
                    p.middle_name,
                    p.voice_id,
                    1 - (v.embedding <=> %s) AS similarity
                FROM voices AS v
                JOIN persons AS p ON p.voice_id = v.id
                WHERE v.model_version = %s
                ORDER BY v.embedding <=> %s
                LIMIT 1
                """,
                (vector, self.model_version, vector),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        similarity = float(row["similarity"])
        if similarity < threshold:
            return None

        return (
            PersonResponse(
                id=row["id"],
                first_name=row["first_name"],
                last_name=row["last_name"],
                middle_name=row["middle_name"],
                voice_id=row["voice_id"],
            ),
            similarity,
        )

    async def get_person(
        self,
        person_id: UUID,
    ) -> PersonResponse | None:
        self._require_initialized()

        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, first_name, last_name, middle_name, voice_id
                FROM persons
                WHERE id = %s
                """,
                (person_id,),
            )
            row = await cursor.fetchone()

        return PersonResponse(**row) if row else None

    async def get_person_embedding(
        self,
        person_id: UUID,
    ) -> EmbeddingResponse | None:
        self._require_initialized()

        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    p.id AS person_id,
                    p.voice_id,
                    v.model_version,
                    v.embedding
                FROM persons AS p
                JOIN voices AS v ON v.id = p.voice_id
                WHERE p.id = %s
                """,
                (person_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        return EmbeddingResponse(
            person_id=row["person_id"],
            voice_id=row["voice_id"],
            model_version=row["model_version"],
            embedding=np.asarray(
                row["embedding"],
                dtype=np.float32,
            ).tolist(),
        )
