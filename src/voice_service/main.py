from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, status

from .config import Settings
from .schemas import (
    EmbeddingResponse,
    IdentifyRequest,
    IdentifyResponse,
    PersonCreate,
    PersonResponse,
)
from .store import VectorIndexedBase


def create_app() -> FastAPI:
    settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = VectorIndexedBase(
            dsn=settings.database_url,
            model_version=settings.model_version,
            min_pool_size=settings.db_pool_min_size,
            max_pool_size=settings.db_pool_max_size,
        )
        await store.open()
        try:
            await store.initialize()
            app.state.store = store
            yield
        finally:
            await store.close()

    app = FastAPI(
        title="Voice Identification Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_store(request: Request) -> VectorIndexedBase:
        return request.app.state.store

    @app.post(
        "/api/v1/voices/identify",
        response_model=IdentifyResponse,
    )
    async def identify_voice(
        payload: IdentifyRequest,
        store: VectorIndexedBase = Depends(get_store),
    ) -> IdentifyResponse:
        match = await store.identify(
            payload.embedding,
            threshold=settings.identification_threshold,
        )
        if match is None:
            return IdentifyResponse(
                identified=False,
                person=None,
                similarity=None,
            )

        person, similarity = match
        return IdentifyResponse(
            identified=True,
            person=person,
            similarity=similarity,
        )

    @app.get(
        "/api/v1/persons/{person_id}",
        response_model=PersonResponse,
    )
    async def get_person(
        person_id: UUID,
        store: VectorIndexedBase = Depends(get_store),
    ) -> PersonResponse:
        person = await store.get_person(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Person not found")
        return person

    @app.get(
        "/api/v1/persons/{person_id}/embedding",
        response_model=EmbeddingResponse,
    )
    async def get_person_embedding(
        person_id: UUID,
        store: VectorIndexedBase = Depends(get_store),
    ) -> EmbeddingResponse:
        result = await store.get_person_embedding(person_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Person not found")
        return result

    @app.post(
        "/api/v1/voices",
        response_model=PersonResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_person(
        payload: PersonCreate,
        store: VectorIndexedBase = Depends(get_store),
    ) -> PersonResponse:
        voice_id, person_id = await store.register_voice(
            embedding=payload.embedding,
            first_name=payload.first_name,
            last_name=payload.last_name,
            middle_name=payload.middle_name,
        )

        return PersonResponse(
            id=person_id,
            first_name=payload.first_name,
            last_name=payload.last_name,
            middle_name=payload.middle_name,
            voice_id=voice_id,
        )

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
