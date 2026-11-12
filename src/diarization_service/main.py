from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.diarize import router as diarize_router
from .config import Settings
from .pipeline.diarization import Diarizer
from .pipeline.embedding import EmbeddingExtractor
from .pipeline.identification import VoiceServiceClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings

    app.state.diarizer = Diarizer(
        hf_token=settings.hf_token,
        use_gpu=settings.use_gpu,
        model_name=settings.pyannote_model,
    )
    app.state.embedding_extractor = EmbeddingExtractor(
        hf_token=settings.hf_token,
        use_gpu=settings.use_gpu,
        model_name=settings.embedding_model,
    )
    app.state.voice_client = VoiceServiceClient(
        base_url=settings.voice_service_url,
    )

    try:
        yield
    finally:
        await app.state.voice_client.close()


app = FastAPI(
    title="Diarization Service",
    version="1.0.0",
    description=(
        "Диаризация аудиозаписей совещаний + идентификация спикеров "
        "через банк голосов (voice_service)."
    ),
    lifespan=lifespan,
)

app.include_router(diarize_router)


@app.get("/health/live", tags=["health"])
async def health_live() -> dict[str, str]:
    return {"status": "ok"}