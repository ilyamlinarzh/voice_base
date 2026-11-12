"""POST /api/v1/diarize — главный эндпоинт.

Принимает аудиофайл, прогоняет через pyannote Pipeline, для каждого сегмента
извлекает эмбеддинг и идентифицирует через voice_service.
"""

import asyncio
import os
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from ..config import Settings
from ..pipeline.diarization import Diarizer
from ..pipeline.embedding import EmbeddingExtractor
from ..pipeline.identification import VoiceServiceClient
from ..schemas import DiarizeResponse, Segment

router = APIRouter(prefix="/api/v1", tags=["diarization"])


def _get_state(request: Request, attr: str):
    if not hasattr(request.app.state, attr):
        raise HTTPException(
            status_code=503,
            detail=f"{attr} not initialized — service still starting up",
        )
    return getattr(request.app.state, attr)


def get_diarizer(request: Request) -> Diarizer:
    return _get_state(request, "diarizer")


def get_extractor(request: Request) -> EmbeddingExtractor:
    return _get_state(request, "embedding_extractor")


def get_voice_client(request: Request) -> VoiceServiceClient:
    return _get_state(request, "voice_client")


def get_settings(request: Request) -> Settings:
    return _get_state(request, "settings")


@router.post("/diarize", response_model=DiarizeResponse)
async def diarize(
    audio: UploadFile = File(..., description="Аудиофайл (WAV/MP3/M4A)"),
    diarizer: Diarizer = Depends(get_diarizer),
    extractor: EmbeddingExtractor = Depends(get_extractor),
    voice_client: VoiceServiceClient = Depends(get_voice_client),
    settings: Settings = Depends(get_settings),
) -> DiarizeResponse:
    suffix = Path(audio.filename or "").suffix or ".wav"
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        written = 0
        while chunk := await audio.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (max {settings.max_upload_size_mb} MB)",
                )
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        raw_segments = await asyncio.to_thread(
            diarizer.diarize,
            tmp_path,
            settings.min_segment_duration,
        )

        def _extract_all() -> list[tuple[dict, list[float]]]:
            result = []
            for seg in raw_segments:
                emb = extractor.extract(tmp_path, seg["start"], seg["end"])
                result.append((seg, emb.tolist()))
            return result

        extracted = await asyncio.to_thread(_extract_all)

        segments: list[Segment] = []
        for seg, emb_list in extracted:
            try:
                id_resp = await voice_client.identify(
                    emb_list, settings.identification_threshold
                )
                identified = bool(id_resp.get("identified"))
                person = id_resp.get("person") or {}
                similarity = id_resp.get("similarity")
            except httpx.HTTPError:
                identified = False
                person = {}
                similarity = None

            segments.append(
                Segment(
                    id=str(uuid.uuid4()),
                    start=seg["start"],
                    end=seg["end"],
                    speaker_label=seg["speaker_label"],
                    embedding=emb_list,
                    person_id=person.get("id") if identified else None,
                    similarity=float(similarity) if identified and similarity is not None else None,
                    needs_moderation=not identified,
                )
            )

        return DiarizeResponse(segments=segments)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass