import asyncio
import os
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ..config import Settings
from ..pipeline.asr import WhisperTranscriber
from ..pipeline.diarization import Diarizer
from ..pipeline.embedding import EmbeddingExtractor
from ..pipeline.identification import VoiceServiceClient
from ..pipeline.merge import merge_segments_with_words
from ..pipeline.protocol import generate_protocol_docx

router = APIRouter(prefix="/api/v1", tags=["protocol"])


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


def get_transcriber(request: Request) -> WhisperTranscriber:
    return _get_state(request, "transcriber")


def get_settings(request: Request) -> Settings:
    return _get_state(request, "settings")


async def _identify_embedding(
    emb_list: list[float],
    voice_client: VoiceServiceClient,
    threshold: float,
) -> tuple[bool, dict, float | None]:
    try:
        id_resp = await voice_client.identify(emb_list, threshold)
        identified = bool(id_resp.get("identified"))
        person = id_resp.get("person") or {}
        similarity = id_resp.get("similarity")
        return identified, person, similarity
    except httpx.HTTPError:
        return False, {}, None


@router.post("/protocol")
async def generate_protocol(
    audio: UploadFile = File(..., description="Аудиофайл"),
    diarizer: Diarizer = Depends(get_diarizer),
    extractor: EmbeddingExtractor = Depends(get_extractor),
    voice_client: VoiceServiceClient = Depends(get_voice_client),
    transcriber: WhisperTranscriber = Depends(get_transcriber),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Сгенерировать DOCX-протокол совещания."""
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
        diarize_task = asyncio.to_thread(
            diarizer.diarize, tmp_path, settings.min_segment_duration
        )
        transcribe_task = asyncio.to_thread(
            transcriber.transcribe, tmp_path, settings.whisper_language
        )
        raw_segments, words = await asyncio.gather(diarize_task, transcribe_task)

        def _extract_all() -> list[tuple[dict, list[float]]]:
            result = []
            for seg in raw_segments:
                clean_intervals = Diarizer.clean_subsegments(seg)
                if not clean_intervals:
                    emb = extractor.extract(tmp_path, seg["start"], seg["end"])
                else:
                    longest = max(clean_intervals, key=lambda iv: iv[1] - iv[0])
                    emb = extractor.extract(tmp_path, longest[0], longest[1])
                result.append((seg, emb.tolist()))
            return result

        extracted = await asyncio.to_thread(_extract_all)

        identified_segments = []
        unique_person_ids: set[str] = set()
        for seg, emb_list in extracted:
            identified, person, similarity = await _identify_embedding(
                emb_list, voice_client, settings.identification_threshold
            )
            person_id = person.get("id") if identified else None
            if person_id:
                unique_person_ids.add(person_id)
            identified_segments.append({
                **seg,
                "embedding": emb_list,
                "person_id": person_id,
                "similarity": (
                    float(similarity)
                    if identified and similarity is not None
                    else None
                ),
                "needs_moderation": not identified,
            })

        merged = merge_segments_with_words(identified_segments, words)

        persons_cache: dict[str, dict | None] = {}
        for pid in unique_person_ids:
            persons_cache[pid] = await voice_client.get_person(pid)

        docx_bytes = generate_protocol_docx(
            segments=merged,
            persons_cache=persons_cache,
            language=settings.whisper_language,
        )

        filename = f"protocol_{uuid.uuid4().hex[:8]}.docx"
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass