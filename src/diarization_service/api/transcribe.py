import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from ..config import Settings
from ..pipeline.asr import WhisperTranscriber
from ..schemas import TranscribeResponse, Word

router = APIRouter(prefix="/api/v1", tags=["transcription"])


def _get_state(request: Request, attr: str):
    if not hasattr(request.app.state, attr):
        raise HTTPException(
            status_code=503,
            detail=f"{attr} not initialized — service still starting up",
        )
    return getattr(request.app.state, attr)


def get_transcriber(request: Request) -> WhisperTranscriber:
    return _get_state(request, "transcriber")


def get_settings(request: Request) -> Settings:
    return _get_state(request, "settings")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(..., description="Аудиофайл"),
    transcriber: WhisperTranscriber = Depends(get_transcriber),
    settings: Settings = Depends(get_settings),
) -> TranscribeResponse:
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
        words = await asyncio.to_thread(
            transcriber.transcribe,
            tmp_path,
            settings.whisper_language,
        )
        return TranscribeResponse(
            language=settings.whisper_language,
            words=[Word(**w) for w in words],
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass