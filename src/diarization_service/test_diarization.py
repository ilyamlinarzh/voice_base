"""Временный скрипт для проверки диаризации без запуска FastAPI.

Запускать из src/:
    cd src
    poetry run python -m diarization_service.test_diarization /path/to/test.wav

После проверки можно удалить.
"""

import os
import sys
from pathlib import Path


from .config import settings
from .pipeline.diarization import Diarizer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python -m diarization_service.test_diarization <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Файл не найден: {audio_path}")
        sys.exit(1)

    print(f"Загрузка модели pyannote (может занять время при первом запуске)...")
    diarizer = Diarizer(
        hf_token=settings.hf_token,
        use_gpu=settings.use_gpu,
        model_name=settings.pyannote_model,
    )

    print(f"Диаризация {audio_path}...")
    segments = diarizer.diarize(
        audio_path,
        min_segment_duration=settings.min_segment_duration,
    )

    print(f"\nНайдено {len(segments)} сегментов:")
    for i, seg in enumerate(segments):
        print(f"  [{i:3d}] {seg['start']:7.2f} - {seg['end']:7.2f}  {seg['speaker_label']}")