FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*
    
COPY pyproject.toml poetry.lock* ./diarization_service/

RUN pip install --no-cache-dir ./diarization_service

COPY . ./diarization_service/

RUN pip install --no-cache-dir --no-deps ./diarization_service

EXPOSE 8001

CMD ["uvicorn", "diarization_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
