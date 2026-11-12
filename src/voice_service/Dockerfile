FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml poetry.lock* ./voice_service/

RUN pip install --no-cache-dir ./voice_service

COPY . ./voice_service/

RUN pip install --no-cache-dir --no-deps ./voice_service

EXPOSE 8000

CMD ["uvicorn", "voice_service.main:app", "--host", "0.0.0.0", "--port", "8000"]