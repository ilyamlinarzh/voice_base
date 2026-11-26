
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Сначала устанавливаем CPU-версии PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.15.0,<3.0.0" \
    "torchaudio>=2.11.1,<3.0.0"
    
RUN pip install --no-cache-dir soundfile
# Копируем исходники
COPY . ./diarization_service/

# Устанавливаем проект и остальные зависимости.
# Уже установленные CPU-версии PyTorch удовлетворяют ограничениям.
RUN pip install --no-cache-dir ./diarization_service

EXPOSE 8001

CMD ["uvicorn", "diarization_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
