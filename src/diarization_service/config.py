from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    hf_token: str = Field(..., description="HuggingFace access token")
    voice_service_url: str = "http://voice_service:8000"

    identification_threshold: float = Field(default=0.80, ge=-1.0, le=1.0)

    min_segment_duration: float = Field(default=0.5, ge=0.0)

    use_gpu: bool = False

    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"

    max_upload_size_mb: int = Field(default=500, ge=1)


settings = Settings()