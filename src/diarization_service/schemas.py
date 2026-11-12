import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMBEDDING_DIM = 256


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="UUID сегмента для модерации")
    start: float = Field(ge=0.0, description="Начало сегмента, секунды")
    end: float = Field(ge=0.0, description="Конец сегмента, секунды")
    speaker_label: str = Field(description="Внутренняя метка pyannote, напр. SPEAKER_00")
    embedding: list[float] = Field(
        min_length=EMBEDDING_DIM,
        max_length=EMBEDDING_DIM,
        description=f"{EMBEDDING_DIM}-мерный эмбеддинг сегмента",
    )

    person_id: str | None = Field(
        default=None,
        description="UUID из voice_service, если спикер идентифицирован",
    )
    similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Косинусное сходство с ближайшим голосом в БД",
    )
    needs_moderation: bool = Field(
        description="true, если спикер не идентифицирован — нужен ручной разбор",
    )


class DiarizeResponse(BaseModel):
    segments: list[Segment]


class ModerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: list[float] = Field(
        min_length=EMBEDDING_DIM,
        max_length=EMBEDDING_DIM,
    )
    action: Literal["register_new", "assign_existing"]

    first_name: str | None = Field(default=None, min_length=1, max_length=200)
    last_name: str | None = Field(default=None, min_length=1, max_length=200)
    middle_name: str | None = Field(default=None, max_length=200)

    person_id: str | None = None

    @field_validator("first_name", "last_name", "middle_name")
    @classmethod
    def _normalize_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


class ModerateResponse(BaseModel):
    person_id: str
    voice_id: str