from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

DIMENSION = 512

Embedding = Annotated[
    list[float],
    Field(min_length=DIMENSION, max_length=DIMENSION),
]


class PersonCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: Embedding
    first_name: str = Field(min_length=1, max_length=200)
    last_name: str = Field(min_length=1, max_length=200)
    middle_name: str | None = Field(default=None, max_length=200)

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be empty")
        return value

    @field_validator("middle_name")
    @classmethod
    def normalize_middle_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class IdentifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: Embedding


class PersonResponse(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    middle_name: str | None
    voice_id: UUID


class IdentifyResponse(BaseModel):
    identified: bool
    person: PersonResponse | None
    similarity: float | None = None


class EmbeddingResponse(BaseModel):
    person_id: UUID
    voice_id: UUID
    model_version: str
    embedding: list[float]
