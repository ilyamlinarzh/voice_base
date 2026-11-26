from typing import Annotated
from uuid import UUID
import re
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DIMENSION = 256
RU_NAME_PATTERN = re.compile(r"^[А-Яа-яЁё]+(-[А-Яа-яЁё]+)?$")
EN_NAME_PATTERN = re.compile(r"^[A-Za-z]+(-[A-Za-z]+)?$")

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
        if not (RU_NAME_PATTERN.match(value) or EN_NAME_PATTERN.match(value)):
            raise ValueError(
                "Name must contain only Russian OR only English letters, "
                "spaces, and hyphens"
            )
        return value

    @field_validator("middle_name")
    @classmethod
    def normalize_middle_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not (RU_NAME_PATTERN.match(value) or EN_NAME_PATTERN.match(value)):
            raise ValueError(
                "Name must contain only Russian OR only English letters, "
                "spaces, and hyphens"
            )
        return value or None

    @model_validator(mode="after")
    def check_consistent_alphabet(self) -> "PersonCreate":
        if RU_NAME_PATTERN.match(self.first_name):
            first_is_ru = True
        else:
            first_is_ru = False
        if RU_NAME_PATTERN.match(self.last_name):
            last_is_ru = True
        else:
            last_is_ru = False

        if first_is_ru != last_is_ru:
            raise ValueError(
                "First name and last name must use the same alphabet (both Russian or both English)"
            )

        if self.middle_name and first_is_ru:
            if not RU_NAME_PATTERN.match(self.middle_name):
                raise ValueError(
                    "Middle name must be in Russian if the first name is Russian"
                )

        return self
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
