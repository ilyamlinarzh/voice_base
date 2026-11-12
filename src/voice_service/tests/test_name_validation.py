import pytest
from pydantic import ValidationError

from voice_service.schemas import DIMENSION, PersonCreate


def make_embedding() -> list[float]:
    return [0.1] * DIMENSION


@pytest.mark.parametrize("invalid_name", [
    "Иван*",
    "Иван123",
    "Иван_Петров"
])
def test_invalid_chars_in_first_name(invalid_name: str):
    with pytest.raises(ValidationError):
        PersonCreate(
            embedding=make_embedding(),
            first_name=invalid_name,
            last_name="Иванов",
        )


@pytest.mark.parametrize("invalid_name", [
    "Петров*",
    "Петров123",
    "Петров!",
])
def test_invalid_chars_in_last_name(invalid_name: str):
    with pytest.raises(ValidationError):
        PersonCreate(
            embedding=make_embedding(),
            first_name="Иван",
            last_name=invalid_name,
        )