from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class CallParticipant(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    call_id: str | None = None
    display_name: str | None = None
    join_time: datetime | None = None
    leave_time: datetime | None = None
    duration: int | None = None


class CallSummary(BaseModel):
    """Элемент списка из GET /api/v4/logs/calls."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(
        ..., description="Уникальный идентификатор сессии звонка (session_id)"
    )
    conference_id: str | None = Field(
        default=None, description="ID конференции (если звонок был конференцией)"
    )
    topic: str | None = None
    type: str | int | None = None
    owner: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = Field(default=None, alias="stop_time")
    duration: int | None = None
    participant_count: int | None = None

    @property
    def is_completed(self) -> bool:
        """Звонок считается завершённым, если зафиксировано время окончания."""
        return self.end_time is not None


class CallDetails(CallSummary):
    """Детальная информация о звонке из GET /api/v4/logs/calls/{session_id}."""

    participants: list[CallParticipant] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict, exclude=True)


class CallRecord(BaseModel):
    """Информация о серверной записи из GET /api/v4/conferences/{conference_id}/records."""

    model_config = ConfigDict(extra="allow")

    id: str
    conference_id: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    duration: int | None = None
    created_at: datetime | None = None
    download_url: str | None = None
