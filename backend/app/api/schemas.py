from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    message: str = Field(min_length=1, max_length=10_000)

    @field_validator("message")
    @classmethod
    def strip_and_validate_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message must not be empty or whitespace-only.")
        return cleaned


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID


class ResetResponse(BaseModel):
    session_id: UUID
    status: Literal["reset"]