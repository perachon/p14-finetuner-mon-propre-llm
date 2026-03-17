from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SFTRecord(BaseModel):
    id: str
    instruction: str
    input: str | None = None
    output: str
    lang: Literal["fr", "en"]
    source: str
    symptoms: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    vitals: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    pii_redacted: bool = True
    created_at: datetime | None = None


class DPORecord(BaseModel):
    id: str
    prompt: str
    chosen: str
    rejected: str
    lang: Literal["fr", "en"]
    source: str
    confidence: float | None = None
    pii_redacted: bool = True
    created_at: datetime | None = None


class MetadataSchema(BaseModel):
    version: str = "1.0"
    fields: dict[str, str]


class TriageRequest(BaseModel):
    patient_message: str = Field(..., description="Message initial du patient")
    lang: Literal["fr", "en"] = "fr"
    context: dict[str, Any] = Field(default_factory=dict)


class TriageDecision(BaseModel):
    priority: Literal["urgence_maximale", "urgence_moderee", "urgence_differee"]
    explanation: str
    recommended_next_steps: list[str]
    red_flags: list[str] = Field(default_factory=list)


class TriageResponse(BaseModel):
    interaction_id: str
    decision: TriageDecision
    follow_up_questions: list[str]
    model_info: dict[str, Any] = Field(default_factory=dict)
