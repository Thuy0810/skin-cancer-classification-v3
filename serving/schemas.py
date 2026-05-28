from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]
