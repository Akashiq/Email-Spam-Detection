from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000, examples=["You won a free iPhone! Click now"])


class PredictResponse(BaseModel):
    prediction_id: str
    label: Literal["spam", "ham"]
    is_spam: bool
    spam_probability: float
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: list[PredictResponse]


class FeedbackRequest(BaseModel):
    prediction_id: str
    true_label: Literal[0, 1] = Field(..., description="1 = spam, 0 = ham (ground truth)")
