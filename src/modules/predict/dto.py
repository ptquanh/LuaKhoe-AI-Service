from pydantic import BaseModel
from typing import Optional, List


class Detection(BaseModel):
    disease: str
    confidence: float


class PredictionRequest(BaseModel):
    image_url: str


class PredictionResult(BaseModel):
    disease: str
    confidence: float
    detections: List[Detection] = []
    status: str
    model_version: str
    latency_ms: float
    low_confidence: Optional[bool] = False
    annotated_image: Optional[str] = None
