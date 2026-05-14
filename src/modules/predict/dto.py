from pydantic import BaseModel
from typing import Optional, List


class Detection(BaseModel):
    disease: str
    confidence: float
    box: Optional[List[float]] = None
    polygon: Optional[List[List[float]]] = None
    color: Optional[str] = None


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
