from pydantic import BaseModel
from typing import Optional

class PredictionResult(BaseModel):
    disease: str
    confidence: float
    status: str
    model_version: str
    latency_ms: float
    saved_path: str
    filename: str
    low_confidence: Optional[bool] = False
