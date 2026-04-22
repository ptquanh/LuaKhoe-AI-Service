from pydantic import BaseModel
from typing import Optional
from src.modules.predict.dto import PredictionResult
from src.modules.advisory.dto import RecommendationResponse

class AnalyzeResponse(BaseModel):
    prediction: PredictionResult
    recommendation: Optional[RecommendationResponse] = None
