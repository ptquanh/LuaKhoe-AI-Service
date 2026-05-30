from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from src.modules.predict.constants import WATER_NORMAL, GROWTH_TILLERING, DENSITY_MEDIUM


class Detection(BaseModel):
    disease: str
    confidence: float
    box: Optional[List[float]] = None
    polygon: Optional[List[List[float]]] = None
    color: Optional[str] = None
    affected_area_ratio: Optional[float] = 0.0


class FieldParams(BaseModel):
    """User-reported field condition parameters."""
    water: str = WATER_NORMAL        # "Ngập úng" | "Bình thường" | "Khô hạn"
    growth: str = GROWTH_TILLERING   # "Mạ" | "Đẻ nhánh" | "Làm đòng" | "Trỗ bông" | "Chín"
    density: str = DENSITY_MEDIUM    # "Dày" | "Vừa" | "Thưa"
    fog: bool = False
    leafhopper: bool = False
    pesticide: bool = False


class PredictionRequest(BaseModel):
    image_url: str
    province: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lng: Optional[float] = None
    field_params: Optional[FieldParams] = None
    weather: Optional[Dict[str, Any]] = None
    confidence_threshold: Optional[float] = None
    ai_model_version: Optional[str] = None


class EnvAdjustment(BaseModel):
    """Environment adjustment metadata returned in response."""
    original_scores: Dict[str, float]
    adjusted_scores: Dict[str, float]
    weather: Dict[str, Any]
    applied: bool


class PredictionResult(BaseModel):
    disease: str
    confidence: float
    detections: List[Detection] = []
    status: str
    model_version: str
    latency_ms: float
    low_confidence: Optional[bool] = False
    annotated_image: Optional[str] = None
    env_adjustment: Optional[EnvAdjustment] = None


class ReScoringDetection(BaseModel):
    disease: str
    confidence: float


class ReScoringRequest(BaseModel):
    original_results: List[ReScoringDetection]
    new_image_url: str
    confidence_threshold: Optional[float] = None
    ai_model_version: Optional[str] = None


class ReScoringResult(BaseModel):
    disease: str
    confidence: float
    detections: List[Detection] = []
    status: str
    model_version: str
    latency_ms: float
    annotated_image: Optional[str] = None
