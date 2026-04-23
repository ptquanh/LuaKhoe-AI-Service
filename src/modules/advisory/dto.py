from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TreatmentProtocol(BaseModel):
    chemical: str = "Không có dữ liệu trong tài liệu tham khảo"
    biological: str = "Không có dữ liệu trong tài liệu tham khảo"
    cultural: str = "Không có dữ liệu trong tài liệu tham khảo"

class RecommendationResult(BaseModel):
    disease_name: str
    severity_assessment: str
    immediate_actions: List[str]
    treatment_protocol: TreatmentProtocol
    npk_adjustment: str
    prevention_measures: List[str]
    sources_used: List[str]
    confidence_note: str

class Lesion(BaseModel):
    mask_area_px: int

class DiseaseEvent(BaseModel):
    disease_class: str
    confidence: float
    lesions: List[Lesion]

class RecommendationRequest(BaseModel):
    events: List[DiseaseEvent]

class RecommendationResponse(BaseModel):
    status: str
    recommendation: RecommendationResult
    latency_ms: float
    rag_chunks_used: int

class IngestionRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text content to ingest")
    source: str = Field(..., description="Source document name (e.g., 'IRRI_blast_2023.pdf')")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional extra metadata")

class IngestionResponse(BaseModel):
    status: str
    message: str
