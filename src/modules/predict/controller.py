# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, Body
import time
from src.modules.predict.dto import PredictionResult, PredictionRequest
from src.modules.predict.service import BaseModelInference
from src.shared.middlewares.dependencies import get_engine, validator
from src.shared.utils.image_loader import download_image_from_url
from src.shared.utils.logger import logger

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=PredictionResult)
async def predict(
    request: PredictionRequest = Body(...),
    engine: BaseModelInference = Depends(get_engine),
):
    try:
        start_time = time.time()
        
        # 1. Download image from URL
        image = download_image_from_url(request.image_url)
        
        # 2. Validation
        if not validator.is_valid_image(image):
            raise HTTPException(status_code=400, detail="Invalid rice leaf image.")
            
        # 3. Model Prediction
        result = engine.predict(
            image=image,
            province=request.province,
            gps_lat=request.gps_lat,
            gps_lng=request.gps_lng,
            field_params=request.field_params.model_dump() if request.field_params else None,
            weather=request.weather,
            confidence_threshold=request.confidence_threshold,
            ai_model_version=request.ai_model_version
        )
        
        # 4. Post-inference Checks
        result = validator.check_confidence(result, confidence_threshold=request.confidence_threshold)
        
        # 5. Metadata
        latency = (time.time() - start_time) * 1000
        result.update({
            "latency_ms": round(latency, 2),
        })
        
        logger.info(f"Predicted {result['disease']} in {latency:.1f}ms")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
