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


from src.modules.predict.dto import ReScoringRequest, ReScoringResult, Detection

@router.post("/re-score", response_model=ReScoringResult)
async def re_score(
    request: ReScoringRequest = Body(...),
    engine: BaseModelInference = Depends(get_engine),
):
    try:
        start_time = time.time()
        
        # 1. Download new close-up image
        image = download_image_from_url(request.new_image_url)
        
        # 2. Validate image
        if not validator.is_valid_image(image):
            raise HTTPException(status_code=400, detail="Invalid rice leaf image.")
            
        # 3. Predict on new close-up image
        conf_threshold = request.confidence_threshold if request.confidence_threshold is not None else 0.25
        pred_res = engine.predict(
            image=image,
            confidence_threshold=conf_threshold,
            ai_model_version=request.ai_model_version
        )
        
        new_detections = pred_res.get("detections", [])
        new_detect_map = {d["disease"]: d for d in new_detections}
        
        epsilon = 0.05
        final_detections = []
        
        # Get union of all unique diseases from both original and new detections
        all_diseases = set(orig.disease for orig in request.original_results) | set(d["disease"] for d in new_detections)
        
        # Apply Bayesian Aggregation formula for each disease
        for disease in all_diseases:
            # Get S_old
            orig_match = next((orig for orig in request.original_results if orig.disease == disease), None)
            s_old = orig_match.confidence if orig_match else epsilon
            
            # Get S_new
            s_new = new_detect_map[disease]["confidence"] if disease in new_detect_map else epsilon
            
            # Compute Bayesian Aggregation
            denominator = (s_old * s_new) + (1 - s_old) * (1 - s_new)
            if denominator == 0:
                s_final = 0.0
            else:
                s_final = (s_old * s_new) / denominator
                
            # Filter out diseases with S_final < 0.3 (30%)
            if s_final < 0.3:
                continue
                
            if disease in new_detect_map:
                new_d = new_detect_map[disease]
                final_detections.append(Detection(
                    disease=disease,
                    confidence=s_final,
                    box=new_d.get("box"),
                    polygon=new_d.get("polygon"),
                    color=new_d.get("color"),
                    affected_area_ratio=new_d.get("affected_area_ratio", 0.0)
                ))
            else:
                final_detections.append(Detection(
                    disease=disease,
                    confidence=s_final,
                    box=None,
                    polygon=None,
                    color=None,
                    affected_area_ratio=0.0
                ))
                
        # 5. Sort final detections by confidence descending
        final_detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Primary disease determination
        if final_detections:
            primary_disease = final_detections[0].disease
            primary_confidence = final_detections[0].confidence
        else:
            primary_disease = "Lúa khỏe mạnh / Không rõ bệnh"
            primary_confidence = 1.0
            
        latency = (time.time() - start_time) * 1000
        
        return ReScoringResult(
            disease=primary_disease,
            confidence=primary_confidence,
            detections=final_detections,
            status="success",
            model_version=pred_res.get("model_version", "re-scored"),
            latency_ms=round(latency, 2),
            annotated_image=pred_res.get("annotated_image")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Re-score endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
