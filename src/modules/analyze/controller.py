import time
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from src.modules.analyze.dto import AnalyzeResponse
from src.modules.predict.service import BaseModelInference
from src.shared.utils.storage import BaseStorage
from src.shared.middlewares.dependencies import get_engine, get_storage, rag_state
from src.shared.utils.helpers import run_prediction_logic
from src.shared.utils.logger import logger

router = APIRouter(tags=["Analyze"])

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    engine: BaseModelInference = Depends(get_engine),
    storage: BaseStorage = Depends(get_storage),
):
    """
    Convenience endpoint: runs disease prediction → RAG recommendation
    in a single call. Returns both results.
    """
    try:
        # Step 1: Predict
        prediction = await run_prediction_logic(file, engine, storage)

        # Step 2: Recommend (only if RAG is available and disease is not Healthy)
        recommendation_response = None
        disease = prediction.get("disease", "")
        confidence = prediction.get("confidence", 0.0)

        if rag_state.recommendation_graph and disease.lower() != "healthy":
            rec_start = time.time()
            initial_state = {
                "disease_classes": [disease],
                "confidence": confidence,
                "rag_context": [],
                "recommendation": None,
                "error": None,
            }
            result = await rag_state.recommendation_graph.ainvoke(initial_state)

            rec = result.get("recommendation")
            if rec:
                rec_latency = (time.time() - rec_start) * 1000
                recommendation_response = {
                    "status": "success",
                    "recommendation": rec,
                    "latency_ms": round(rec_latency, 2),
                    "rag_chunks_used": len(result.get("rag_context", [])),
                }

        return {
            "prediction": prediction,
            "recommendation": recommendation_response,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
