from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from src.modules.predict.dto import PredictionResult
from src.modules.predict.service import BaseModelInference
from src.shared.utils.storage import BaseStorage
from src.shared.middlewares.dependencies import get_engine, get_storage
from src.shared.utils.helpers import run_prediction_logic
from src.shared.utils.logger import logger

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    engine: BaseModelInference = Depends(get_engine),
    storage: BaseStorage = Depends(get_storage),
):
    try:
        return await run_prediction_logic(file, engine, storage)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
