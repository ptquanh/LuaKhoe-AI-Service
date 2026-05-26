import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from src.shared.utils.logger import logger
from src.modules.predict import controller as predict
import uvicorn

app = FastAPI(
    title="Lúa Khỏe AI — Rice Disease Inference Service",
    description="Lightweight microservice for ONNX-based rice disease detection via image URL.",
    version="3.0.0",
)

# CORS — allow backend service to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(predict.router)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "online",
        "service": "luakhoe-ai-inference",
        "model_strategy": settings.ACTIVE_MODEL_STRATEGY,
    }


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Lua Khoe AI Inference Service v3.0")
    logger.info(f"Model Strategy: {settings.ACTIVE_MODEL_STRATEGY}")
    logger.info(f"Model Path: {settings.MODEL_WEIGHTS_PATH}")
    logger.info(f"Confidence Threshold: {settings.CONFIDENCE_THRESHOLD}")


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
