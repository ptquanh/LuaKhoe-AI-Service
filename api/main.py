import os
import sys
import time

# Ensure project root is in path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.core_inference import get_inference_engine, BaseModelInference
from src.storage import get_storage_service, BaseStorage
from src.preprocessor import ImagePreprocessor
from src.validation import DataValidator
from src.logger import logger
from api.schemas import APIStatus, PredictionResult
from config import settings
import uvicorn
import uuid
import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # 1. Đọc ảnh bằng hệ màu RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. THÊM VIỀN ĐỂ ẢNH THÀNH HÌNH VUÔNG (Letterboxing)
    max_size = max(img.size)
    new_image = Image.new("RGB", (max_size, max_size), (0, 0, 0))
    new_image.paste(img, ((max_size - img.size[0]) // 2,
                           (max_size - img.size[1]) // 2))
    
    # 3. Resize về 300x300 an toàn không bị bóp méo
    new_image = new_image.resize((300, 300))
    
    # 4. Chuyển thành numpy array và chuẩn hóa (InceptionResNetV2)
    image_np = np.array(new_image)
    image_np = (image_np.astype(np.float32) / 127.5) - 1.0
    
    # 5. Thêm chiều batch (1) -> (1, 300, 300, 3)
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

app = FastAPI(
    title="Lúa Khỏe AI - Rice Disease Detection",
    description="Scalable API with Strategy Pattern for both AI Inference and Storage.",
    version="1.2.0"
)

# Enable CORS for frontend/backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency Providers
def get_engine() -> BaseModelInference:
    return get_inference_engine()

def get_storage() -> BaseStorage:
    return get_storage_service()

# Initialize Persistent Components
preprocessor = ImagePreprocessor()
validator = DataValidator()

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Lúa Khỏe AI Service...")
    logger.info(f"AI Strategy: {settings.ACTIVE_MODEL_STRATEGY}")
    logger.info(f"Storage Strategy: {settings.STORAGE_TYPE}")

@app.get("/status", response_model=APIStatus)
async def get_status():
    return {
        "status": "online",
        "ai_strategy": settings.ACTIVE_MODEL_STRATEGY,
        "storage_strategy": settings.STORAGE_TYPE,
        "labels": settings.labels_list
    }

@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    engine: BaseModelInference = Depends(get_engine),
    storage: BaseStorage = Depends(get_storage)
):
    start_time = time.time()
    
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        # 1. Image Loading & Initial Validation
        image_bytes = await file.read()
        image = preprocessor.load_image_from_bytes(image_bytes)
        
        if not validator.is_valid_image(image):
            raise HTTPException(status_code=400, detail="Invalid rice leaf image.")
        
        # 2. Save Image (for traceability/active learning)
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        saved_path = await storage.save_image(file, unique_filename)
        
        # 3. Model Prediction
        image_np = preprocess_image(image_bytes)
        result = engine.predict(image_np)
        
        # 4. Post-inference Checks
        result = validator.check_confidence(result)
        
        # 5. Metadata
        latency = (time.time() - start_time) * 1000
        result.update({
            "latency_ms": round(latency, 2),
            "saved_path": saved_path,
            "filename": unique_filename
        })
        
        logger.info(f"Predicted {result['disease']} in {latency:.1f}ms (Saved to {settings.STORAGE_TYPE})")
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Prediction endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
