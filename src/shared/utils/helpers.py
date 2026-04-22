import time
import uuid
import io
import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException

from src.modules.predict.service import BaseModelInference
from src.shared.utils.storage import BaseStorage
from src.shared.middlewares.dependencies import preprocessor, validator
from src.shared.utils.logger import logger
from config import settings

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # 1. Read image as RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Add padding to make image square (Letterboxing)
    max_size = max(img.size)
    new_image = Image.new("RGB", (max_size, max_size), (0, 0, 0))
    new_image.paste(img, ((max_size - img.size[0]) // 2,
                           (max_size - img.size[1]) // 2))
    
    # 3. Resize to 300x300 without distortion
    new_image = new_image.resize((300, 300))
    
    # 4. Convert to numpy array and normalize
    image_np = np.array(new_image)
    image_np = (image_np.astype(np.float32) / 127.5) - 1.0
    
    # 5. Add batch dimension (1) -> (1, 300, 300, 3)
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

async def run_prediction_logic(
    file: UploadFile,
    engine: BaseModelInference,
    storage: BaseStorage,
) -> dict:
    """Core prediction logic shared by /predict and /analyze."""
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

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
        "filename": unique_filename,
    })

    logger.info(
        f"Predicted {result['disease']} in {latency:.1f}ms "
        f"(Saved to {settings.STORAGE_TYPE})"
    )
    return result
