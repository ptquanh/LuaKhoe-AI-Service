# Implementation Plan - Rice Plant Disease Detection (LuaKhoe-AI)

Deploy a rice plant disease detection system using deep learning and FastAPI, designed for integration with backend services.

## Proposed Changes

### Environment & Setup
- [NEW] [.gitignore](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/.gitignore): Ignore `venv`, `__pycache__`, `.env`, `data/`, `.idea/`, and `models/*.onnx`.
- [NEW] [requirements.txt](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/requirements.txt): Added `onnxruntime`, `pydantic-settings`, `loguru`.
- [NEW] [Dockerfile](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/Dockerfile) & [docker-compose.yml](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/docker-compose.yml): Containerization for independent deployment.
- [NEW] [config.py](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/config.py): Centralized configuration management using `pydantic-settings`.
- [NEW] `.env`: Added `STORAGE_TYPE`, `UPLOAD_DIR`, `LOG_LEVEL`, and `IS_PRODUCTION` flags.

### Core Logic & Training (src/)
- [NEW] `src/storage.py`: Strategy Pattern for image storage (Local/S3).
- [NEW] [src/validation.py](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/src/validation.py): Logic to filter out non-leaf/non-rice images.
- [NEW] [src/logger.py](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/src/logger.py): Logging configuration using `loguru`.
- [NEW] [src/preprocessor.py](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/src/preprocessor.py): Decoupled image preprocessing logic.
- [NEW] [src/core_inference.py](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/src/core_inference.py): Strategy Pattern implementation with [BaseModelInference](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/src/core_inference.py#10-18) and [ONNXInference](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/src/core_inference.py#20-67).
- [NEW] `train_rice_diseases.ipynb`: Google Colab notebook for training and ONNX export.

### API Layer (api/)
- [NEW] [api/main.py](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/api/main.py): FastAPI application with `/predict` and `/status` endpoints.
- [NEW] `api/schemas.py`: Pydantic models for API requests/responses.

### Testing (tests/)
- [NEW] [tests/download_test_images.py](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/tests/download_test_images.py): Script to download specific "strange" rice disease images from URLs for testing.
- [NEW] `tests/test_api.py`: Script to test the FastAPI endpoint using `requests`.

## Verification Plan

### Automated Tests
- Run `python tests/test_api.py` to verify the prediction endpoint returns the expected JSON structure.
- Run `pytest` for unit tests of the inference logic.

### Manual Verification
1. Start the API: `uvicorn api.main:app --reload`
2. Access `http://localhost:8000/docs` to test via Swagger UI.
3. Test with a pretrained model (weighted for Rice Disease if available, otherwise generic ImageNet as baseline).
4. Run [tests/download_test_images.py](file:///d:/CODE%20PLAYGROUND/Projects/Fullstack/L%C3%BAa%20Kh%E1%BB%8Fe/LuaKhoe-AI/tests/download_test_images.py) and upload for classification analysis.
