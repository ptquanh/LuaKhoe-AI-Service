import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # API Settings
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # Environment
    IS_PRODUCTION: bool = False
    LOG_LEVEL: str = "INFO"

    # Storage Settings
    STORAGE_TYPE: str = "local"
    UPLOAD_DIR: str = "data/uploads"
    TEST_IMAGE_DIR: str = "tests/test_images"

    # Model Strategy
    ACTIVE_MODEL_STRATEGY: str = "onnx_runtime"
    MODEL_WEIGHTS_PATH: str = "models/v1_cnn_mvp.onnx"

    # Input Specifications
    IMAGE_INPUT_SIZE: int = 224

    # Inference Settings
    CONFIDENCE_THRESHOLD: float = 0.75

    # Disease Labels
    DISEASE_LABELS: str = "Healthy,Bacterial_Leaf_Blight,Brown_Spot,Leaf_Blast,Leaf_Smut"

    # Database (pgvector via asyncpg)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/luakhoe"

    # Google Gemini (Embeddings + LLM)
    GOOGLE_API_KEY: str = ""
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    # Groq LLM
    GROQ_API_KEY: str = ""
    GROQ_LLM_MODEL: str = "gemma2-9b-it"

    # RAG Pipeline Tuning
    RAG_SIMILARITY_THRESHOLD: float = 0.65
    RAG_TOP_K: int = 5
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200

    @property
    def labels_list(self) -> List[str]:
        return [label.strip() for label in self.DISEASE_LABELS.split(",")]

    @property
    def is_rag_enabled(self) -> bool:
        return bool(self.GOOGLE_API_KEY and self.DATABASE_URL and self.GROQ_API_KEY)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
