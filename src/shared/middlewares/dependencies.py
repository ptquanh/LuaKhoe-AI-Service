from fastapi import HTTPException
from src.modules.predict.service import get_inference_engine, BaseModelInference
from src.shared.utils.storage import get_storage_service, BaseStorage
from src.shared.utils.preprocessor import ImagePreprocessor
from src.shared.utils.validation import DataValidator

# Persistent Components
preprocessor = ImagePreprocessor()
validator = DataValidator()

# Dependency Providers
def get_engine() -> BaseModelInference:
    return get_inference_engine()

def get_storage() -> BaseStorage:
    return get_storage_service()

class RAGState:
    vector_store = None
    recommendation_graph = None
    ingestion_pipeline = None

rag_state = RAGState()

def require_rag():
    """Guard: raises 503 if the RAG pipeline was not initialized."""
    if rag_state.recommendation_graph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG pipeline is not available. "
                "Ensure GOOGLE_API_KEY and DATABASE_URL are configured."
            ),
        )
