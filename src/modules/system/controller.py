from fastapi import APIRouter
from src.modules.system.dto import APIStatus
from config import settings
from src.shared.middlewares.dependencies import rag_state

router = APIRouter(tags=["System"])

@router.get("/status", response_model=APIStatus)
async def get_status():
    rag_chunks = 0
    if rag_state.vector_store:
        try:
            rag_chunks = await rag_state.vector_store.get_chunk_count()
        except Exception:
            pass

    return {
        "status": "online",
        "ai_strategy": settings.ACTIVE_MODEL_STRATEGY,
        "storage_strategy": settings.STORAGE_TYPE,
        "labels": settings.labels_list,
        "rag_enabled": rag_state.recommendation_graph is not None,
        "rag_chunks_count": rag_chunks,
    }
