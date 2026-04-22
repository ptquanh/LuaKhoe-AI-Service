import os
import sys

# Ensure project root is in path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from src.shared.utils.logger import logger
import uvicorn

# Import Routers
from src.modules.system import controller as system
from src.modules.predict import controller as predict
from src.modules.advisory import controller as rag
from src.modules.analyze import controller as analyze
from src.shared.middlewares.dependencies import rag_state

app = FastAPI(
    title="Lúa Khỏe AI - Rice Disease Detection & Advisory",
    description=(
        "Scalable API with Strategy Pattern for AI Inference, Storage, "
        "and RAG-based agronomic recommendations."
    ),
    version="2.0.0"
)

# Enable CORS for frontend/backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(system.router)
app.include_router(predict.router)
app.include_router(rag.router)
app.include_router(analyze.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Lúa Khỏe AI Service v2.0...")
    logger.info(f"AI Strategy: {settings.ACTIVE_MODEL_STRATEGY}")
    logger.info(f"Storage Strategy: {settings.STORAGE_TYPE}")

    # Initialize RAG pipeline if credentials are present
    if settings.is_rag_enabled:
        try:
            from src.shared.database import init_db
            from src.modules.advisory.repository import VectorStoreService
            from src.modules.advisory.recommendation_graph import RecommendationGraphBuilder
            from src.modules.advisory.ingestion import IngestionPipeline

            await init_db()

            rag_state.vector_store = VectorStoreService()
            builder = RecommendationGraphBuilder(rag_state.vector_store)
            rag_state.recommendation_graph = builder.build()
            rag_state.ingestion_pipeline = IngestionPipeline(rag_state.vector_store)

            chunk_count = await rag_state.vector_store.get_chunk_count()
            logger.info(
                f"RAG pipeline initialized — "
                f"Groq ({settings.GROQ_LLM_MODEL}) + "
                f"Gemini Embeddings ({settings.GEMINI_EMBEDDING_MODEL}) + pgvector "
                f"({chunk_count} chunks in DB)"
            )
        except Exception as e:
            logger.error(f"RAG pipeline initialization failed: {e}")
            logger.warning("Continuing without RAG — /recommend will return 503")
    else:
        logger.warning(
            "RAG pipeline disabled — GOOGLE_API_KEY, GROQ_API_KEY, or DATABASE_URL not set"
        )

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
