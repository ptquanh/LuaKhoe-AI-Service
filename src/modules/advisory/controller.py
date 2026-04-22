import time
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from src.modules.advisory.dto import (
    RecommendationRequest,
    RecommendationResponse,
    IngestionRequest,
    IngestionResponse,
)
from src.shared.middlewares.dependencies import require_rag, rag_state
from src.shared.utils.logger import logger
from src.modules.advisory.document_parser import extract_text_from_file

router = APIRouter(tags=["RAG"])

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Generate RAG-grounded agronomic recommendation for a detected disease."""
    require_rag()
    start_time = time.time()

    initial_state = {
        "disease_classes": [request.disease],
        "confidence": request.confidence,
        "rag_context": [],
        "recommendation": None,
        "error": None,
    }

    try:
        result = await rag_state.recommendation_graph.ainvoke(initial_state)

        recommendation = result.get("recommendation")
        if recommendation is None:
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline error: {result.get('error', 'Unknown')}",
            )

        latency = (time.time() - start_time) * 1000
        return {
            "status": "success",
            "recommendation": recommendation,
            "latency_ms": round(latency, 2),
            "rag_chunks_used": len(result.get("rag_context", [])),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Recommendation endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

async def _safe_ingest_background_task(text: str, source: str, metadata: dict):
    """Safe background worker wrapper with explicit error handling."""
    try:
        logger.info(f"Background ingestion started for source: {source}")
        chunks_created = await rag_state.ingestion_pipeline.ingest_text(
            text=text,
            source=source,
            metadata=metadata,
        )
        total = await rag_state.vector_store.get_chunk_count()
        logger.info(
            f"Background ingestion successful for '{source}'. "
            f"Created {chunks_created} chunks. Total in DB: {total}."
        )
    except Exception as e:
        logger.exception("Background ingestion failed for '" + source + "'")


@router.post("/ingest", response_model=IngestionResponse, status_code=202)
async def ingest(request: IngestionRequest, background_tasks: BackgroundTasks):
    """Ingest raw text into the pgvector knowledge base in the background."""
    require_rag()

    # Add the ingestion job to the background queue
    background_tasks.add_task(
        _safe_ingest_background_task,
        text=request.text,
        source=request.source,
        metadata=request.metadata,
    )

    return {
        "status": "processing",
        "message": f"Document ingestion started in the background for '{request.source}'",
    }

async def _safe_ingest_file_background_task(file_bytes: bytes, filename: str):
    """Background task to extract, clean, and ingest a document."""
    try:
        logger.info(f"Background document parsing started for: {filename}")
        
        cleaned_text = extract_text_from_file(file_bytes, filename)
        
        if not cleaned_text:
            raise ValueError("Extracted text is empty after cleaning.")

        chunks_created = await rag_state.ingestion_pipeline.ingest_text(
            text=cleaned_text,
            source=filename,
            metadata={"filename": filename},
        )
        total = await rag_state.vector_store.get_chunk_count()
        logger.info(
            f"Background file ingestion successful for '{filename}'. "
            f"Created {chunks_created} chunks. Total in DB: {total}."
        )
    except Exception as e:
        logger.exception("Background file ingestion failed for '" + filename + "'")


@router.post("/ingest/file", response_model=IngestionResponse, status_code=202)
async def ingest_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Ingest a raw document (.pdf, .docx, .txt) into the knowledge base."""
    require_rag()

    # Read the file into memory
    file_bytes = await file.read()
    filename = file.filename

    # Add the file processing job to the background queue
    background_tasks.add_task(
        _safe_ingest_file_background_task,
        file_bytes=file_bytes,
        filename=filename,
    )

    return {
        "status": "processing",
        "message": f"Document ingestion started in the background for '{filename}'",
    }
