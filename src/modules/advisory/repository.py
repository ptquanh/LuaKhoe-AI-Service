import uuid
from datetime import datetime, timezone

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from sqlalchemy import text

from src.shared.database import async_session_factory
from src.modules.advisory.models import DocumentChunk
from src.shared.utils.logger import logger
from config import settings
from src.modules.system.config_service import ConfigService


class VectorStoreService:
    """Manages document embeddings and pgvector similarity search."""

    def __init__(self):
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    async def embed_text(self, text_input: str) -> list[float]:
        return await self._embeddings.aembed_query(text_input)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)

    async def similarity_search(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        k = k or ConfigService.get_int("RAG_TOP_K", settings.RAG_TOP_K)
        score_threshold = score_threshold or ConfigService.get_float("RAG_SIMILARITY_THRESHOLD", settings.RAG_SIMILARITY_THRESHOLD)

        query_embedding = await self.embed_text(query)

        async with async_session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        id, content, source, chunk_metadata,
                        1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                    FROM document_chunks
                    WHERE 1 - (embedding <=> CAST(:query_vec AS vector)) >= :threshold
                    ORDER BY embedding <=> CAST(:query_vec AS vector)
                    LIMIT :k
                """),
                {
                    "query_vec": str(query_embedding),
                    "threshold": score_threshold,
                    "k": k,
                },
            )
            rows = result.fetchall()

        documents: list[tuple[Document, float]] = []
        for row in rows:
            doc = Document(
                page_content=row.content,
                metadata={
                    "source": row.source,
                    "similarity": float(row.similarity),
                    **(row.chunk_metadata or {}),
                },
            )
            documents.append((doc, float(row.similarity)))

        logger.info(
            f"Vector search: query='{query[:80]}...' → "
            f"{len(documents)} chunks (threshold={score_threshold})"
        )
        return documents

    async def add_documents(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0

        texts = [chunk.page_content for chunk in chunks]
        embeddings = await self.embed_texts(texts)

        async with async_session_factory() as session:
            for chunk, embedding in zip(chunks, embeddings):
                db_chunk = DocumentChunk(
                    id=uuid.uuid4(),
                    content=chunk.page_content,
                    source=chunk.metadata.get("source", "unknown"),
                    chunk_metadata={
                        k: v for k, v in chunk.metadata.items() if k != "source"
                    },
                    embedding=embedding,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(db_chunk)
            await session.commit()

        logger.info(f"Stored {len(chunks)} chunks in pgvector")
        return len(chunks)

    async def get_chunk_count(self) -> int:
        async with async_session_factory() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM document_chunks")
            )
            return result.scalar_one()
