import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.modules.advisory.repository import VectorStoreService
from src.shared.utils.logger import logger
from config import settings


class IngestionPipeline:
    """Chunks raw text, embeds via Gemini, and stores in pgvector."""

    def __init__(self, vector_store: VectorStoreService):
        self._vector_store = vector_store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def ingest_text(
        self,
        text: str,
        source: str,
        metadata: dict | None = None,
    ) -> int:
        metadata = metadata or {}
        metadata["source"] = source

        raw_chunks = self._splitter.split_text(text)

        documents = [
            Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": i},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

        logger.info(
            f"[Ingest] Splitting '{source}' → {len(documents)} chunks "
            f"(size={settings.RAG_CHUNK_SIZE}, overlap={settings.RAG_CHUNK_OVERLAP})"
        )

        BATCH_SIZE = 20  # Giảm số lượng mỗi lô xuống thấp hơn
        SLEEP_TIME = 8   # Tăng thời gian nghỉ mặc định lên
        total_added = 0

        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            
            # Logic thử lại (Retry) khi gặp 429 Limit
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    count = await self._vector_store.add_documents(batch)
                    total_added += count
                    logger.info(f"[Ingest] Đã xử lý: {total_added}/{len(documents)} chunks từ '{source}'")
                    success = True
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        wait_time = (attempt + 1) * 30  # Lần 1: 30s, Lần 2: 60s, Lần 3: 90s
                        logger.warning(
                            f"[Ingest] Lỗi 429 Quota/Limit (Lần thử {attempt+1}/{max_retries}). "
                            f"Ngủ đông {wait_time}s rồi thử lại..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        # Nếu là lỗi khác (ví dụ sai dimensions 768 vs 3072), ném ra ngoài luôn
                        raise e
            
            if not success:
                logger.error(f"[Ingest] Thất bại ở lô từ {i} đến {i+BATCH_SIZE} sau {max_retries} lần thử.")
                # Ném lỗi để dừng tiến trình, hoặc bạn có thể bỏ dòng `raise` nếu muốn bỏ qua lô này
                raise Exception("Ingestion failed due to persistent 429 Rate Limit errors.")
                
            if i + BATCH_SIZE < len(documents):
                logger.info(f"[Ingest] Tạm nghỉ {SLEEP_TIME}s giữa các lô...")
                await asyncio.sleep(SLEEP_TIME)

        logger.info(f"[Ingest] Hoàn tất lưu {total_added} chunks từ '{source}'")
        return total_added
