import uuid
from datetime import datetime, timezone

from sqlalchemy import Text, String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

from src.shared.database import Base

EMBEDDING_DIMENSIONS = 3072  # Gemini text-embedding-004


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column(
        "chunk_metadata", JSON, default=dict
    )
    embedding = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        preview = self.content[:60] if self.content else ""
        return f"<DocumentChunk source='{self.source}' content='{preview}...'>"
