from pydantic import BaseModel
from typing import List

class APIStatus(BaseModel):
    status: str
    ai_strategy: str
    storage_strategy: str
    labels: List[str]
    rag_enabled: bool = False
    rag_chunks_count: int = 0
