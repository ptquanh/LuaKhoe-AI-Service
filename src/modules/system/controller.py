from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.modules.system.dto import APIStatus
from config import settings
from src.shared.middlewares.dependencies import rag_state
from src.modules.system.config_service import ConfigService

router = APIRouter(tags=["System"])
admin_router = APIRouter(prefix="/admin/configs", tags=["Admin Configs"])

# Placeholder auth dependency
async def admin_required():
    pass

class ConfigCreate(BaseModel):
    key: str
    value: str
    description: str | None = None

class ConfigUpdate(BaseModel):
    value: str
    description: str | None = None

@admin_router.get("/", dependencies=[Depends(admin_required)])
async def list_configs():
    return await ConfigService.get_all_configs()

@admin_router.put("/{key}", dependencies=[Depends(admin_required)])
async def update_config(key: str, data: ConfigUpdate):
    await ConfigService.set_config(key, data.value, data.description)
    return {"status": "success", "message": f"Config {key} updated"}

@admin_router.post("/", dependencies=[Depends(admin_required)])
async def add_config(data: ConfigCreate):
    await ConfigService.set_config(data.key, data.value, data.description)
    return {"status": "success", "message": f"Config {data.key} added"}


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
