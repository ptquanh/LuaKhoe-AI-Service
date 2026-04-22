from fastapi import APIRouter
from api.routers import system, predict, rag, analyze

router = APIRouter()
router.include_router(system.router)
router.include_router(predict.router)
router.include_router(rag.router)
router.include_router(analyze.router)
