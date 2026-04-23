from typing import Dict, Any, Optional
from sqlalchemy import select
from datetime import datetime, timezone

from src.shared.database import async_session_factory
from src.modules.system.models import SystemConfig
from src.shared.utils.logger import logger

class ConfigService:
    _cache: Dict[str, Any] = {}
    
    @classmethod
    async def load_all(cls):
        """Loads all configs from DB into the in-memory cache."""
        async with async_session_factory() as session:
            result = await session.execute(select(SystemConfig))
            configs = result.scalars().all()
            for config in configs:
                cls._cache[config.key] = config.value
        logger.info(f"Loaded {len(cls._cache)} configurations into cache.")

    @classmethod
    def get_str(cls, key: str, default: str = "") -> str:
        return str(cls._cache.get(key, default))

    @classmethod
    def get_int(cls, key: str, default: int = 0) -> int:
        val = cls._cache.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
        return default

    @classmethod
    def get_float(cls, key: str, default: float = 0.0) -> float:
        val = cls._cache.get(key)
        if val is not None:
            try:
                return float(val)
            except ValueError:
                pass
        return default

    @classmethod
    async def set_config(cls, key: str, value: str, description: Optional[str] = None):
        """Sets or updates a config in the DB and refreshes the cache."""
        async with async_session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(SystemConfig).where(SystemConfig.key == key)
                )
                config = result.scalar_one_or_none()
                if config:
                    config.value = value
                    if description is not None:
                        config.description = description
                    config.updated_at = datetime.now(timezone.utc)
                else:
                    config = SystemConfig(
                        key=key,
                        value=value,
                        description=description,
                        updated_at=datetime.now(timezone.utc)
                    )
                    session.add(config)
            
            # Refresh cache after commit
            cls._cache[key] = value
            logger.info(f"Updated configuration '{key}'.")

    @classmethod
    async def get_all_configs(cls) -> list[dict]:
        """Fetches all configurations directly from DB for Admin API."""
        async with async_session_factory() as session:
            result = await session.execute(select(SystemConfig))
            configs = result.scalars().all()
            return [
                {
                    "key": c.key,
                    "value": c.value,
                    "description": c.description,
                    "updated_at": c.updated_at
                }
                for c in configs
            ]
