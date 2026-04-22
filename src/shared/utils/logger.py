import sys
from loguru import logger
from config import settings

def setup_logger():
    # Remove default handler
    logger.remove()
    
    # Add stdout handler with color/format
    logger.add(
        sys.stdout, 
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler for production logs
    if settings.IS_PRODUCTION:
        logger.add(
            "logs/api_{time}.log",
            rotation="10 MB",
            retention="10 days",
            compression="zip",
            level="WARNING"
        )

# Initialize on import
setup_logger()
