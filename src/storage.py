from abc import ABC, abstractmethod
import os
import shutil
from fastapi import UploadFile
from config import settings
from src.logger import logger

# 1. Base Class for Storage Strategy
class BaseStorage(ABC):
    @abstractmethod
    async def save_image(self, file: UploadFile, filename: str) -> str:
        """Saves image and returns the path or URL."""
        pass

# 2. Local Storage Implementation
class LocalStorage(BaseStorage):
    def __init__(self, upload_dir: str = settings.UPLOAD_DIR):
        self.upload_dir = upload_dir
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)

    async def save_image(self, file: UploadFile, filename: str) -> str:
        try:
            file_path = os.path.join(self.upload_dir, filename)
            # Ensure seek is at start if file was read previously
            await file.seek(0)
            
            with open(file_path, "wb") as buffer:
                # Use shutil for efficient copying from the spoold file
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"Image saved locally to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save image locally: {e}")
            raise e

# 3. Cloud Storage Placeholder (Expansion point)
class S3CloudStorage(BaseStorage):
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        # Placeholder for Boto3 init
        logger.warning("S3CloudStorage initialized in placeholder mode.")

    async def save_image(self, file: UploadFile, filename: str) -> str:
        # Placeholder for S3 upload logic
        logger.info(f"Mock S3 upload for {filename} to bucket {self.bucket_name}")
        return f"https://{self.bucket_name}.s3.amazonaws.com/{filename}"

# 4. Factory Function
def get_storage_service() -> BaseStorage:
    storage_type = settings.STORAGE_TYPE.lower()
    
    if storage_type == "local":
        return LocalStorage()
    elif storage_type == "s3":
        # In actual implementation, bucket name would come from config
        return S3CloudStorage(bucket_name="luakhoe-production-images")
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
