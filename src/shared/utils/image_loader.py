import io
import requests
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from fastapi import HTTPException
from src.shared.utils.logger import logger


def download_image_from_url(url: str) -> Image.Image:
    """Download an image from a URL and return a PIL Image (RGB)."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="URL does not point to a valid image."
            )

        return Image.open(io.BytesIO(response.content)).convert("RGB")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from URL {url}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing downloaded image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file format.")
