import numpy as np
from PIL import Image
from config import settings
from src.shared.utils.logger import logger

class DataValidator:
    """
    Validates input data to ensure it's a valid rice leaf image 
    before intensive processing.
    """
    @staticmethod
    def is_valid_image(image: Image.Image) -> bool:
        # 1. Basic check: Dimensions
        width, height = image.size
        if width < 100 or height < 100:
            logger.warning(f"Image too small: {width}x{height}")
            return False
        
        # 2. Basic check: Aspect Ratio (Extreme aspect ratios are likely not photos)
        aspect_ratio = width / height
        if aspect_ratio > 4 or aspect_ratio < 0.25:
            logger.warning(f"Extreme aspect ratio: {aspect_ratio}")
            return False
            
        return True

    @staticmethod
    def check_confidence(prediction_result: dict):
        """
        Check if AI is confident enough. If not, log for future training.
        """
        confidence = prediction_result.get("confidence", 0)
        if confidence < settings.CONFIDENCE_THRESHOLD:
            logger.info(
                f"Low confidence prediction: {prediction_result['disease']} "
                f"({confidence:.2f} < {settings.CONFIDENCE_THRESHOLD})"
            )
            # Tag as low confidence for future data collection
            prediction_result["low_confidence"] = True
            
        return prediction_result
