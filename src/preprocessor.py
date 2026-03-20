import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from config import settings

class ImagePreprocessor:
    def __init__(self, target_size: int = settings.IMAGE_INPUT_SIZE):
        self.target_size = (target_size, target_size)
        
        # Standard normalization for ImageNet-based models (MobileNet, EfficientNet)
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_pil(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for ONNX Runtime (returns numpy array)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Standardize using torchvision transforms then convert to numpy
        tensor = self.transform(image)
        return tensor.unsqueeze(0).numpy()

    def preprocess_torch(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for PyTorch models
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
        from io import BytesIO
        return Image.open(BytesIO(image_bytes))
