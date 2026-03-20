from abc import ABC, abstractmethod
import onnxruntime as ort
import numpy as np
import torch
import os
from config import settings
from typing import Dict, Any, Union

# 1. Base Class for Model Inference Strategy
class BaseModelInference(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        pass

# 2. ONNX Implementation (Recommended for Production)
class ONNXInference(BaseModelInference):
    def __init__(self):
        self.session = None
        self.input_name = None
        self.labels = settings.labels_list

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Running in mock mode.")
            return
        
        # Initialize ONNX Runtime Session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_np: np.ndarray) -> Dict[str, Any]:
        if self.session is None:
            # Mock return for development if file is missing
            return self._mock_predict()
            
        # Run inference
        outputs = self.session.run(None, {self.input_name: image_np})
        logits = outputs[0]
        
        # Softmax and Extract Result
        probs = self._softmax(logits[0])
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        
        return {
            "disease": self.labels[class_idx],
            "confidence": confidence,
            "status": "success",
            "model_version": settings.ACTIVE_MODEL_STRATEGY
        }

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _mock_predict(self):
        return {
            "disease": "Healthy", 
            "confidence": 0.99, 
            "status": "mock_success",
            "model_version": "mock_v1.0",
            "note": "Pretrained ONNX model file not found, returned mock result."
        }

# 3. Simple CNN (PyTorch) Implementation
class PyTorchInference(BaseModelInference):
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = settings.labels_list

    def load_model(self, model_path: str):
        # Implementation for loading .pth weights would go here
        # For now, we remain flexible to the architecture
        pass

    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        # Logic for PyTorch inference
        return {"disease": "Brown_Spot", "confidence": 0.88}

# 4. Factory Function
def get_inference_engine() -> BaseModelInference:
    strategy = settings.ACTIVE_MODEL_STRATEGY
    
    if strategy == "onnx_runtime":
        instance = ONNXInference()
    elif strategy == "torch":
        instance = PyTorchInference()
    else:
        raise ValueError(f"Unknown inference strategy: {strategy}")
    
    instance.load_model(settings.MODEL_WEIGHTS_PATH)
    return instance
