from abc import ABC, abstractmethod
import base64
import io
import os
from PIL import Image
from config import settings
from typing import Dict, Any, List

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
        self.model = None
        self.labels = settings.labels_list

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Running in mock mode.")
            return
        
        # Initialize ultralytics YOLO with the ONNX model
        # ultralytics natively supports .onnx files and handles preprocessing automatically! 
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='segment')

    def predict(self, image) -> Dict[str, Any]:
        if self.model is None:
            # Mock return for development if file is missing
            return self._mock_predict()
            
        # Run pure inference, similar to the provided snippet
        preds = self.model.predict(
            source=image,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.IMAGE_INPUT_SIZE,
            save=False,
            verbose=False,
        )
        
        result = preds[0]

        # Parse all detections
        num_detections = len(result.boxes) if result.boxes is not None else 0
        detections = self._aggregate_detections(result, num_detections)

        # Primary disease = highest confidence detection (or Healthy)
        if detections:
            disease = detections[0]["disease"]
            confidence = detections[0]["confidence"]
        else:
            disease = "Healthy"
            confidence = 1.0

        # Generate annotated image with masks, bboxes, and labels
        annotated_image_b64 = self._render_annotated_image(result, num_detections)

        return {
            "disease": disease,
            "confidence": confidence,
            "detections": detections,
            "status": "success",
            "model_version": settings.ACTIVE_MODEL_STRATEGY,
            "annotated_image": annotated_image_b64,
        }

    def _aggregate_detections(self, result, num_detections: int) -> List[Dict[str, Any]]:
        """Aggregate detections by disease class, keeping highest confidence per class."""
        if num_detections == 0 or result.boxes is None:
            return []

        cls_array = result.boxes.cls.cpu().numpy()
        conf_array = result.boxes.conf.cpu().numpy()

        # Group by disease class → keep max confidence
        disease_map: Dict[str, float] = {}
        for i in range(len(cls_array)):
            conf = float(conf_array[i])
            if conf < settings.CONFIDENCE_THRESHOLD:
                continue

            class_idx = int(cls_array[i])
            name = self._resolve_class_name(class_idx)

            # Get box (xyxy)
            box = result.boxes.xyxy[i].cpu().numpy().tolist()
            
            # Get polygon (mask) if available
            polygon = None
            if result.masks is not None:
                polygon = result.masks.xyn[i].tolist() # Normalized coordinates

            if name not in disease_map or conf > disease_map[name]["confidence"]:
                disease_map[name] = {
                    "confidence": conf,
                    "box": box,
                    "polygon": polygon
                }

        # Sort by confidence descending
        return [
            {
                "disease": name, 
                "confidence": data["confidence"],
                "box": data["box"],
                "polygon": data["polygon"]
            }
            for name, data in sorted(disease_map.items(), key=lambda x: x[1]["confidence"], reverse=True)
        ]

    def _resolve_class_name(self, class_idx: int) -> str:
        """Resolve class index to disease name."""
        if hasattr(self.model, "names") and class_idx in self.model.names:
            return self.model.names[class_idx]
        if class_idx < len(self.labels):
            return self.labels[class_idx]
        return f"Class_{class_idx}"

    def _render_annotated_image(self, result, num_detections: int) -> str | None:
        """Render YOLO result with masks/bboxes/labels onto the image and return as base64 PNG."""
        if num_detections == 0:
            return None

        try:
            # result.plot() returns a BGR numpy array with annotations drawn
            annotated_bgr = result.plot(
                conf=True,
                line_width=2,
                font_size=None,
                pil=False,
                img=None,
            )
            # Convert BGR (OpenCV) → RGB (PIL)
            annotated_rgb = annotated_bgr[..., ::-1]
            pil_img = Image.fromarray(annotated_rgb)

            # Encode to base64 PNG
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG", optimize=True)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")
        except Exception:
            return None

    def _mock_predict(self):
        return {
            "disease": "Healthy", 
            "confidence": 0.99,
            "detections": [],
            "status": "mock_success",
            "model_version": "mock_v1.0",
            "annotated_image": None,
            "note": "Pretrained ONNX model file not found, returned mock result."
        }


# 3. Factory Function
def get_inference_engine() -> BaseModelInference:
    strategy = settings.ACTIVE_MODEL_STRATEGY
    
    if strategy == "onnx_runtime":
        instance = ONNXInference()
    else:
        raise ValueError(f"Unknown inference strategy: {strategy}")
    
    instance.load_model(settings.MODEL_WEIGHTS_PATH)
    return instance
