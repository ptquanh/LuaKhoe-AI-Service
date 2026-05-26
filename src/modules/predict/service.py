from abc import ABC, abstractmethod
import base64
import io
import os
# pyrefly: ignore [missing-import]
from PIL import Image
from config import settings
from typing import Dict, Any, List
from src.modules.predict.env_adjustment import adjust_prediction
from src.modules.predict.constants import DISEASE_HEALTHY

# 1. Base Class for Model Inference Strategy
class BaseModelInference(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        pass

    @abstractmethod
    def predict(
        self, 
        image: Any, 
        province: str = None, 
        gps_lat: float = None, 
        gps_lng: float = None, 
        field_params: dict = None,
        weather: dict = None
    ) -> Dict[str, Any]:
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
        # pyrefly: ignore [missing-import]
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='segment')

    def predict(
        self, 
        image, 
        province: str = None, 
        gps_lat: float = None, 
        gps_lng: float = None, 
        field_params: dict = None,
        weather: dict = None
    ) -> Dict[str, Any]:
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

        env_adjustment_result = None

        if detections and settings.ENV_ADJUSTMENT_ENABLED:
            # Build original scores dict {disease: max_confidence}
            original_scores = {d["disease"]: d["confidence"] for d in detections}
            
            # Apply env adjustment
            adjustment = adjust_prediction(
                yolo_scores=original_scores,
                province=province,
                gps_lat=gps_lat,
                gps_lng=gps_lng,
                field_params=field_params,
                weather=weather
            )
            
            # Update detections with adjusted scores
            adjusted_scores = adjustment["all_scores"]
            for d in detections:
                if d["disease"] in adjusted_scores:
                    d["confidence"] = adjusted_scores[d["disease"]]
                    
            # Sort detections by new confidence
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            
            env_adjustment_result = {
                "original_scores": original_scores,
                "adjusted_scores": adjusted_scores,
                "weather": adjustment["weather"],
                "applied": True,
            }

        # Primary disease = highest confidence detection (or Healthy)
        if detections:
            disease = detections[0]["disease"]
            confidence = detections[0]["confidence"]
        else:
            disease = DISEASE_HEALTHY
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
            "env_adjustment": env_adjustment_result,
        }

    def _aggregate_detections(self, result, num_detections: int) -> List[Dict[str, Any]]:
        """Aggregate detections by disease class, keeping highest confidence per class."""
        if num_detections == 0 or result.boxes is None:
            return []

        import numpy as np

        cls_array = result.boxes.cls.cpu().numpy()
        conf_array = result.boxes.conf.cpu().numpy()
        
        # Compute combined mask for each class to get accurate total affected area ratio
        class_masks: Dict[str, np.ndarray] = {}
        if result.masks is not None and result.masks.data is not None:
            for i in range(len(cls_array)):
                conf = float(conf_array[i])
                if conf < settings.CONFIDENCE_THRESHOLD:
                    continue
                class_idx = int(cls_array[i])
                name = self._resolve_class_name(class_idx)
                
                # result.masks.data is a tensor of shape [N, H, W]
                mask_np = result.masks.data[i].cpu().numpy()
                if name not in class_masks:
                    class_masks[name] = np.zeros_like(mask_np, dtype=bool)
                class_masks[name] |= (mask_np > 0)

        # pyrefly: ignore [missing-import]
        from ultralytics.utils.plotting import colors

        # Group by disease class → keep max confidence
        disease_map: Dict[str, Any] = {}
        for i in range(len(cls_array)):
            conf = float(conf_array[i])
            if conf < settings.CONFIDENCE_THRESHOLD:
                continue

            class_idx = int(cls_array[i])
            name = self._resolve_class_name(class_idx)
            
            c = colors(class_idx, bgr=False)
            hex_color = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

            # Get box (xyxy)
            box = result.boxes.xyxy[i].cpu().numpy().tolist()
            
            # Get polygon (mask) if available
            polygon = None
            if result.masks is not None:
                polygon = result.masks.xyn[i].tolist() # Normalized coordinates

            affected_area_ratio = 0.0
            if name in class_masks:
                mask_union = class_masks[name]
                affected_area_ratio = float(np.sum(mask_union) / mask_union.size)

            if name not in disease_map or conf > disease_map[name]["confidence"]:
                disease_map[name] = {
                    "confidence": conf,
                    "box": box,
                    "polygon": polygon,
                    "color": hex_color,
                    "affected_area_ratio": affected_area_ratio
                }

        # Sort by confidence descending
        return [
            {
                "disease": name, 
                "confidence": data["confidence"],
                "box": data["box"],
                "polygon": data["polygon"],
                "color": data["color"],
                "affected_area_ratio": data["affected_area_ratio"]
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
                labels=False,
                conf=False,
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
            "disease": DISEASE_HEALTHY, 
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
