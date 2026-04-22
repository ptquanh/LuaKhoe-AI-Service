import torch
import torchvision.models as models
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def export_dummy_model():
    """
    Exports a pretrained MobileNetV3-Small to ONNX format 
    with a classification head matching our disease labels.
    """
    print("Initiating dummy model export...")
    
    # 1. Load Pretrained MobileNetV3
    model = models.mobilenet_v3_small(weights='DEFAULT')
    
    # 2. Modify Head for Rice Diseases
    num_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(num_features, len(settings.labels_list))
    
    model.eval()
    
    # 3. Create dummy input
    dummy_input = torch.randn(1, 3, settings.IMAGE_INPUT_SIZE, settings.IMAGE_INPUT_SIZE)
    
    # 4. Export to ONNX
    os.makedirs(os.path.dirname(settings.MODEL_WEIGHTS_PATH), exist_ok=True)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        settings.MODEL_WEIGHTS_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported successfully to {settings.MODEL_WEIGHTS_PATH}")

if __name__ == "__main__":
    export_dummy_model()
