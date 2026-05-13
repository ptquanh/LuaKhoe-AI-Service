from src.modules.predict.service import get_inference_engine, BaseModelInference
from src.shared.utils.validation import DataValidator

# Persistent Components
validator = DataValidator()

# Dependency Providers
def get_engine() -> BaseModelInference:
    return get_inference_engine()
