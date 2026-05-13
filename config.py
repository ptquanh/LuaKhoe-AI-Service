from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # API Settings
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Environment
    IS_PRODUCTION: bool = False
    LOG_LEVEL: str = "INFO"

    # Model Configuration
    ACTIVE_MODEL_STRATEGY: str = "onnx_runtime"
    MODEL_WEIGHTS_PATH: str = "models/v1_cnn_mvp.onnx"

    # Input Specifications
    IMAGE_INPUT_SIZE: int = 224

    # Inference Settings
    CONFIDENCE_THRESHOLD: float = 0.75
    IOU_THRESHOLD: float = 0.45

    # Disease Labels (comma-separated)
    DISEASE_LABELS: str = "Healthy,Bacterial_Leaf_Blight,Brown_Spot,Leaf_Blast,Leaf_Smut"

    @property
    def labels_list(self) -> List[str]:
        return [label.strip() for label in self.DISEASE_LABELS.split(",")]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
