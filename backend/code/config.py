from dataclasses import dataclass, field


@dataclass
class Settings:
    """
    Configuration Constants
    """

    CLASSIFIER_IMAGE_SIZE: tuple = (128, 128)
    CLASSIFIER_WEIGHTS_PATH: str = "backend/static/weights/classifier_v2.h5"
    AUTOENCODER_WEIGHTS_PATH: str = "backend/static/weights/autoencoder_v1.h5"
    PORT: int = 9000
    APP_NAME: str = "crop-disease-prediction-api"
    APP_VERSION: str = "0.1.0"
    API_V1: str = "v1"
