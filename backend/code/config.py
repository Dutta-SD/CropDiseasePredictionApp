from dataclasses import dataclass


@dataclass
class Settings:
    """
    Configuration Constants
    """

    IMAGE_SIZE: tuple = (256, 256)
    CLASSIFIER_WEIGHTS_PATH: str = "backend/static/weights/classifier_v2.h5"
    AUTOENCODER_WEIGHTS_PATH: str = "backend/static/weights/autoencoder_v1.h5"
    PORT: int = 9000
    APP_NAME = "crop-disease-prediction-api"
    APP_VERSION = "0.1.0"
