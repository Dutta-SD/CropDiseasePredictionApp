from dataclasses import dataclass


@dataclass
class Settings:
    """
    Configuration Constants
    """

    IMAGE_SIZE: tuple = (256, 256)
    CLASSIFIER_WEIGHTS_PATH: str = "static/weights/classifier_v2.h5"
    AUTOENCODER_WEIGHTS_PATH: str = "static/weights/autoencoder_v1.h5"


settings = Settings()
