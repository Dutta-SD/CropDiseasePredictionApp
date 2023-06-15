from dataclasses import dataclass
from pydantic import BaseSettings


class Settings(BaseSettings):
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
    REDIS_URL: str = "redis-19211.c246.us-east-1-4.ec2.cloud.redislabs.com"
    REDIS_TTL: int = 100
    REDIS_PASSWORD: str
    LOGGER_NAME: str = "crop-logger"

    class Config:
        env_file = "./backend/.env"


app_config = Settings()


@dataclass
class ModelDetails:
    """Information regarding deep learning model"""

    DISEASE_INDEX_TO_LABEL: tuple = (
        "Tomato Late Blight",
        "Tomato healthy",
        "Grape healthy",
        "Orange Haunglongbing (Citrus greening)",
        "Soybean healthy",
        "Squash Powdery mildew",
        "Potato healthy",
        "Corn (maize) Northern Leaf Blight",
        "Tomato Early blight",
        "Tomato Septoria leaf spot",
        "Corn (maize) Cercospora leaf spot Gray leaf spot",
        "Strawberry Leaf scorch",
        "Peach healthy",
        "Apple Apple scab",
        "Tomato Tomato Yellow Leaf Curl Virus",
        "Tomato Bacterial spot",
        "Apple Black rot",
        "Blueberry healthy",
        "Cherry (including sour) Powdery mildew",
        "Peach Bacterial spot",
        "Apple Cedar apple rust",
        "Tomato Target Spot",
        "Pepper, bell healthy",
        "Grape Leaf blight (Isariopsis Leaf Spot)",
        "Potato Late blight",
        "Tomato Tomato mosaic virus",
        "Strawberry healthy",
        "Apple healthy",
        "Grape Black rot",
        "Potato Early blight",
        "Cherry (including sour) healthy",
        "Corn (maize) Common rust ",
        "Grape Esca (Black Measles)",
        "Raspberry healthy",
        "Tomato Leaf Mold",
        "Tomato Spider mites Two-spotted spider mite",
        "Pepper, bell Bacterial spot",
        "Corn (maize) healthy",
    )
