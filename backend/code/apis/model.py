import io
from fastapi import UploadFile
from fastapi.routing import APIRouter
from typing import Dict
from code.config import app_config
from code.utilities import api_utils
from PIL import Image
from code.service import classification


prediction_router = APIRouter(prefix=f"/{app_config.API_V1}/model")


@prediction_router.get("", status_code=200, tags=["prediction"])
async def get_model_homepage() -> Dict:
    """Fetch Sample Output to check if API working or not

    Returns:
        Dict: JSON response
    """
    return api_utils.success_response({"response": "OK"})


@prediction_router.post("/predict", status_code=201, tags=["prediction"])
async def get_prediction(img: UploadFile) -> Dict:
    """Gets classification result for crop disease

    Args:
        request (Request): User Request

    Returns:
        Dict: JSON prediction for disease
    """
    image: Image.Image = Image.open(io.BytesIO(img.file.read()))
    np_image = classification.preprocess_image(image)
    class_prediction = classification.predict_image_class(np_image)
    return api_utils.success_response(
        {
            "class_index": class_prediction,
        }
    )
