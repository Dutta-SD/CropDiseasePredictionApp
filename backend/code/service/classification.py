import keras.models
import numpy as np
from PIL import Image
from code.config import ModelDetails
from code.config import app_config
from code.service.components.customF1 import custom_f1


crop_disease_classifier: keras.Model = keras.models.load_model(
    app_config.CLASSIFIER_WEIGHTS_PATH,
    custom_objects={
        "custom_f1": custom_f1,
    },
)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess Image for classifier

    Args:
        img (Image.Image): Image file from frontend

    Returns:
        np.ndarray: Image as 3 dimensional array
    """
    img_resized = img.resize(app_config.CLASSIFIER_IMAGE_SIZE)
    arr = np.array(img_resized)[:, :, :3]
    return np.expand_dims(arr, 0).astype(np.float32)


def predict_image_class(img_array: np.ndarray) -> str:
    """Predicts disease of the crop

    Args:
        img_array (np.ndarray): Image as 3D numpy array

    Returns:
        str: class of the disease
    """
    output = crop_disease_classifier.predict(img_array)
    class_index = np.argmax(output.flatten())
    return ModelDetails.DISEASE_INDEX_TO_LABEL[class_index]
