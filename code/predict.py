from typing import Dict

import keras.models
import numpy as np
from PIL import Image
from keras.losses import mean_squared_error

from code.constants import IMAGE_SIZE, CLASSIFIER_WEIGHTS_PATH, AUTOENCODER_WEIGHTS_PATH

OUTLIER_DETECTION_MODEL = keras.models.load_model(AUTOENCODER_WEIGHTS_PATH)
# TODO: Unknown metric f1. Fix
# TODO: See Autoencoder in Kaggle
CLASSIFIER_MODEL = keras.models.load_model(CLASSIFIER_WEIGHTS_PATH)


def preprocess_image(img: Image.Image):
    # TODO: Add more preprocessing steps

    img_resized = img.resize(IMAGE_SIZE)
    arr = np.array(img_resized)[:, :, :3]
    return np.expand_dims(arr, 0).astype(np.float)


def check_if_outlier_or_not(img_array: np.array):
    print("Outlier Checking")
    print(f"Image array shape is {img_array.shape}")
    output = OUTLIER_DETECTION_MODEL.predict(img_array)
    print(output.shape)
    output_final, img_final = np.ravel(output), np.ravel(img_array)
    print(f"output final shape {output_final.shape} and input final shape {img_final.shape}")
    err = mean_squared_error(output_final, img_final)
    print(f"MSE is {err}")
    return float(err) < 0.5


def predict_image_class(img_array: np.array):
    return None


def make_final_prediction(prediction, is_outlier) -> Dict:
    return {"is_crop": is_outlier}


def predict(raw_img_array: Image.Image) -> Dict:
    """
    Prediction for the image
    :param raw_img_array: Image.Image file that is uploaded
    :return: Response
    """

    img_array = preprocess_image(raw_img_array)
    is_outlier = check_if_outlier_or_not(img_array)
    print(is_outlier)
    prediction = None
    # if not is_outlier:
    #     prediction = predict_image_class(img_array)

    return make_final_prediction(prediction, is_outlier)
