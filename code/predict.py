from typing import Dict
import keras.models
import numpy as np
from PIL import Image
from keras.losses import mean_squared_error
from code.config import settings
from code.model.customF1 import custom_f1


outlier_detector = keras.models.load_model(settings.AUTOENCODER_WEIGHTS_PATH)
disease_classifier = keras.models.load_model(
    settings.CLASSIFIER_WEIGHTS_PATH,
    custom_objects={
        "custom_f1": custom_f1,
    },
)


def preprocess_image(img: Image.Image):
    # TODO: Add more preprocessing steps

    img_resized = img.resize(settings.IMAGE_SIZE)
    arr = np.array(img_resized)[:, :, :3]
    return np.expand_dims(arr, 0).astype(np.float32)


def check_if_outlier_or_not(img_array: np.ndarray):
    print("Outlier Checking")
    print(f"Image array shape is {img_array.shape}")
    output = outlier_detector.predict(img_array)
    print(output.shape)
    output_final, img_final = np.ravel(output), np.ravel(img_array)
    print(
        f"output final shape {output_final.shape} and input final shape {img_final.shape}"
    )
    err = mean_squared_error(output_final, img_final)
    print(f"MSE is {err}")
    return float(err) < 0.5


def predict_image_class(img_array: np.ndarray):
    return


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
    prediction = None
    if not is_outlier:
        prediction = predict_image_class(img_array)

    return make_final_prediction(prediction, is_outlier)
