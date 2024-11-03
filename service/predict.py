import numpy as np
import torch
from ml.app.config import ModelConfig
from ml.app.models.classification import DiseaseClassificationModel
import torchvision.transforms.functional as F

from service.external import get_gemini_response

ID2LABEL = [
    "Apple Apple scab",
    "Apple Black rot",
    "Apple Cedar apple rust",
    "Apple healthy",
    "Blueberry healthy",
    "Cherry Powdery mildew",
    "Cherry healthy",
    "Corn Cercospora leaf spot Gray leaf spot",
    "Corn Common rust",
    "Corn Northern Leaf Blight",
    "Corn healthy",
    "Grape Black rot",
    "Grape Esca Black Measles",
    "Grape Leaf blight Isariopsis Leaf Spot",
    "Grape healthy",
    "Orange Haunglongbing Citrus greening",
    "Peach Bacterial spot",
    "Peach healthy",
    "Pepper bell Bacterial spot",
    "Pepper bell healthy",
    "Potato Early blight",
    "Potato Late blight",
    "Potato healthy",
    "Raspberry healthy",
    "Soybean healthy",
    "Squash Powdery mildew",
    "Strawberry Leaf scorch",
    "Strawberry healthy",
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites Two spotted spider mite",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato mosaic virus",
    "Tomato healthy",
]


def load_my_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]

    # Create a new state dict with the correct keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.model."):
            new_key = k.replace("model.model.", "model.")
            new_state_dict[new_key] = v

    # Load the new state dict
    model.load_state_dict(new_state_dict)
    return model


CLF_MODEL = DiseaseClassificationModel(model_name=ModelConfig.PRETRAINED_MODEL_NAME)
CLF_MODEL = load_my_model(ModelConfig.CLASSIFY_MODEL_CHECKPOINT, CLF_MODEL)


def transform_for_prediction(img):
    z = img
    z = F.resize(img, ModelConfig.IMG_SIZE)
    z = F.to_tensor(z)
    z = F.normalize(z, mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD)
    return z


def generate_prediction_from_image(image: np.array):
    image_tensor = transform_for_prediction(image).unsqueeze(0)

    with torch.no_grad():
        outputs = CLF_MODEL(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()
        
        # OOD detector
        # TODO
        
    pred_text = ID2LABEL[prediction]
    return pred_text, get_gemini_response(pred_text)
