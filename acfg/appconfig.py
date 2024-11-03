from dataclasses import dataclass

import torch

from acfg.modelconfig import ModelConfig
from ml.app.anomaly import DiseaseOODModule
from ml.app.models.classification import DiseaseClassificationModel
from ml.app.models.ood import Autoencoder


def get_device():
    """Gets the appropriate device for PyTorch operations.

    Checks for CUDA GPU availability first, then Apple M1/M2 MPS, falling back to CPU.

    Returns:
        tuple: A tuple containing two strings:
            - First string indicates the device type ('cuda', 'mps', or 'cpu')
            - Second string indicates the specific device ('cuda:0', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda", "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps", "mps"
    else:
        return "cpu", "cpu"


@dataclass
class ServiceConfig:
    LLM_MODEL_KEY = "gemini"
    OOD_THRESHOLD = 0.034
    ID2LABEL = (
        "Apple scab",
        "Apple Black rot",
        "Apple Cedar rust",
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
    )


def load_my_model(checkpoint_path, model):
    """Loads a PyTorch model from a checkpoint file with state dict key remapping.

    Args:
        checkpoint_path (str): Path to the checkpoint file containing model weights
        model (torch.nn.Module): PyTorch model instance to load the weights into

    Returns:
        torch.nn.Module: Model with loaded weights

    Notes:
        - Loads checkpoint using appropriate device (CUDA/MPS/CPU)
        - Remaps state dict keys by removing 'model.model.' prefix
        - Only keeps state dict entries that start with 'model.model.'
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(get_device()[1]))
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
CLF_MODEL = load_my_model(ModelConfig.CLASSIFY_MODEL_CHECKPOINT, CLF_MODEL).to(
    get_device()[1]
)

OOD_MODEL = DiseaseOODModule.load_from_checkpoint(
    ModelConfig.OOD_MODEL_CHECKPOINT
).model.to(get_device()[1])
