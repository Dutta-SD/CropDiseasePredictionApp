import PIL
import numpy as np
import torch
from acfg.modelconfig import ModelConfig
import torchvision.transforms.functional as F
from torch.nn import functional as Fx


from acfg.appconfig import CLF_MODEL, OOD_MODEL, ServiceConfig, get_device
from service.external import llm_strategy


def transform_for_prediction(img: PIL.Image):
    """Transforms a PIL image for model prediction.

    This function applies a series of transformations to prepare an image for model inference:
    1. Resizes the image to the model's expected input size
    2. Converts the image to a tensor
    3. Normalizes the tensor using preconfigured mean and std values

    Args:
        img (PIL.Image): Input image to transform

    Returns:
        torch.Tensor: Transformed image tensor ready for model inference
    """
    z = img
    z = F.resize(img, [ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE])
    z = F.to_tensor(z)
    z = F.normalize(z, mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD)
    return z.to(get_device()[1])


def classify_disease(image):
    image_tensor = transform_for_prediction(image).unsqueeze(0)

    with torch.no_grad():
        outputs = CLF_MODEL(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    return ServiceConfig.ID2LABEL[prediction]


def img_in_distribution(image):
    image_tensor = transform_for_prediction(image).unsqueeze(0)

    with torch.no_grad():
        output = OOD_MODEL(image_tensor)
        mse_loss_value = Fx.mse_loss(output, image_tensor)
        print("MSE", mse_loss_value)
        
    return mse_loss_value < ServiceConfig.OOD_THRESHOLD


def workflow(image: np.array):
    if not img_in_distribution(image):
        disease_name = "Unknown"
        remedy = "We do not know the remedy to this one. Sorry!"
    else:
        disease_name = classify_disease(image)
        remedy = "No remedy needed. Plant is Healthy"
        print(disease_name)

        if "healthy" in disease_name:
            return disease_name, remedy

        else:
            remedy = llm_strategy(ServiceConfig.LLM_MODEL_KEY, disease_name)

    return disease_name, remedy
