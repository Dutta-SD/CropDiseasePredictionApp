from torchvision import transforms as T
from PIL import Image

from ml.app.config import ModelConfig
import torch

from ml.app.lm import ClassificationModule


model = ClassificationModule.load_from_checkpoint(
    ModelConfig.CLASSIFY_MODEL_CHECKPOINT,
    model=ModelConfig.PRETRAINED_MODEL_NAME,
    num_classes=ModelConfig.NUM_OUTPUT_CLASSES,
)
model.eval()


def get_transforms(img_size):
    return T.Compose(
        [
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD),
        ]
    )


def generate_prediction_from_image(image_path, checkpoint_path, model_class, num_classes):
    return "1"
    transform = get_transforms(ModelConfig.IMG_SIZE)

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    return prediction
