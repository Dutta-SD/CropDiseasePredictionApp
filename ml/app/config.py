import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the Image Classification Model."""

    TRAIN_DATA_PATH: str = "ml/input/PlantDiseaseClassificationDataset/train"
    VAL_DATA_PATH: str = "ml/input/PlantDiseaseClassificationDataset/valid"
    TEST_DATA_PATH: str = "ml/input/PlantDiseaseClassificationDataset/test"
    IMG_SIZE: int = 196
    BATCH_SIZE: int = 16
    NUM_OUTPUT_CLASSES: int = 38
    NUM_WORKERS: int = os.cpu_count() // 2
    IMG_STD: tuple = (0.485, 0.456, 0.406)
    IMG_MEAN: tuple = (0.229, 0.224, 0.225)
    VAL_LOSS: str = "VL"
    PRETRAINED_MODEL_NAME: str = "resnet_50"
    CLASSIFY_MODEL_CHECKPOINT: str = "service/static/PlantDiseaseClassificationModel/classify_best.ckpt"
    PRED2LABEL={-1: "-1"}
