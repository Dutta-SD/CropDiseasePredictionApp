import torch
import torchvision
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights

from acfg.modelconfig import ModelConfig


# TODO: Uncomment if needed
# Pytorch fix for hash mismatch
# def get_state_dict(self, *args, **kwargs):
#     kwargs.pop("check_hash")
#     return load_state_dict_from_url(self.url, *args, **kwargs)


# WeightsEnum.get_state_dict = get_state_dict


class MLPHead(nn.Module):
    def __init__(self, in_features: int, num_output_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_output_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PretrainedModelFactory:
    @staticmethod
    def _freeze_pretrained_weights(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def _efficientnet_b0():
        model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        __class__._freeze_pretrained_weights(model)
        model.classifier = MLPHead(
            in_features=model.classifier[1].in_features,
            num_output_classes=ModelConfig.NUM_OUTPUT_CLASSES,
        )
        return model

    @staticmethod
    def _resnet_50():
        model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        __class__._freeze_pretrained_weights(model)
        model.fc = MLPHead(
            in_features=model.fc.in_features,
            num_output_classes=ModelConfig.NUM_OUTPUT_CLASSES,
        )
        return model

    @staticmethod
    def _mobilenet_v3_small():
        raise NotImplementedError

    @staticmethod
    def _vit_b_16():
        raise NotImplementedError

    def __init__(self):
        self.available_models = {
            "efficientnet_b0": self._efficientnet_b0,
            "resnet_50": self._resnet_50,
            "vit_b_16": self._vit_b_16,
            "mobilenet_v3_small": self._mobilenet_v3_small,
        }

    def get_model(self, model_name: str) -> nn.Module:
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available. Choose from {self.available_models.keys()}")
        return self.available_models[model_name]()


class DiseaseClassificationModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        factory = PretrainedModelFactory()
        self.model = factory.get_model(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
