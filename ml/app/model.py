import torch
import torchvision
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)
from torchvision.models._api import WeightsEnum


# fix for hash
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict


class PretrainedModelFactory:
    def __init__(self):
        self.available_models = {
            "efficientnet_b0": (
                torchvision.models.efficientnet_b0,
                EfficientNet_B0_Weights.DEFAULT,
                lambda x: x.classifier[1].in_features,
            ),
            "resnet_50": (
                torchvision.models.resnet50,
                ResNet50_Weights.DEFAULT,
            ),
            "vit_b_16": (
                torchvision.models.vit_b_16,
                ViT_B_16_Weights.DEFAULT,
            ),  # Parameter Heavy
            "mobilenet_v3_small": (
                torchvision.models.mobilenet_v3_small,
                MobileNet_V3_Small_Weights.DEFAULT,
            ),
        }

    def get_model(self, model_name: str) -> nn.Module:
        if model_name not in self.available_models:
            raise ValueError(
                f"Model '{model_name}' not available. Choose from {self.available_models.keys()}"
            )
        model_builder, weights, get_out_features = self.available_models[model_name]
        return model_builder(weights=weights), get_out_features


class MLPHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DiseaseClassificationModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        factory = PretrainedModelFactory()
        self.feature_extractor, get_out_features = factory.get_model(model_name)
        self._freeze_pretrained_weights()
        self.classifier = MLPHead(get_out_features(self.feature_extractor), num_classes)

    def _freeze_pretrained_weights(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
