from logging import getLogger

import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import timm

__all__ = ["get_model"]

model_names = [
    'bird_base',
]
logger = getLogger(__name__)


def get_model(
    name: str, output_dim:int = 2,
) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            "You have to choose %s as a model." % (", ").join(model_names)
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))

    if name == "bird_base":
        model = BirdNet(model_name = "tf_efficientnet_b0_ns", pretrained=True, output_dim=output_dim)
    else:
        logger.error( "There is no model appropriate to your choice. ")
    return model


class BirdNet(nn.Module):
    def __init__(self, model_name:str = 'tf_efficientnet_b0_ns', pretrained:bool = True, output_dim = 264) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, output_dim)
        )
    def forward(self, x):
        output = self.backbone(x)
        return output
        