from logging import getLogger
from .BirdNet_SED import BirdNet_SED
from .BirdNet_maxpooling import BirdNetwMaxpool

import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import timm

__all__ = ["get_model"]

model_names = [
    'bird_base',
    'bird_sed',
    'bird_sed_b1',
    'bird_maxpool'
]
logger = getLogger(__name__)


def get_model(
    name: str, output_dim:int = 2,pretrained_path = '',
) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            "model_name is %s" %name,
            "You have to choose %s as a model." % (", ").join(model_names)
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))
    if name == "bird_base":
        model = BirdNet(model_name = "tf_efficientnet_b0_ns", pretrained=True, output_dim=output_dim)
    elif name == "bird_sed":
        model = BirdNet_SED(model_name = "tf_efficientnet_b0_ns", pretrained=True, output_dim=output_dim)
    elif name == 'bird_sed_b1':
        model = BirdNet_SED(model_name = "tf_efficientnet_b1_ns", pretrained=False, output_dim=output_dim)
    elif name == 'bird_maxpool':
        model = BirdNetwMaxpool(model_name = "tf_efficientnet_b0_ns", pretrained=False, output_dim=output_dim)
    else:
        logger.error( "There is no model appropriate to your choice. ")
    if pretrained_path != '':
        logger.info(f'weights loading from:{pretrained_path}')
        model.load_state_dict(torch.load(pretrained_path))
        output_dim = 264
        if 'base' in pretrained_path:
            model.backbone.classifier = nn.Sequential(
                nn.Linear(model.in_features, output_dim)
            )
        elif 'sed' in pretrained_path:
            logger.info('FC layer updating...')
            model.att_block.att = nn.Conv1d(
                in_channels=model.in_features,
                out_channels=output_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
            model.att_block.cla = nn.Conv1d(
                in_channels=model.in_features,
                out_channels=output_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
            
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
        clipwise_logits = self.backbone(x)
        output_dict = {
            "logit": clipwise_logits, # (batch_size, out_dim)
            'clipwise_output': nn.Softmax(dim = -1)(clipwise_logits)
        }
        return output_dict
        

                
            