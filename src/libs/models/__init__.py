from logging import getLogger
from .BirdNet_SED import BirdNet_SED

import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import timm

__all__ = ["get_model"]

model_names = [
    "bird_base",
    "bird_base_b1",
    "bird_sed",
    "bird_sed_b1",
    "bird_base_gem_b1",
    'bird_sed_eca',
]
logger = getLogger(__name__)


def get_model(
    name: str,
    output_dim: int = 2,
    pretrained_path="",
) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. " "model_name is %s" % name,
            "You have to choose %s as a model." % (", ").join(model_names),
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))
    if name == "bird_base":
        model = BirdNet(model_name = "tf_efficientnet_b0_ns", pretrained=True, output_dim=output_dim)
    if name == "bird_base_b1":
        model = BirdNet(model_name = "tf_efficientnet_b1_ns", pretrained=True, output_dim=output_dim)
    elif name == 'bird_base_gem_b1':
        model = BirdNet_gem(model_name= "tf_efficientnet_b1_ns", pretrained=True, output_dim=output_dim)
    elif name == "bird_base_b1":
        model = BirdNet(model_name = "tf_efficientnet_b1_ns", pretrained=True, output_dim=output_dim)
    elif name == "bird_sed_eca":
        model = BirdNet_SED(
            model_name="eca_nfnet_l0", pretrained=True, output_dim=output_dim
        )
    elif name == "bird_sed":
        model = BirdNet_SED(
            model_name="tf_efficientnet_b0_ns", pretrained=True, output_dim=output_dim
        )
    elif name == "bird_sed_b1":
        model = BirdNet_SED(
            model_name="tf_efficientnet_b1_ns", pretrained=True, output_dim=output_dim
        )
    else:
        logger.error("There is no model appropriate to your choice. ")
    if pretrained_path != "":
        logger.info(f"weights loading from:{pretrained_path}")
        if 'gem' in name:
            model = BirdNet_SED(model_name="tf_efficientnet_b1_ns", pretrained=True, output_dim=output_dim)
            output_dim = 264 + 1
            model.load_state_dict(torch.load(pretrained_path), strict=False)
            model2 = BirdNet_gem(model_name= "tf_efficientnet_b1_ns", pretrained=True, output_dim=output_dim)
            model2.encoder = model.encoder
            model = model2
        else:
            output_dim = 264 + 1
            model.load_state_dict(torch.load(pretrained_path), strict=False)
            if "base" in pretrained_path:
                model.backbone.classifier = nn.Sequential(
                    nn.Linear(model.in_features, output_dim)
                )
            elif "sed" in pretrained_path:
                logger.info("FC layer updating...")
                model.att_block.att = nn.Conv1d(
                    in_channels=model.in_features,
                    out_channels=output_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                model.att_block.cla = nn.Conv1d(
                    in_channels=model.in_features,
                    out_channels=output_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )

    return model


class BirdNet(nn.Module):
    def __init__(
        self,
        model_name: str = "tf_efficientnet_b0_ns",
        pretrained: bool = True,
        output_dim=264,
    ) -> None:
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
            'clipwise_output': nn.Sigmoid()(clipwise_logits)
        }
        return output_dict
    
def gem(x: torch.Tensor, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )
        
class BirdNet_gem(nn.Module):
    def __init__(
        self,
        model_name: str = "tf_efficientnet_b0_ns",
        pretrained: bool = True,
        output_dim=264,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.in_features = self.backbone.classifier.in_features
        layers = list(self.backbone.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.gem = GeM()
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, output_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.gem(feat)
        feat = feat[:,:,0,0]
        clipwise_logits = self.classifier(feat)
        output_dict = {
            "logit": clipwise_logits, # (batch_size, out_dim)
            'clipwise_output': nn.Sigmoid()(clipwise_logits),
            'clipwise_logit':clipwise_logits
        }
        return output_dict
