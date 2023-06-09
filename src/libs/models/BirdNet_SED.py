import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import timm
from torchlibrosa.augmentation import SpecAugmentation


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


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


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class BirdNet_SED(nn.Module):
    def __init__(
        self,
        model_name: str = "tf_efficientnet_b0_ns",
        pretrained: bool = True,
        output_dim=264,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if 'eca' in model_name:
            self.in_features = self.backbone.head.fc.in_features
        else:
            self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, output_dim)
        )

        self.bn0 = nn.BatchNorm2d(128)

        layers = list(self.backbone.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        self.fc1 = nn.Linear(self.in_features, self.in_features, bias=True)
        self.att_block = AttBlockV2(self.in_features, output_dim, activation="sigmoid")
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=2, time_stripes_num=2, freq_drop_width=2, freq_stripes_num=2
        )
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def forward(self, x):
        # (batch_size, 3, mel_bins, time_steps)
        frames_num = x.shape[3]

        # if self.training:
        #     x = self.spec_augmenter(x)

        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        x = torch.mean(x, dim=2)  # (batch_size, channels, frames)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)
        bg_att = 1-norm_att
        background_logit = torch.sum(bg_att * self.att_block.cla(x), dim = 2)
        output_dict = {
            "framewise_output": framewise_output, # (batch_size, time_steps, out_dim)
            "segmentwise_output": segmentwise_output, # (batch_size, 4 よくわからん, out_dim])
            "logit": logit, # (batch_size, out_dim)
            "framewise_logit": framewise_logit, # (batch_size, time_steps, out_dim)
            "clipwise_output": clipwise_output, # (batch_size, out_dim)
            "norm_att":norm_att,
            "background_logit": background_logit,
        }

        return output_dict
