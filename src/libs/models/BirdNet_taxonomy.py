import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import timm
from torchlibrosa.augmentation import SpecAugmentation

from libs.models.BirdNet_SED import *

class Base(nn.Module):
    def __init__(self, in_features, encoder, output_dim = 264) -> None:
        super().__init__()
        self.in_features = in_features
        self.encoder = encoder
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_features, output_dim)
            )
        
    def forward(self, x, frames_num):
        clipwise_logits = self.backbone(x)
        output_dict = {
            "clipwise_logit": clipwise_logits, # (batch_size, out_dim)
            'clipwise_output': nn.Softmax(dim = -1)(clipwise_logits)
        }
        return output_dict
    
class Base_SED(nn.Module):
    def __init__(self, in_features, encoder, output_dim = 264) -> None:
        super().__init__()
        self.in_features = in_features
        self.encoder = encoder
        self.bn0 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(self.in_features, self.in_features, bias=True)
        self.att_block = AttBlockV2(
            self.in_features, output_dim, activation="sigmoid")
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=2, time_stripes_num=2,
                                               freq_drop_width=2, freq_stripes_num=2)        
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)
        
    def forward(self, x, frames_num):
        # (batch_size, 3, mel_bins, time_steps)
        #frames_num = x.shape[3]

        # if self.training:
        #     x = self.spec_augmenter(x)
        
        # (batch_size, channels, freq, frames)
        #x = self.encoder(x)

        x = torch.mean(x, dim=2) # (batch_size, channels, frames)

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
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output, # (batch_size, time_steps, out_dim)
            "segmentwise_output": segmentwise_output, # (batch_size, 4 よくわからん, out_dim])
            "logit": logit, # (batch_size, out_dim)
            "framewise_logit": framewise_logit, # (batch_size, time_steps, out_dim)
            "clipwise_output": clipwise_output # (batch_size, out_dim)
        }

        return output_dict

class BirdNet_Taxonomy(nn.Module):
    def __init__(self, model_name:str = 'tf_efficientnet_b0_ns', pretrained:bool = True, output_dim = 264, order_dim = 41, fam_dim = 249, is_train = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, output_dim)
        )

        layers = list(self.backbone.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        self.models = {'species':Base_SED(in_features=self.in_features, encoder=self.encoder, output_dim=output_dim), 
                       'order':Base(in_features=self.in_features, encoder=self.encoder, output_dim=order_dim), 
                       'family':Base(in_features=self.in_features, encoder=self.encoder, output_dim=fam_dim)}
        self.is_train = is_train

    def init_weight(self):
        self.models['species'].init_weight()
        self.models['order'].init_weight()
        self.models['family'].init_weight()

    def forward(self, x):
        frames_num = x.shape[3]
        y = self.encoder(x)
        if self.is_train:
            outputs_dicts = {}
            labels = ['species', 'order', 'family']
            for label in labels:
                if label in ['order', 'family']:
                    __output_dict = self.models[label](y.detach().clone(), frames_num)
                else:
                    __output_dict = self.models[label](y, frames_num)
                outputs_dicts[label] = __output_dict

            return outputs_dicts    
        else:
            return self.models['species'](x)
        
    def to(self, device):
        self.models['species'].to(device)
        self.models['order'].to(device)
        self.models['family'].to(device)