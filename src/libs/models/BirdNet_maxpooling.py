import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import timm
from torchlibrosa.augmentation import SpecAugmentation

class BirdNetwMaxpool(nn.Module):
    def __init__(self, model_name:str = 'tf_efficientnet_b0_ns', pretrained:bool = True, output_dim = 264, part_size = 50) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        #print(vars(self.backbone))
        self.in_features = self.backbone.num_features

        self.part_size = part_size
        self.pool = nn.MaxPool1d(part_size)
        self.classifier = nn.Linear(self.in_features, output_dim)
        
    def forward(self, x):
        """
        Returns tensor (bs, num_classes)

        Argument:
        x - tensor (bs, part_size, time_segment, freq)
        """
        x = x.view((x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])) #(bs, part_size, 3, time_segment, freq)→(bs*part_size, time_segment, freq)
        x = self.backbone(x) #(bs*part_size, time_segment, freq)→(bs*part_size, in_features)
        x = x.view((-1, self.in_features, self.part_size)) #(bs*part_size, in_features) → (bs, in_features, part_size)
        x = self.pool(x) #(bs, in_features, part_size) → (bs, in_features, 1)
        x = x.view(x.shape[0], -1) #(bs, in_features, 1) → (bs, in_features)
        clipwise_logits = self.classifier(x) #(bs, in_features) → (bs, out_size)

        output_dict = {
            "clipwise_logit": clipwise_logits, # (batch_size, out_dim)
            'clipwise_output': nn.Softmax(dim = -1)(clipwise_logits)
        }
        return output_dict