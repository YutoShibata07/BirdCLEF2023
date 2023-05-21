from logging import getLogger
from typing import Optional

import torch.nn as nn
import torch
import numpy as np


__all__ = ["get_criterion"]
logger = getLogger(__name__)

def get_criterion(
    loss_fn:str = 'ce',
) -> nn.Module:
    if loss_fn == 'ce':
        criterion = CrossEntropyLoss()
    elif loss_fn == 'focal_bce':
        criterion = BCEFocalLoss(output_dict=True)
    elif loss_fn == 'bce':
        criterion = BinaryCrossEntropyLoss()
    elif loss_fn == 'bcef_2way':
        criterion = BCEFocal2WayLoss()
    elif loss_fn == 'focal_clip_max':
        criterion = BCEFocalLoss(output_dict=True, clip_max=True)
    elif loss_fn == 'focal_clip_max_v2':
        criterion = BCEFocalLoss_v2(output_dict=True, clip_max=True)
    elif loss_fn == 'focal_clip_max_taxonomy':
        criterion = BCEFocalLoss_Group(output_dict=True, clip_max=True)
    else:
        message = "loss function not found"
        logger.error(message)
        print(loss_fn)
        raise ValueError(message)
    return criterion

class CrossEntropyLoss(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, preds, targets, rating):
        ce_loss = nn.CrossEntropyLoss()(preds['clipwise_logit'], targets)
        return ce_loss
    
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, preds, targets, rating):
        ce_loss = nn.BCEWithLogitsLoss()(preds['clipwise_logit'], targets)
        return ce_loss

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
# For clip_max part: https://www.kaggle.com/competitions/birdclef-2021/discussion/243293
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, output_dict = False, clip_max = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.output_dict = output_dict
        self.clip_max = clip_max

    def forward(self, preds, targets, rating):
        if self.output_dict == True:
            if self.clip_max==True:
                preds_clip_max = preds["framewise_logit"].max(1)[0]
            preds = preds['logit']
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        if self.clip_max == True:
            bce_loss_clip_max = nn.BCEWithLogitsLoss(reduction='none')(preds_clip_max, targets)
            probas = torch.sigmoid(preds_clip_max)
            loss_clip_max = targets * self.alpha * \
                (1. - probas)**self.gamma * bce_loss_clip_max + \
                (1. - targets) * probas**self.gamma * bce_loss_clip_max
            loss = loss + 0.5 * loss_clip_max
        loss = loss.mean(axis=1)
        # loss = loss * (0.7 + 0.3 * rating / 5.0)
        loss = loss.mean()
        
        return loss
    
class BCEFocalLoss_v2(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, output_dict = False, clip_max = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.output_dict = output_dict
        self.clip_max = clip_max

    def forward(self, preds, targets, rating):
        if self.output_dict == True:
            if self.clip_max==True:
                preds_clip_max = preds["framewise_logit"].max(1)[0]
            preds = torch.logit(preds['clipwise_output'])
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss

        if self.clip_max == True:
            bce_loss_clip_max = nn.BCEWithLogitsLoss(reduction='none')(preds_clip_max, targets)
            probas = torch.sigmoid(preds_clip_max)
            loss_clip_max = targets * self.alpha * \
                (1. - probas)**self.gamma * bce_loss_clip_max + \
                (1. - targets) * probas**self.gamma * bce_loss_clip_max
            loss = loss + 0.5 * loss_clip_max
        loss = loss.mean(axis=1)
        # loss = loss * (0.7 + 0.3 * rating / 5.0)        
        loss = loss.mean()
        
        return loss
    
class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()
        self.aw_loss = AutomaticWeightedLoss()

        self.weights = weights

    def forward(self, input, target, rating):
        input_ = input["logit"]
        target = target.float()
        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)
        loss_sum = self.aw_loss(loss, aux_loss)
        loss_sum = loss + 0.5 * loss_sum
        return loss_sum

        return self.weights[0] * loss + self.weights[1] * aux_loss
    
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class BCELossV2_nocall(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, output_dict = False, clip_max = False, att_loss_w = 10):
        super().__init__()
        self.output_dict = output_dict
        self.clip_max = clip_max
        self.att_loss_w = att_loss_w

    def forward(self, preds, targets):
        if self.output_dict == True:
            if self.clip_max==True:
                preds_clip_max = preds["framewise_logit"].max(1)[0]
            norm_att = preds['norm_att']
            preds = torch.logit(preds['clipwise_output'])
        bce_loss = nn.BCEWithLogitsLoss()(preds, targets)
        if self.clip_max == True:
            bce_loss_clip_max = nn.BCEWithLogitsLoss()(preds_clip_max, targets)
            bce_loss = bce_loss + 0.5 * bce_loss_clip_max
        is_nocall = targets[:,-1] == 1
        norm_loss = torch.std(norm_att, dim = -1) #[bs, 265]
        norm_loss = torch.mean(norm_loss, dim = -1) * is_nocall
        norm_loss = torch.mean(norm_loss, dim = -1)
        bce_loss = bce_loss + norm_loss * self.att_loss_w
        return bce_loss
        
class BCEFocalLoss_Group(nn.Module):    
    def __init__(self, alpha=1, gamma=2.0, output_dict = False, clip_max = False, w_order = 0.01, w_family = 0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.output_dict = output_dict
        self.clip_max = clip_max
        self.w_order = w_order
        self.w_family = w_family

        #self.focal_loss = BCEFocalLoss(alpha=self.alpha, gamma=self.gamma, output_dict=self.output_dict, clip_max=self.clip_max)
        self.focal_loss = BCELossV2_nocall(lpha=self.alpha, gamma=self.gamma, output_dict=self.output_dict, clip_max=self.clip_max)
        self.bce_loss = BinaryCrossEntropyLoss()

    def forward(self, preds, targets, rating):
        species_loss = self.focal_loss(preds['species'], targets['target'], rating)
        order_loss = self.bce_loss(preds['order'], targets['order_target'], rating)
        family_loss = self.bce_loss(preds['family'], targets['family_target'], rating)

        return species_loss + self.w_order*order_loss + self.w_family*family_loss