import numpy as np
import torch


def mixup_data(x, y, alpha=0.2, use_cuda=True, use_taxonomy=False):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    if use_taxonomy:
        y_a = y
        y_b = {}
        keys = y.keys()
        for key in keys:
            y_b[key] = y[key][index]
    else:
        y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, rating):
    return lam * criterion(pred, y_a, rating) + (1 - lam) * criterion(pred, y_b, rating)