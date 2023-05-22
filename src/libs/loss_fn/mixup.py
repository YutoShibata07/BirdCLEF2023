import numpy as np
import torch


def mixup_data(x, y, alpha=0.6, use_cuda=True):
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
    y_a = y + y[index] - y * y[index]
    nocall_flg = y_a[:,-1] > torch.Tensor([0.999] * y_a.shape[0]).to('cuda')#torch.gt(y_a[:,-1], 0.999)
    same_target_flg = y_a.sum(axis = -1) < torch.Tensor([2.0] * y_a.shape[0]).to('cuda')
    y_a[(nocall_flg==True)&(same_target_flg==False),-1] = 0.002
    # if ((same_target_flg==False) * (nocall_flg)).sum()>0:
    #     print(y_a)
    # print((same_target_flg==False).sum())
    # print(nocall_flg.sum()[])
    return mixed_x, y_a, y_a, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
