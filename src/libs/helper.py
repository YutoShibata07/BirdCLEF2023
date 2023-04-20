import os, time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from libs.meter import AverageMeter, ProgressMeter
from libs.metric import angular_dist_score, angular_dist_score_numpy
from libs.convertAngle import pred_to_angle_azshift
from libs.convertAngle import y_to_onehot
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)

def softmax(x):
    u = np.sum(np.exp(x), axis=-1)
    return np.exp(x)/u.reshape(-1, 1)

def do_one_iteration(
    sample: List,
    model: nn.Module,
    criterion: Any,
    device: str,
    iter_type: str,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler = None,
    bin_num:int = 24
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and optimizer is None:
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    x = sample[0].to(device).float()
    azimuth = sample[1][:,0]
    zenith = sample[1][:,1]
    t = sample[2].to(device)
    

    batch_size = x.shape[0]
    output = model(x)
    loss = criterion(output, t)
    # print(output)
    # print(loss)
    # # assert False
    if iter_type == "train" and optimizer is not None:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # keep predicted results and gts for calculate F1 Score
    gt = np.stack([azimuth, zenith]).T
    pred = output.to("cpu").detach().numpy() #[batch_size, bin_num * bin_num]
    return batch_size, loss.item(), gt, pred


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    scheduler, 
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
    bin_num:int = 16
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size, loss, gt, pred = do_one_iteration(
            sample, model, criterion, device, "train", optimizer, scheduler = scheduler, bin_num=bin_num
        )

        losses.update(loss, batch_size)
        # top1.update(acc1, batch_size)

        # save the ground truths and predictions in lists
        if i < 125:
            gts += list(gt)
            preds += list(pred)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
    # calculate F1 Score
    # f1s = f1_score(gts, preds, average="macro")
    gts = np.array(gts)
    preds = np.array(preds)
    preds = preds[:200000 * 5]
    gts = gts[:200000 * 5]
    preds = softmax(preds)
    preds_angles = pred_to_angle_azshift(preds, bin_num=bin_num)
    score = angular_dist_score_numpy(preds_angles, gts)
    # score = score.to('cpu').detach().numpy()[0]
    return losses.get_average(), gts, preds, score


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    device: str,
    mode: str = "train",
    bin_num:int = 16
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    losses = AverageMeter("Loss", ":.4e")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in loader:
            
            batch_size, loss, gt, pred = do_one_iteration(
                sample, model, criterion, device, "evaluate", bin_num=bin_num
            )

            losses.update(loss, batch_size)
            # top1.update(acc1, batch_size)

            # keep predicted results and gts for calculate F1 Score
            gts += list(gt)
            preds += list(pred)

    # f1s = f1_score(gts, preds, average="macro")
    gts = np.array(gts)
    preds = np.array(preds)
    preds = softmax(preds)
    pred_angles = pred_to_angle_azshift(preds, bin_num=bin_num)
    score = angular_dist_score_numpy(pred_angles, gts)
    return losses.get_average(), gts, pred_angles, score