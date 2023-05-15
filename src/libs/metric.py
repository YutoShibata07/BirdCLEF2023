from typing import Any, Dict, List, Tuple

import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import pandas as pd
import sklearn.metrics


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution.drop(["row_id"], axis=1, errors="ignore")
    submission = submission.drop(["row_id"], axis=1, errors="ignore")
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average="macro",
    )
    return score


def padded_cmap_numpy(gts, predictions, padding_factor=5):
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(predictions.shape[-1])])

    new_samples = np.array(new_rows)
    tmp_gts = np.identity(predictions.shape[1])[gts]
    padded_predictoins = np.concatenate([predictions, new_samples])
    padded_gts = np.concatenate([tmp_gts, new_samples])
    score = sklearn.metrics.average_precision_score(
        padded_gts.astype(bool),
        padded_predictoins,
        average="macro",
    )
    return score
