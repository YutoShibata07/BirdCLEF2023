from logging import getLogger
from typing import Optional

import torch.nn as nn
import torch


__all__ = ["get_criterion"]
logger = getLogger(__name__)

def get_criterion(
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    return criterion