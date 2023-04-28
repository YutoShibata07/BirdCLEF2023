from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os
__all__ = ["get_dataloader"]

logger = getLogger(__name__)


def get_dataloader(
    files:List,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[transforms.Compose] = None,
    bird_label_map:dict = None,
) -> DataLoader:

    if split not in ["train", "val", "test"]:
        message = "split should be selected from ['train', 'val', 'test']."
        logger.error(message)
        raise ValueError(message)
    data = BirdClefDataset(files=files, transform=transform, bird_label_map=bird_label_map, split=split)
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = (V - _min) / (_max - _min)
        # V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    return V

class BirdClefDataset(Dataset):
    def __init__(
        self,
        files: List,
        transform: Optional[transforms.Compose] = None,
        bird_label_map:dict = None,
        split:str = 'train',
    ) -> None:
        super().__init__()

        self.files = files
        self.transform = transform
        self.bird_label_dict = bird_label_map
        self.split = split
        logger.info(f"the number of samples: {len(self.files)}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sound = np.load(self.files[idx])
        sound = mono_to_color(sound)
        target = self.bird_label_dict[self.files[idx].split('/')[-2]]
        if self.split == 'train':
            sound = sound[np.random.choice(sound.shape[0]), :, :]
        else:
            sound = sound[0,:,:]
        sound = np.stack([sound, sound, sound])
        if self.transform is not None:
            sound = self.transform(sound)
        sample = {
            "sound": sound,
            "target": target,
        }

        return sample