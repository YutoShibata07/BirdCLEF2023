from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import random
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
    aug_list:List = [],
    duration:int = 5
) -> DataLoader:

    if split not in ["train", "val", "test"]:
        message = "split should be selected from ['train', 'val', 'test']."
        logger.error(message)
        raise ValueError(message)
    data = BirdClefDataset(files=files, transform=transform, bird_label_map=bird_label_map, split=split, aug_list = aug_list, duration=duration)
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

def random_power(images, power = 1.5, c= 0.7):
    images = images - images.min()
    images = images/(images.max()+0.0000001)
    images = images**(random.random()*power + c)
    return images

class BirdClefDataset(Dataset):
    def __init__(
        self,
        files: List,
        transform: Optional[transforms.Compose] = None,
        bird_label_map:dict = None,
        split:str = 'train',
        aug_list:List = [],
        duration:int = 5,
    ) -> None:
        super().__init__()

        self.files = files
        self.transform = transform
        self.bird_label_dict = bird_label_map
        self.split = split
        self.aug_list = aug_list
        if 'soundscape' in self.aug_list:
            self.soundscape_df = pd.read_csv('../data_2021/train_soundscape_labels.csv')
            self.soundscape_df = self.soundscape_df[self.soundscape_df.birds=='nocall']
            self.soundscape_df['row_id'] = self.soundscape_df['row_id'].apply(lambda x:'_'.join(x.split('_')[:-1]))
            self.soundscape_df['filepath'] = self.soundscape_df['row_id'].apply(lambda x:glob.glob(os.path.join('../dataset_soundscapes_2021/logmel/train_soundscapes', str(x) + '*.npy'))[0])
            self.soundscape_df = self.soundscape_df.reset_index(drop=True)
            logger.info(f"the number of soundscape sound images:{self.soundscape_df.shape[0]}")
        self.duration = duration
        logger.info(f"the number of samples: {len(self.files)}")
        logger.info(f'augmentation list:{self.aug_list}')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sound = np.load(self.files[idx])
        target = self.bird_label_dict[self.files[idx].split('/')[-2]]
        labels = np.zeros(len(self.bird_label_dict.keys()), dtype=float)
        labels[target] = 1.0
        sound_size = int(self.duration//5)
        if self.split == 'train':
            start_idx = np.random.choice(sound.shape[0])
        else:
            start_idx = 0
        if start_idx + sound_size > sound.shape[0]:
            pad_size = start_idx+sound_size-sound.shape[0]
            sound = np.concatenate([sound[start_idx:], np.zeros((pad_size, sound.shape[1], sound.shape[2]))])
        else:
            sound = sound[start_idx:start_idx+sound_size, :, :]
        sound = sound.transpose(0, 2, 1)
        sound = sound.reshape([-1, sound.shape[-1]]).T
        if self.split == 'train':
            if ('soundscape' in self.aug_list) & (np.random.rand() > 0.5):
                soundscapes = []
                for i in range(sound_size):
                    ss_idx = np.random.choice(self.soundscape_df.shape[0])
                    ss_file = self.soundscape_df.loc[ss_idx, 'filepath']
                    ss_num = self.soundscape_df['seconds'].values[ss_idx] // 5 - 1
                    soundscape = np.load(ss_file)
                    soundscapes.append(soundscape[int(ss_num)])
                soundscapes = np.array(soundscapes) #[sound_size, freq_num, time_dim]
                soundscapes = soundscapes.transpose(0, 2, 1)
                soundscapes = soundscapes.reshape([-1, soundscapes.shape[-1]]).T
                ratio = 0.5 * np.random.rand()
                sound = sound * (1-ratio) + soundscapes * ratio
            if ('random_power' in self.aug_list) & (np.random.rand() > 0.5):
                sound = random_power(images=sound, power = 3, c= 0.5)
            if ('white' in self.aug_list) & (np.random.rand() > 0.8):
                noise_level = 0.05
                noise = (np.random.sample((sound.shape[0], sound.shape[1])) + 9) * sound.mean() * noise_level * (np.random.sample() + 0.3)
                sound = sound + noise
            if ('pink' in self.aug_list) & (np.random.rand() > 0.8):
                r = random.randint(1,128)
                # [1, 0.98795181, ..., 0]
                pink_noise = np.array([np.concatenate((1 - np.arange(r)/r,np.zeros(128 - r)))]).T
                sound = sound + (np.random.sample((sound.shape[0], sound.shape[1])).astype(np.float32) + 9) * 2 * pink_noise * sound.mean() * 0.05 * (np.random.sample() + 0.3)
            if ('upper_freq_decay' in self.aug_list) & (np.random.rand() > 0.5):
                r = random.randint(128//2,128)
                x = random.random()/2
                pink_noise = np.array([np.concatenate((1-np.arange(r)*x/r,np.zeros(128-r)-x+1))]).T
                sound = sound*pink_noise
        sound = mono_to_color(sound)
        sound = sound.astype(float)
        sound = np.stack([sound, sound, sound])
        if self.transform is not None:
            sound = self.transform(sound)
        sample = {
            "sound": sound,
            "target": labels,
        }

        return sample