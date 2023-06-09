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
import librosa
import torch.nn as nn
import torchaudio
__all__ = ["get_dataloader"]

logger = getLogger(__name__)

import albumentations as A

def get_train_transform():
    return A.Compose([
        A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=8),
                A.CoarseDropout(max_holes=4),
            ], p=0.5),
    ])


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
    duration:int = 5,
    cleaning_path :str = '',
    is_hard:bool =False
) -> DataLoader:

    if split not in ["train", "val", "test", 'oof']:
        message = "split should be selected from ['train', 'val', 'test']."
        logger.error(message)
        raise ValueError(message)
    if is_hard:
        data = BirdClefDataset_hard(files=files, transform=transform, bird_label_map=bird_label_map, split=split, aug_list = aug_list, duration=duration, cleaning_path=cleaning_path)
    else:
        data = BirdClefDataset(files=files, transform=transform, bird_label_map=bird_label_map, split=split, aug_list = aug_list, duration=duration, cleaning_path=cleaning_path)
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

# 残響特性のモデル化 (指数関数型)
def decay_curve(t, rt60, freq):
    return np.exp(-t * freq * 0.5 / (rt60 / np.log(10)))


class BirdClefDataset(Dataset):
    def __init__(
        self,
        files: List,
        transform: Optional[transforms.Compose] = None,
        bird_label_map:dict = None,
        split:str = 'train',
        aug_list:List = [],
        duration:int = 5,
        cleaning_path:str = '',
    ) -> None:
        super().__init__()

        self.files = files
        self.transform = transform
        self.bird_label_dict = bird_label_map
        self.split = split
        self.aug_list = aug_list
        self.df_meta = self.get_metadata()
        if 'soundscape' in self.aug_list:
            self.soundscape_df = pd.read_csv('../data_2021/train_soundscape_labels.csv')
            self.soundscape_df = self.soundscape_df[self.soundscape_df.birds=='nocall']
            self.soundscape_df['row_id'] = self.soundscape_df['row_id'].apply(lambda x:'_'.join(x.split('_')[:-1]))
            self.soundscape_df['filepath'] = self.soundscape_df['row_id'].apply(lambda x:glob.glob(os.path.join('../dataset_soundscapes_2021/logmel/train_soundscapes', str(x) + '*.npy'))[0])
            self.soundscape_df = self.soundscape_df.reset_index(drop=True)
            logger.info(f"the number of soundscape sound images:{self.soundscape_df.shape[0]}")
        self.duration = duration
        self.cleaning_path = cleaning_path
        logger.info(f"the number of samples: {len(self.files)}")
        logger.info(f'augmentation list:{self.aug_list}')
        if len(self.cleaning_path) > 0:
            self.oof = pd.read_csv(os.path.join('../result', self.cleaning_path, 'oof.csv'))
            self.oof['soundname'] = self.oof['soundname'].apply(lambda x:x.split('_')[-1])
            self.oof['image_idx'] = self.oof['filename'].apply(lambda x:int(x.split('_')[-1].split('.')[0]))
            self.oof = self.oof.reset_index(drop=True)
        else:
            self.oof = pd.DataFrame(columns=['soundname'])
    def __len__(self) -> int:
        return len(self.files)
    
    @staticmethod
    def get_metadata():
        df_2023 = pd.read_csv('../data/train_metadata.csv')
        df_2021 = pd.read_csv('../data_2021/train_metadata.csv')
        df_2022 = pd.read_csv('../data_2022/train_metadata.csv')
        df_2020 = pd.read_csv('../data_2020/train_extended.csv')
        df_2023['soundname'] = df_2023['filename'].map(lambda x: '2023_' + x[:-4])
        df_2022['soundname'] = df_2022['filename'].map(lambda x: '2022_' + x[:-4])
        df_2021['soundname'] = "2021_" + df_2021.primary_label + '/' + df_2021['filename'].map(lambda x: x[:-4])
        df_2020['soundname'] = "2020_" + df_2020.ebird_code + '/' + df_2020['filename'].map(lambda x: x.split('.')[0])
        df_2023 = df_2023[['soundname', 'rating', 'secondary_labels']]
        df_2022 = df_2022[['soundname', 'rating', 'secondary_labels']]
        df_2021 = df_2021[['soundname', 'rating', 'secondary_labels']]
        df_2020 = df_2020[['soundname', 'rating', 'secondary_labels']]
        return pd.concat([df_2020, df_2021, df_2022, df_2023], axis=0)
    
    def get_sound(self, file_path: int, start_idx:int = None):
        sound = np.load(file_path)
        sound_size = int(self.duration//5)
        if start_idx == None:
            start_idx = np.random.choice(sound.shape[0])
        # if start_idx + sound_size > sound.shape[0]:
        #     pad_size = start_idx+sound_size-sound.shape[0]
        #     # 10秒だと仮定
        #     # sound = np.concatenate([sound[start_idx:], sound[start_idx:]], axis=-1)
        #     sound = np.concatenate([sound[start_idx:], np.zeros((pad_size, sound.shape[1], sound.shape[2]))])
        # else:
        #     sound = sound[start_idx:start_idx+sound_size, :, :]
        sound = sound[start_idx,:,:]
        # sound = sound.transpose(0, 2, 1)
        # sound = sound.reshape([-1, sound.shape[-1]]).T
        return sound, start_idx
    
    def __getitem__(self, idx: int):
        ds_name = self.files[idx].split('/')[-4].split('_')[-1]
        if ds_name == 'dataset':
            ds_name = "2023"
        if self.split == 'train':
            sound, start_idx = self.get_sound(self.files[idx])
            tmp_soundfile = ds_name + '_' + self.files[idx].split('/')[-2]+'/'+self.files[idx].split('/')[-1][:-4]
        elif self.split=='oof':
            #検証データは "_{second}.npy"となっている
            start_idx = int(self.files[idx].split('_')[-1].split('.npy')[0])
            sound,_ = self.get_sound(self.files[idx].split('_')[-2] + '.npy', start_idx=start_idx)
            tmp_soundfile = ds_name + '_' +self.files[idx].split('/')[-2]+'/'+self.files[idx].split('/')[-1].split('_')[-2]
        else:
            start_idx = 0
            sound, _ = self.get_sound(self.files[idx], start_idx=start_idx)
            tmp_soundfile = ds_name + '_' + self.files[idx].split('/')[-2]+'/'+self.files[idx].split('/')[-1][:-4]
        # assert False
        target = self.bird_label_dict[self.files[idx].split('/')[-2]]
        meta = self.df_meta[self.df_meta['soundname']==tmp_soundfile].iloc[0]
        if self.split == 'train':
            soundname = self.files[idx].split('/')[-2]+'/'+self.files[idx].split('/')[-1][:-4]
            if (len(self.cleaning_path) > 0) & (soundname in self.oof.soundname.unique()):
                labels = np.zeros(len(self.bird_label_dict.keys()), dtype=float)
                # ファイルごとにtrain, valを分けているのでoof_trainに絞らなくても良い
                oof_pred = self.oof[(self.oof['soundname']==soundname) & (self.oof.image_idx==start_idx)].iloc[0, 4:-1]
                if oof_pred[target]>0.5:
                    labels[target] += 1.0
                elif oof_pred[target] > 0.1:
                    labels[target] += 0.9975
                elif oof_pred[target] > 0.01:
                    labels[target] += 0.5
                else:
                    labels[target] += 0.2
                for slabel in eval(meta['secondary_labels']):
                    if slabel in self.bird_label_dict.keys():
                        sec_idx = self.bird_label_dict[slabel]
                        if oof_pred[sec_idx]>0.5:
                            labels[sec_idx] = 0.9975
                        elif oof_pred[sec_idx] > 0.1:
                            labels[sec_idx] = 0.8
                        elif oof_pred[sec_idx] > 0.05:
                            labels[sec_idx] = 0.3
                        else:
                            labels[sec_idx] = 0.0025
            else:
                labels = np.zeros(len(self.bird_label_dict.keys()), dtype=float) + 0.0001
                labels[target] += 0.9999
                for slabel in eval(meta['secondary_labels']):
                    if slabel in self.bird_label_dict.keys():
                        labels[self.bird_label_dict[slabel]] += 0.2999
        else:
            labels = np.zeros(len(self.bird_label_dict.keys()), dtype=float)
            labels[target] = 1.0
        sound_size = int(self.duration//5)
        # intra mixup
        if (self.split=='train') & (np.random.rand() > 0.75):
            sound2, start_idx = self.get_sound(self.files[idx])
            ratio = np.random.rand()
            sound = librosa.db_to_power(sound) * ratio + librosa.db_to_power(sound2) * (1 - ratio)
            sound = librosa.power_to_db(sound) 
        if self.split == 'train':
            if ('reverberation' in self.aug_list) & (np.random.rand() > 0.5):
                sound_p = librosa.db_to_power(sound)
                sr = 32000
                # 残響時間 (秒)
                rt60 = np.random.rand() + 0.2
                hop_duration = (512 +  1024)/ sr
                n_frames = sound_p.shape[1]

                # FIRフィルタ長
                filter_length = int(rt60 * sr / (512 + 1024))
                # 残響フィルタを適用
                sound_rev = sound_p.copy()
                freq_decay = np.random.rand() + 0.5
                for freq_bin in range(sound_p.shape[0]):
                    decay = decay_curve(np.arange(filter_length) * hop_duration, rt60, freq=freq_bin * freq_decay)
                    sound_rev[freq_bin, filter_length-1:] = np.convolve(sound_p[freq_bin, :], decay, mode='valid')
                ratio = 0.5 * np.random.rand()
                sound_p = sound_p * (1-ratio) + sound_rev * ratio
                sound = librosa.power_to_db(sound_p)
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
                # 12.5%の確率でnocallとする
                if np.random.rand() > 0.75:
                    sound = soundscapes.copy()
                    labels = np.zeros(len(self.bird_label_dict.keys()), dtype=float) + 0.001
                    labels[-1] = 1.0
                else:
                    ratio = 0.5 * np.random.rand()
                    sound = librosa.db_to_power(sound) * (1-ratio) + librosa.db_to_power(soundscapes) * ratio
                    sound = librosa.power_to_db(sound)
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
            if ('bandpass' in self.aug_list) & (np.random.rand() > 0.3):
                start = random.randint(1, 128-10)
                leng = random.randint(0, 128-start)
                sound[start:start + leng,:] = sound[start:start + leng,:] + (np.random.sample((leng, sound.shape[1])).astype(np.float32) + 9) * 2 * sound.mean() * 0.05 * (np.random.sample() + 0.3)
        sound = mono_to_color(sound)
        sound = sound.astype(float)
        if self.transform is not None:
            sound = self.transform(sound)
        sound = np.stack([sound, sound, sound])
        sound = sound.transpose(2,1,0)
        if (self.split == 'train') & ('cutout' in self.aug_list):
            sound = get_train_transform()(image=sound)['image']
        sound = sound.transpose(2,1,0)
        sound = torch.from_numpy(sound)
        if (self.split == 'train') & (np.random.rand() > 0.5):
            if ('time_mask' in self.aug_list):
                transforms_time_mask = nn.Sequential(
                        torchaudio.transforms.TimeMasking(time_mask_param=10),
                )
                time_mask_num = 2 # number of time masking
                for _ in range(time_mask_num): # tima masking
                    sound = transforms_time_mask(sound)
            if 'freq_mask' in self.aug_list:
                transforms_freq_mask = nn.Sequential(
                        torchaudio.transforms.FrequencyMasking(freq_mask_param=8),
                )
                freq_mask_num = 1 # number of frequency masking
                for _ in range(freq_mask_num): # frequency masking
                    sound = transforms_freq_mask(sound)
        
        sample = {
            "sound": sound,
            "target": labels,
            "rating": meta['rating'],
        }

        return sample
    
class BirdClefDataset_hard(Dataset):
    def __init__(
        self,
        files: List,
        transform: Optional[transforms.Compose] = None,
        bird_label_map:dict = None,
        split:str = 'train',
        aug_list:List = [],
        duration:int = 5,
        cleaning_path:str = '',
    ) -> None:
        super().__init__()

        self.files = files
        self.transform = transform
        self.bird_label_dict = bird_label_map
        self.split = split
        self.aug_list = aug_list
        self.df_meta = self.get_metadata()
        if 'soundscape' in self.aug_list:
            self.soundscape_df = pd.read_csv('../data_2021/train_soundscape_labels.csv')
            self.soundscape_df = self.soundscape_df[self.soundscape_df.birds=='nocall']
            self.soundscape_df['row_id'] = self.soundscape_df['row_id'].apply(lambda x:'_'.join(x.split('_')[:-1]))
            self.soundscape_df['filepath'] = self.soundscape_df['row_id'].apply(lambda x:glob.glob(os.path.join('../dataset_soundscapes_2021/logmel/train_soundscapes', str(x) + '*.npy'))[0])
            self.soundscape_df = self.soundscape_df.reset_index(drop=True)
            logger.info(f"the number of soundscape sound images:{self.soundscape_df.shape[0]}")
        self.duration = duration
        self.cleaning_path = cleaning_path
        logger.info(f"the number of samples: {len(self.files)}")
        logger.info(f'augmentation list:{self.aug_list}')
        if 'esc50' in self.aug_list:
            self.esc_df = pd.read_csv('../data_esc50/esc50.csv')
            self.esc_df = self.esc_df[self.esc_df.category != 'chirping_birds'].reset_index()
            self.esc_files = glob.glob('../dataset_soundscapes_esc50/logmel/44100/*.npy')
        if len(self.cleaning_path) > 0:
            self.oof = pd.read_csv(os.path.join('../result', self.cleaning_path, 'oof.csv'))
            self.oof['soundname'] = self.oof['soundname'].apply(lambda x:x.split('_')[-1])
            self.oof['image_idx'] = self.oof['filename'].apply(lambda x:int(x.split('_')[-1].split('.')[0]))
            self.oof = self.oof.reset_index(drop=True)
        else:
            self.oof = pd.DataFrame(columns=['soundname'])
    def __len__(self) -> int:
        return len(self.files)
    
    @staticmethod
    def get_metadata():
        df_2023 = pd.read_csv('../data/train_metadata.csv')
        df_2021 = pd.read_csv('../data_2021/train_metadata.csv')
        df_2022 = pd.read_csv('../data_2022/train_metadata.csv')
        df_2020 = pd.read_csv('../data_2020/train_extended.csv')
        df_2023['soundname'] = df_2023['filename'].map(lambda x: '2023_' + x[:-4])
        df_2022['soundname'] = df_2022['filename'].map(lambda x: '2022_' + x[:-4])
        df_2021['soundname'] = "2021_" + df_2021.primary_label + '/' + df_2021['filename'].map(lambda x: x[:-4])
        df_2020['soundname'] = "2020_" + df_2020.ebird_code + '/' + df_2020['filename'].map(lambda x: x.split('.')[0])
        df_2023 = df_2023[['soundname', 'rating', 'secondary_labels']]
        df_2022 = df_2022[['soundname', 'rating', 'secondary_labels']]
        df_2021 = df_2021[['soundname', 'rating', 'secondary_labels']]
        df_2020 = df_2020[['soundname', 'rating', 'secondary_labels']]
        return pd.concat([df_2020, df_2021, df_2022, df_2023], axis=0)
    
    def get_sound(self, file_path: int, start_idx:int = None):
        sound = np.load(file_path)
        sound_size = int(self.duration//5)
        if start_idx == None:
            start_idx = np.random.choice(sound.shape[0])

        sound = sound[start_idx,:,:]
        return sound, start_idx

    def get_label(self,target, meta, soundname, start_idx=0):
        if (len(self.cleaning_path) > 0) & (soundname in self.oof.soundname.unique()):
            labels = np.zeros(len(self.bird_label_dict.keys()), dtype=float)
            # ファイルごとにtrain, valを分けているのでoof_trainに絞らなくても良い
            oof_pred = self.oof[(self.oof['soundname']==soundname) & (self.oof.image_idx==start_idx)].iloc[0, 4:-1]
            if oof_pred[target]>0.5:
                labels[target] += 1.0
            elif oof_pred[target] > 0.1:
                labels[target] += 0.9975
            elif oof_pred[target] > 0.01:
                labels[target] += 0.5
            else:
                labels[target] += 0.2
            for slabel in eval(meta['secondary_labels']):
                if slabel in self.bird_label_dict.keys():
                    sec_idx = self.bird_label_dict[slabel]
                    if oof_pred[sec_idx]>0.5:
                        labels[sec_idx] = 0.9975
                    elif oof_pred[sec_idx] > 0.1:
                        labels[sec_idx] = 0.8
                    elif oof_pred[sec_idx] > 0.05:
                        labels[sec_idx] = 0.3
                    else:
                        labels[sec_idx] = 0.0025
        else:
            labels = np.zeros(len(self.bird_label_dict.keys()), dtype=float)
            labels[target] += 0.9999
            for slabel in eval(meta['secondary_labels']):
                if slabel in self.bird_label_dict.keys():
                    labels[self.bird_label_dict[slabel]] += 0.2999
        return labels
    
    def __getitem__(self, idx: int):
        idx2, idx3 = np.random.choice(len(self.files)), np.random.choice(len(self.files))
        label_all = np.zeros(len(self.bird_label_dict.keys()), dtype=float) + 0.0001
        image = np.zeros((128, 626))
        exit_flg = False
        for num, idx_ in enumerate([idx, idx2, idx3]):
            ds_name = self.files[idx_].split('/')[-4].split('_')[-1]
            if ds_name == 'dataset':
                ds_name = "2023"
            if self.split == 'train':
                sound, start_idx = self.get_sound(self.files[idx_])
                tmp_soundfile = ds_name + '_' + self.files[idx_].split('/')[-2]+'/'+self.files[idx_].split('/')[-1][:-4]
            else:
                start_idx = 0
                sound, _ = self.get_sound(self.files[idx_], start_idx=start_idx)
                tmp_soundfile = ds_name + '_' + self.files[idx_].split('/')[-2]+'/'+self.files[idx_].split('/')[-1][:-4]
            target = self.bird_label_dict[self.files[idx_].split('/')[-2]]
            meta = self.df_meta[self.df_meta['soundname']==tmp_soundfile].iloc[0]
            soundname = self.files[idx_].split('/')[-2]+'/'+self.files[idx_].split('/')[-1][:-4]
            label = self.get_label(target, meta, soundname, start_idx)
            sound_size = int(self.duration//5)
            # intra mixup
            if (self.split=='train') & (np.random.rand() > 0.75):
                sound2, start_idx = self.get_sound(self.files[idx_])
                ratio = np.random.rand()
                sound = librosa.db_to_power(sound) * ratio + librosa.db_to_power(sound2) * (1 - ratio)
                sound = librosa.power_to_db(sound) 
                label = label + self.get_label(target, meta, soundname, start_idx)
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
                if np.random.rand() > 0.75:
                    sound = soundscapes
                    label_all = np.zeros(len(self.bird_label_dict.keys()), dtype=float) + 0.001
                    label_all[-1] = 1.0
                    exit_flg = True
                else:
                    ratio = 0.5 * np.random.rand()
                    sound = librosa.db_to_power(sound) * (1-ratio) + librosa.db_to_power(soundscapes) * ratio
                    sound = librosa.power_to_db(sound)
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
            if ('bandpass' in self.aug_list) & (np.random.rand() > 0.3):
                start = random.randint(1, 128-10)
                leng = random.randint(0, 128-start)
                sound[start:start + leng,:] = sound[start:start + leng,:] + (np.random.sample((leng, sound.shape[1])).astype(np.float32) + 9) * 2 * sound.mean() * 0.05 * (np.random.sample() + 0.3)
            start = np.random.randint(0, sound.shape[1] - 100)
            length = np.random.randint(70, sound.shape[1])
            if exit_flg:
                image = sound
                break
            elif num == 0:
                image[:, start:start + length] = sound[:,start:start+length]
            else:
                image[:, start:start + length] = image[:,start:start+length] + sound[:,start:start+length]/(0.0000001+sound[:,start:start+length].max()) * (random.random()*1+0.5) * image.max()
            label_all = label_all + label
            if np.random.rand() > 0.8:
                break
        label_all = label_all.clip(0, 1)
        if ('esc50' in self.aug_list) & (np.random.rand() > 0.75):
            esc_img = np.load(self.esc_files[np.random.choice(len(self.esc_files))])[0]
            start = np.random.randint(0, esc_img.shape[1] - 100)
            length = np.random.randint(70, esc_img.shape[1]-start)
            image[:, start:start + length] = image[:,start:start+length] + esc_img[:,start:start+length]/(0.0000001+esc_img[:,start:start+length].max()) * (random.random()*1+0.5) * image.max()
        image = mono_to_color(image)
        sound = image.astype(float)
        if self.transform is not None:
            sound = self.transform(sound)
        sound = np.stack([sound, sound, sound])
        sound = sound.transpose(2,1,0)
        if (self.split == 'train') & ('cutout' in self.aug_list):
            sound = get_train_transform()(image=sound)['image']
        sound = sound.transpose(2,1,0)
        sound = torch.from_numpy(sound)
        if (self.split == 'train') & (np.random.rand() > 0.5):
            if ('time_mask' in self.aug_list):
                transforms_time_mask = nn.Sequential(
                        torchaudio.transforms.TimeMasking(time_mask_param=10),
                )
                time_mask_num = 2 # number of time masking
                for _ in range(time_mask_num): # tima masking
                    sound = transforms_time_mask(sound)
            if 'freq_mask' in self.aug_list:
                transforms_freq_mask = nn.Sequential(
                        torchaudio.transforms.FrequencyMasking(freq_mask_param=8),
                )
                freq_mask_num = 1 # number of frequency masking
                for _ in range(freq_mask_num): # frequency masking
                    sound = transforms_freq_mask(sound)
        
        sample = {
            "sound": sound,
            "target": label_all,
            "rating": meta['rating'],
        }

        return sample