from logging import getLogger

import argparse
import glob
import os
import sys
from typing import Dict, List, Union
import tqdm

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tqdm
from multiprocessing import Pool

logger = getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make sound dataset for sound pose estimation"
    )
    parser.add_argument(
        "--sound_dir",
        type=str,
        default="../data/train_audio",
        help="path to a sound dirctory",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="../data/train_metadata.csv",
        help="a path where annotations are",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../dataset",
        help="a directory where sound dataset will be saved",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=32000,
        help="sampling rate of sound data",
    )
    parser.add_argument(
        "--processed_method",
        type=str,
        default="logmel",
        help="a path where processed mic data",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=128,
        help="the number of mel bins",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=2048,
        help="the number of mel bins",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="the number of mel bins",
    )
    return parser.parse_args()

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y

class LogMelIntensityExtractor:
    def __init__(self, sr, nfft, n_mels, fmin = 0, fmax=24000, duration=5, resample=True, save_dir = ''):
        self.n_mels = n_mels
        self.nfft = nfft
        self.sr = sr
        self.melW = librosa.filters.mel(
            sr=self.sr,
            n_fft=nfft,
        )
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.step = self.audio_length
        self.resample = resample
        self.save_dir = save_dir
        
    def logmel(self, sig):
        melspec = librosa.feature.melspectrogram(
            y=sig, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
        )
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        return melspec
    
    def transform(self, sig_path):
        sound, original_sr = sf.read(sig_path)
        if (self.resample==True) & (original_sr != self.sr):
            sound = librosa.resample(sound, original_sr, self.sr, res_type="kaiser_fast")
        sounds = [sound[i:i+self.audio_length] for i in range(0, max(1, len(sound) - self.audio_length + 1), self.step)]
        sounds[-1] = crop_or_pad(sounds[-1] , length=self.audio_length)
        images = [self.logmel(sound) for sound in sounds]
        images = np.stack(images)
        return images
    
    def save_soundfile(self, sig_path):
        sound, original_sr = sf.read(sig_path)
        if (self.resample==True) & (original_sr != self.sr):
            sound = librosa.resample(sound, original_sr, self.sr, res_type="kaiser_fast")
        sounds = [sound[i:i+self.audio_length] for i in range(0, max(1, len(sound) - self.audio_length + 1), self.step)]
        sounds[-1] = crop_or_pad(sounds[-1] , length=self.audio_length)
        images = [self.logmel(sound) for sound in sounds]
        images = np.stack(images)
        label = sig_path.split('/')[-2]
        file_id = sig_path.split('/')[-1].split('.')[0]
        np.save(os.path.join(self.save_dir, label + '_' + file_id + '.npy'), images)
        return images


def main() -> None:
    args = get_arguments()

    # 保存ディレクトリがなければ，作成
    os.makedirs(args.save_dir, exist_ok=True)
    if args.processed_method == "logmel":
        dataset_name = "logmel"
    else:
        message = f"preprocess method is not prepared"
        raise ValueError(message)
    
    dataset_dir = os.path.join(args.save_dir, dataset_name)

    # すでにデータセットが存在しているなら終了
    if os.path.exists(dataset_dir):
        print("Sound dataset exists.")
        # return
    else:
        os.mkdir(dataset_dir)
    DEBUG = False
    meta_df = pd.read_csv(args.meta_path)
    meta_df['filename'] = meta_df['filename'].apply(lambda x:os.path.join(args.sound_dir, x))
    feature_extractor = LogMelIntensityExtractor(sr = args.sr, nfft=args.nfft, n_mels=args.n_mels)
    filepath_list = meta_df['filename']
    
    # for idx in range(len(meta_df)):
    #     filepath = meta_df.iloc[idx]['filename']
    #     melspec = feature_extractor.transform(filepath)
    #     print(filepath)
    #     label = filepath.split('/')[-2]
    #     file_id = filepath.split('/')[-1].split('.')[0]
    #     # assert False
    #     np.save(os.path.join(args.save_dir, label + '_' + file_id + '.npy'), melspec)
        
    print("Finished making sound dataset.")
    
if __name__ == "__main__":
    main()