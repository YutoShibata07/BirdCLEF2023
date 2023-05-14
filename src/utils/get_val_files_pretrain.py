import glob
import numpy as np
import os

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
from tqdm import tqdm
import json
import random
logger = getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make sound dataset for sound pose estimation"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../dataset",
        help="path to a sound dirctory",
    )
    parser.add_argument(
        "--input_feature",
        type=str,
        default="logmel",
        help="a directory where sound dataset will be saved",
    )
    parser.add_argument(
        '--csv_dir',
        type=str,
        default='../csv_2021_2022'
    )
    
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    logger.info("Finished setting up seed.")
    
def get_val_files(dataset_dir:str='', input_feature:str = 'logmel'):
    files_2023 = np.concatenate([np.load('../csv/val_files.npy'), np.load('../csv/train_files.npy')])
    files_2023 = ['/'.join(file.split('/')[-2:]) for file in files_2023]
    assert len(files_2023) == 16941
    species_2021 = glob.glob(os.path.join('../dataset_2021', input_feature,'*'))
    species_2022 = glob.glob(os.path.join('../dataset_2022', input_feature,'*'))
    species_2020 = glob.glob(os.path.join('../dataset_2020', input_feature,'*'))
    all_species = np.concatenate([species_2020, species_2021,species_2022])
    # print(all_species)
    train_files = []
    val_files = []
    for bird_path in all_species:
        bird_samples = glob.glob(os.path.join(bird_path, '*.npy'))
        bird_samples = [sample for sample in bird_samples if '/'.join(sample.split('/')[-2:]) not in files_2023]
        if len(bird_samples) == 1:
            train_files.extend(bird_samples)
        else:
            np.random.shuffle(bird_samples)
            val_idx = int(len(bird_samples) * 0.2)
            val_files.extend(bird_samples[:val_idx])
            train_files.extend(bird_samples[val_idx:])
    print(len(train_files))
    print(len(val_files))
    return train_files, val_files


if __name__ == '__main__':
    args = get_arguments()
    set_seed(seed=44)
    os.makedirs(args.csv_dir, exist_ok=True)
    args_dict = vars(args)
    f = open(args.csv_dir + '/' + 'args.txt', 'w')
    f.write(json.dumps(args_dict))
    f.close()
    train_files, val_files = get_val_files(args.dataset_dir, input_feature=args.input_feature)
    np.save(os.path.join(args.csv_dir, 'train_files.npy'), train_files)
    np.save(os.path.join(args.csv_dir, 'val_files.npy'), val_files)
    
    