import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import datetime
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

from libs.config import get_config
from libs.device import get_device
from libs.dataset import get_dataloader
from libs.models import get_model
from libs.logger import TrainLogger
from libs.loss_fn import get_criterion
from libs.helper import train, evaluate
from libs.seed import set_seed
from libs.get_augmentatoins import get_augmentations
from libs.dataset import BirdClefDataset
from libs.metric import padded_cmap, padded_cmap_numpy

import os
import math
import time
import random
import shutil
import gc
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import glob 

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.model_selection import KFold, StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

import wandb
import argparse
from logging import DEBUG, INFO, basicConfig, getLogger
import torch.nn as nn

logger = getLogger(__name__)

def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for sound pose estimation with Sound Pose Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Add --use_wandb option if you want to use wandb.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    return parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_arguments()
    # save log files in the directory which contains config file.
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    # setting logger configuration
    logname = os.path.join(result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log")
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # fix seed
    set_seed()

    # configuration
    config = get_config(args.config)
    if config.training_year == '2020_2021_2022':
        train_file_path = '../csv_2020_2021_2022/train_files.npy'
        val_file_path = '../csv_2020_2021_2022/val_files.npy'
    elif config.training_year == '2021_2022':
        train_file_path = '../csv_2021_2022/train_files.npy'
        val_file_path = '../csv_2021_2022/val_files.npy'
    else:
        train_file_path = '../csv/train_files.npy'
        val_file_path = '../csv/val_files.npy'
    train_files = np.load(train_file_path)
    val_files = np.load(val_file_path)
    if args.debug:
        config.max_epoch = 1
        logger.info('DEBUG Mode')
        train_files = train_files[:300]
        
        # config.batch_size = 96
    
    # cpu or cuda
    device = get_device(allow_only_gpu=False)
    birds_2021_2022 = ['lesgol', 'rinphe', 'afrsil1', 'buggna', 'bugtan', 'blkfra', 'masboo', 'acowoo', 'akekee', 'amecro', 'grepew', 'hawpet1', 'labwoo', 'oaktit', 'sonspa', 'brbmot1', 'creoro1', 'bulori', 'hawhaw', 'rorpar', 'stvhum2', 'golphe', 'blctan1', 'gcrwar', 'belkin1', 'killde', 'akepa1', 'eawpew', 'rucwar1', 'naswar', 'chispa', 'rudtur', 'mouchi', 'leater1', 'rethaw', 'brwpar1', 'lesvio1', 'hawgoo', 'comrav', 'woothr', 'pavpig2', 'cangoo', 'tromoc', 'amepip', 'whfpar1', 'comloo', 'fiespa', 'phivir', 'skylar', 'perfal', 'amekes', 'whevir', 'blbthr1', 'ducfly', 'leasan', 'ruff', 'sopsku1', 'norcar', 'dowwoo', 'crfpar', 'cobtan1', 'amerob', 'incter1', 'eurwig', 'treswa', 'hoomer', 'semplo', 'azaspi1', 'bknsti', 'tuftit', 'yetvir', 'canwar', 'canvas', 'bkbplo', 'ovenbi1', 'btbwar', 'apapan', 'plaxen1', 'herwar', 'rthhum', 'redpha1', 'blbgra1', 'reshaw', 'jabwar', 'oliwoo1', 'wetshe', 'sumtan', 'lesyel', 'redcro', 'compau', 'ruboro1', 'soulap1', 'japqua', 'grethr1', 'commyn', 'brnjay', 'wlswar', 'pygnut', 'snogoo', 'barpet', 'brtcur', 'bncfly', 'thbkin', 'bawswa1', 'btywar', 'wegspa1', 'crehon', 'yelwar', 'runwre1', 'pibgre', 'pabspi1', 'bnhcow', 'gnttow', 'sooshe', 'rettro', 'eastow', 'wiltur', 'norwat', 'rubrob', 'swaspa', 'whcsee1', 'mouwar', 'banana', 'bbwduc', 'stejay', 'bawwar', 'rugdov', 'hofwoo1', 'houfin', 'roahaw', 'linspa', 'grnher', 'gretin1', 'savspa', 'astfly', 'scptyr1', 'robgro', 'pinsis', 'ribgul', 'flrtan1', 'whtspa', 'whttro', 'brncre', 'chbant1', 'kebtou1', 'saypho', 'comsan', 'elepai', 'rcatan1', 'strfly1', 'osprey', 'yehbla', 'refboo', 'mastro1', 'fragul', 'wilsni1', 'shicow', 'shtsan', 'rucwar', 'yebori1', 'bkbwar', 'marwre', 'y00475', 'carchi', 'socfly1', 'btnwar', 'palila', 'plawre1', 'hudgod', 'willet1', 'kauama', 'yefgra1', 'chemun', 'wooduc', 'plsvir', 'haiwoo', 'whfibi', 'babwar', 'prowar', 'barswa', 'bucmot2', 'grhcha1', 'mallar3', 'ercfra', 'strcuc1', 'blcjay1', 'rebwoo', 'tropew1', 'lucwar', 'ameavo', 'relpar', 'yebcha', 'orfpar', 'compot1', 'chbwre1', 'gwfgoo', 'comwax', 'wantat1', 'goftyr1', 'bkhgro', 'calqua', 'brnowl', 'aldfly', 'grekis', 'caskin', 'foxspa', 'cubthr', 'rubwre1', 'grbani', 'wbwwre1', 'easblu', 'yegvir', 'smbani', 'reccar', 'trokin', 'pilwoo', 'norhar2', 'sthwoo1', 'botgra', 'rebnut', 'blknod', 'sobtyr1', 'chcant2', 'cotfly1', 'purfin', 'hawama', 'grnjay', 'yebfly', 'oahama', 'iiwi', 'leafly', 'baleag', 'comgal1', 'brbsol1', 'orcwar', 'chswar', 'wewpew', 'grasal1', 'grtgra', 'sinwre1', 'easmea', 'hawcoo', 'caltow', 'yeofly1', 'ccbfin', 'gnwtea', 'scrtan1', 'verdin', 'grcfly', 'goowoo1', 'laugul', 'batpig1', 'amered', 'gamqua', 'sooter1', 'puaioh', 'whwdov', 'fepowl', 'cacgoo1', 'merlin', 'rumfly1', 'cacwre', 'rewbla', 'gocspa', 'norsho', 'gohque1', 'logshr', 'wesmea', 'banswa', 'gryhaw2', 'warwhe1', 'tenwar', 'dusfly', 'whbman1', 'lessca', 'brcvir1', 'tropar', 'magpet1', 'higmot1', 'yeteup1', 'lazbun', 'sheowl', 'houspa', 'eucdov', 'clanut', 'bushti', 'sthant1', 'orbspa1', 'rinduc', 'comyel', 'blkpho', 'acafly', 'thbeup1', 'comgol', 'colcha1', 'herthr', 'pagplo', 'brnnod', 'whtdov', 'greyel', 'yebcar', 'dunlin', 'balori', 'rotbec', 'yebsap', 'parjae', 'eletro', 'commer', 'omao', 'grycat', 'chukar', 'palwar', 'rempar', 'wesant1', 'bobfly1', 'carwre', 'sora', 'pirfly1', 'brwhaw', 'yehcar1', 'burpar', 'gadwal', 'rawwre1', 'magwar', 'casfin', 'gocfly1', 'macwar', 'wrenti', 'spvear1', 'subfly', 'rebsap', 'easpho', 'gresca', 'bulpet', 'gartro1', 'peflov', 'sltred', 'akikik', 'brnthr', 'rinkin1', 'burwar1', 'layalb', 'warvir', 'monoro1', 'madpet', 'akiapo', 'houwre', 'compea', 'towsol', 'annhum', 'norpar', 'scatan', 'barant1', 'obnthr1', 'coopet', 'hawcre', 'melbla1', 'nutwoo', 'buffle', 'casvir', 'blsspa1', 'sposan', 'linwoo1', 'swathr', 'eursta', 'rudpig', 'moudov', 'stbori', 'thswar1', 'orbeup1', 'bkbmag1', 'coltro1', 'comgra', 'veery', 'maupar', 'glwgul', 'hergul', 'belvir', 'heptan', 'kalphe', 'sancra', 'coohaw', 'mitpar', 'norfli', 'lobgna5', 'blhpar1', 'chbsan', 'whcpar', 'lotman1', 'brratt1', 'strsal1', 'blugrb1', 'cintea', 'bewwre', 'mouela1', 'scamac1', 'yerwar', 'gbbgul', 'sibtan2', 'spotow', 'andsol1', 'lcspet', 'larspa', 'redjun', 'amtspa', 'lotjae', 'nutman', 'pecsan', 'banwre1', 'wilfly', 'rehbar1', 'whbnut', 'amewig', 'blujay', 'gockin', 'cliswa', 'paltan1', 'gbwwre1', 'lobdow', 'pasfly', 'daejun', 'whcspa', 'saffin', 'buwtea', 'wesblu', 'caster1', 'rucspa1', 'rtlhum', 'purgal2', 'cedwax', 'putfru1', 'weskin', 'bkmtou1', 'sander', 'cocwoo1', 'zebdov', 'whwbec1', 'bkcchi', 'bubsan', 'arcter', 'rubpep1', 'indbun', 'redava', 'lotduc', 'laufal1', 'olsfly', 'butsal1', 'reevir1', 'spodov', 'mauala', 'clcrob', 'grbher3', 'brant', 'royter1', 'nrwswa', 'scbwre1', 'aniani', 'hutvir', 'yebsee1', 'normoc', 'webwoo1', 'incdov', 'wessan', 'mouqua', 'chbchi', 'yelgro', 'westan', 'gryfra', 'whiter', 'rutjac1', 'cowscj1', 'gilwoo', 'pomjae', 'trogna1', 'plupig2', 'whimbr', 'ocbfly1', 'cregua1', 'yebela1', 'brnboo', 'littin1', 'lesgre1', 'bcnher', 'grefri', 'amegfi', 'bkwpet', 'rocpig', 'baywre1', 'solsan', 'bongul', 'meapar', 'vigswa', 'ruckin', 'rufhum', 'squcuc1', 'tunswa', 'orcpar', 'grhowl', 'cogdov', 'easkin', 'whiwre1', 'cinfly2', 'norpin', 'brebla', 'buhvir', 'mutswa']
    birds_2020_2021_2022 = ['purfin', 'soulap1', 'yerwar', 'yebori1', 'rthhum', 'eargre', 'yebela1', 'azaspi1', 'grnjay', 'yebsap', 'sinwre1', 'whcpar', 'rettro', 'lazbun', 'ruckin', 'blbgra1', 'akepa1', 'rorpar', 'yefcan', 'savspa', 'tropew1', 'lucwar', 'eawpew', 'cacgoo1', 'ocbfly1', 'rubwre1', 'lotman1', 'scptyr1', 'elepai', 'snobun', 'compot1', 'sander', 'scoori', 'easblu', 'dowwoo', 'pasfly', 'amtspa', 'orbeup1', 'tropar', 'wooscj2', 'annhum', 'aldfly', 'solsan', 'juntit1', 'webwoo1', 'coohaw', 'whcspa', 'mouwar', 'rucwar1', 'chbant1', 'lecthr', 'wilsni1', 'yetvir', 'haiwoo', 'rawwre1', 'rebsap', 'phivir', 'wetshe', 'commyn', 'pinwar', 'zebdov', 'yebfly', 'wbwwre1', 'linspa', 'chbwre1', 'squcuc1', 'comwax', 'sltred', 'foxspa', 'horlar', 'orcpar', 'hamfly', 'vigswa', 'ruboro1', 'rusbla', 'fiscro', 'royter1', 'sancra', 'bugtan', 'stbori', 'ribgul', 'magpet1', 'blcjay1', 'prawar', 'aniani', 'hawama', 'warwhe1', 'ducfly', 'cintea', 'sthwoo1', 'orcwar', 'comnig', 'sagspa1', 'cedwax', 'easmea', 'categr', 'tenwar', 'cangoo', 'comgol', 'gocfly1', 'labwoo', 'gnwtea', 'blctan1', 'rtlhum', 'buggna', 'canwre', 'dusfly', 'mastro1', 'wrenti', 'bcnher', 'goftyr1', 'warvir', 'easpho', 'norcar', 'blhpar1', 'wessan', 'moublu', 'indbun', 'layalb', 'lesvio1', 'swathr', 'ruff', 'amegfi', 'comyel', 'putfru1', 'whiwre1', 'westan', 'coltro1', 'japqua', 'ameavo', 'lessca', 'norpin', 'wegspa1', 'comred', 'amekes', 'canwar', 'marwre', 'bkbplo', 'norfli', 'rucspa1', 'willet1', 'eletro', 'hawgoo', 'sooshe', 'gresca', 'subfly', 'clanut', 'rinduc', 'gbwwre1', 'nutwoo', 'leabit', 'runwre1', 'linwoo1', 'monoro1', 'brwhaw', 'calgul', 'plaxen1', 'whwdov', 'goowoo1', 'plawre1', 'comsan', 'brespa', 'mutswa', 'comrav', 'redpha1', 'puaioh', 'larspa', 'baywre1', 'wiltur', 'wesmea', 'rewbla', 'rufhum', 'grcfly', 'brnnod', 'mouela1', 'banana', 'brbmot1', 'grepew', 'whfibi', 'btnwar', 'reevir1', 'scatan', 'pinjay', 'rinkin1', 'brwpar1', 'nrwswa', 'osprey', 'rutjac1', 'thswar1', 'btbwar', 'comgra', 'rubpep1', 'baisan', 'andsol1', 'lobgna5', 'relpar', 'shicow', 'whiter', 'yeteup1', 'acowoo', 'skylar', 'thbeup1', 'herwar', 'grhowl', 'phaino', 'truswa', 'pagplo', 'rudtur', 'rumfly1', 'bkcchi', 'brdowl', 'cregua1', 'lewwoo', 'littin1', 'lobdow', 'perfal', 'scbwre1', 'whimbr', 'hawcre', 'pinsis', 'glwgul', 'comloo', 'hudgod', 'yebsee1', 'heptan', 'butsal1', 'norhar2', 'cobtan1', 'gadwal', 'whevir', 'coshum', 'hoowar', 'moudov', 'blsspa1', 'canvas', 'grethr1', 'refboo', 'amebit', 'gockin', 'sumtan', 'meapar', 'trogna1', 'sagthr', 'ccbfin', 'pilwoo', 'veery', 'camwar', 'fepowl', 'sthant1', 'reshaw', 'scamac1', 'olsfly', 'melbla1', 'bncfly', 'comgal1', 'strsal1', 'bnhcow', 'gocspa', 'obnthr1', 'bkbcuc', 'blkfra', 'grycat', 'buwwar', 'lesnig', 'semplo', 'merlin', 'bawswa1', 'scrtan1', 'sobtyr1', 'wesgre', 'grekis', 'botgra', 'amerob', 'brebla', 'fiespa', 'yehcar1', 'evegro', 'btywar', 'ovenbi1', 'rethaw', 'batpig1', 'trokin', 'smbani', 'spodov', 'flrtan1', 'killde', 'chbchi', 'lobcur', 'strfly1', 'buwtea', 'crfpar', 'whtspa', 'iiwi', 'normoc', 'yeofly1', 'bkwpet', 'rocpig', 'burwar1', 'oliwoo1', 'incdov', 'sooter1', 'bknsti', 'tunswa', 'whbnut', 'belkin1', 'brcvir1', 'higmot1', 'pibgre', 'mitpar', 'bulpet', 'burpar', 'cotfly1', 'spotow', 'strcuc1', 'pygnut', 'gwfgoo', 'gcrwar', 'chemun', 'bewwre', 'sheowl', 'chiswi', 'chswar', 'leafly', 'gretin1', 'chispa', 'rudduc', 'prowar', 'hofwoo1', 'grbher3', 'yebcha', 'nutman', 'blknod', 'hawcoo', 'weskin', 'lesgre1', 'mauala', 'whfpar1', 'oahama', 'pomjae', 'barswa', 'bulori', 'mallar3', 'bkmtou1', 'gohque1', 'balori', 'roahaw', 'bobfly1', 'greroa', 'rehbar1', 'sposan', 'gnttow', 'shtsan', 'pecsan', 'hawpet1', 'jabwar', 'naswar', 'chcant2', 'gryhaw2', 'chukar', 'bkpwar', 'treswa', 'bbwduc', 'redcro', 'yehbla', 'bkbmag1', 'maupar', 'lotjae', 'magwar', 'coopet', 'houwre', 'vesspa', 'kalphe', 'louwat', 'compau', 'laugul', 'yelgro', 'eucdov', 'rebwoo', 'goleag', 'thbkin', 'verdin', 'colcha1', 'towsol', 'tromoc', 'macwar', 'buhvir', 'grbani', 'blujay', 'lesyel', 'wesant1', 'gilwoo', 'cowscj1', 'banswa', 'bktspa', 'cubthr', 'semsan', 'gbbgul', 'reccar', 'laufal1', 'wantat1', 'cogdov', 'pingro', 'easkin', 'brnthr', 'grasal1', 'amepip', 'parjae', 'comter', 'sora', 'brnowl', 'gryfly', 'leasan', 'grtgra', 'sopsku1', 'pirfly1', 'bkchum', 'robgro', 'carchi', 'brbsol1', 'amered', 'apapan', 'amewig', 'swahaw', 'herthr', 'whtswi', 'dunlin', 'saffin', 'madpet', 'hawhaw', 'socfly1', 'amecro', 'blugrb1', 'grnher', 'hutvir', 'rugdov', 'rubrob', 'paltan1', 'arcter', 'omao', 'kebtou1', 'barpet', 'lcspet', 'akekee', 'clcrob', 'yefgra1', 'eastow', 'brthum', 'rucwar', 'yebcar', 'bushti', 'tuftit', 'plsvir', 'cliswa', 'renpha', 'wooduc', 'norsho', 'wilfly', 'peflov', 'whttro', 'eursta', 'spvear1', 'chbsan', 'bawwar', 'palwar', 'lotduc', 'whcsee1', 'wewpew', 'eurwig', 'buffle', 'saypho', 'whbman1', 'blbthr1', 'yegvir', 'yelwar', 'rebnut', 'wesblu', 'rufgro', 'whwbec1', 'daejun', 'golphe', 'casvir', 'greegr', 'gryfra', 'whtdov', 'caltow', 'rcatan1', 'astfly', 'palila', 'rotbec', 'caskin', 'mouchi', 'purgal2', 'belspa2', 'bongul', 'cacwre', 'ercfra', 'mouqua', 'boboli', 'babwar', 'winwre3', 'afrsil1', 'incter1', 'acafly', 'norwat', 'kauama', 'rempar', 'rinphe', 'gartro1', 'grefri', 'logshr', 'norpar', 'brnjay', 'masboo', 'creoro1', 'hergul', 'commer', 'bubsan', 'houfin', 'brncre', 'crehon', 'grhcha1', 'rebmer', 'caster1', 'brratt1', 'brant', 'cinfly2', 'calqua', 'banwre1', 'bucmot2', 'brtcur', 'akikik', 'pabspi1', 'houspa', 'casfin', 'greyel', 'barant1', 'y00475', 'lesgol', 'gamqua', 'fragul', 'snogoo', 'akiapo', 'compea', 'orbspa1', 'shshaw', 'baleag', 'blkpho', 'orfpar', 'carwre', 'cocwoo1', 'rudpig', 'wlswar', 'rocwre', 'swaspa', 'stvhum2', 'oaktit', 'plupig2', 'pavpig2', 'leater1', 'amewoo', 'bkbwar', 'bkhgro', 'woothr', 'belvir', 'redjun', 'hoomer', 'stejay', 'sonspa', 'sibtan2', 'redava', 'brnboo']
    if config.training_year == '2023':
        ss = pd.read_csv('../../BirdCLEF2023/data/sample_submission.csv')
        birds = list(ss.columns[1:])
    elif config.training_year == '2020_2021_2022':
        birds = birds_2020_2021_2022
    elif config.training_year == '2021_2022':
        birds = birds_2021_2022
    
    bird_label_map = {birds[i]:i for i in range(len(birds))}
    bird_label_map['nocall'] = 264
    oof_df = pd.DataFrame()
    all_df = pd.DataFrame(columns  = ['filename', 'primary_label'])
    all_df['filename'] = np.concatenate([train_files, val_files], 0)
    all_df['primary_label'] = all_df['filename'].apply(lambda x: x.split('/')[-2])
    all_df['soundname'] = all_df['filename'].map(lambda x: os.path.join(x.split('/')[-4].split('_')[-1] + '_' + x.split('/')[-2], x.split('/')[-1])[:-4])
    meta_df = BirdClefDataset.get_metadata()
    meta_df_2023 = pd.read_csv('../data/train_metadata.csv')
    meta_df_count = meta_df_2023.groupby('primary_label').size()
    few_labels = meta_df_count[meta_df_count<config.n_split].index.tolist()
    all_df = all_df.merge(meta_df, on='soundname', how='left')
    all_df[birds] = 0
    few_df = all_df[all_df.primary_label.isin(few_labels)]
    all_df = all_df[~all_df.primary_label.isin(few_labels)].reset_index()
    kf = StratifiedKFold(n_splits=config.n_split, random_state=args.seed, shuffle=True)
    log_path = os.path.join(result_path,'log.csv')
    train_logger = TrainLogger(log_path, resume=args.resume)
    final_result = pd.DataFrame(columns=['fold','final_score','best_score'])
    one_sample_files = meta_df
    for fold, (train_index, val_index) in enumerate(kf.split(all_df, y=all_df['primary_label'])):
        # モデルを毎回リセット
        if "2021_2022" in config.model_path:
            if '2020' in config.model_path:
                past_birds = birds_2020_2021_2022
            else:
                past_birds = birds_2021_2022
            if 'exp055' in config.model_path:
                nocall_flg = 0
            else:
                nocall_flg = 1
            base_path = '/'.join(args.config.split('/')[:-2])
            pretrained_path = glob.glob(os.path.join(base_path, config.model_path) + '/*best*.prm')[0]
            print(pretrained_path)
            # pretrained_path = os.path.join(base_path, config.model_path, 'fold0_best_model.prm')
            model = get_model(
                config.model,
                output_dim=len(past_birds) + nocall_flg,
                pretrained_path=pretrained_path
            )
        else:
            model = get_model(
                config.model,
                output_dim=len(birds) + 1
            )
            
        if args.use_wandb:
            wandb.init(
                name=experiment_name + '_fold{}'.format(fold),
                config=config,
                project="BirdCLEF2023",
                job_type="training",
                # dirs="./wandb_result/",
            )
            # Magic
            wandb.watch(model, log="all")
            
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = config.lr_max)

        # ToDo: Schedulerの作成
        if config.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=config.max_epoch, eta_min=config.lr_min, last_epoch=-1)
        
        # criterion for loss
        criterion = get_criterion(loss_fn = config.loss_fn)
        begin_epoch = 0
        if (config.n_split==2):
            train_files_fold = train_files
            val_files_fold = val_files
        else:
            train_files_fold = list(all_df.iloc[train_index]['filename'].values) + list(few_df['filename'].values)
            val_files_fold = list(all_df.iloc[val_index]['filename'].values)
        #　検証データに関しては全時間のサンプルを取得する
        new_val_files_fold = []
        for file in val_files_fold:
            tmp_sound = np.load(file)
            seconds = tmp_sound.shape[0]
            new_files = ['..' + file.split('.')[-2] + f'_{i}.npy' for i in range(seconds)]
            new_val_files_fold.extend(new_files)
        train_loader = get_dataloader(
            files = train_files_fold,
            batch_size=config.batch_size,
            split='train',
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            transform=None,
            bird_label_map = bird_label_map,
            shuffle=True,
            aug_list=get_augmentations(config.aug_ver),
            duration=config.duration,
            cleaning_path=config.cleaning_path,
            is_hard=config.is_hard
        )
        val_loader = get_dataloader(
            files = val_files_fold,
            batch_size=config.batch_size,
            split='val',
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            transform=None,
            bird_label_map = bird_label_map,
            shuffle=False,
            aug_list=[],
            duration=config.duration,
            cleaning_path='',
            is_hard=False
        ) 
        if (config.training_year != '2023'):
            val_oof_loader = val_loader
            new_val_files_fold = val_files_fold
        else:
            val_oof_loader = get_dataloader(
                files = new_val_files_fold,
                batch_size=config.batch_size,
                split='oof',
                num_workers=2,
                pin_memory=True,
                drop_last=False,
                transform=None,
                bird_label_map = bird_label_map,
                shuffle=False,
                aug_list=[],
                duration=config.duration,
                cleaning_path='',
                is_hard=False,
            )    
        fold_df = all_df.iloc[val_index]
        fold_df = pd.DataFrame(columns  = ['filename', 'primary_label'])
        fold_df['filename'] = new_val_files_fold
        if (config.training_year == '2023')&(config.is_hard==False):
            fold_df['second'] = fold_df['filename'].apply(lambda x:int(x.split('_')[-1].split('.')[0]))
        else:
            fold_df['second'] = 0
        fold_df['primary_label'] = fold_df['filename'].apply(lambda x: x.split('/')[-2])
        fold_df['soundname'] = fold_df['filename'].map(lambda x: x.split('/')[-4].split('_')[-1] + '_' + os.path.join(x.split('/')[-2], x.split('/')[-1].split('_')[0]))    
        is_done = (os.path.isfile(os.path.join(result_path, f'fold{fold}_final_model.prm'))) | (os.path.isfile(os.path.join(result_path, f'fold{fold}_final_model_.prm')))           
        if is_done == False:
            logger.info(f'Fold {fold} Start training')
            best_score = 0
            best_epoch = 0
            for epoch in range(begin_epoch, config.max_epoch):
                start = time.time()
                train_loss, gts, preds, train_score = train(
                    train_loader, model, criterion, optimizer, scheduler, epoch, device, do_mixup=config.do_mixup
                )
                train_time = int(time.time()-start)
                
                start = time.time()
                val_loss, val_gts, val_preds, val_score = evaluate(
                    val_loader, model, criterion, device, do_mixup=False
                )
                # tmp_val_df = fold_df.copy()
                # tmp_val_df[birds] = val_preds
                # tmp_val_df['second'] = tmp_val_df['filename'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))
                # tmp_val_df['primary_label_num'] = tmp_val_df['primary_label'].apply(lambda x:bird_label_map[x])
                # first_samples = tmp_val_df[tmp_val_df.second == 0]
                # val_score = padded_cmap_numpy(predictions=first_samples[birds].values, gts=first_samples['primary_label_num'].values)
                val_time = int(time.time() - start)
                if val_score > best_score:
                    best_preds = val_preds
                    best_score = val_score
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(result_path, f'fold{fold}_best_model.prm'))
                    
                train_logger.update(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    train_time,
                    train_loss,
                    train_score,
                    val_time,
                    val_loss,
                    val_score,
                )

                # save logs to wandb
                if args.use_wandb:
                    wandb.log(
                        {
                            f"fold_{fold}_lr": optimizer.param_groups[0]["lr"],
                            f"fold_{fold}_train_time[sec]": train_time,
                            f"fold_{fold}_train_loss": train_loss,
                            f"fold_{fold}_val_time[sec]": val_time,
                            f"fold_{fold}_val_loss": val_loss,
                            f"fold_{fold}_val_score": val_score
                        },
                        step=epoch,
                    )
                if epoch > best_epoch + config.early_stop_epoch:
                    break
              
            torch.save(model.state_dict(), os.path.join(result_path, f'fold{fold}_final_model.prm'))   
            logger.info(f'fold_{fold}_best score:{best_score}')
            best_model = model.load_state_dict(torch.load(os.path.join(result_path, f'fold{fold}_best_model.prm')))
            _, _, best_preds, val_score = evaluate(
                    val_oof_loader, model, criterion, device, do_mixup=False
            )
            fold_df[birds] = best_preds   
            oof_df = pd.concat([oof_df, fold_df])
            
        else:
            model_path = os.path.join(result_path, f'fold_{fold}_best_model.prm')
            logger.info(model_path)
            model.load_state_dict(torch.load(os.path.join(result_path, f'fold{fold}_best_model.prm')))
            val_loss, val_gts, _, best_score = evaluate(
                    val_loader, model, criterion, device, do_mixup=False
            )
            val_score = best_score
            logger.info(f'fold{fold}_best score:{best_score}')
            _, _, best_preds, _ = evaluate(
                val_oof_loader, model, criterion, device, do_mixup=False
            )
            fold_df[birds] = best_preds   
            oof_df = pd.concat([oof_df, fold_df]) 
            logger.info(f"Best score:{best_score}")   
            wandb.finish()
            del train_loader, val_loader
            gc.collect()
            fold_df[birds] = best_preds   
            oof_df = pd.concat([oof_df, fold_df])
        tmp = pd.Series(
                [fold, val_score, best_score],
                index=['fold','final_score','best_score'],
            )
        final_result = final_result.append(tmp,ignore_index=True)
        if config.n_split == 2:
            break
    final_result.to_csv(os.path.join(result_path, f'final_result.csv'), index=False)
    oof_df.sort_index().to_csv(os.path.join(result_path, 'oof.csv'), index=False)
    logger.info('Make oof_pred Done !') 
    logger.info('ALL Done !')     
        
    
if __name__ == '__main__':
    main()