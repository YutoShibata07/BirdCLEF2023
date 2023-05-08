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

import os
import math
import time
import random
import shutil
import gc
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

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
    
    if config.training_year == '2021_2022':
        train_file_path = '../csv_2021_2022/train_files.npy'
        val_file_path = '../csv_2021_2022/val_files.npy'
    else:
        train_file_path = '../csv/train_files.npy'
        val_file_path = '../csv/val_files.npy'
    train_files = np.load(train_file_path)
    val_files = np.load(val_file_path)
    if args.debug:
        config.max_epoch = 1
        config.file_limit = 1
        logger.info('DEBUG Mode')
        config.val_ver = 0
        train_files = train_files[:300]
        
        # config.batch_size = 96
    
    # cpu or cuda
    device = get_device(allow_only_gpu=False)
    birds_2021_2022 = ['lesgol', 'rinphe', 'afrsil1', 'buggna', 'bugtan', 'blkfra', 'masboo', 'acowoo', 'akekee', 'amecro', 'grepew', 'hawpet1', 'labwoo', 'oaktit', 'sonspa', 'brbmot1', 'creoro1', 'bulori', 'hawhaw', 'rorpar', 'stvhum2', 'golphe', 'blctan1', 'gcrwar', 'belkin1', 'killde', 'akepa1', 'eawpew', 'rucwar1', 'naswar', 'chispa', 'rudtur', 'mouchi', 'leater1', 'rethaw', 'brwpar1', 'lesvio1', 'hawgoo', 'comrav', 'woothr', 'pavpig2', 'cangoo', 'tromoc', 'amepip', 'whfpar1', 'comloo', 'fiespa', 'phivir', 'skylar', 'perfal', 'amekes', 'whevir', 'blbthr1', 'ducfly', 'leasan', 'ruff', 'sopsku1', 'norcar', 'dowwoo', 'crfpar', 'cobtan1', 'amerob', 'incter1', 'eurwig', 'treswa', 'hoomer', 'semplo', 'azaspi1', 'bknsti', 'tuftit', 'yetvir', 'canwar', 'canvas', 'bkbplo', 'ovenbi1', 'btbwar', 'apapan', 'plaxen1', 'herwar', 'rthhum', 'redpha1', 'blbgra1', 'reshaw', 'jabwar', 'oliwoo1', 'wetshe', 'sumtan', 'lesyel', 'redcro', 'compau', 'ruboro1', 'soulap1', 'japqua', 'grethr1', 'commyn', 'brnjay', 'wlswar', 'pygnut', 'snogoo', 'barpet', 'brtcur', 'bncfly', 'thbkin', 'bawswa1', 'btywar', 'wegspa1', 'crehon', 'yelwar', 'runwre1', 'pibgre', 'pabspi1', 'bnhcow', 'gnttow', 'sooshe', 'rettro', 'eastow', 'wiltur', 'norwat', 'rubrob', 'swaspa', 'whcsee1', 'mouwar', 'banana', 'bbwduc', 'stejay', 'bawwar', 'rugdov', 'hofwoo1', 'houfin', 'roahaw', 'linspa', 'grnher', 'gretin1', 'savspa', 'astfly', 'scptyr1', 'robgro', 'pinsis', 'ribgul', 'flrtan1', 'whtspa', 'whttro', 'brncre', 'chbant1', 'kebtou1', 'saypho', 'comsan', 'elepai', 'rcatan1', 'strfly1', 'osprey', 'yehbla', 'refboo', 'mastro1', 'fragul', 'wilsni1', 'shicow', 'shtsan', 'rucwar', 'yebori1', 'bkbwar', 'marwre', 'y00475', 'carchi', 'socfly1', 'btnwar', 'palila', 'plawre1', 'hudgod', 'willet1', 'kauama', 'yefgra1', 'chemun', 'wooduc', 'plsvir', 'haiwoo', 'whfibi', 'babwar', 'prowar', 'barswa', 'bucmot2', 'grhcha1', 'mallar3', 'ercfra', 'strcuc1', 'blcjay1', 'rebwoo', 'tropew1', 'lucwar', 'ameavo', 'relpar', 'yebcha', 'orfpar', 'compot1', 'chbwre1', 'gwfgoo', 'comwax', 'wantat1', 'goftyr1', 'bkhgro', 'calqua', 'brnowl', 'aldfly', 'grekis', 'caskin', 'foxspa', 'cubthr', 'rubwre1', 'grbani', 'wbwwre1', 'easblu', 'yegvir', 'smbani', 'reccar', 'trokin', 'pilwoo', 'norhar2', 'sthwoo1', 'botgra', 'rebnut', 'blknod', 'sobtyr1', 'chcant2', 'cotfly1', 'purfin', 'hawama', 'grnjay', 'yebfly', 'oahama', 'iiwi', 'leafly', 'baleag', 'comgal1', 'brbsol1', 'orcwar', 'chswar', 'wewpew', 'grasal1', 'grtgra', 'sinwre1', 'easmea', 'hawcoo', 'caltow', 'yeofly1', 'ccbfin', 'gnwtea', 'scrtan1', 'verdin', 'grcfly', 'goowoo1', 'laugul', 'batpig1', 'amered', 'gamqua', 'sooter1', 'puaioh', 'whwdov', 'fepowl', 'cacgoo1', 'merlin', 'rumfly1', 'cacwre', 'rewbla', 'gocspa', 'norsho', 'gohque1', 'logshr', 'wesmea', 'banswa', 'gryhaw2', 'warwhe1', 'tenwar', 'dusfly', 'whbman1', 'lessca', 'brcvir1', 'tropar', 'magpet1', 'higmot1', 'yeteup1', 'lazbun', 'sheowl', 'houspa', 'eucdov', 'clanut', 'bushti', 'sthant1', 'orbspa1', 'rinduc', 'comyel', 'blkpho', 'acafly', 'thbeup1', 'comgol', 'colcha1', 'herthr', 'pagplo', 'brnnod', 'whtdov', 'greyel', 'yebcar', 'dunlin', 'balori', 'rotbec', 'yebsap', 'parjae', 'eletro', 'commer', 'omao', 'grycat', 'chukar', 'palwar', 'rempar', 'wesant1', 'bobfly1', 'carwre', 'sora', 'pirfly1', 'brwhaw', 'yehcar1', 'burpar', 'gadwal', 'rawwre1', 'magwar', 'casfin', 'gocfly1', 'macwar', 'wrenti', 'spvear1', 'subfly', 'rebsap', 'easpho', 'gresca', 'bulpet', 'gartro1', 'peflov', 'sltred', 'akikik', 'brnthr', 'rinkin1', 'burwar1', 'layalb', 'warvir', 'monoro1', 'madpet', 'akiapo', 'houwre', 'compea', 'towsol', 'annhum', 'norpar', 'scatan', 'barant1', 'obnthr1', 'coopet', 'hawcre', 'melbla1', 'nutwoo', 'buffle', 'casvir', 'blsspa1', 'sposan', 'linwoo1', 'swathr', 'eursta', 'rudpig', 'moudov', 'stbori', 'thswar1', 'orbeup1', 'bkbmag1', 'coltro1', 'comgra', 'veery', 'maupar', 'glwgul', 'hergul', 'belvir', 'heptan', 'kalphe', 'sancra', 'coohaw', 'mitpar', 'norfli', 'lobgna5', 'blhpar1', 'chbsan', 'whcpar', 'lotman1', 'brratt1', 'strsal1', 'blugrb1', 'cintea', 'bewwre', 'mouela1', 'scamac1', 'yerwar', 'gbbgul', 'sibtan2', 'spotow', 'andsol1', 'lcspet', 'larspa', 'redjun', 'amtspa', 'lotjae', 'nutman', 'pecsan', 'banwre1', 'wilfly', 'rehbar1', 'whbnut', 'amewig', 'blujay', 'gockin', 'cliswa', 'paltan1', 'gbwwre1', 'lobdow', 'pasfly', 'daejun', 'whcspa', 'saffin', 'buwtea', 'wesblu', 'caster1', 'rucspa1', 'rtlhum', 'purgal2', 'cedwax', 'putfru1', 'weskin', 'bkmtou1', 'sander', 'cocwoo1', 'zebdov', 'whwbec1', 'bkcchi', 'bubsan', 'arcter', 'rubpep1', 'indbun', 'redava', 'lotduc', 'laufal1', 'olsfly', 'butsal1', 'reevir1', 'spodov', 'mauala', 'clcrob', 'grbher3', 'brant', 'royter1', 'nrwswa', 'scbwre1', 'aniani', 'hutvir', 'yebsee1', 'normoc', 'webwoo1', 'incdov', 'wessan', 'mouqua', 'chbchi', 'yelgro', 'westan', 'gryfra', 'whiter', 'rutjac1', 'cowscj1', 'gilwoo', 'pomjae', 'trogna1', 'plupig2', 'whimbr', 'ocbfly1', 'cregua1', 'yebela1', 'brnboo', 'littin1', 'lesgre1', 'bcnher', 'grefri', 'amegfi', 'bkwpet', 'rocpig', 'baywre1', 'solsan', 'bongul', 'meapar', 'vigswa', 'ruckin', 'rufhum', 'squcuc1', 'tunswa', 'orcpar', 'grhowl', 'cogdov', 'easkin', 'whiwre1', 'cinfly2', 'norpin', 'brebla', 'buhvir', 'mutswa']
    if config.training_year == '2023':
        ss = pd.read_csv('../../BirdCLEF2023/data/sample_submission.csv')
        birds = list(ss.columns[1:])
    else:
        # birds = [record_path.split('/')[-2] for record_path in train_files] + [record_path.split('/')[-2] for record_path in val_files]
        # birds = list(set(birds)) 
        birds = birds_2021_2022
        

    bird_label_map = {birds[i]:i for i in range(len(birds))}
    train_loader = get_dataloader(
        files = train_files,
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
    )

    val_loader = get_dataloader(
        files = val_files,
        batch_size=config.batch_size,
        split='val',
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        transform=None,
        bird_label_map = bird_label_map,
        shuffle=False,
        aug_list=[],
        duration=config.duration
    )
    if "2021_2022" in config.model_path:
        base_path = '/'.join(args.config.split('/')[:-2])
        pretrained_path = os.path.join(base_path, config.model_path, 'best_model.prm')
        model = get_model(
            config.model,
            output_dim=len(birds_2021_2022),
            pretrained_path=pretrained_path
        )
    else:
        model = get_model(
            config.model,
            output_dim=len(birds)
        )
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = config.lr_max)
    
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="BirdCLEF2023",
            job_type="training",
            # dirs="./wandb_result/",
        )
        # Magic
        wandb.watch(model, log="all")
    
    # ToDo: Schedulerの作成
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=config.max_epoch, eta_min=config.lr_min, last_epoch=-1)
    log_path = os.path.join(result_path,'log.csv')
    train_logger = TrainLogger(log_path, resume=args.resume)
    
    # criterion for loss
    criterion = get_criterion(loss_fn = config.loss_fn)
    begin_epoch = 0
    is_done = (os.path.isfile(os.path.join(result_path, f'final_model.prm'))) | (os.path.isfile(os.path.join(result_path, f'final_model_.prm')))
    oof_df = pd.DataFrame(columns  = ['filename', 'primary_label'] + birds)
    oof_df['filename'] = val_files
    oof_df['primary_label'] = oof_df['filename'].apply(lambda x: x.split('/')[-2])
    oof_df[birds] = 0
    if is_done == False:
        logger.info(f'Start training')
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
            val_time = int(time.time() - start)
            if val_score > best_score:
                best_preds = val_preds
                best_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(result_path, f'best_model.prm'))
                
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
                        f"lr": optimizer.param_groups[0]["lr"],
                        f"train_time[sec]": train_time,
                        f"train_loss": train_loss,
                        f"val_time[sec]": val_time,
                        f"val_loss": val_loss,
                        f"val_score": val_score
                    },
                    step=epoch,
                )
            if epoch > best_epoch + config.early_stop_epoch:
                break
        torch.save(model.state_dict(), os.path.join(result_path, f'final_model.prm'))
        logger.info(f'best score:{best_score}')
        oof_df[birds] = best_preds
    else:
        model_path = os.path.join(result_path, f'best_model.prm')
        logger.info(model_path)
        model.load_state_dict(torch.load(model_path))
        val_loss, val_gts, best_preds, best_score = evaluate(
                val_loader, model, criterion, device, do_mixup=False
        )
        val_score = best_score
        logger.info(f'best score:{best_score}')
        oof_df[birds] = best_preds
        
    wandb.finish()
    del model, train_loader, val_loader
    gc.collect()
    final_result = pd.DataFrame(columns=['final_score','best_score'])
    logger.info(f"Final score:{val_score}")
    logger.info(f"Best score:{best_score}")
    tmp = pd.Series(
            [val_score, best_score],
            index=['final_score','best_score'],
        )
    final_result = final_result.append(tmp,ignore_index=True)
    final_result.to_csv(os.path.join(result_path, 'final_result.csv'), index=False)
    oof_df.to_csv(os.path.join(result_path, 'oof_df.csv'), index=False)
    logger.info('ALL Done !')
    
if __name__ == '__main__':
    main()