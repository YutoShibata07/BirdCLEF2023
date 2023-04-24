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
    
    train_files = np.load('/home/ubuntu/slocal/BirdCLEF2023/csv/train_files.npy')
    val_files = np.load('/home/ubuntu/slocal/BirdCLEF2023/csv/val_files.npy')
    if args.debug:
        config.max_epoch = 1
        config.file_limit = 1
        logger.info('DEBUG Mode')
        config.val_ver = 0
        train_files = train_files[:300]
        
        # config.batch_size = 96
    
    # cpu or cuda
    device = get_device(allow_only_gpu=False)
        
    ss = pd.read_csv('/home/ubuntu/slocal/BirdCLEF2023/data/sample_submission.csv')
    birds = list(ss.columns[1:])
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
    )
    model = get_model(
        config.model,
        output_dim=264
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
    criterion = get_criterion()
    begin_epoch = 0
    is_done = os.path.isfile(os.path.join(result_path, f'final_model.prm'))
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
                train_loader, model, criterion, optimizer, scheduler, epoch, device
            )
            train_time = int(time.time()-start)
            
            start = time.time()
            val_loss, val_gts, val_preds, val_score = evaluate(
                val_loader, model, criterion, device
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
        model.load_state_dict(torch.load(os.path.join(result_path, f'best_model.prm')))
        val_loss, val_gts, best_preds, best_score = evaluate(
                val_loader, model, criterion, device, bin_num=config.bin_num
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