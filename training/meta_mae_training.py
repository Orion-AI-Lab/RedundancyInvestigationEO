# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from utilities.mae_utilities import *
import timm.optim.optim_factory as optim_factory
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import models.mae as models_mae
import utilities.utils as utils

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    configs=None):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = configs['accum_iter']

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, configs)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=configs['mask_ratio'])

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train(configs):
    print('='*20)
    print('Initializing MAE')
    print('='*20)
    configs['output_dir'] = configs['checkpoint_path']
    configs['log_dir'] = configs['checkpoint_path']

    configs = init_distributed_mode(configs)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(configs).replace(', ', ',\n'))

    device = torch.device(configs['device'])

    # fix the seed for reproducibility
    seed = configs['seed'] + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_data = utils.load_dataset(configs,mode='train')
    val_data = utils.load_dataset(configs,mode='val')
    dataset_train = torch.utils.data.ConcatDataset([train_data, val_data])
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and configs['log_dir'] is not None:
        os.makedirs(configs['log_dir'], exist_ok=True)
        log_writer = SummaryWriter(log_dir=configs['log_dir'])
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=configs['batch_size'],
        num_workers=configs['num_workers'],
        pin_memory=configs['pin_mem'],
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[configs['model']](norm_pix_loss=configs['norm_pix_loss'])

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = configs.batch_size * configs.accum_iter * get_world_size()
    
    if configs['lr'] is None:  # only base_lr is specified
        configs['lr'] = configs['blr'] * eff_batch_size / 256

    print("base lr: %.2e" % (configs['lr'] * 256 / eff_batch_size))
    print("actual lr: %.2e" % configs['lr'])

    print("accumulate grad iterations: %d" % configs['accum_iter'])
    print("effective batch size: %d" % eff_batch_size)

    if configs['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs['gpu']], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, configs['weight_decay'])
    optimizer = torch.optim.AdamW(param_groups, lr=configs['lr'], betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScalerWithGradNormCount()

    load_model(args=configs, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {configs['epochs']} epochs")
    start_time = time.time()
    for epoch in range(configs['start_epoch'], configs['epochs']):
        if configs['distributed']:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            configs=configs
        )
        if configs['output_dir'] and (epoch % 20 == 0 or epoch + 1 == configs['epochs']):
            save_model(
                args=configs, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if configs['output_dir'] and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(configs['output_dir'], "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))