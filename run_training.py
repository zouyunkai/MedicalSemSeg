import datetime
import json
import os
import time
from pathlib import Path

import neptune.new as neptune
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
from monai.data import DataLoader
from monai.losses import DiceCELoss
from tensorboardX import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import utils.misc as misc
from data.dataset_builder import build_train_and_val_datasets
from engine.train import train_one_epoch
from engine.val import run_validation
from models.model_builder import build_model
from utils.arguments import get_args


@record
def main(cfg):
    # -- Initialize distributed mode and hardware --
    misc.init_distributed_mode(cfg)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Fix the seed for reproducibility --
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -- Setup config --
    cfg_dict = vars(cfg)

    # -- Enable logging to file and online logging to Neptune --
    if misc.get_rank() == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=cfg.log_dir)
    else:
        log_writer = None
    if misc.get_rank() == 1:
        neptune_logger = neptune.init()
        neptune_logger['parameters'] = cfg_dict

    # -- Setup data --
    dataset_train, dataset_val = build_train_and_val_datasets(cfg)

    if cfg.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
        )
        #sampler_val = DistributedEvalSampler(
        #    dataset_val, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    print("Sampler_train = %s" % str(sampler_train))
    print("Sampler_val = %s" % str(sampler_val))

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=cfg.batch_size_train,
        num_workers=cfg.n_workers_train,
        pin_memory=cfg.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=cfg.batch_size_val,
        num_workers=cfg.n_workers_val,
        pin_memory=cfg.pin_mem,
        drop_last=False,
    )

    # Setup model
    model = build_model(cfg)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = torch.cuda.amp.GradScaler()

    misc.load_model(cfg=cfg, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)

    # Run training
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            #data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, criterion, device, epoch,
            loss_scaler, cfg, log_writer=log_writer)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if not(epoch % cfg.val_interval):
            val_stats = run_validation(
                model, data_loader_val, criterion, device, epoch,
                log_writer=log_writer, cfg=cfg)
            log_stats_val = {**{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch, }
            log_stats = {**log_stats, **log_stats_val}

        if cfg.output_dir and (epoch % cfg.save_ckpt_freq == 0 or epoch + 1 == cfg.epochs):
            misc.save_model(
                cfg=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if misc.is_main_process():
            misc.log_to_neptune(neptune_logger, log_stats)

        if cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if misc.is_main_process():
        neptune_logger.stop()


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
