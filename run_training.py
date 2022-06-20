import datetime
import json
import os
import time
from pathlib import Path

import monai
import neptune.new as neptune
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
from monai.data import ThreadDataLoader
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss
from tensorboardX import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import utils.misc as misc
from data.dataset_builder import build_train_and_val_datasets
from engine.train import train_one_epoch
from engine.val import run_validation
from models.model_builder import build_model
from models.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
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
    monai.utils.set_determinism(seed=seed, additional_settings=None)

    # -- Setup config --
    cfg_dict = vars(cfg)

    # -- Enable logging to file and online logging to Neptune --
    if misc.get_rank() == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=cfg.log_dir)
    else:
        log_writer = None
    if misc.get_rank() == 0 and cfg.neptune_logging:
        tags = misc.tag_builder(cfg)
        neptune_logger = neptune.init(description=cfg.description)
        neptune_logger['parameters'] = cfg_dict
        neptune_logger['sys/tags'].add(tags)

    # -- Setup data --
    dataset_train, dataset_val = build_train_and_val_datasets(cfg)
    #sampler_val = DistSampler(dataset_val, shuffle=False) if cfg.distributed else None
    #print("Sampler_val = %s" % str(sampler_val))

    data_loader_train = ThreadDataLoader(
        dataset_train,
        batch_size=cfg.n_images_per_batch,
        num_workers=0,
        pin_memory=cfg.pin_mem,
        drop_last=True,
    )

    data_loader_val = ThreadDataLoader(
        dataset_val,
        batch_size=1,
        num_workers=0,
        pin_memory=cfg.pin_mem,
        drop_last=False,
    )

    # Setup model
    model = build_model(cfg)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-6)
    print(optimizer)
    loss_scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision, init_scale=4096)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=args.warmup_epochs,
                                              max_epochs=args.epochs)

    misc.load_model(cfg=cfg, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,
                    scheduler=scheduler)

    if cfg.loss_fn == 'DiceCE':
        criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=cfg.smooth_nr,
                               smooth_dr=cfg.smooth_dr)
    elif cfg.loss_fn == 'Tversky':
        criterion = TverskyLoss(to_onehot_y=True, softmax=True, alpha=cfg.tversky_alpha, beta=cfg.tversky_beta,
                                smooth_nr=cfg.smooth_nr, smooth_dr=cfg.smooth_dr)
    elif cfg.loss_fn == 'DiceFocal':
        criterion = DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=cfg.smooth_nr,
                                  smooth_dr=cfg.smooth_dr)
    else:
        raise RuntimeError('Could not parse loss function argument.')


    if cfg.t_normalize:
        air_cval = (0.0 - cfg.t_norm_mean)/cfg.t_norm_std
    else:
        air_cval = 0.0

    inferer = SlidingWindowInferer(
        roi_size=cfg.vol_size,
        sw_batch_size=cfg.batch_size_val,
        overlap=cfg.val_infer_overlap,
        mode='gaussian',
        cval=air_cval
    )

    # Run training
    start_time = time.time()
    #dataset_train.start()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            #data_loader_train.sampler.set_epoch(epoch)
            #data_loader_val.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, criterion, device, epoch,
            loss_scaler, cfg, log_writer=log_writer)
        log_stats = {**{f'{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if not((epoch + 1) % cfg.val_interval):
            torch.cuda.empty_cache()
            if args.distributed:
                torch.distributed.barrier()
            val_stats = run_validation(inferer,
                model, data_loader_val, criterion, device, epoch,
                log_writer=log_writer, cfg=cfg)
            log_stats_val = {**{f'{k}': v for k, v in val_stats.items()},
                     'epoch': epoch, }
            log_stats = {**log_stats, **log_stats_val}

        if cfg.output_dir and (epoch % cfg.save_ckpt_freq == 0 or epoch + 1 == cfg.epochs):
            misc.save_model(
                cfg=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, scheduler=scheduler)

        if misc.is_main_process() and cfg.neptune_logging:
            misc.log_to_neptune(neptune_logger, log_stats)

        if cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        scheduler.step()
        #dataset_train.update_cache()
    #dataset_train.shutdown()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if misc.is_main_process() and cfg.neptune_logging:
        neptune_logger.stop()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
