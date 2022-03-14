import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from tensorboardX import SummaryWriter
from pathlib import Path
from utils.arguments import get_args
import utils.misc as misc
import numpy as np
import neptune.new as neptune
import os
from data.dataset_builder import build_training_dataset, build_validation_dataset
from models.segmentors.builder import build_segmentor


def main(args):
    # -- Initialize distributed mode and hardware --
    misc.init_distributed_mode(args)
    torch.backend.cudnn.benchmark = True

    # -- Fix the seed for reproducibility --
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -- Setup config --
    cfg = misc.read_config(args.config)
    cfg.update_from_args(args)

    # -- Enable logging to file and online logging to Neptune --
    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=args.log_dir)
    else:
        log_writer = None
    if misc.get_rank() == 0:
        neptune_logger = neptune.init()
        neptune_logger['parameters'] = cfg

    # -- Setup data --
    dataset_train = build_training_dataset(cfg)
    dataset_val = build_validation_dataset(cfg)

    if cfg.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
    )

    # Setup model
    model = build_segmentor(cfg)

    # Run training

if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
