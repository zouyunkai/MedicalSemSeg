import datetime
import os
import time
from pathlib import Path

import torch
from monai.data import ThreadDataLoader
from tensorboardX import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import utils.misc as misc
from data.dataset_builder import build_test_dataset
from engine.test import test_model
from models.model_builder import build_model
from utils.arguments import get_args


@record
def main(cfg):
    # -- Initialize hardware --
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Enable logging to file --
    if misc.get_rank() == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=cfg.log_dir)
    else:
        log_writer = None

    # -- Setup data --
    dataset_test = build_test_dataset(cfg)

    data_loader_eval = ThreadDataLoader(
        dataset_test,
        batch_size=1,
        num_workers=0,
        pin_memory=cfg.pin_mem,
        drop_last=False,
    )

    # Setup model
    model = build_model(cfg)
    model.to(device)
    print("Model = %s" % str(model))

    model_state_dict = torch.load(cfg.resume)['model']
    model.load_state_dict(model_state_dict)

    # Run evaluation
    start_time = time.time()

    test_model(model, data_loader_eval, device, cfg, log_writer=log_writer)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
