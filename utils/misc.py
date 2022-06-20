import builtins
import datetime
import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return 0.0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        if self.deque:
            return self.fmt.format(
                median=self.median,
                avg=self.avg,
                global_avg=self.global_avg,
                max=self.max,
                value=self.value)
        else:
            return self.fmt.format(
                median=0.0,
                avg=0.0,
                global_avg=0.0,
                max=0.0,
                value=0.0)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None or v is np.nan:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def log_all_average(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(cfg):
    if cfg.dist_on_itp:
        cfg.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        cfg.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        cfg.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        cfg.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(cfg.gpu)
        os.environ['RANK'] = str(cfg.rank)
        os.environ['WORLD_SIZE'] = str(cfg.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
        print("Using OpenMPI distributed settings")
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.gpu = int(os.environ['LOCAL_RANK'])
        print("Using distributed settings from environment variables")
    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ['SLURM_PROCID'])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
        print("Using slurm distributed settings")
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        cfg.distributed = False
        return

    cfg.distributed = True

    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = cfg.backend
    print('| distributed init (rank {}): {}, gpu {}'.format(
        cfg.rank, cfg.dist_url, cfg.gpu), flush=True)
    if 'SLURM_PROCID' in os.environ:
        srank = int(os.environ['SLURM_PROCID'])
        devc = torch.cuda.device_count()
        print("Slurm rank is: {} and device count is: {}".format(srank, devc))
    torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                         world_size=cfg.world_size, rank=cfg.rank)
    torch.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)

def save_model(cfg, epoch, model, model_without_ddp, optimizer, loss_scaler, scheduler):
    output_dir = Path(cfg.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'scheduler': scheduler.state_dict(),
                'cfg': cfg,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=cfg.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

def load_model(cfg, model_without_ddp, optimizer, loss_scaler, scheduler):
    if cfg.resume:
        if cfg.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % cfg.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(cfg, 'eval') and cfg.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            cfg.start_epoch = checkpoint['epoch'] + 1
            print("With optim!")
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With loss scaler!")
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("With scheduler!")

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def log_to_neptune(neptune_logger, metric_dict):
    epoch = metric_dict['epoch']
    for k, v in metric_dict.items():
        if not k == 'epoch':
            neptune_logger[k].log(v, epoch)

def tag_builder(cfg):
    tags = list()
    tags.append('Finetuning')
    tags.append(cfg.model)
    if cfg.input_dim == 3:
        tags.append('3D')
    else:
        tags.append('2D')
    tags.append(cfg.task)
    return tags

def get_1d_sincos_embed_from_range(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / (embed_dim*10)**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def save_decathlon_datalist(org_json_path, train_files, val_files, log_dir):
    jsonf = open(org_json_path)
    json_data = json.load(jsonf)

    train_files_fixed = []
    val_files_fixed = []

    for tf in train_files:
        train_files_fixed.append(clean_strings(tf))

    for vf in val_files:
        val_files_fixed.append(clean_strings(vf))

    json_data['training'] = train_files_fixed
    json_data['validation'] = val_files_fixed
    json_data['numTraining'] = len(train_files_fixed)
    json_data['numValidation'] = len(val_files_fixed)

    with open(os.path.join(log_dir, 'dataset_split.json'), 'w') as fp:
        json.dump(json_data, fp, indent=4)



def clean_strings(dict_obj):
    clean_string_img = re.sub(r'^.*?imagesTr', './imagesTr', dict_obj['image'])
    clean_string_img = re.sub(r'\\', '/', clean_string_img)

    clean_string_label = re.sub(r'^.*?labelsTr', './labelsTr', dict_obj['label'])
    clean_string_label = re.sub(r'\\', '/', clean_string_label)

    clean_data = {'image': clean_string_img, 'label': clean_string_label}

    return clean_data
