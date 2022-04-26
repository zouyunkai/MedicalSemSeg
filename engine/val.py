import math
import sys

import numpy as np
import torch
import torch.distributed as dist
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, compute_meandice
from monai.metrics.utils import do_metric_reduction
from monai.transforms import AsDiscrete
from torch import autograd

import utils.misc as misc


def run_validation(inferer,
        model, data_loader, criterion, device, epoch, cfg, log_writer=None):

    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mHdorffDist', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mDice', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    for c in range(cfg.output_dim):
        name = 'class' + str(c) + 'Dice'
        metric_logger.add_meter(name, misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Validation for epoch: [{}]'.format(epoch)
    print_freq = 5

    post_label = AsDiscrete(to_onehot=cfg.output_dim)
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.output_dim)
    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)
    haus_dist_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean", get_not_nans=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    ### Test of new calculations for val dice scores ###
    metric = torch.zeros((cfg.output_dim - 1) * 2, dtype=torch.float, device=device)
    metric_sum = 0.0
    metric_count = 0
    metric_mat = []
    ####################################################


    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        autograd.set_detect_anomaly(cfg.anomaly_detection)

        inputs, labels = (batch["image"], batch["label"])
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                outputs = inferer(inputs=inputs, network=model)
                loss = criterion(outputs, labels)

        ### Test of new calculations for val dice scores ###
        ct = 1.0
        val_outputs = outputs / ct
        val_outputs = post_pred(val_outputs[0, ...])
        val_outputs = val_outputs[None, ...]
        val_labels = post_label(val_labels[0, ...])
        val_labels = val_labels[None, ...]
        value = compute_meandice(
            y_pred=val_outputs,
            y=val_labels,
            include_background=False
        )
        metric_count += len(value)
        metric_sum += value.sum().item()
        metric_vals = value.cpu().numpy()
        if len(metric_mat) == 0:
            metric_mat = metric_vals
        else:
            metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

        for _c in range(cfg.output_dim - 1):
            val0 = torch.nan_to_num(value[0, _c], nan=0.0)
            val1 = 1.0 - torch.isnan(value[0, 0]).float()
            metric[2 * _c] += val0 * val1
            metric[2 * _c + 1] += val1
        ####################################################


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping validation".format(loss_value))
            sys.exit(1)

        labels_list = decollate_batch(labels)
        labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
        outputs_list = decollate_batch(outputs)
        output_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
        dice_metric(y_pred=output_convert, y=labels_convert)
        haus_dist_metric(y_pred=output_convert, y=labels_convert)
        dice_scores, dice_not_nans = dice_metric.aggregate()
        hdorf_dist, hdorf_not_nans = haus_dist_metric.aggregate()

        mDice, _ = do_metric_reduction(dice_scores, reduction='mean')

        metric_logger.update(loss=loss_value)
        metric_logger.update(mHdorffDist=hdorf_dist.item())
        metric_logger.update(mDice=mDice.item())
        for c in range(cfg.output_dim):
            if dice_not_nans[0][c] > 0:
                class_dice = dice_scores[0][c].item()
            else:
                class_dice = None
            keyword_args = {'class' + str(c) + 'Dice': class_dice}
            metric_logger.update(**keyword_args)

        dice_metric.reset()
        haus_dist_metric.reset()

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)

    ### Test of new calculations for val dice scores ###
    dist.barrier()
    dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
    metric = metric.tolist()
    new_dice_dict = {}
    if dist.get_rank() == 0:
        for _c in range(cfg.output_dim - 1):
            name = 'val/class{}DiceNew'.format(_c)
            val = metric[2 * _c] / metric[2 * _c + 1]
            new_dice_dict[name] = val
            print("evaluation metric - class {0:d}: {}".format(_c, val))
        avg_metric = 0
        for _c in range(cfg.output_dim - 1):
            avg_metric += metric[2 * _c] / metric[2 * _c + 1]
        avg_metric = avg_metric / float(cfg.output_dim - 1)
        name = 'val/mDiceNew'
        new_dice_dict[name] = avg_metric
        print("evaluation metric - Mean dice: {}".format(avg_metric))
    ####################################################
    # gather the stats from all processes
    torch.cuda.synchronize()
    metric_logger.synchronize_between_processes()
    print("Validation averaged stats:", metric_logger.log_all_average())
    val_dict = {'val/' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    new_val_dict = {**val_dict, **new_dice_dict}
    return new_val_dict