import math
import sys

import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from torch import autograd

import utils.misc as misc


def train_one_epoch(
            model, data_loader,
            optimizer, criterion, device, epoch,
            loss_scaler, cfg, log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mHdorffDist', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mDice', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    haus_dist_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        autograd.set_detect_anomaly(cfg.anomaly_detection)

        inputs, labels = (batch["image"], batch["label"])
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)

        print("Output shape is {}".format(outputs.shape))
        print("Output min is {} and max is {}".format(outputs.min(), outputs.max()))
        print("Output has nan {}".format(torch.isnan(outputs).any()))
        loss = criterion(outputs, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backwards step using loss scaler
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        optimizer.zero_grad()

        torch.cuda.synchronize()

        labels_list = decollate_batch(labels)
        labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
        outputs_list = decollate_batch(outputs)
        output_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
        dice_metric(y_pred=output_convert, y=labels_convert)
        haus_dist_metric(y_pred=output_convert, y=labels_convert)
        dice_score = dice_metric.aggregate().item()
        hdorf_dist = haus_dist_metric.aggregate().item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mHdorffDist=hdorf_dist)
        metric_logger.update(mDice=dice_score)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'train/' + k: meter.global_avg for k, meter in metric_logger.meters.items()}