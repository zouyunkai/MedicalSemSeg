import math
import sys
import time

import numpy as np
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch import autograd

import utils.misc as misc


def train_one_epoch(
            model, data_loader,
            optimizer, criterion, device, epoch,
            loss_scaler, cfg, log_writer=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    #metric_logger.add_meter('mHdorffDist', misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    metric_logger.add_meter('mDice', misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    for c in range(cfg.output_dim):
        name = 'class' + str(c) + 'Dice'
        metric_logger.add_meter(name, misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    post_label = AsDiscrete(to_onehot=cfg.output_dim)
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.output_dim)
    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)
    #haus_dist_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean", get_not_nans=True)

    iters = len(data_loader)
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    tid_data_start = time.time()
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        autograd.set_detect_anomaly(cfg.anomaly_detection)

        inputs, labels = (batch["image"], batch["label"])
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        tid_forward_start = time.time()
        tid_data = tid_forward_start - tid_data_start
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        tid_backward_start = time.time()
        tid_forward = tid_backward_start - tid_forward_start

        # Backwards step using loss scaler.
        # If scaler is disabled this acts as a simple backwards pass on loss as loss_scaler.scale(loss)
        # simply returns the loss in that scenario
        loss_scaler.scale(loss).backward()


        tid_step_start = time.time()
        tid_backward = tid_step_start - tid_backward_start

        if cfg.gradient_clipping is not None:
            # Unscale here is recorded by the scaler and thus is not performed again in the upcoming step
            # If scaler is disabled then this is a no-op
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)
        # If scaler is disabled, returns optimizer.step()
        loss_scaler.step(optimizer)
        # If scaled is disabled then this is a no-op
        loss_scaler.update()

        optimizer.zero_grad()

        torch.cuda.synchronize()

        tid_metric_start = time.time()
        tid_step = tid_metric_start - tid_step_start

        labels_list = decollate_batch(labels)
        labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
        outputs_list = decollate_batch(outputs)
        output_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
        dice_metric(y_pred=output_convert, y=labels_convert)
        #haus_dist_metric(y_pred=output_convert, y=labels_convert)
        dice_scores, dice_not_nans = dice_metric.aggregate()
        #hdorf_dist, hdorf_not_nans = haus_dist_metric.aggregate()

        class_means = torch.zeros(cfg.output_dim)
        for c in range(cfg.output_dim):
            if dice_not_nans[:,c].sum() > 0:
                class_dice = dice_scores[:,c].nanmean()
            else:
                class_dice = np.nan
            class_means[c] = class_dice
            keyword_args = {'class' + str(c) + 'Dice': class_dice}
            metric_logger.update(**keyword_args)
        
        mDice = class_means.nanmean()

        metric_logger.update(loss=loss_value)
        #metric_logger.update(mHdorffDist=hdorf_dist.item())
        metric_logger.update(mDice=mDice.item())

        dice_metric.reset()
        #haus_dist_metric.reset()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / iters + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        tid_data_start = time.time()
        tid_metric = tid_data_start - tid_metric_start
        print("Timers for one batch: Data: {:.5f} Forward: {:.5f} Backward: {:.5f} Step: {:.5f} Metric: {:.5f}".format(tid_data, tid_forward, tid_backward, tid_step, tid_metric))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Training averaged stats:", metric_logger.log_all_average())
    return {'train/' + k: meter.global_avg for k, meter in metric_logger.meters.items()}