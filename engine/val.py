import math
import sys

import numpy as np
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch import autograd

import utils.misc as misc
from engine.utils import sliding_window_inference


def run_validation(model, data_loader, criterion, device, epoch, cfg, log_writer=None):

    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    metric_logger.add_meter('mDice', misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    for c in range(cfg.output_dim):
        name = 'class' + str(c) + 'Dice'
        metric_logger.add_meter(name, misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Validation for epoch: [{}]'.format(epoch)
    print_freq = 1

    post_label = AsDiscrete(to_onehot=cfg.output_dim)
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.output_dim)
    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)

    if cfg.t_normalize:
        air_cval = (0.0 - cfg.t_norm_mean)/cfg.t_norm_std
    else:
        air_cval = 0.0

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        autograd.set_detect_anomaly(cfg.anomaly_detection)

        inputs, labels = (batch["image"], batch["label"])
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        org_affine = batch['image_meta_dict']['original_affine']
        aff_xyz = misc.get_affine_xyz(org_affine)
        aff_xyz = aff_xyz.float()
        aff_xyz = aff_xyz.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                outputs = sliding_window_inference(inputs=inputs,
                                                   affine=aff_xyz,
                                                   predictor=model,
                                                   roi_size=cfg.vol_size,
                                                   sw_batch_size=cfg.batch_size_val,
                                                   overlap=cfg.val_infer_overlap,
                                                   mode='gaussian',
                                                   device=device,
                                                   sw_device=device,
                                                   cval=air_cval
                                                   )
                loss = criterion(outputs, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping validation".format(loss_value))
            sys.exit(1)

        labels_list = decollate_batch(labels)
        labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
        outputs_list = decollate_batch(outputs)
        output_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
        dice_metric(y_pred=output_convert, y=labels_convert)
        dice_scores, dice_not_nans = dice_metric.aggregate()

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
        metric_logger.update(mDice=mDice.item())

        dice_metric.reset()

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)

    # gather the stats from all processes
    torch.cuda.synchronize()
    metric_logger.synchronize_between_processes()
    print("Validation averaged stats:", metric_logger.log_all_average())
    val_dict = {'val/' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return val_dict


