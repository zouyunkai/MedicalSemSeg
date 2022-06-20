import os

import nibabel as nib
import numpy as np
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from torch import autograd

import utils.misc as misc


def eval_model(inferer, model, data_loader, criterion, device, cfg, log_writer=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mHdorffDist', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mDice', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    for c in range(cfg.output_dim):
        name = 'class' + str(c) + 'Dice'
        metric_logger.add_meter(name, misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Evaluation starting'
    print_freq = 1

    post_label = AsDiscrete(to_onehot=cfg.output_dim)
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.output_dim)
    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)
    haus_dist_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean", get_not_nans=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    autograd.set_detect_anomaly(cfg.anomaly_detection)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.cuda.empty_cache()

        inputs, labels = (batch["image"], batch["label"])
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        original_affine = batch['label_meta_dict']['affine'][0].numpy()
        img_name = os.path.split(batch['image_meta_dict']['filename_or_obj'][0])[-1]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                outputs = inferer(inputs=inputs, network=model)
                loss = criterion(outputs.cpu(), labels.cpu())

        labels_list = decollate_batch(labels)
        labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
        outputs_list = decollate_batch(outputs)
        output_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
        dice_metric(y_pred=output_convert, y=labels_convert)
        haus_dist_metric(y_pred=output_convert, y=labels_convert)
        dice_scores, dice_not_nans = dice_metric.aggregate()
        hdorf_dist, hdorf_not_nans = haus_dist_metric.aggregate()

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
        loss_value = loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mHdorffDist=hdorf_dist.item())
        metric_logger.update(mDice=mDice.item())

        dice_metric.reset()
        haus_dist_metric.reset()

        if cfg.save_eval_output:
            # Save pred
            outputs_formatted = torch.softmax(outputs, 1).cpu().numpy()
            outputs_formatted = np.argmax(outputs_formatted, axis=1).astype(np.uint8)[0]
            nib.save(nib.Nifti1Image(outputs_formatted.astype(np.uint8), original_affine),
                     os.path.join(cfg.output_dir, 'pred_' + img_name))
            # Save transformed image
            nib.save(nib.Nifti1Image(inputs.squeeze().cpu().numpy(), original_affine), os.path.join(cfg.output_dir, 'img_' + img_name))
            # Save transformed label
            nib.save(nib.Nifti1Image(labels.squeeze().cpu().numpy(), original_affine),
                     os.path.join(cfg.output_dir, 'gt_' + img_name))

    # gather the stats from all processes
    print("Evaluation averaged stats:", metric_logger.log_all_average())
    eval_dict = {'eval/' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return eval_dict