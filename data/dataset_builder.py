import os
import random

import monai
import numpy as np
from monai.apps import CrossValidation, DecathlonDataset
from monai.data import (
    CacheDataset,
    Dataset,
    load_decathlon_datalist,
    partition_dataset
)

from data.transforms import ScaleCubedIntensityRanged
from utils.misc import get_world_size, is_main_process, get_rank


def build_training_transforms(cfg):
    transforms = [
        monai.transforms.LoadImaged(keys=["image", "label"]),
        monai.transforms.AddChanneld(keys=["image", "label"])
    ]
    if cfg.t_voxel_spacings:
        transforms.append(
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=cfg.t_voxel_dims,
                mode=("bilinear", "nearest"),
            ))
    if cfg.t_cubed_ct_intensity:
        transforms.append(
            ScaleCubedIntensityRanged(
                keys=["image"],
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
        ))
    elif cfg.t_fixed_ct_intensity:
        transforms.append(
            monai.transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
        ))
    else:
        transforms.append(
            monai.transforms.ScaleIntensityRangePercentilesD(
                keys=['image'],
                lower=5,
                upper=95,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                relative=False
            )
        )
    if cfg.t_crop_foreground:
        transforms.append(
            monai.transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image"
        ))
    if cfg.t_spatial_pad:
        transforms.append(
            monai.transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=cfg.vol_size,
        ))
    if cfg.t_rand_crop_fgbg:
        transforms.append(
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.vol_size,
                pos=1,
                neg=1,
                num_samples=cfg.t_n_samples,
                image_key="image",
                image_threshold=0,
            )
        )
    elif cfg.t_rand_crop_classes:
        transforms.append(
            monai.transforms.RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.vol_size,
                num_classes=cfg.output_dim,
                num_samples=cfg.t_n_samples,
                image_key="image",
                image_threshold=0,
            )
        )
    transforms.append(
        monai.transforms.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=cfg.t_flip_prob,
        )
    )
    transforms.append(
        monai.transforms.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=cfg.t_flip_prob,
        )
    )
    transforms.append(
        monai.transforms.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=cfg.t_flip_prob,
        )
    )
    transforms.append(
        monai.transforms.RandRotate90d(
            keys=["image", "label"],
            prob=cfg.t_rot_prob,
            max_k=3,
        )
    )
    transforms.append(
        monai.transforms.RandShiftIntensityd(
            keys=["image"],
            offsets=cfg.t_intensity_shift_os,
            prob=cfg.t_intensity_shift_prob,
        )
    )
    transforms.append(
        monai.transforms.RandScaleIntensityd(
            keys=["image"],
            factors=cfg.t_intensity_scale_factors,
            prob=cfg.t_intensity_scale_prob,
        )
    )
    if cfg.t_normalize:
        transforms.append(
            monai.transforms.NormalizeIntensityd(
                keys=['image'],
                subtrahend=cfg.t_norm_mean,
                divisor = cfg.t_norm_std
            )
        )
    transforms.append(
        monai.transforms.ToTensord(keys=["image", "label"])
    )
    return monai.transforms.Compose(transforms)


def build_validation_transforms(cfg):
    transforms = [
        monai.transforms.LoadImaged(keys=["image", "label"]),
        monai.transforms.AddChanneld(keys=["image", "label"])
    ]
    if cfg.t_voxel_spacings:
        transforms.append(
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=cfg.t_voxel_dims,
                mode=("bilinear", "nearest"),
            ))
    if cfg.t_cubed_ct_intensity:
        transforms.append(
            ScaleCubedIntensityRanged(
                keys=["image"],
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
        ))
    elif cfg.t_fixed_ct_intensity:
        transforms.append(
            monai.transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
        ))
    else:
        transforms.append(
            monai.transforms.ScaleIntensityRangePercentilesD(
                keys=['image'],
                lower=5,
                upper=95,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                relative=False
            )
        )
    if cfg.t_crop_foreground:
        transforms.append(
            monai.transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="label"
        ))
    if cfg.t_spatial_pad:
        transforms.append(
            monai.transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=cfg.vol_size,
        ))
    if cfg.t_normalize:
        transforms.append(
            monai.transforms.NormalizeIntensityd(
                keys=['image'],
                subtrahend=cfg.t_norm_mean,
                divisor = cfg.t_norm_std
            )
        )
    transforms.append(
        monai.transforms.ToTensord(keys=["image", "label"])
    )
    return monai.transforms.Compose(transforms)


def build_train_dataset(data_path, transform, dstype='training', cache_rate=1.0, num_workers=8):
    data_json = os.path.join(data_path, 'dataset_val.json')
    data_files = load_decathlon_datalist(data_json, True, dstype)
    if is_main_process():
        print("Number of files in total training dataset: {}".format(len(data_files)))

    data_partition = partition_dataset(data=data_files,
                                       num_partitions=get_world_size(),
                                       shuffle=True,
                                       even_divisible=True)[get_rank()]
    print("Number of files in training dataset partition for rank {}:{}".format(get_rank(), len(data_partition)), force=True)

    dataset = CacheDataset(
        data=data_partition,
        transform=transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    print("Number of files in training CacheDataset for rank {}:{}".format(get_rank(), len(dataset)), force=True)
    return dataset

def build_val_cachedataset(data_path, transform, dstype='validation', cache_rate=1.0, num_workers=4):
    data_json = os.path.join(data_path, 'dataset_val.json')
    data_files = load_decathlon_datalist(data_json, True, dstype)
    if is_main_process():
        print("Number of files in total validation dataset: {}".format(len(data_files)))
    data_partition = partition_dataset(data=data_files,
                                       num_partitions=get_world_size(),
                                       shuffle=False,
                                       even_divisible=False)[get_rank()]
    print("Number of files in validation dataset partition for rank {}:{}".format(get_rank(), len(data_partition)), force=True)
    dataset = CacheDataset(
        data=data_partition,
        transform=transform,
        cache_rate=cache_rate,
        num_workers=num_workers
    )
    print("Number of files in validation CacheDataset for rank {}:{}".format(get_rank(), len(dataset)), force=True)
    return dataset

def build_val_dataset(data_path, transform, dstype='validation'):
    data_json = os.path.join(data_path, 'dataset_val.json')
    data_files = load_decathlon_datalist(data_json, True, dstype)
    if is_main_process():
        print("Number of files in total validation dataset: {}".format(len(data_files)))
    dataset = Dataset(
        data=data_files,
        transform=transform,
    )
    return dataset

def build_decathlon_cv_datasets_dist(cfg, train_transform, val_transform):
    data_json = os.path.join(cfg.data_path, cfg.task, 'dataset.json')
    data_files = load_decathlon_datalist(data_json, True, 'training')
    if is_main_process():
        print("Number of files in total training dataset: {}".format(len(data_files)))

    # Split for Cross Validation
    random.Random(cfg.seed).shuffle(data_files)
    cv_splits = np.array_split(data_files, cfg.cv_folds)
    train_folds = list(range(cfg.cv_folds))
    train_folds.pop(cfg.curr_fold)
    train_files = [cv_splits[i] for i in train_folds]
    train_files = [file for files in train_files for file in files]
    val_files = cv_splits[cfg.curr_fold]
    if is_main_process():
        print("Number of files in training cv split: {}".format(len(train_files)))
        print("Number of files in val cv split: {}".format(len(val_files)))

    # Split for GPU's
    partition_train = partition_dataset(data=train_files,
                                       num_partitions=get_world_size(),
                                       shuffle=False,
                                       even_divisible=False)[get_rank()]
    print("Number of files in training dataset partition for rank {}:{}".format(get_rank(), len(partition_train)), force=True)
    partition_val = partition_dataset(data=val_files,
                                       num_partitions=get_world_size(),
                                       shuffle=False,
                                       even_divisible=True)[get_rank()]
    print("Number of files in validation dataset partition for rank {}:{}".format(get_rank(), len(partition_val)), force=True)

    # Create datasets
    dataset_train = CacheDataset(
        data=partition_train,
        transform=train_transform,
        cache_rate=1.0,
        num_workers=cfg.n_workers_train
    )

    dataset_val = CacheDataset(
        data=partition_val,
        transform=val_transform,
        cache_rate=1.0,
        num_workers=cfg.n_workers_val
    )

    return dataset_train, dataset_val

def build_decathlon_cv_datasets(cfg, train_transform, val_transform):
    msd_dataset = CrossValidation(
        dataset_cls=DecathlonDataset,
        nfolds=cfg.cv_folds,
        root_dir=cfg.data_path,
        task=cfg.task,
        transform=train_transform,
        seed=cfg.seed,
        section='training',
        download=False,
        val_frac=0.0,
        cache_rate=1.0,
        num_workers=cfg.n_workers_train
    )
    if is_main_process():
        print("Number of files in total training dataset: {}".format(len(msd_dataset)))
    train_folds = list(range(cfg.cv_folds)).pop(cfg.curr_fold)
    dataset_train = msd_dataset.get_dataset(folds=train_folds)
    dataset_val = msd_dataset.get_dataset(folds=cfg.curr_fold, transform=val_transform)
    if is_main_process():
        print("Number of files in training cv split: {}".format(len(dataset_train)))
        print("Number of files in val cv split: {}".format(len(dataset_val)))
    return dataset_train, dataset_val





def build_train_and_val_datasets(cfg):
    train_transform = build_training_transforms(cfg)
    val_transform = build_validation_transforms(cfg)
    if cfg.distributed:
        dataset_train, dataset_val = build_decathlon_cv_datasets_dist(cfg, train_transform, val_transform)
    else:
        dataset_train, dataset_val = build_decathlon_cv_datasets(cfg, train_transform, val_transform)
    return dataset_train, dataset_val