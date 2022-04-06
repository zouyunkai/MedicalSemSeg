import os

import monai
from monai.data import (
    CacheDataset,
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
                source_key="label"
        ))
    if cfg.t_spatial_pad:
        transforms.append(
            monai.transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=cfg.vol_size,
        ))
    '''
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
    '''
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
    data_json = os.path.join(data_path, 'dataset_val_cancer.json')
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
    print("Number of files in training SmartCacheDataset for rank {}:{}".format(get_rank(), len(dataset)), force=True)
    return dataset

def build_val_dataset(data_path, transform, dstype='validation', cache_rate=1.0, num_workers=4):
    data_json = os.path.join(data_path, 'dataset_val_cancer.json')
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


def build_train_and_val_datasets(cfg):
    train_transform = build_training_transforms(cfg)
    train_dataset = build_train_dataset(cfg.data_path, train_transform, dstype='training', num_workers=cfg.n_workers_train)
    val_transform = build_validation_transforms(cfg)
    val_dataset = build_val_dataset(cfg.data_path, val_transform, dstype='validation', num_workers=cfg.n_workers_val)
    return train_dataset, val_dataset