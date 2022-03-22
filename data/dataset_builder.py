import os

import monai
from monai.data import (
    CacheDataset,
    load_decathlon_datalist,
)

from data.transforms import ScaleCubedIntensityRanged


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
    else:
        transforms.append(
            monai.transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
        ))
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
    transforms.append(
        monai.transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cfg.vol_size,
            pos=1,
            neg=0,
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
    else:
        transforms.append(
            monai.transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
        ))
    if cfg.t_crop_foreground:
        transforms.append(
            monai.transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image"
        ))
    transforms.append(
        monai.transforms.ToTensord(keys=["image", "label"])
    )
    return monai.transforms.Compose(transforms)


def build_dataset(data_path, transform, dstype='training', cache_num=24, num_workers=8):
    data_json = os.path.join(data_path, 'dataset_val.json')
    data_files = load_decathlon_datalist(data_json, True, dstype)
    dataset = CacheDataset(
        data=data_files,
        transform=transform,
        cache_num=cache_num,
        cache_rate=1.0,
        num_workers=num_workers
    )
    return dataset


def build_train_and_val_datasets(cfg):
    train_transform = build_training_transforms(cfg)
    train_dataset = build_dataset(cfg.data_path, train_transform, dstype='training', cache_num=8, num_workers=cfg.n_workers_train)
    val_transform = build_validation_transforms(cfg)
    val_dataset = build_dataset(cfg.data_path, val_transform, dstype='validation', cache_num=4, num_workers=cfg.n_workers_val)
    return train_dataset, val_dataset