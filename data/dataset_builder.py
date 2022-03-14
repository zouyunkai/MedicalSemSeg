import monai
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from transforms import ScaleCubedIntensityRanged

def build_training_transforms(cfg):
    transforms = [
        monai.LoadImaged(keys=["image", "label"]),
        monai.AddChanneld(keys=["image", "label"])
    ]
    if cfg.t_norm_voxel_spacings:
        transforms.append(
            monai.Spacingd(
                keys=["image", "label"],
                pixdim=cfg.t_voxel_spacings,
                mode=("bilinear", "nearest"),
            ))
    if cfg.t_cubed_ct_intensity:
        transforms.append(
            monai.ScaleCubedIntensityRanged(
                keys=["image"],
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
        ))
    else:
        transforms.append(
            monai.ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
        ))
    if cfg.t_crop_foreground:
        transforms.append(
            monai.CropForegroundd(
                keys=["image", "label"],
                source_key="image"
        ))
    transforms.append(
        monai.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cfg.input_size,
            pos=1,
            neg=1,
            num_samples=cfg.n_samples,
            image_key="image",
            image_threshold=0,
        )
    )
    transforms.append(
        monai.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=cfg.t_flip_prob,
        )
    )
    transforms.append(
        monai.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=cfg.t_flip_prob,
        )
    )
    transforms.append(
        monai.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=cfg.t_flip_prob,
        )
    )
    transforms.append(
        monai.RandRotate90d(
            keys=["image", "label"],
            prob=cfg.t_rot_prob,
            max_k=3,
        )
    )
    transforms.append(
        monai.RandShiftIntensityd(
            keys=["image"],
            offsets=cfg.t_intensity_sift_os,
            prob=cfg.t_intensity_shift_prob,
        )
    )
    transforms.append(
        monai.ToTensord(keys=["image", "label"])
    )
    return monai.transforms.Compose(transforms)

def build_validation_transforms(cfg):

def build_validation_dataset(cfg):

def build_training_dataset(cfg):
