import monai

from models.backbones.focalnet_3d import FocalNet
from models.backbones.gc_vit_3d import GCViT
from models.backbones.segformer_backbone import MixVisionTransformer
from models.backbones.swin_nnformer import SwinTransformerNNFormer
from models.backbones.swinception import SwInception
from models.backbones.swindepth import SwinDepth
from models.segmentors.segformer_head import SegFormerHead
from models.segmentors.segformer_head_official import SegFormerHeadOfficial
from models.segmentors.swin_unetr import SwinUNETRCustom


def build_model(cfg):
    if cfg.model == 'nnFormerUNETR':
        if cfg.t_fixed_ct_intensity:
            transform = monai.transforms.ScaleIntensityRange(
                    a_min=cfg.t_ct_min,
                    a_max=cfg.t_ct_max,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                    )
        else:
            transform = monai.transforms.ScaleIntensityRangePercentiles(
                    lower=5,
                    upper=95,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                    relative=False
                    )
        encoder = SwinTransformerNNFormer(
            pretrain_img_size=cfg.vol_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.hidden_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            qkv_bias=cfg.qkv_bias,
            use_learned_cls_vectors=cfg.learned_cls_vectors,
            lcv_transform=transform,
            lcv_vector_dim=cfg.lcv_vector_dim,
            lcv_sincos_emb=cfg.lcv_sincos_emb,
            lcv_final_layer=cfg.lcv_final_layer,
            lcv_concat_vector=cfg.lcv_concat_vector,
            lcv_only=cfg.lcv_only,
            lcv_linear_comb=cfg.lcv_linear_comb,
            lcv_patch_voxel_mean=cfg.lcv_patch_voxel_mean,
            rel_pos_bias_affine=cfg.rel_pos_bias_affine,
            rel_crop_pos_emb=cfg.rel_crop_pos_emb,
            use_abs_pos_emb=cfg.abs_pos_emb,
            global_token=cfg.global_token
        )
        if cfg.learned_cls_vectors:
            print("LCV Intensity intervals: {}".format(encoder.lcv.intensity_intervals))
            print("LCV onehot intervals: {}".format(encoder.lcv.interval_onehot))
        model = SwinUNETRCustom(
            encoder,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            hidden_size=cfg.hidden_dim,
            patch_size=cfg.patch_size
        )
    elif cfg.model == 'SwInception':
        if cfg.t_fixed_ct_intensity:
            transform = monai.transforms.ScaleIntensityRange(
                    a_min=cfg.t_ct_min,
                    a_max=cfg.t_ct_max,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                    )
        else:
            transform = monai.transforms.ScaleIntensityRangePercentiles(
                    lower=5,
                    upper=95,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                    relative=False
                    )
        encoder = SwInception(
            pretrain_img_size=cfg.vol_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.hidden_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            qkv_bias=cfg.qkv_bias,
            mlp_ratio=cfg.mlp_ratio,
            use_learned_cls_vectors=cfg.learned_cls_vectors,
            lcv_transform=transform,
            lcv_vector_dim=cfg.lcv_vector_dim,
            lcv_sincos_emb=cfg.lcv_sincos_emb,
            lcv_final_layer=cfg.lcv_final_layer,
            lcv_concat_vector=cfg.lcv_concat_vector,
            lcv_only=cfg.lcv_only,
            lcv_linear_comb=cfg.lcv_linear_comb,
            lcv_patch_voxel_mean=cfg.lcv_patch_voxel_mean,
            rel_pos_bias_affine=cfg.rel_pos_bias_affine,
            rel_crop_pos_emb=cfg.rel_crop_pos_emb,
            use_abs_pos_emb=cfg.abs_pos_emb,
            global_token=cfg.global_token
        )
        if cfg.learned_cls_vectors:
            print("LCV Intensity intervals: {}".format(encoder.lcv.intensity_intervals))
            print("LCV onehot intervals: {}".format(encoder.lcv.interval_onehot))
        model = SwinUNETRCustom(
            encoder,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            hidden_size=cfg.hidden_dim,
            patch_size=cfg.patch_size,
        )
    elif cfg.model == 'SwinDepth':
        if cfg.t_fixed_ct_intensity:
            transform = monai.transforms.ScaleIntensityRange(
                a_min=cfg.t_ct_min,
                a_max=cfg.t_ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        else:
            transform = monai.transforms.ScaleIntensityRangePercentiles(
                lower=5,
                upper=95,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                relative=False
            )
        encoder = SwinDepth(
            pretrain_img_size=cfg.vol_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.hidden_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            qkv_bias=cfg.qkv_bias,
            mlp_ratio=cfg.mlp_ratio,
            use_learned_cls_vectors=cfg.learned_cls_vectors,
            lcv_transform=transform,
            lcv_vector_dim=cfg.lcv_vector_dim,
            lcv_sincos_emb=cfg.lcv_sincos_emb,
            lcv_final_layer=cfg.lcv_final_layer,
            lcv_concat_vector=cfg.lcv_concat_vector,
            lcv_only=cfg.lcv_only,
            lcv_linear_comb=cfg.lcv_linear_comb,
            lcv_patch_voxel_mean=cfg.lcv_patch_voxel_mean,
            rel_pos_bias_affine=cfg.rel_pos_bias_affine,
            rel_crop_pos_emb=cfg.rel_crop_pos_emb,
            use_abs_pos_emb=cfg.abs_pos_emb,
            global_token=cfg.global_token
        )
        if cfg.learned_cls_vectors:
            print("LCV Intensity intervals: {}".format(encoder.lcv.intensity_intervals))
            print("LCV onehot intervals: {}".format(encoder.lcv.interval_onehot))
        model = SwinUNETRCustom(
            encoder,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            hidden_size=cfg.hidden_dim,
            patch_size=cfg.patch_size,
        )
    elif cfg.model == 'SwinSegFormer':
        encoder = SwinTransformerNNFormer(
            pretrain_img_size=cfg.vol_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.hidden_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            qkv_bias=cfg.qkv_bias,
            use_abs_pos_emb=cfg.abs_pos_emb
        )
        model = SegFormerHead(
            encoder=encoder,
            in_channels=[cfg.hidden_dim * 2**i for i in range(0, len(cfg.depths)+1)],
            num_classes=cfg.output_dim
        )
    elif cfg.model == 'SegFormer3D':
        encoder = MixVisionTransformer(
            img_size=cfg.vol_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.hidden_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            sr_ratios=[8, 4, 2, 1],
            qkv_bias=cfg.qkv_bias
        )
        model = SegFormerHeadOfficial(
            encoder=encoder,
            in_channels=[cfg.hidden_dim * 2**i for i in range(0, len(cfg.depths))],
            num_classes=cfg.output_dim
        )
    elif cfg.model == 'GCViTUNETR':
        encoder = GCViT(
            resolution=cfg.vol_size,
            in_chans=cfg.in_chans,
            dim=cfg.hidden_dim,
            mlp_ratio=3.,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            qkv_bias=cfg.qkv_bias,
        )
        model = SwinUNETRCustom(
            encoder,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            hidden_size=cfg.hidden_dim,
            patch_size=cfg.patch_size,
        )
    elif cfg.model == 'FocalNetUNETR':
        encoder = FocalNet(
            pretrain_img_size=cfg.vol_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.hidden_dim,
            depths=cfg.depths,
            focal_windows=cfg.window_size
        )
        model = SwinUNETRCustom(
            encoder,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            hidden_size=cfg.hidden_dim,
            patch_size=cfg.patch_size,
        )
    return model

