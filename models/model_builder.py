import monai
from monai.networks.nets import UNETR

from models.backbones.swin_3d import SwinTransformer3D
from models.backbones.swin_nnformer import SwinTransformerNNFormer
from models.backbones.vit_mae import ViTMAE
from models.segmentors.nnformer_official.nnformer_official import nnFormer
from models.segmentors.swin_unetr import SwinUNETRCustom
from models.segmentors.swin_unetr_official import SwinUNETR
from models.segmentors.unetr import UNETRC
from models.segmentors.unetr_official import UNETROC


def build_model(cfg):
    if cfg.model == 'UNETR_Custom':

        encoder = ViTMAE(
                        vol_size=cfg.vol_size,
                        patch_size=cfg.patch_size,
                        in_chans=cfg.in_chans,
                        input_dim=cfg.input_dim,
                        qkv_bias=cfg.qkv_bias,
                        use_abs_pos_emb=cfg.abs_pos_emb,
                        use_rel_pos_bias=cfg.rel_pos_bias#,
                        #out_indices=[3, 7, 11]
                         )
        encoder.init_weights(cfg.pretrained)
        model = UNETRC(encoder, in_chans=cfg.in_chans, output_dim=cfg.output_dim)
    elif cfg.model == 'UNETR_Official':
        model = UNETR(
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            feature_size=16,
            hidden_size=cfg.hidden_dim,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    elif cfg.model == 'UNETR_OfficialCustom':
        encoder = ViTMAE(
                        vol_size=cfg.vol_size,
                        patch_size=cfg.patch_size,
                        in_chans=cfg.in_chans,
                        input_dim=cfg.input_dim,
                        qkv_bias=cfg.qkv_bias,
                        use_abs_pos_emb=cfg.abs_pos_emb,
                        use_rel_pos_bias=cfg.rel_pos_bias#,
                        #out_indices=[3, 7, 11]
                         )
        encoder.init_weights(cfg.pretrained)
        model = UNETROC(
            encoder,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            feature_size=16,
            hidden_size=cfg.hidden_dim,
            num_heads=12,
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    elif cfg.model == 'OfficialSwinUNETR':
        model = SwinUNETR(
            img_size=cfg.vol_size,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=cfg.hidden_dim,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
        )
    elif cfg.model == 'SwinUNETR':
        encoder = SwinTransformer3D(
            vol_size=cfg.vol_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.hidden_dim,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=cfg.window_size,
            qkv_bias=cfg.qkv_bias
        )
        model = SwinUNETRCustom(
            encoder,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            hidden_size=cfg.hidden_dim,
            patch_size=cfg.patch_size,
        )
    elif cfg.model == 'nnFormerUNETR':
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
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
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
            lcv_patch_voxel_mean=cfg.lcv_patch_voxel_mean
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
            input_downsampled=cfg.downsample_volume
        )
    elif cfg.model == 'nnFormer':
        model = nnFormer(
            crop_size=cfg.vol_size,
            embedding_dim=cfg.hidden_dim,
            input_channels=cfg.in_chans,
            num_classes=cfg.output_dim,
            deep_supervision=False,
            window_size=cfg.window_size,
            patch_size=cfg.patch_size
        )
    return model

