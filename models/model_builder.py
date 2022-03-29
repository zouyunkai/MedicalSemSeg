from monai.networks.nets import UNETR

from models.backbones.vit_mae import ViTMAE
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
            hidden_size=768,
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
            hidden_size=768,
            num_heads=12,
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    return model
