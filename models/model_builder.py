from models.backbones.vit_mae import ViTMAE
from models.segmentors.unetr import UNETR


def build_model(cfg):
    encoder = ViTMAE(
                    vol_size=cfg.vol_size,
                    patch_size=cfg.patch_size,
                    in_chans=cfg.in_chans,
                    input_dim=cfg.input_dim,
                    qkv_bias=cfg.qkv_bias,
                    use_abs_pos_emb=cfg.abs_pos_emb,
                    use_rel_pos_bias=cfg.rel_pos_bias
                     )
    model = UNETR(encoder, output_dim=cfg.output_dim)
    return model