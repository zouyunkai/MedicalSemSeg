from math import ceil
import operator
import functools

import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from torch import nn


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LRGAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_local = nn.Linear(dim, dim * 3, qkv_bias)
        self.qkv_region = nn.Linear(dim, dim * 3, qkv_bias)
        self.qkv_global = nn.Linear(dim, dim * 3, qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_local = nn.Linear(dim, dim)
        self.proj_region = nn.Linear(dim, dim)
        self.proj_global = nn.Linear(dim, dim)

    def forward(self, local_tokens, region_tokens, global_token):

        B, N_local, C = local_tokens.shape
        _, N_region, _ = region_tokens.shape

        qkv_local = self.qkv_local(local_tokens).reshape(B, N_local, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_local, k_local, v_local = qkv_local[0], qkv_local[1], qkv_local[2]  # make torchscript happy (cannot use tensor as tuple)

        qkv_region = self.qkv_region(region_tokens).reshape(B, N_region, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_region, k_region, v_region = qkv_region[0], qkv_region[1], qkv_region[2]  # make torchscript happy (cannot use tensor as tuple)

        qkv_global = self.qkv_global(global_token).reshape(B, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_global, k_global, v_global = qkv_global[0], qkv_global[1], qkv_global[2]  # make torchscript happy (cannot use tensor as tuple)

        q_local = q_local * self.scale
        q_region = q_region * self.scale
        q_global = q_global * self.scale

        q = torch.cat([q_local, q_region, q_global], dim=1)
        k = torch.cat([k_local, k_region, k_global], dim=1)
        v = torch.cat([v_local, v_region, v_global], dim=1)

        attn = (q @ k.transpose(-2, -1))

        x = (attn @ v).transpose(1, 2).reshape(B, (N_local + N_region + 1), -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LRGTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 local_resolution,
                 region_resolution,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.dim = dim
        self.local_resolution = local_resolution
        self.region_resolution = region_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = LRGAttention(
            dim,
            local_resolution=local_resolution,
            region_resolution=region_resolution,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, local_tokens, region_tokens, global_token):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return local_tokens, region_tokens, global_token

class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 local_resolution,
                 region_resolution,
                 depth,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 ):
        super().__init__()
        self.depth = depth

        self.blocks = nn.ModuleList([
            LRGTransformerBlock(
                dim=dim,
                local_resolution=local_resolution,
                region_resolution=region_resolution,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample_local = downsample(dim=dim, norm_layer=norm_layer)
            self.downsample_region = downsample(dim=dim, norm_layer=norm_layer)
            self.downsample_global = nn.Linear(dim, dim * 2)
            self.downsample = True
        else:
            self.downsample = None

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            local_tokens = self.downsample_local(local_tokens)
            region_tokens = self.downsample_region(region_tokens)
            global_token = self.downsample_global(global_token)
        return local_tokens, region_tokens, global_token

class LRGFormer(nn.Module):

    def __init__(self,
                 vol_size=512,
                 local_size=24,
                 region_size=64,
                 embed_dim=96,
                 in_chans=1,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                ):
        super().__init__()

        self.num_layers = len(depths)
        self.vol_size = to_3tuple(vol_size)
        self.local_size = to_3tuple(local_size)
        self.n_local_tokens = functools.reduce(operator.mul, self.local_size)
        self.region_size = to_3tuple(region_size)
        self.n_region_tokens = functools.reduce(operator.mul, self.region_size)
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed_local = PatchEmbed3D(
            vol_size=self.local_size,
            patch_size=(1, 1, 1),
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_region = PatchEmbedRegion(
            region_size=self.region_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_global = PatchEmbedGlobal(
            vol_size=self.vol_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                local_resolution=(
                    self.local_size[0] // 2 ** i_layer,
                    self.local_size[1] // 2 ** i_layer,
                    self.local_size[2] // 2 ** i_layer),
                region_resolution=(
                    self.region_size[0] // 2 ** i_layer,
                    self.region_size[1] // 2 ** i_layer,
                    self.region_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** (i+1)) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    def forward(self, x, coords):
        outputs = []

        local_tokens = self.patch_embed_local(x, coords)
        region_tokens = self.patch_embed_region(x)
        global_token = self.patch_embed_global(x)

        outputs.append(local_tokens)

        local_tokens = local_tokens.flatten(2).transpose(1, 2).contiguous()
        region_tokens = region_tokens.flatten(2).transpose(1, 2).contiguous()
        global_token = global_token.flatten(2).tranpose(1, 2).contiguous()

        x = torch.cat([local_tokens, region_tokens, global_token], dim=-1)


        for i in range(self.num_layers):
            layer = self.layers[i]
            x = layer(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                local_tokens = x[:,0:self.n_local_tokens,:]
                out = norm_layer(local_tokens)
                Ls, Lh, Lw = layer.local_resolution
                out = out.view(-1, Ls, Lh, Lw, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()

                outputs.append(out)


