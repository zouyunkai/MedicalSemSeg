# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn=nn.GELU, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)
        self.act = act_fn()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, encoder, in_channels, num_classes, dropout_ratio=0.1, embedding_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.encoder = encoder

        self.in_channels = [in_channels * 2**i for i in range(0, 5)]
        c0_in_channels, c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=embedding_dim)

        self.linear_fuse_0 = BasicConv3d(
            in_channels=embedding_dim*2,
            out_channels=embedding_dim,
            kernel_size=1
        )

        self.linear_fuse_1 = BasicConv3d(
            in_channels=embedding_dim*2,
            out_channels=embedding_dim,
            kernel_size=1
        )

        self.linear_fuse_2 = BasicConv3d(
            in_channels=embedding_dim*2,
            out_channels=embedding_dim,
            kernel_size=1
        )

        self.linear_fuse_3 = BasicConv3d(
            in_channels=embedding_dim*2,
            out_channels=embedding_dim,
            kernel_size=1
        )


        self.dropout = nn.Dropout3d(dropout_ratio)

        self.linear_pred = nn.Conv3d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        org_shape = inputs[0].size()[2:]
        x = self.encoder(inputs)
        c0, c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w, d = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
        _c4 = nn.functional.interpolate(_c4, size=c3.size()[2:],mode='trilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
        _c3 = self.linear_fuse_3(torch.cat([_c4, _c3], dim=1))
        _c3 = nn.functional.interpolate(_c3, size=c2.size()[2:],mode='trilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
        _c2 = self.linear_fuse_2(torch.cat([_c3, _c2], dim=1))
        _c2 = nn.functional.interpolate(_c2, size=c1.size()[2:],mode='trilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])
        _c1 = self.linear_fuse_1(torch.cat([_c2, _c1], dim=1))
        _c1 = nn.functional.interpolate(_c1, size=c0.size()[2:], mode='trilinear', align_corners=False)

        _c0 = self.linear_c0(c0).permute(0, 2, 1).reshape(n, -1, c0.shape[2], c0.shape[3], c0.shape[4])
        _c = self.linear_fuse_0(torch.cat([_c1, _c0], dim=1))
        _c = nn.functional.interpolate(_c, size=org_shape, mode='trilinear', align_corners=False)

        #_c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)

        #x = nn.functional.interpolate(x, size=self.patch_resolution, mode='trilinear', align_corners=False)

        x = self.linear_pred(x)

        #x = nn.functional.interpolate(x, size=self.input_resolution, mode='trilinear', align_corners=False)

        return x