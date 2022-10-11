# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Tuple, Union

import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep


class SwinUNETRCustom(nn.Module):
    """
    Swin-UNETR based on: "Yucheng et al.,
    Self-Supervised Pre-Training of Swin Transformers
for 3D Medical Image Analysis"
    """

    def __init__(
        self,
        encoder,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int] = [96, 96, 96],
        hidden_size: int = 48,
        patch_size: Union[Sequence[int], int] = [2, 2, 2],
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")


        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.img_size = img_size
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.hidden_size = hidden_size
        self.classification = False
        self.encoder = encoder
        self.fl_out_size = tuple(img_d // (p_d*(2**4)) for img_d, p_d in zip(self.img_size, self.patch_size))

        self.unet_encoders = nn.ModuleList()
        self.unet_decoders = nn.ModuleList()

        encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        decoder0 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            upsample_kernel_size=self.patch_size,
            norm_name=norm_name,
            res_block=True,
        )

        self.unet_encoders.append(encoder0)
        self.unet_encoders.append(encoder1)
        self.unet_decoders.append(decoder0)

        for i_layer in range(self.encoder.num_layers):
            enc = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size * 2 ** (i_layer + 1),
                out_channels=hidden_size * 2 ** (i_layer + 1),
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )
            self.unet_encoders.append(enc)

            dec = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size * 2 ** (i_layer + 1),
                out_channels=hidden_size * 2 ** i_layer,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
            self.unet_decoders.append(dec)

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=hidden_size, out_channels=out_channels)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        z = self.encoder(x_in)

        x = self.unet_decoders[-1](self.unet_encoders[-1](z[-1]), self.unet_encoders[-2](z[-2]))
        for i in range(1, self.encoder.num_layers):
            enc = self.unet_encoders[-(i+2)]
            dec = self.unet_decoders[-(i+1)]
            x = dec(x, enc(z[-(i+2)]))
        x = self.unet_decoders[0](x, self.unet_encoders[0](x_in[0]))
        return self.out(x)
